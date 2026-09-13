"""MirobodyTargetAgent —— 被测系统是一个自部署的 mirobody 实例

注册名称: "mirobody"

补的是 mirobody README 里那句话的另一半。它写着 "we don't say 'trust us', we ship
the eval"，但本仓库原有的 target agent 里没有一个打 mirobody —— 有基模、有几个 RAG
系统，唯独没有它自己。也就是说这套 eval 从没考过 mirobody：跑出来的分数证明的是
ESL-Bench 这个基准好，不是 mirobody 好。

实现流程:
1. 拿 user_email 去 `health_app_user` 换数字 id
2. 用部署自己的 `JWT_KEY` 给那个 id 签一张 access token
3. `POST {base_url}/api/chat`，按 SSE 累积 `type=="reply"` 的分片
4. 顺带记录 `queryTitle`（agent 调了哪些工具），写进 session_info 便于回看

**为什么自己签票而不走登录**：mirobody 的登录只认 `EMAIL_PREDEFINE_CODES` 里的邮箱
（没配 SMTP 时），而 seed 出来的合成用户（`user5086@demo`）不在那份名单里，**根本没有
验证码可用**。所以签票不是抄近路，是唯一可行路径。要签票就得读部署的 JWT_KEY，也就
必须能 import 目标部署的 mirobody —— 这跟 seed 的前提一致（见
`generator/eslbench/seed_mirobody.py`）。

前置条件:
- `uv sync --extra mirobody --python 3.12`（引擎要 3.12+，本项目自身跑 3.11，
  所以这个 extra 在 3.11 上什么都不装），且配置指向要评测的那个部署
- 该部署的 HTTP 服务在跑（默认 http://localhost:18080）
- 用户已 seed 过: `python -m generator.eslbench.seed_mirobody --users user5086@demo`

基础设施参数从环境变量读，不走 config:
- `MIROBODY_CONFIG`    被测部署的 `config.{ENV}.yaml` 绝对路径。不给就按当前工作
                       目录找，而 runner 的工作目录是本仓库、不是那个部署 —— 那样
                       会静默退回内置默认值，错误信息指向一个不存在的库名
- `MIROBODY_BASE_URL`  被测部署地址（默认 http://localhost:18080）
- `MIROBODY_TIMEOUT`   单轮超时秒数。不给就用框架统一口径 `AGENT_LLM_TIMEOUT`
                       （默认 840）—— 一轮是一次 `/api/chat`，部署侧在里面跑完
                       整个 agent 循环（重题实测 61 次 LLM 调用 / 567s），所以
                       这个预算必须比单次调用的大，而不是更小
- `MIROBODY_PROVIDER`  覆盖 agent 的 LLM provider。留空用部署自己的默认值 ——
                       它按哪个 key 可用来选（OPENROUTER 走 claude-sonnet，
                       DASHSCOPE 走 qwen），所以配好了 key 的部署通常不需要设这个。
                       部署解不开自己那份密钥时（`CONFIG_ENCRYPTION_KEY` 缺失），
                       它会把密文当 token 发出去，agent 报的是上游 401 而不是配置错
"""

import json
import logging
import os
import time
import uuid
from typing import Any, Literal, Optional

import aiohttp
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.ai import UsageMetadata
from pydantic import BaseModel, ConfigDict, Field

from evaluator.core.interfaces.abstract_target_agent import AbstractTargetAgent
from evaluator.core.schema import SessionInfo, TargetAgentReaction, TestAgentAction
from evaluator.utils.config import get_agent_llm_timeout

logger = logging.getLogger(__name__)

# JWT 有效期：够跑完一整轮评测，又不至于签出一张长期有效的凭证。
_TOKEN_TTL_SECONDS = 6 * 60 * 60

# mirobody 的 chat 是 SSE，每个事件是 `data: {json}\n\n`。
_SSE_PREFIX = "data:"


# ============================================================
# 配置模型
# ============================================================


class MirobodyTargetInfo(BaseModel):
    """自部署 mirobody 被测目标配置"""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "_comment": "默认配置 — 打本地 mirobody 的 DeepAgent",
                    "type": "mirobody",
                    "agent": "Deep",
                    "user_email": "user5086@demo",
                },
            ],
        },
    )
    type: Literal["mirobody"] = Field(description="目标类型")
    agent: Literal["Deep", "Mix", "Base"] = Field("Deep", description="mirobody agent 类型")
    user_email: Optional[str] = Field(None, description="被问的用户邮箱（需已 seed 进该部署）")
    provider: str = Field("", description="覆盖 agent 的 LLM provider，空则用部署默认")
    language: str = Field("en", description="响应语言")
    timezone: str = Field("UTC", description="时区")


# ============================================================
# Agent 实现
# ============================================================


class MirobodyTargetAgent(AbstractTargetAgent, name="mirobody", params_model=MirobodyTargetInfo):
    """自部署 mirobody 实例（HTTP + SSE）"""

    _display_meta = {
        "icon": (
            "M12 2.25c-5.385 0-9.75 4.365-9.75 9.75s4.365 9.75 9.75 9.75 9.75-4.365"
            " 9.75-9.75S17.385 2.25 12 2.25zm0 3a6.75 6.75 0 016.75 6.75A6.75 6.75 0"
            " 0112 18.75 6.75 6.75 0 015.25 12A6.75 6.75 0 0112 5.25zm0 3a3.75 3.75 0"
            " 100 7.5 3.75 3.75 0 000-7.5z"
        ),
        "color": "#0ea5e9",
        "features": ["自部署", "mirobody", "SSE"],
    }
    _cost_meta = {
        # 部署侧自己调模型，token 成本不经本进程。
        "est_input_tokens": 0,
        "est_output_tokens": 0,
    }

    def __init__(self, target_config: MirobodyTargetInfo, history: list[BaseMessage] | None = None):
        super().__init__(target_config, history=history)
        self.config: MirobodyTargetInfo = target_config

        self.base_url = os.environ.get("MIROBODY_BASE_URL", "http://localhost:18080").rstrip("/")
        # 默认取框架统一口径（`AGENT_LLM_TIMEOUT`），而不是本模块自己的字面量。
        #
        # 上一版默认 300s，比框架给「单次 LLM 调用」的预算还小 —— 而这里一轮是
        # 一次 `/api/chat`，部署侧在里面自己跑完整个 agent 循环:跨全量统计的题
        # 实测 61 次 LLM 调用 + 41 次沙箱计算、耗时 567s。所以 300s 不是配得紧,
        # 是量级不对,后果是重题被判 0 分而不是被判错 —— 同一道题在 300s 下得
        # 0.00、在 900s 下得 0.89,一个随评测方配置变化的数不是被测系统的属性。
        #
        # `MIROBODY_TIMEOUT` 保留:它和 `AGENT_LLM_TIMEOUT` 是两个粒度(整轮 vs
        # 单次调用),不是重复的旋钮。
        self.timeout = float(os.environ.get("MIROBODY_TIMEOUT") or get_agent_llm_timeout())
        # provider 属于"打哪个部署、用它哪个模型"这类基础设施参数，从 env 读，
        # 不写进题库 —— 题库不该锚定某个模型。config 里显式给的优先。
        self.provider = target_config.provider or os.environ.get("MIROBODY_PROVIDER", "")

        # 同一 case 内多轮共用一个 session，让部署侧的会话记忆生效。
        self.session_id = f"holyeval-{uuid.uuid4().hex[:16]}"

        self._token: Optional[str] = None
        self._user_id: Optional[str] = None
        self._http: Optional[aiohttp.ClientSession] = None
        self._tools_used: list[str] = []

        self._cost = UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)

        # 预加载历史（history 不经过对话循环，直接作为上下文回放）
        self._history_messages: list[dict[str, str]] = []
        for msg in self.history:
            if isinstance(msg, HumanMessage):
                self._history_messages.append({"role": "user", "content": str(msg.content)})
            elif isinstance(msg, AIMessage):
                self._history_messages.append({"role": "assistant", "content": str(msg.content)})

    # ---------------- 属性 ----------------

    @property
    def rate_limit_key(self) -> str | None:
        # 打的是自己的部署，不需要限速。
        return None

    @property
    def cost(self) -> UsageMetadata:
        return self._cost

    def get_session_info(self) -> SessionInfo:
        # has_user_data=True：被测部署确实持有该用户的健康档案（seed 灌进去的），
        # 这正是这个 target 与 llm_api 之类基线的区别。
        return SessionInfo(
            user_id=self._user_id or "",
            user_token=self._token or "",
            has_user_data=True,
        )

    # ---------------- 鉴权 ----------------

    @staticmethod
    async def _ensure_mirobody_config() -> None:
        """确保被测部署的 mirobody 配置已加载。

        本仓库的 runner 初始化的是自己的配置，不会碰 mirobody 的。而
        `execute_query` 和 JWT 签票都要读 mirobody 的配置（PG 连接、`JWT_KEY`），
        没初始化就是 `ValueError: no configuration found`。

        实现在 `evaluator.utils.mirobody_config`，与 seeder 共用一份 —— 这两处
        必须一致，而它们曾经不一致：seeder 调的是无参数的 `Config.init()`，于是
        README 里「先 seed 再评测」的流程，第一步就连到了一个谁也没配过的库
        （名字取自操作系统用户），而它后面那步评测却是好的。
        """
        from evaluator.utils.mirobody_config import ensure_mirobody_config

        await ensure_mirobody_config()

    async def _resolve_user_id(self, email: str) -> str:
        """邮箱 → `health_app_user.id`。查不到就报错，不静默当成"没有数据"。"""
        from mirobody.utils import execute_query

        rows = await execute_query(
            "SELECT id FROM health_app_user WHERE email = :email AND is_del = false",
            {"email": email},
        )
        if not rows:
            raise RuntimeError(
                f"该部署里没有用户 {email!r}。先 seed:\n"
                f"    python -m generator.eslbench.seed_mirobody --users {email}"
            )
        return str(rows[0]["id"])

    def _mint_token(self, user_id: str) -> str:
        """用部署的 JWT_KEY 给 *user_id* 签一张 access token。

        `mirobody/server/auth.py` 只从 `sub` 取用户 id（要求是正整数字符串），
        HS256 验签，校验 `exp`。所以最小可用 claim 就是 sub + exp。
        """
        import jwt
        from mirobody.utils.config import global_config

        key = global_config().get("JWT_KEY")
        if not key:
            raise RuntimeError(
                "该部署没有配 JWT_KEY，无法为合成用户签票。"
                "`deploy.sh` 生成的 config.{env}.yaml 里应当有这一项。"
            )

        now = int(time.time())
        return jwt.encode(
            {
                "sub": str(user_id),
                "iat": now,
                "exp": now + _TOKEN_TTL_SECONDS,
                "token_type": "oauth_access_token",
            },
            key,
            algorithm="HS256",
        )

    async def _ensure_initialized(self) -> None:
        if self._token is not None:
            return
        if not self.config.user_email:
            raise RuntimeError(
                "未配置 user_email，无法确定要问哪个用户。题库需要在 "
                "`user.target_overrides.mirobody.user_email` 里带上它。"
            )

        await self._ensure_mirobody_config()
        self._user_id = await self._resolve_user_id(self.config.user_email)
        self._token = self._mint_token(self._user_id)
        logger.info(
            "[mirobody] %s → user_id=%s @ %s",
            self.config.user_email,
            self._user_id,
            self.base_url,
        )

    # ---------------- HTTP ----------------

    async def _get_http(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self._http

    async def _close_http(self) -> None:
        if self._http and not self._http.closed:
            await self._http.close()
        self._http = None

    async def _call_chat(self, question: str) -> str:
        """POST /api/chat，按 SSE 累积回复文本。"""
        client = await self._get_http()

        payload: dict[str, Any] = {
            "question": question,
            "agent": self.config.agent,
            "session_id": self.session_id,
            "trace_id": uuid.uuid4().hex,
            "language": self.config.language,
            "timezone": self.config.timezone,
            "file_list": [],
        }
        if self.provider:
            payload["provider"] = self.provider

        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        reply: list[str] = []
        error = ""

        async with client.post(f"{self.base_url}/api/chat", json=payload, headers=headers) as resp:
            if resp.status == 401:
                raise RuntimeError(
                    "/api/chat 返回 401 —— 签出的票被拒。通常是 eval 侧读到的 JWT_KEY "
                    "和被测部署实际在用的不是同一个（ENV / config.{env}.yaml 对不上）。"
                )
            if resp.status != 200:
                body = (await resp.text())[:400]
                raise RuntimeError(f"/api/chat 返回 {resp.status}: {body}")

            async for raw in resp.content:
                line = raw.decode("utf-8", "replace").strip()
                if not line or not line.startswith(_SSE_PREFIX):
                    continue
                try:
                    chunk = json.loads(line[len(_SSE_PREFIX):].strip())
                except json.JSONDecodeError:
                    continue

                kind = chunk.get("type")
                content = chunk.get("content")

                if kind == "reply" and content:
                    reply.append(str(content))
                elif kind == "queryTitle" and content:
                    self._tools_used.append(str(content))
                elif kind == "error":
                    error = str(content or "unknown error")
                elif kind == "end":
                    break

        if error and not reply:
            raise RuntimeError(f"mirobody agent 报错: {error}")
        if error:
            logger.warning("[mirobody] 回复中带错误分片，仍采用已收到的文本: %s", error)

        return "".join(reply)

    # ---------------- 主流程 ----------------

    async def _generate_next_reaction(self, test_action: Optional[TestAgentAction]) -> TargetAgentReaction:
        try:
            await self._ensure_initialized()

            if test_action is None:
                return TargetAgentReaction(
                    type="message",
                    message_list=[{"content": "Hello, I am your health assistant"}],
                )

            question = self._extract_user_input(test_action)
            if not question.strip():
                return TargetAgentReaction(type="message", message_list=[{"content": ""}])

            before = len(self._tools_used)
            try:
                text = await self._call_chat(question)
            finally:
                # 框架没有 teardown 钩子，agent 对象被 runner 一直持有，所以每轮结束
                # 就关掉 HTTP 会话，否则批量跑分会攒下一堆 "Unclosed client session"。
                # 打的是本地部署，重建连接的代价可以忽略。
                await self._close_http()
            # 只带本轮新增的工具调用，便于逐轮回看 agent 用了什么。
            turn_tools = self._tools_used[before:]

            return TargetAgentReaction(
                type="message",
                message_list=[{"content": text}],
                tool_calls=[{"name": name} for name in turn_tools] or None,
            )

        except Exception as e:
            logger.error("[mirobody] 调用失败: %s", e, exc_info=True)
            await self._close_http()
            raise

    @staticmethod
    def _extract_user_input(test_action: TestAgentAction) -> str:
        if test_action.type == "semantic":
            return test_action.semantic_content or ""
        if test_action.type == "message":
            msg = test_action.message_content
            if isinstance(msg, dict):
                return msg.get("content", "")
            return ""
        return str(test_action.custom_content) if test_action.custom_content else ""
