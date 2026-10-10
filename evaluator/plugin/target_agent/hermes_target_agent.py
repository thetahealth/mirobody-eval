"""
HermesTargetAgent — Calls Hermes Agent via its OpenAI-compatible HTTP API

Registered name: "hermes"

Hermes Agent (https://github.com/nousresearch/hermes-agent) 是 Nous Research 的自主学习
AI Agent 框架，通过 gateway 暴露 OpenAI 兼容的 /v1/chat/completions 端点。
内部自动执行 tool calling loop（搜索、代码执行、记忆检索等），返回最终回答。

请求格式:
    POST /v1/chat/completions
    {
        "model": "hermes-agent",
        "messages": [{"role": "user", "content": "..."}],
        "stream": false
    }

响应格式:
    {
        "choices": [{"message": {"role": "assistant", "content": "..."}}],
        "usage": {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}
    }

模型探测:
    启动时通过 GET /v1/models 自动探测 Hermes 当前配置的模型名。
    用户在 Hermes 侧通过 `hermes model` 或修改 ~/.hermes/config.yaml 切换模型后，
    下一次创建 HermesTargetAgent 实例会自动读取到新模型名，无需修改 mirobody-eval 配置。

前置依赖:
    1. 安装 Hermes: curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
    2. 配置模型: hermes model
    3. 启动 API server: hermes gateway start（默认 http://127.0.0.1:8642）
    4. 在 .env 中配置 HERMES_API_BASE_URL（可选，默认 http://127.0.0.1:8642）

User-level params (HermesTargetInfo):
    system_prompt: System prompt (optional) — 作为 ephemeral system prompt 叠加在 Hermes 核心 prompt 之上

Infrastructure params (read from .env):
    HERMES_API_BASE_URL: API server 地址（默认 http://127.0.0.1:8642）
    HERMES_API_KEY: API key（可选，对应 Hermes 的 API_SERVER_KEY）
"""

import logging
import uuid
from typing import Literal, Optional

import aiohttp
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.ai import UsageMetadata
from pydantic import BaseModel, ConfigDict, Field

from evaluator.core.interfaces.abstract_target_agent import AbstractTargetAgent
from evaluator.core.schema import TargetAgentReaction, TestAgentAction
from evaluator.utils import paths
from evaluator.utils.config import get_config

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://127.0.0.1:8642"


class HermesTargetInfo(BaseModel):
    """Hermes Agent target config — calls Hermes via OpenAI-compatible API"""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"_comment": "Minimal config", "type": "hermes"},
                {"_comment": "With system prompt", "type": "hermes", "system_prompt": "You are a health assistant."},
                {
                    "_comment": "ESLBench (user_email via target_overrides)",
                    "type": "hermes",
                    "user_email": "user110@demo",
                },
            ]
        },
    )
    type: Literal["hermes"] = Field(description="Target type")
    model: Optional[str] = Field(None, description="Model name (optional, auto-detected from Hermes config if not set)")
    system_prompt: Optional[str] = Field(None, description="System prompt (layered on top of Hermes core prompt)")
    user_email: Optional[str] = Field(None, description="User email for data directory resolution (ESLBench etc.)")


class HermesTargetAgent(AbstractTargetAgent, name="hermes", params_model=HermesTargetInfo):
    """
    Hermes Agent Target — calls Hermes via OpenAI-compatible /v1/chat/completions

    Hermes 内部自动执行 tool calling loop，返回最终回答（final_response）。
    模型名通过 GET /v1/models 自动探测，跟随 Hermes 侧配置实时变化。
    """

    _display_meta = {
        "icon": (
            "M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0"
            " 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0"
            " 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25"
            " 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456"
            " 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25"
            " 2.25 0 00-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394"
            " 1.183a2.25 2.25 0 001.423 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z"
        ),
        "color": "#8b5cf6",
        "features": ["OpenAI Compat", "Tool Calling", "Self-improving"],
        "icon_url": "/static/images/agents/hermes.png",
    }
    _cost_meta = {
        "est_input_tokens": 0,  # Agent 内部调用 LLM，token 从 usage 汇总返回
        "est_output_tokens": 0,
    }

    def __init__(self, target_config: HermesTargetInfo, history: list[BaseMessage] | None = None):
        super().__init__(target_config, history=history)
        self.config: HermesTargetInfo = target_config

        # Infrastructure config
        self.base_url = get_config("HERMES_API_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")
        self._api_key: str | None = get_config("HERMES_API_KEY")

        # 如果有 user_email，解析 data_dir 并注入 system_prompt
        if self.config.user_email:
            user_dir = self.config.user_email.replace("@", "_AT_")
            data_dir = paths.user_data_dir("eslbench") / user_dir
            data_block = f"\n\nUser data directory: {data_dir}"
            if self.config.system_prompt:
                self.config = self.config.model_copy(update={"system_prompt": self.config.system_prompt + data_block})
            else:
                self.config = self.config.model_copy(update={"system_prompt": data_block.strip()})
            logger.info(
                "[HermesTargetAgent] user_dir=%s exists=%s",
                data_dir,
                data_dir.is_dir(),
            )
            if not data_dir.is_dir():
                logger.warning(
                    "[HermesTargetAgent] error_code=USER_DIR_NOT_FOUND user_dir=%s data_group=eslbench",
                    data_dir,
                )

        # Session management
        self._session_id = f"holyeval_{uuid.uuid4().hex}"

        # 将 history (List[BaseMessage]) 转为 OpenAI messages 格式
        self._conversation: list[dict[str, str]] = []
        for msg in self.history:
            if isinstance(msg, HumanMessage):
                self._conversation.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                self._conversation.append({"role": "assistant", "content": msg.content})

        # HTTP client
        self._http_client: aiohttp.ClientSession | None = None

        # Cost tracking — 如果 config.model 显式指定则使用，否则等 _detect_model 探测
        self._cost = UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)
        if self.config.model:
            self._model = self.config.model
            self._model_detected = True
        else:
            self._model = "hermes-agent"
            self._model_detected = False

    @property
    def cost(self) -> UsageMetadata:
        return self._cost

    @property
    def model(self) -> str:
        return self._model

    def _accumulate_cost(self, usage: dict[str, UsageMetadata] | None) -> None:
        from evaluator.utils.llm import accumulate_usage

        if usage:
            for model_name in usage:
                self._model = model_name
        self._cost = accumulate_usage(self._cost, usage)

    async def _get_http_client(self) -> aiohttp.ClientSession:
        if self._http_client is None or self._http_client.closed:
            # Hermes agent loop 可能跑很久（tool calling），给足超时
            self._http_client = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=600),
            )
        return self._http_client

    async def _close_http_client(self) -> None:
        if self._http_client and not self._http_client.closed:
            await self._http_client.close()
            self._http_client = None

    async def _detect_model(self) -> None:
        """通过 GET /v1/models 探测 Hermes 当前配置的模型名。

        Hermes API server 的 /v1/models 返回当前 config.yaml 配置的模型，
        用户通过 `hermes model` 切换后，下次探测自动获取新值。
        探测失败不影响功能，仅保留默认 "hermes-agent"。
        """
        if self._model_detected:
            return
        self._model_detected = True

        try:
            client = await self._get_http_client()
            headers: dict[str, str] = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            async with client.get(f"{self.base_url}/v1/models", headers=headers) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()
                models = data.get("data", [])
                if models:
                    detected = models[0].get("id", "")
                    if detected:
                        self._model = detected
                        logger.info("[HermesTargetAgent] Detected model: %s", detected)
        except Exception as e:
            logger.debug("[HermesTargetAgent] Model detection failed (non-fatal): %s", e)

    # 类级别标记 — 同一进程只检测一次
    _hermes_setup_checked: bool = False

    async def _ensure_hermes(self) -> None:
        """确保 Hermes Agent 已安装、配置并运行。首次调用时自动检测，必要时触发安装流程。"""
        if HermesTargetAgent._hermes_setup_checked:
            return
        HermesTargetAgent._hermes_setup_checked = True

        from evaluator.utils.hermes_setup import check_hermes_status, ensure_hermes_ready

        status = await check_hermes_status(self.base_url)
        if status.ready:
            logger.info("[HermesTargetAgent] Hermes 已就绪 (model=%s)", status.model_name)
            return

        logger.warning("[HermesTargetAgent] Hermes 未就绪: %s — 启动自动配置...", status.issues)
        await ensure_hermes_ready(self.base_url)

    async def _generate_next_reaction(self, test_action: TestAgentAction | None) -> TargetAgentReaction:
        # 首次调用时确保 Hermes 就绪 + 探测模型
        await self._ensure_hermes()
        await self._detect_model()

        if test_action is None:
            return TargetAgentReaction(
                type="message",
                message_list=[{"content": "Hello, I am Hermes Agent. How can I help you?"}],
            )

        user_text = self._extract_user_input(test_action)

        try:
            content, usage = await self._call_chat_completions(user_text)
        except Exception as e:
            logger.error("[HermesTargetAgent] API call failed: %s", e, exc_info=True)
            await self._close_http_client()
            raise

        self._accumulate_cost(usage)

        # 追加到对话历史
        self._conversation.append({"role": "user", "content": user_text})
        self._conversation.append({"role": "assistant", "content": content})

        return TargetAgentReaction(
            type="message",
            message_list=[{"content": content}],
            usage=usage,
        )

    async def _call_chat_completions(self, user_input: str) -> tuple[str, dict[str, UsageMetadata] | None]:
        """POST /v1/chat/completions（非 streaming）

        Returns:
            (assistant_content, usage_dict)
        """
        client = await self._get_http_client()

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        # 构建 messages
        messages: list[dict[str, str]] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.extend(self._conversation)
        messages.append({"role": "user", "content": user_input})

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }

        logger.debug(
            "[HermesTargetAgent] /v1/chat/completions: model=%s, %d messages, last_user=%s...",
            self._model,
            len(messages),
            user_input[:50],
        )

        async with client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"/v1/chat/completions returned {resp.status}: {text[:300]}")

            data = await resp.json()

        # 解析 response
        choices = data.get("choices", [])
        if not choices:
            raise Exception(f"Empty choices in response: {data}")

        content = choices[0].get("message", {}).get("content", "")

        # 解析 usage
        usage: dict[str, UsageMetadata] | None = None
        raw_usage = data.get("usage")
        if raw_usage:
            model_name = data.get("model", self._model)
            usage = {
                model_name: UsageMetadata(
                    input_tokens=raw_usage.get("prompt_tokens", 0),
                    output_tokens=raw_usage.get("completion_tokens", 0),
                    total_tokens=raw_usage.get("total_tokens", 0),
                )
            }
            logger.info(
                "[HermesTargetAgent] Turn cost: model=%s, in=%d, out=%d, total=%d",
                model_name,
                raw_usage.get("prompt_tokens", 0),
                raw_usage.get("completion_tokens", 0),
                raw_usage.get("total_tokens", 0),
            )

        # 从响应 header 更新 session_id（Hermes 返回 X-Hermes-Session-Id）
        resp_session_id = resp.headers.get("X-Hermes-Session-Id")
        if resp_session_id:
            self._session_id = resp_session_id

        return content, usage

    def _extract_user_input(self, test_action: TestAgentAction) -> str:
        if test_action.type == "semantic":
            return test_action.semantic_content or ""
        elif test_action.type == "message":
            msg = test_action.message_content
            if isinstance(msg, dict):
                return msg.get("content", "")
            return ""
        else:
            return str(test_action.custom_content) if test_action.custom_content else ""

    async def cleanup(self) -> None:
        await self._close_http_client()
        logger.debug("[HermesTargetAgent] Cleanup completed")
