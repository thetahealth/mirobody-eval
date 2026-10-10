"""
DygRagApiTargetAgent — DyG-RAG 动态图检索增强问答

注册名称: "dyg_rag_api"

基于 DyG-RAG (Dynamic Graph RAG) 论文的评测逻辑:
1. 加载用户数据 → 转换为 event_docs 格式（[PROFILE] + [DATA] 时序健康记录）
2. 通过 GraphRAG 引擎构建动态事件图 + 时序向量索引（首次自动构建，后续使用缓存）
3. 查询阶段: 时间/实体解析 → 时间加权向量检索 → 事件图随机游走 → Time-CoT → LLM 回答


源自: DyG-RAG 上游实现 的 GraphRAG 引擎
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.ai import UsageMetadata
from pydantic import BaseModel, ConfigDict, Field

from evaluator.core.interfaces.abstract_target_agent import AbstractTargetAgent
from evaluator.core.schema import (
    SessionInfo,
    TargetAgentReaction,
    TestAgentAction,
)
from evaluator.utils import paths
from evaluator.utils.config import get_agent_llm_timeout
from evaluator.utils.llm import BasicMessage

# 把 dyg vendor 的 GEMINI_CALL_TIMEOUT 默认值（120s）对齐到框架统一的 AGENT_LLM_TIMEOUT（默认 840s）。
# vendor 的 _llm.py 在 import 时读这个 env，所以必须在 evaluator.vendor.dyg_graphrag 被首次 import 之前
# （即此 plugin 模块顶层）写入；用 setdefault 不覆盖用户显式设置。
os.environ.setdefault("GEMINI_CALL_TIMEOUT", str(get_agent_llm_timeout()))

logger = logging.getLogger(__name__)

# 缓存固定写入仓库内 benchmark/data/<bench>/.dyg_work_dirs/，跟 user_data 物理分离（不挂 PVC）
_BENCHMARK_DATA_DIR = Path(__file__).resolve().parents[3] / "benchmark" / "data"

# GraphRAG 实例缓存 (key: user_dir path string)
_GRAPHRAG_CACHE: dict[str, Any] = {}
# 引用计数：DYG_RELEASE_AFTER_USE=1 时启用，用完即释放
_REF_COUNT: dict[str, int] = {}
# Per-user 锁：防止同一用户的并发首次建图撞车（一个 user 多 case 各自调 ainsert_typed → 重复抽事件）。
# 同一 user 的 N 个 case 在锁里串行：第一个建完 _GRAPHRAG_CACHE，后到的直接命中。
_GRAPHRAG_LOCKS: dict[str, asyncio.Lock] = {}


# ============================================================
# 数据转换: 用户数据 → DyG-RAG event_docs
# ============================================================


def _email_to_dir(email: str) -> str:
    """邮箱 → 目录名: user110@demo → user110_AT_demo"""
    return email.replace("@", "_AT_")


def _prepare_llm_chunks(user_dir: Path) -> list[str]:
    """将 profile + events + exam_indicator 拆分为需要 LLM 事件提取的 chunk 列表。

    返回 ~27 个 chunk（1 profile + ~20 events + ~6 exams），跳过 device_indicator。
    """
    from datetime import datetime as _dt

    chunks: list[str] = []

    # === Profile chunk ===
    profile_file = user_dir / "profile.json"
    if profile_file.exists():
        with open(profile_file, encoding="utf-8") as f:
            profile = json.load(f)
        parts: list[str] = ["[PROFILE]"]
        demographics = profile.get("demographics", {})
        health_profile = profile.get("health_profile", {})
        personality = profile.get("personality", "")
        if demographics:
            parts.append(json.dumps(demographics, ensure_ascii=False, indent=2))
        if health_profile:
            parts.append(f"\nHealth Profile:\n{json.dumps(health_profile, ensure_ascii=False, indent=2)}")
        if personality:
            parts.append(f"\nPersonality: {personality[:500]}")
        chunks.append("\n".join(parts))

    # === 读取 timeline ===
    timeline_file = user_dir / "timeline.json"
    if not timeline_file.exists():
        return chunks

    with open(timeline_file, encoding="utf-8") as f:
        tl = json.load(f)

    entries = tl.get("entries", []) if isinstance(tl, dict) else tl if isinstance(tl, list) else []
    if not entries:
        return chunks

    # 分类收集
    event_lines: list[str] = []
    exam_by_date: dict[str, list[str]] = {}  # date -> [indicator lines]

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        etype = entry.get("entry_type", "")
        time_str = entry.get("time", "")

        if etype == "event":
            ev = entry.get("event", {})
            line = (
                f"[{time_str}] Event: {ev.get('event_name', '?')}, "
                f"Type: {ev.get('event_type', '?')}, "
                f"Start: {ev.get('start_date', '?')}, "
                f"Duration: {ev.get('duration_days', '?')} days, "
                f"Interrupted: {ev.get('interrupted', False)}"
            )
            if ev.get("interrupted") and ev.get("interruption_date"):
                line += f", Interruption date: {ev['interruption_date']}"
            event_lines.append(line)
        elif etype == "exam_indicator":
            date_key = time_str[:10]
            indicator_line = f"{entry.get('indicator', '?')} = {entry.get('value', '?')} {entry.get('unit', '')}"
            exam_by_date.setdefault(date_key, []).append(indicator_line)

    # === Event chunks: 每 10 条 event 为一个 chunk ===
    group_size = 10
    for i in range(0, len(event_lines), group_size):
        batch = event_lines[i : i + group_size]
        chunks.append("[DATA] Health Events\n" + "\n".join(batch))

    # === Exam chunks: 按日期分组 ===
    for date_key in sorted(exam_by_date.keys()):
        indicator_lines = exam_by_date[date_key]
        chunks.append(f"[DATA] Exam Results on {date_key}\n" + "\n".join(indicator_lines))

    return chunks


def _prepare_device_events(user_dir: Path) -> list[dict]:
    """将 device_indicator 按 ISO 周聚合，直接生成 event node dict 列表（跳过 LLM 提取）。

    每个 dict 包含 sentence, timestamp, event_id, entities_involved 等字段，
    可直接写入事件图和向量数据库。
    """
    from datetime import datetime as _dt, timedelta

    from evaluator.vendor.dyg_graphrag._utils import compute_mdhash_id

    timeline_file = user_dir / "timeline.json"
    if not timeline_file.exists():
        return []

    with open(timeline_file, encoding="utf-8") as f:
        tl = json.load(f)

    entries = tl.get("entries", []) if isinstance(tl, dict) else tl if isinstance(tl, list) else []
    if not entries:
        return []

    # 按 ISO 周聚合 device_indicator
    device_by_week: dict[str, dict[str, list[tuple[float, str]]]] = {}  # week_key -> indicator -> [(value, unit)]
    week_first_day: dict[str, str] = {}  # week_key -> first day (YYYY-MM-DD)

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("entry_type") != "device_indicator":
            continue
        time_str = entry.get("time", "")
        day = time_str[:10]
        try:
            dt = _dt.fromisoformat(day)
            week_key = dt.strftime("%Y-W%W")
        except ValueError:
            continue
        name = entry.get("indicator", "?")
        val = entry.get("value")
        unit = entry.get("unit", "")
        if val is not None and val != "None" and isinstance(val, (int, float)):
            device_by_week.setdefault(week_key, {}).setdefault(name, []).append((val, unit))
            # 记录该周的第一天（ISO 周一）
            if week_key not in week_first_day:
                iso_year, iso_week, _ = dt.isocalendar()
                monday = _dt.fromisocalendar(iso_year, iso_week, 1)
                week_first_day[week_key] = monday.strftime("%Y-%m-%d")

    # 生成 event node dicts
    result: list[dict] = []
    for week_key in sorted(device_by_week.keys()):
        indicators = device_by_week[week_key]
        # 构建 sentence: 所有指标用 "; " 连接
        parts: list[str] = []
        sorted_names = sorted(indicators.keys())
        for name in sorted_names:
            values = [v for v, _ in indicators[name]]
            unit = indicators[name][0][1]
            m = sum(values) / len(values)
            mn = min(values)
            mx = max(values)
            parts.append(f"{name}: mean={m:.1f}, min={mn:.1f}, max={mx:.1f} {unit}")
        sentence = f"Device indicators for {week_key}: " + "; ".join(parts)
        timestamp = week_first_day.get(week_key, "")
        event_id = compute_mdhash_id(f"{sentence}-{timestamp}", prefix="event-")

        result.append(
            {
                "event_id": event_id,
                "sentence": sentence,
                "timestamp": timestamp,
                "entities_involved": sorted_names,
                "context": "",
                "source_id": f"device_weekly_{week_key}",
                "participants": "",
                "location": "",
            }
        )

    return result


# ============================================================
# Config Model
# ============================================================


class DygRagApiTargetInfo(BaseModel):
    """DyG-RAG 被测目标配置 — 动态图检索增强生成"""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "_comment": "默认配置（使用 Gemini 进行事件提取和 QA）",
                    "type": "dyg_rag_api",
                    "model": "gemini-3-pro-preview",
                    "data_group": "eslbench",
                    "user_email": "user110@demo",
                },
            ],
        },
    )
    type: Literal["dyg_rag_api"] = Field(description="目标类型")
    model: str = Field(
        "gemini-3-pro-preview",
        description="QA 生成模型（默认 gemini-3-pro-preview，用于最终答案生成）",
    )
    data_group: str = Field(description="数据目录（benchmark 名称，如 'eslbench'）")
    user_email: Optional[str] = Field(None, description="用户邮箱（映射到 .data/{user_dir}/）")
    top_k: int = Field(20, description="DyG-RAG 种子事件数量（用于图遍历的起始节点数）", ge=1, le=100)
    enable_graph_traversal: bool = Field(True, description="是否启用图随机游走（关闭则仅用向量检索）")


# ============================================================
# Target Agent
# ============================================================


class DygRagApiTargetAgent(
    AbstractTargetAgent,
    name="dyg_rag_api",
    params_model=DygRagApiTargetInfo,
):
    """DyG-RAG 动态图检索增强问答 — 基于时序事件图的健康数据 RAG

    流程:
    1. 首次查询时自动构建索引（LLM 事件提取 → 动态事件图 + 时序 VDB）
    2. 查询: 时间/实体解析 → 时间加权向量检索 → 图随机游走 → Time-CoT → LLM 回答
    """

    _display_meta = {
        "icon": (
            "M12 21a9.004 9.004 0 008.716-6.747M12 21a9.004 9.004 0 01-8.716-6.747M12 21c2.485 0"
            " 4.5-4.03 4.5-9S14.485 3 12 3m0 18c-2.485 0-4.5-4.03-4.5-9S9.515 3 12 3m0 0a8.997"
            " 8.997 0 017.843 4.582M12 3a8.997 8.997 0 00-7.843 4.582m15.686 0A11.953 11.953 0"
            " 0112 10.5c-2.998 0-5.74-1.1-7.843-2.918m15.686 0A8.959 8.959 0 0121 12c0 .778-.099"
            " 1.533-.284 2.253m0 0A17.919 17.919 0 0112 16.5a17.92 17.92 0 01-8.716-2.247m0"
            " 0A9.015 9.015 0 003 12c0-1.605.42-3.113 1.157-4.418"
        ),
        "color": "#10b981",
        "features": ["DyG-RAG", "动态事件图", "时序检索", "图遍历"],
    }
    _cost_meta = {
        "est_input_tokens": 8000,
        "est_output_tokens": 800,
    }

    def __init__(
        self,
        target_config: DygRagApiTargetInfo,
        history: list[BaseMessage] | None = None,
    ):
        super().__init__(target_config, history=history)
        self.config: DygRagApiTargetInfo = target_config

        # 对话历史
        self._conversation: list[BasicMessage] = []
        for msg in self.history:
            if isinstance(msg, HumanMessage):
                self._conversation.append(BasicMessage(role="user", content=msg.content))
            elif isinstance(msg, AIMessage):
                self._conversation.append(BasicMessage(role="assistant", content=msg.content))

        # 用户数据目录
        self._user_dir: Path | None = None
        if target_config.user_email:
            dir_name = _email_to_dir(target_config.user_email)
            self._user_dir = paths.user_data_dir(target_config.data_group) / dir_name
            logger.info(
                "[DyG-RAG] user_dir=%s exists=%s data_group=%s",
                self._user_dir,
                self._user_dir.is_dir(),
                target_config.data_group,
            )
            if not self._user_dir.is_dir():
                logger.warning(
                    "[DyG-RAG] error_code=USER_DIR_NOT_FOUND user_dir=%s data_group=%s",
                    self._user_dir,
                    target_config.data_group,
                )

        # GraphRAG 实例（延迟初始化）
        self._graphrag: Any = None
        self._cost = UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)

    async def _ensure_graphrag(self) -> None:
        """确保 GraphRAG 实例已初始化并完成索引构建"""
        if self._graphrag is not None:
            return

        cache_key = str(self._user_dir)
        release_mode = os.environ.get("DYG_RELEASE_AFTER_USE", "0") == "1"

        lock = _GRAPHRAG_LOCKS.get(cache_key)
        if lock is None:
            lock = asyncio.Lock()
            _GRAPHRAG_LOCKS[cache_key] = lock

        async with lock:
            # 锁内重新检查缓存：同一 user 排队的后到 case 直接命中
            if cache_key in _GRAPHRAG_CACHE:
                self._graphrag = _GRAPHRAG_CACHE[cache_key]
                logger.info("[DyG-RAG] 使用缓存的 GraphRAG 实例: %s", self._user_dir.name)
                if release_mode:
                    _REF_COUNT[cache_key] = _REF_COUNT.get(cache_key, 0) + 1
                return

            await self._build_graphrag(cache_key, release_mode)

    async def _build_graphrag(self, cache_key: str, release_mode: bool) -> None:
        """实际构建 GraphRAG 实例（在 _GRAPHRAG_LOCKS[cache_key] 锁内调用）"""
        from evaluator.vendor.dyg_graphrag import GraphRAG

        # 索引存储目录：固定写入仓库内 benchmark/data/<bench>/.dyg_work_dirs/，跟 user data 物理分离
        work_dir = _BENCHMARK_DATA_DIR / self.config.data_group / ".dyg_work_dirs" / self._user_dir.name
        work_dir.mkdir(parents=True, exist_ok=True)
        index_done_flag = work_dir / ".index_done"

        # 设置模型 — 注意 vendor 的 gemini_best_complete 在 query 阶段也会读 os.getenv("GEMINI_MODEL"),
        # 默认 fallback 是 gemini-3-pro-preview。所以必须在整个 plugin 生命周期里保持 GEMINI_MODEL=
        # config.model，不能在 GraphRAG() 构造完毕就 restore，否则 query 时会自动回到 pro。
        if self.config.model:
            os.environ["GEMINI_MODEL"] = self.config.model

        # 初始化 GraphRAG（自动读取 GEMINI_API_KEY / GOOGLE_API_KEY 等环境变量）
        # 禁用 cross-encoder（需要 PyTorch），改用 BM25 重排序
        graphrag_kwargs: dict[str, Any] = {
            "working_dir": str(work_dir),
            "enable_ce_rerank": False,
            "enable_bm25_reranking": True,
            "enable_graph_traversal": self.config.enable_graph_traversal,
            "embedding_func_max_async": 4,
            "embedding_batch_num": 10,
            "event_extract_max_gleaning": 1,
        }
        # 非 Gemini 模型（gpt-*, openrouter 等）走 vendor 的 openai_complete/openai_embedding 路径；
        # 默认 GraphRAG 的 best/cheap/embedding 都指向 Gemini，环境没法用时这里整体切到 OpenAI 兼容。
        if self.config.model and not self.config.model.startswith("gemini"):
            from evaluator.vendor.dyg_graphrag._llm import (
                openai_complete_if_cache,
                openai_embedding,
            )

            qa_model = self.config.model

            def _translate_gemini_kwargs(kw: dict) -> dict:
                """vendor 内若干路径（如 _op.py NER batch）会硬编码传 Gemini 专有 kwargs，
                OpenAI ChatCompletion 不认得，这里转换/剥离。"""
                kw = dict(kw)  # 不污染调用方
                # response_mime_type="application/json" → response_format={"type":"json_object"}
                mime = kw.pop("response_mime_type", None)
                if mime == "application/json" and "response_format" not in kw:
                    kw["response_format"] = {"type": "json_object"}
                # Gemini 结构化输出（response_json_schema / response_schema）OpenAI 走 json_schema 走 tool calling，
                # 这里简单退化为 json_object 模式 + prompt 里描述 schema（vendor 的 prompt 已经包含描述）。
                kw.pop("response_json_schema", None)
                kw.pop("response_schema", None)
                # Gemini thinking budget 不适用于 OpenAI 普通 chat completion
                kw.pop("thinking_level", None)
                return kw

            async def _openai_best_complete(prompt, system_prompt=None, history_messages=[], **kw):
                return await openai_complete_if_cache(
                    qa_model, prompt, system_prompt=system_prompt,
                    history_messages=history_messages, **_translate_gemini_kwargs(kw),
                )

            async def _openai_cheap_complete(prompt, system_prompt=None, history_messages=[], **kw):
                return await openai_complete_if_cache(
                    qa_model, prompt, system_prompt=system_prompt,
                    history_messages=history_messages, **_translate_gemini_kwargs(kw),
                )

            _openai_best_complete.__name__ = f"{qa_model}_complete"
            _openai_cheap_complete.__name__ = f"{qa_model}_cheap_complete"

            graphrag_kwargs.update(
                {
                    "best_model_func": _openai_best_complete,
                    "cheap_model_func": _openai_cheap_complete,
                    "embedding_func": openai_embedding,
                    "using_gemini": False,
                }
            )

        rag = GraphRAG(**graphrag_kwargs)

        # 自动构建索引（首次耗时较长，后续使用缓存）
        if not index_done_flag.exists():
            logger.info("[DyG-RAG] 开始为 %s 构建索引（首次需要数分钟）...", self._user_dir.name)
            llm_chunks = _prepare_llm_chunks(self._user_dir)
            device_events = _prepare_device_events(self._user_dir)
            if not llm_chunks and not device_events:
                raise ValueError(f"用户数据为空: {self._user_dir}")
            await rag.ainsert_typed(llm_chunks, device_events)
            index_done_flag.touch()
            logger.info("[DyG-RAG] 索引构建完成: %s", self._user_dir.name)
        else:
            logger.info("[DyG-RAG] 加载已有索引: %s", work_dir)

        self._graphrag = rag
        _GRAPHRAG_CACHE[cache_key] = rag
        if release_mode:
            _REF_COUNT[cache_key] = _REF_COUNT.get(cache_key, 0) + 1

    def _release_graphrag(self) -> None:
        """用完释放 GraphRAG 实例（DYG_RELEASE_AFTER_USE=1 时生效，按引用计数决定）"""
        if os.environ.get("DYG_RELEASE_AFTER_USE", "0") != "1":
            return
        if self._graphrag is None:
            return
        cache_key = str(self._user_dir)
        if cache_key in _REF_COUNT:
            _REF_COUNT[cache_key] -= 1
            if _REF_COUNT[cache_key] <= 0:
                _GRAPHRAG_CACHE.pop(cache_key, None)
                _REF_COUNT.pop(cache_key, None)
                logger.info("[DyG-RAG] 释放用户缓存: %s", self._user_dir.name)
        self._graphrag = None

    async def _generate_next_reaction(self, test_action: Optional[TestAgentAction]) -> TargetAgentReaction:
        if test_action is None:
            return TargetAgentReaction(
                type="message",
                message_list=[{"content": "你好，有什么可以帮你的？"}],
            )

        user_text = self._extract_user_input(test_action)

        if not self._user_dir or not self._user_dir.is_dir():
            return TargetAgentReaction(
                type="message",
                message_list=[{"content": f"错误: 用户数据目录不存在 ({self._user_dir})"}],
            )

        try:
            # 初始化 GraphRAG（含自动索引构建）
            try:
                await self._ensure_graphrag()
            except Exception as e:
                logger.error("[DyG-RAG] GraphRAG 初始化失败: %s", e, exc_info=True)
                return TargetAgentReaction(
                    type="message",
                    message_list=[{"content": f"GraphRAG 初始化错误: {e}"}],
                )

            from evaluator.vendor.dyg_graphrag import QueryParam

            logger.info("[DyG-RAG] 查询: %s...", user_text[:80])

            try:
                answer = await self._graphrag.aquery(
                    user_text,
                    param=QueryParam(mode="dynamic", top_k=self.config.top_k),
                )
            except Exception as e:
                logger.error("[DyG-RAG] 查询失败: %s", e, exc_info=True)
                return TargetAgentReaction(
                    type="message",
                    message_list=[{"content": f"DyG-RAG 查询错误: {e}"}],
                )

            # 追加对话历史
            self._conversation.append(BasicMessage(role="user", content=user_text))
            self._conversation.append(BasicMessage(role="assistant", content=answer))

            logger.info("[DyG-RAG] 响应: %d 字符", len(answer))

            return TargetAgentReaction(
                type="message",
                message_list=[{"content": answer}],
            )
        finally:
            self._release_graphrag()

    def get_session_info(self) -> SessionInfo:
        return SessionInfo(has_user_data=bool(self._user_dir and self._user_dir.is_dir()))

    def _extract_user_input(self, test_action: TestAgentAction) -> str:
        if test_action.type == "semantic":
            return test_action.semantic_content or ""
        elif test_action.type == "message":
            msg = test_action.message_content
            if isinstance(msg, dict):
                return msg.get("content", "")
            return ""
        return str(test_action.custom_content) if test_action.custom_content else ""
