"""
Mem0RagApiTargetAgent — mem0 记忆增强问答

注册名称: "mem0_rag_api"

实现流程:
1. 加载用户数据文件 → 复用 user_data_chunker 智能切分（排除 events.json）
2. 通过 mem0.add(chunk, user_id, infer=True) 批量写入记忆（LLM 自动提取关键事实）
3. 查询时 mem0.search(query, user_id) 检索相关记忆
4. 拼接 RAG prompt + few-shot 示例 → LLM 生成答案

配置与 HippoRAG 对齐:
    model: gemini-3-flash-preview（生成模型）
    embedding_model: text-embedding-3-large（向量编码，通过 mem0 config 传入）
    top_k: 10（检索记忆条数）
"""

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.ai import UsageMetadata
from pydantic import BaseModel, ConfigDict, Field

from evaluator.core.interfaces.abstract_target_agent import AbstractTargetAgent
from evaluator.core.schema import (
    TargetAgentReaction,
    TestAgentAction,
)
from evaluator.utils.llm import BasicMessage, do_execute
from evaluator.utils.user_data_chunker import BENCHMARK_DATA_DIR, load_user_documents, resolve_user_dir

logger = logging.getLogger(__name__)

# 全局已加载用户集合（避免同一用户重复灌入记忆）
_LOADED_USERS: set[str] = set()
# 全局 Memory 实例缓存
_MEM0_INSTANCE: Any = None
# 每用户加载锁（防止并发重复加载）
_USER_LOCKS: dict[str, asyncio.Lock] = {}


_HEALTH_DATA_EXTRACTION_PROMPT = """\
You are a Health Data Preservation System. Your task is to extract ALL factual information \
from the input and store each piece as a separate, self-contained fact.

CRITICAL RULES:
1. Preserve ALL numerical values, dates, units, and identifiers EXACTLY as they appear — never round, convert, or paraphrase.
2. Each fact must be self-contained: include the subject (patient name/ID, indicator name, event name) so the fact is useful without surrounding context.
3. For health indicators, always include: indicator name, value, unit, and date/time period.
4. For health events, always include: event name, start date, duration, medications, and affected indicators with their expected changes.
5. For profile data, preserve all fields: demographics, family history, conditions, medications.
6. Do NOT summarize or merge multiple data points into one fact. One data point = one fact.
7. Do NOT skip data that seems routine or repetitive — every measurement matters.

Example input:
"Heart Rate on 2025-03-15: 72 bpm. Blood Pressure on 2025-03-15: 120/80 mmHg. Patient started Metformin 500mg on 2025-03-10 for Type 2 Diabetes."

Example output:
{"facts": [
  "Heart Rate on 2025-03-15: 72 bpm",
  "Blood Pressure (systolic) on 2025-03-15: 120 mmHg",
  "Blood Pressure (diastolic) on 2025-03-15: 80 mmHg",
  "Patient started Metformin 500mg on 2025-03-10",
  "Metformin prescribed for Type 2 Diabetes"
]}

Return ALL facts as JSON with key "facts" containing an array of strings.
"""


def _get_mem0(embedding_model: str = "text-embedding-3-large", extraction_prompt: str | None = None) -> Any:
    """获取或创建 mem0 Memory 单例（配置 text-embedding-3-large）"""
    global _MEM0_INSTANCE
    if _MEM0_INSTANCE is not None:
        return _MEM0_INSTANCE

    from mem0 import Memory

    config: dict[str, Any] = {
        "embedder": {
            "provider": "openai",
            "config": {
                "model": embedding_model,
            },
        },
    }
    if extraction_prompt:
        config["custom_fact_extraction_prompt"] = extraction_prompt
    _MEM0_INSTANCE = Memory.from_config(config)
    return _MEM0_INSTANCE


async def _ensure_user_loaded(
    mem0_instance: Any,
    user_id: str,
    user_dir: Path,
    max_chunk_chars: int,
    marker_dir: Path,
) -> int:
    """确保用户数据已灌入 mem0（幂等，有磁盘标记文件 + 并发锁）

    返回写入的记忆条数。
    """
    # 快速路径：内存缓存命中
    if user_id in _LOADED_USERS:
        return 0

    # 获取每用户锁，防止同一用户并发加载
    if user_id not in _USER_LOCKS:
        _USER_LOCKS[user_id] = asyncio.Lock()
    async with _USER_LOCKS[user_id]:
        # 双重检查
        if user_id in _LOADED_USERS:
            return 0

        # 检查磁盘标记
        marker_dir.mkdir(parents=True, exist_ok=True)
        marker_file = marker_dir / ".mem0_loaded"

        # 只加载语义 chunks（openie_indices 标记的: profile, events, exams, 周摘要）
        # 跳过 device raw data（数值数据量大且 LLM 难以有效提取记忆）
        all_chunks, openie_indices = load_user_documents(user_dir, max_chunk_chars)
        semantic_chunks = [all_chunks[i] for i in sorted(openie_indices) if i < len(all_chunks)]
        data_hash = hashlib.md5("".join(semantic_chunks).encode()).hexdigest()[:12]

        if marker_file.exists():
            try:
                meta = json.loads(marker_file.read_text())
                if meta.get("data_hash") == data_hash and meta.get("chunk_count") == len(semantic_chunks):
                    _LOADED_USERS.add(user_id)
                    logger.info("[mem0] 用户 %s 数据已加载（缓存命中），跳过", user_id)
                    return 0
            except (json.JSONDecodeError, OSError):
                pass

        # 清除旧记忆
        logger.info(
            "[mem0] 清除用户 %s 旧记忆并重新加载 %d/%d 块语义数据",
            user_id, len(semantic_chunks), len(all_chunks),
        )
        try:
            await asyncio.to_thread(mem0_instance.delete_all, user_id=user_id)
        except Exception:
            pass  # 首次加载时可能无旧记忆

        # 逐条写入（mem0 无 batch API，用 to_thread 避免阻塞事件循环）
        # infer=False: 直接存原始 chunk 为向量嵌入，保留完整原始数据，速度快 10-20x
        count = 0
        for i, chunk in enumerate(semantic_chunks):
            try:
                await asyncio.to_thread(mem0_instance.add, chunk, user_id=user_id, infer=False)
                count += 1
                if (i + 1) % 50 == 0:
                    logger.info("[mem0] 用户 %s 已写入 %d/%d 块", user_id, i + 1, len(semantic_chunks))
            except Exception as e:
                logger.warning("[mem0] 写入记忆失败 (chunk %d): %s", i, e)

        # 写标记文件
        marker_file.write_text(
            json.dumps(
                {
                    "data_hash": data_hash,
                    "chunk_count": len(semantic_chunks),
                    "memory_count": count,
                }
            )
        )
        _LOADED_USERS.add(user_id)
        logger.info("[mem0] 用户 %s 加载完成，写入 %d 条记忆", user_id, count)
        return count


# ============================================================
# RAG prompt — 复用 HippoRAG 的 system prompt 和 few-shot
# ============================================================

from evaluator.plugin.target_agent.hippo_rag_api_target_agent import (
    _RAG_SYSTEM_PROMPT,
    _SHOT_LIST_INPUT,
    _SHOT_LIST_OUTPUT,
    _SHOT_NUMERIC_INPUT,
    _SHOT_NUMERIC_OUTPUT,
)


def _build_rag_input(query: str, memories: list[dict]) -> str:
    """将 mem0 search 结果拼接为 RAG 输入"""
    if not memories:
        context = "(No relevant memories found)"
    else:
        context = "\n\n".join(m.get("memory", "") for m in memories)
    return f"{context}\n\nQuestion: {query}\nThought: "


# ============================================================
# 配置模型
# ============================================================


class Mem0RagApiTargetInfo(BaseModel):
    """mem0 记忆增强被测目标配置"""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "_comment": "默认配置（embedding 与 HippoRAG 一致）",
                    "type": "mem0_rag_api",
                    "model": "gemini-3-flash-preview",
                    "data_group": "eslbench",
                    "user_email": "user110@demo",
                },
            ],
        },
    )
    type: Literal["mem0_rag_api"] = Field(description="目标类型")
    model: Literal[
        "gpt-5.4",
        "gpt-5.2",
        "gpt-4.1",
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        "anthropic/claude-opus-4.6",
        "anthropic/claude-sonnet-4.6",
        "minimax/minimax-m2.7",
        "z-ai/glm-5.1",
    ] = Field(description="生成模型名称")
    embedding_model: str = Field(
        "text-embedding-3-large",
        description="嵌入模型（与 HippoRAG 一致）",
    )
    data_group: str = Field(description="数据目录（benchmark 名称，如 'eslbench'）")
    user_email: Optional[str] = Field(None, description="用户邮箱（映射到 .data/{user_dir}/）")
    top_k: int = Field(10, description="检索记忆条数", ge=1, le=100)
    max_chunk_chars: int = Field(15000, description="文档切分字符数", ge=1000, le=50000)
    system_prompt: Optional[str] = Field(None, description="自定义系统提示词（默认使用 HippoRAG 同款）")
    extraction_prompt: Optional[str] = Field(None, description="自定义 mem0 fact extraction 提示词（默认使用健康数据保留模式）")


# ============================================================
# Agent 实现
# ============================================================


class Mem0RagApiTargetAgent(AbstractTargetAgent, name="mem0_rag_api", params_model=Mem0RagApiTargetInfo):
    """mem0 记忆增强问答"""

    _display_meta = {
        "icon": (
            "M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375"
            " 3.375 0 00-3.375-3.375H8.25m5.231 13.481L15 17.25m-4.5-15H5.625c-.621 0-1.125.504-1.125"
            " 1.125v16.5c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0"
            " 00-9-9zm3.75 11.625a2.625 2.625 0 11-5.25 0 2.625 2.625 0 015.25 0z"
        ),
        "color": "#f59e0b",
        "features": ["RAG", "记忆管理", "mem0"],
        "icon_url": "/static/images/agents/mem0.png",
    }
    _cost_meta = {
        "est_input_tokens": 8000,
        "est_output_tokens": 800,
    }

    def __init__(self, target_config: Mem0RagApiTargetInfo, history: list[BaseMessage] | None = None):
        super().__init__(target_config, history=history)
        self.config: Mem0RagApiTargetInfo = target_config

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
            self._user_dir = resolve_user_dir(target_config.data_group, target_config.user_email)
            if not self._user_dir.is_dir():
                logger.warning("[mem0] 用户数据目录不存在: %s", self._user_dir)

        # mem0 缓存目录（磁盘标记文件）
        self._marker_dir: Path | None = None
        if self._user_dir:
            self._marker_dir = BENCHMARK_DATA_DIR / target_config.data_group / ".mem0_work_dirs" / self._user_dir.name

        # 成本统计
        self._cost = UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)

    @property
    def cost(self) -> UsageMetadata:
        return self._cost

    def _accumulate_cost(self, usage: dict[str, UsageMetadata] | None) -> None:
        from evaluator.utils.llm import accumulate_usage

        self._cost = accumulate_usage(self._cost, usage)

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

        # 1. 获取 mem0 实例 & 确保用户数据已加载
        extraction_prompt = self.config.extraction_prompt or _HEALTH_DATA_EXTRACTION_PROMPT
        m = _get_mem0(self.config.embedding_model, extraction_prompt=extraction_prompt)
        user_id = self._user_dir.name  # e.g. "user110_AT_demo"

        await _ensure_user_loaded(
            mem0_instance=m,
            user_id=user_id,
            user_dir=self._user_dir,
            max_chunk_chars=self.config.max_chunk_chars,
            marker_dir=self._marker_dir,
        )

        # 2. 检索相关记忆（同步 API，放入线程池避免阻塞，带重试防 Qdrant 并发竞争）
        memories: list[dict] = []
        for attempt in range(3):
            try:
                results = await asyncio.to_thread(m.search, query=user_text, user_id=user_id, limit=self.config.top_k)
                memories = results.get("results", results) if isinstance(results, dict) else results
                break
            except (IndexError, Exception) as e:
                logger.warning("[mem0] search 失败 (attempt %d/3): %s", attempt + 1, e)
                if attempt < 2:
                    await asyncio.sleep(1 + attempt)
                else:
                    logger.error("[mem0] search 重试耗尽，使用空记忆继续")
                    memories = []

        logger.debug(
            "[mem0] Query: %s, Retrieved %d memories, calling %s",
            user_text[:80],
            len(memories) if memories else 0,
            self.config.model,
        )

        # 3. 构建 RAG prompt
        rag_input = _build_rag_input(user_text, memories)
        system_prompt = self.config.system_prompt or _RAG_SYSTEM_PROMPT

        # 4. 调用 LLM（带 few-shot 示例，与 HippoRAG 一致）
        history_with_oneshot = [
            BasicMessage(role="user", content=_SHOT_NUMERIC_INPUT),
            BasicMessage(role="assistant", content=_SHOT_NUMERIC_OUTPUT),
            BasicMessage(role="user", content=_SHOT_LIST_INPUT),
            BasicMessage(role="assistant", content=_SHOT_LIST_OUTPUT),
        ]
        if self._conversation:
            history_with_oneshot.extend(self._conversation)

        result = await do_execute(
            model=self.config.model,
            system_prompt=system_prompt,
            input=rag_input,
            history_messages=history_with_oneshot,
        )

        self._accumulate_cost(result.usage)

        # 更新对话历史
        self._conversation.append(BasicMessage(role="user", content=user_text))
        self._conversation.append(BasicMessage(role="assistant", content=result.content))

        return TargetAgentReaction(
            type="message",
            message_list=[{"content": result.content}],
            usage=result.usage,
        )

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
