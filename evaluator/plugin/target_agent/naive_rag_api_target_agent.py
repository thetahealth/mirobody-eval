"""
NaiveRagApiTargetAgent — 朴素向量检索增强问答（纯 embedding + cosine top-k）

注册名称: "naive_rag_api"

作为 HippoRAG 的 baseline 对照组:
1. 加载用户数据文件 (profile.json + timeline.json)
2. 展平切分: profile→1chunk, event/exam→每条独立, device→按天聚合
3. OpenAI embedding 构建向量索引（per-user 磁盘缓存）
4. 查询时 cosine top-k 检索
5. 通用 RAG prompt 调用 LLM 生成回答

与 HippoRAG 的区别: 无 OpenIE、无知识图谱、无 PPR 图检索。
"""

import asyncio
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Literal, Optional

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
from evaluator.utils.llm import BasicMessage, do_execute

logger = logging.getLogger(__name__)

# 缓存固定写入仓库内 benchmark/data/<bench>/.X_work_dirs/，跟 user_data 物理分离（不挂 PVC）
_BENCHMARK_DATA_DIR = Path(__file__).resolve().parents[3] / "benchmark" / "data"

# Embedding 维度统一为 1536（text-embedding-3-large 显式降维），减少存储和计算成本
EMBEDDING_DIM = 1536

# Per-user 锁：防止同一用户的并发首次构建撞车（首批 N 个 case 各自打 N×embedding API）。
# Key=user_dir。asyncio 单线程，dict 的 get/setdefault 无 race。
_INDEX_LOCKS: dict[str, asyncio.Lock] = {}


# ============================================================
# 文档加载与展平切分
# ============================================================


def _email_to_dir(email: str) -> str:
    return email.replace("@", "_AT_")


def _flatten_load(user_dir: Path) -> list[str]:
    """加载 profile.json + timeline.json，展平切分

    策略:
    - profile: 整体 1 chunk
    - event: 每个事件独立 1 chunk
    - exam_indicator: 每条独立 1 chunk
    - device_indicator: 按天聚合 1 chunk
    """
    chunks: list[str] = []

    # --- profile ---
    profile_file = user_dir / "profile.json"
    if profile_file.exists():
        try:
            with open(profile_file, encoding="utf-8") as f:
                profile = json.load(f)
            chunks.append(f"[PROFILE]\n{json.dumps(profile, ensure_ascii=False, indent=2)}")
        except (json.JSONDecodeError, OSError):
            pass

    # --- timeline ---
    timeline_file = user_dir / "timeline.json"
    if not timeline_file.exists():
        return chunks

    try:
        with open(timeline_file, encoding="utf-8") as f:
            tl = json.load(f)
    except (json.JSONDecodeError, OSError):
        return chunks

    entries = tl.get("entries", []) if isinstance(tl, dict) else tl if isinstance(tl, list) else []
    if not entries:
        return chunks

    # 分类
    device_by_day: dict[str, list[dict]] = {}

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        etype = entry.get("entry_type", "")

        if etype == "event":
            ev = entry.get("event", {})
            text = (
                f"[EVENT] {ev.get('event_name', '?')}"
                f" | Type: {ev.get('event_type', '?')}"
                f" | Event ID: {ev.get('event_id', '?')}"
                f" | Start: {ev.get('start_date', '?')}"
                f" | Duration: {ev.get('duration_days', '?')} days"
                f" | Interrupted: {ev.get('interrupted', False)}"
            )
            if ev.get("interrupted") and ev.get("interruption_date"):
                text += f" | Interruption date: {ev['interruption_date']}"
            chunks.append(text)

        elif etype == "exam_indicator":
            text = (
                f"[EXAM] {entry.get('time', '?')[:10]}"
                f" | {entry.get('indicator', '?')} = {entry.get('value', '?')} {entry.get('unit', '')}"
                f" | Location: {entry.get('exam_location', '?')}"
                f" | Type: {entry.get('exam_type', '?')}"
            )
            chunks.append(text)

        elif etype == "device_indicator":
            day = entry.get("time", "")[:10]
            device_by_day.setdefault(day, []).append(entry)

    # device 按天聚合
    for day in sorted(device_by_day.keys()):
        indicators = device_by_day[day]
        lines = [f"[DEVICE] Date: {day}"]
        for ind in indicators:
            lines.append(f"  {ind.get('indicator', '?')} = {ind.get('value', '?')} {ind.get('unit', '')}")
        chunks.append("\n".join(lines))

    return chunks


# ============================================================
# 向量检索
# ============================================================


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _get_embeddings(texts: list[str], model: str = "text-embedding-3-large") -> list[list[float]]:
    import os

    from openai import AsyncOpenAI

    api_key = os.getenv("OPENAI_API_KEY_EVAL") or os.getenv("OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=api_key, timeout=120.0)

    all_embeddings: list[list[float]] = []
    batch_size = 20
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch = [t[:24000] if len(t) > 24000 else t for t in batch]
        resp = await client.embeddings.create(
            input=batch, model=model, dimensions=EMBEDDING_DIM
        )
        all_embeddings.extend([d.embedding for d in resp.data])

    return all_embeddings


def _chunks_hash(chunks: list[str]) -> str:
    h = hashlib.md5()
    for c in sorted(chunks):
        h.update(c.encode("utf-8"))
    return h.hexdigest()


async def _build_or_get_index(
    user_dir: Path,
    data_group: str,
    embedding_model: str,
) -> tuple[list[str], list[list[float]]]:
    """构建或获取用户的向量索引（磁盘持久化，不驻留内存）"""
    import numpy as np

    lock_key = f"{user_dir}:{embedding_model}"
    lock = _INDEX_LOCKS.get(lock_key)
    if lock is None:
        lock = asyncio.Lock()
        _INDEX_LOCKS[lock_key] = lock

    async with lock:
        chunks = _flatten_load(user_dir)
        if not chunks:
            raise FileNotFoundError(f"用户数据目录为空: {user_dir}")

        # --- 磁盘缓存 ---
        work_dir = _BENCHMARK_DATA_DIR / data_group / ".naive_rag_work_dirs" / user_dir.name
        meta_file = work_dir / "index_meta.json"
        chunks_file = work_dir / "chunks.json"
        emb_file = work_dir / "chunk_embeddings.npz"

        current_hash = _chunks_hash(chunks)

        if meta_file.exists() and chunks_file.exists() and emb_file.exists():
            with open(meta_file, encoding="utf-8") as f:
                meta = json.load(f)
            # meta 中无 dimensions 字段视为 1536（向后兼容旧 cache）
            cached_dim = meta.get("dimensions", EMBEDDING_DIM)
            if (meta.get("chunks_hash") == current_hash
                    and meta.get("embedding_model") == embedding_model
                    and cached_dim == EMBEDDING_DIM):
                logger.info("[NaiveRAG] Index disk cache hit for %s", user_dir.name)
                with open(chunks_file, encoding="utf-8") as f:
                    cdata = json.load(f)
                embeddings = np.load(emb_file)["embeddings"].tolist()
                return (cdata["chunks"], embeddings)
            else:
                logger.info("[NaiveRAG] Index disk cache invalid, rebuilding %s", user_dir.name)

        # --- 构建 ---
        logger.info("[NaiveRAG] Indexing %d chunks for %s (model=%s, dim=%d)",
                    len(chunks), user_dir.name, embedding_model, EMBEDDING_DIM)
        embeddings = await _get_embeddings(chunks, model=embedding_model)

        # --- 保存 ---
        work_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({"chunks_hash": current_hash, "embedding_model": embedding_model, "dimensions": EMBEDDING_DIM}, f)
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump({"chunks": chunks}, f, ensure_ascii=False)
        np.savez_compressed(str(emb_file), embeddings=np.array(embeddings, dtype=np.float32))
        logger.info("[NaiveRAG] Index saved to disk: %s", work_dir)
        return (chunks, embeddings)


async def _retrieve(
    query: str,
    chunks: list[str],
    embeddings: list[list[float]],
    top_k: int,
    embedding_model: str,
) -> list[str]:
    query_embedding = (await _get_embeddings([query], model=embedding_model))[0]
    scored = [(_cosine_similarity(query_embedding, emb), i) for i, emb in enumerate(embeddings)]
    scored.sort(reverse=True)
    return [chunks[idx] for _, idx in scored[:top_k]]


# ============================================================
# RAG QA Prompt — 通用检索问答
# ============================================================

_RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based ONLY on the provided context.\n\n"
    "## Rules\n"
    "1. Answer strictly from the context. If the information is not present, say so.\n"
    "2. Be precise: extract exact values, dates, and names from the data.\n"
    "3. When counting or listing, scan ALL context chunks systematically.\n"
    "4. Start with \"Thought: \" for reasoning, then \"Answer: \" for the final response.\n"
)


def _build_rag_input(query: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    return f"Context:\n{context}\n\nQuestion: {query}\nThought: "


# ============================================================
# Config + Agent
# ============================================================


class NaiveRagApiTargetInfo(BaseModel):
    """NaiveRAG 被测目标配置 — 纯向量检索增强生成"""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "_comment": "默认配置",
                    "type": "naive_rag_api",
                    "model": "gemini-3-flash-preview",
                    "data_group": "eslbench",
                    "user_email": "user110@demo",
                },
            ],
        },
    )
    type: Literal["naive_rag_api"] = Field(description="目标类型")
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
    embedding_model: str = Field("text-embedding-3-large", description="嵌入模型")
    data_group: str = Field(description="数据目录（benchmark 名称，如 'eslbench'）")
    user_email: Optional[str] = Field(None, description="用户邮箱（映射到 .data/{user_dir}/）")
    top_k: int = Field(10, description="检索 top-k 文档数量", ge=1, le=50)
    system_prompt: Optional[str] = Field(None, description="自定义系统提示词")


class NaiveRagApiTargetAgent(AbstractTargetAgent, name="naive_rag_api", params_model=NaiveRagApiTargetInfo):
    """朴素向量检索增强生成（RAG）— HippoRAG baseline 对照"""

    _display_meta = {
        "icon": (
            "M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375"
            " 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125"
            " 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0"
            " 00-9-9z"
        ),
        "color": "#6366f1",
        "features": ["RAG", "向量检索", "Baseline"],
    }
    _cost_meta = {
        "est_input_tokens": 6000,
        "est_output_tokens": 600,
    }

    def __init__(self, target_config: NaiveRagApiTargetInfo, history: list[BaseMessage] | None = None):
        super().__init__(target_config, history=history)
        self.config: NaiveRagApiTargetInfo = target_config

        self._conversation: list[BasicMessage] = []
        for msg in self.history:
            if isinstance(msg, HumanMessage):
                self._conversation.append(BasicMessage(role="user", content=msg.content))
            elif isinstance(msg, AIMessage):
                self._conversation.append(BasicMessage(role="assistant", content=msg.content))

        self._user_dir: Path | None = None
        if target_config.user_email:
            dir_name = _email_to_dir(target_config.user_email)
            self._user_dir = paths.user_data_dir(target_config.data_group) / dir_name
            logger.info(
                "[NaiveRAG] user_dir=%s exists=%s data_group=%s",
                self._user_dir,
                self._user_dir.is_dir(),
                target_config.data_group,
            )
            if not self._user_dir.is_dir():
                logger.warning(
                    "[NaiveRAG] error_code=USER_DIR_NOT_FOUND user_dir=%s data_group=%s",
                    self._user_dir,
                    target_config.data_group,
                )

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

        # 1. 构建/获取向量索引
        chunks, embeddings = await _build_or_get_index(
            self._user_dir,
            data_group=self.config.data_group,
            embedding_model=self.config.embedding_model,
        )

        # 2. 纯向量检索
        context_chunks = await _retrieve(
            query=user_text,
            chunks=chunks,
            embeddings=embeddings,
            top_k=self.config.top_k,
            embedding_model=self.config.embedding_model,
        )

        # 3. 构建 RAG prompt
        rag_input = _build_rag_input(user_text, context_chunks)
        system_prompt = self.config.system_prompt or _RAG_SYSTEM_PROMPT

        logger.debug(
            "[NaiveRAG] Query: %s, Retrieved %d/%d chunks, calling %s",
            user_text[:80],
            len(context_chunks),
            len(chunks),
            self.config.model,
        )

        # 4. 调用 LLM（无 few-shot，纯通用 prompt）
        result = await do_execute(
            model=self.config.model,
            system_prompt=system_prompt,
            input=rag_input,
            history_messages=self._conversation if self._conversation else None,
        )

        assistant_content = result.content
        self._accumulate_cost(result.usage)

        self._conversation.append(BasicMessage(role="user", content=user_text))
        self._conversation.append(BasicMessage(role="assistant", content=assistant_content))

        return TargetAgentReaction(
            type="message",
            message_list=[{"content": assistant_content}],
            usage=result.usage,
        )

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
        else:
            return str(test_action.custom_content) if test_action.custom_content else ""
