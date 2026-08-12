"""
HippoRagApiTargetAgent — HippoRAG retrieval-augmented QA (with OpenIE + graph retrieval)

Registered name: "hippo_rag_api"

Implements the full HippoRAG pipeline:
1. Load user data files -> smart document chunking
2. Build vector index with OpenAI text-embedding-3-large (per-user cache)
3. LLM performs OpenIE: NER named entity recognition + RDF triple extraction (disk cache)
4. Build knowledge graph: entity<->entity (fact) + passage<->entity + entity<->entity (synonymy)
5. At query time: query->entity matching + Personalized PageRank graph retrieval + dense retrieval fusion
6. Call LLM with improved prompt to generate Thought + Answer

Set use_graph=False (default) for pure vector RAG, use_graph=True for full graph retrieval pipeline.

Default params:
    model: gemini-3-flash-preview
    embedding_model: text-embedding-3-large
    qa_top_k: 10
    max_chunk_chars: 15000
    use_graph: False
"""

import asyncio
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Literal, Optional

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

# Per-user 锁：防止同一用户的并发首次构建撞车（一个 user 5 个 case 并发各打 5 次 embedding）。
# 分两组锁：向量索引 vs 图（不同阶段不互锁，避免无谓串行）。
_INDEX_LOCKS: dict[str, asyncio.Lock] = {}
_GRAPH_LOCKS: dict[str, asyncio.Lock] = {}


# ============================================================
# Document loading and chunking (replicates run_hipporag.py load_event_docs logic)
# ============================================================


def _email_to_dir(email: str) -> str:
    """Email -> directory name: user110@demo -> user110_AT_demo"""
    return email.replace("@", "_AT_")


def _load_user_documents(user_dir: Path, max_chunk_chars: int = 15000) -> tuple[list[str], set[int]]:
    """Load user data files with type-aware smart chunking

    Returns (chunks, openie_indices):
    - chunks: All text chunks (all get embedded)
    - openie_indices: Set of chunk indices that need OpenIE

    Chunking strategy:
    - profile: Single chunk, with OpenIE
    - event: Each event as independent chunk, with OpenIE
    - exam_indicator: Grouped by exam date, with OpenIE
    - device_indicator weekly summary: Statistical aggregation (mean/min/max/trend/anomaly), with OpenIE
    - device_indicator raw data: Grouped by day, embedding only
    """
    chunks: list[str] = []
    openie_indices: set[int] = set()

    # --- Load profile ---
    profile_file = user_dir / "profile.json"
    if profile_file.exists():
        try:
            with open(profile_file, encoding="utf-8") as f:
                profile = json.load(f)
            chunks.append(f"[PROFILE]\n{json.dumps(profile, ensure_ascii=False, indent=2)}")
            openie_indices.add(len(chunks) - 1)
        except (json.JSONDecodeError, OSError):
            pass

    # --- Load timeline ---
    timeline_file = user_dir / "timeline.json"
    if not timeline_file.exists():
        return chunks, openie_indices

    try:
        with open(timeline_file, encoding="utf-8") as f:
            tl = json.load(f)
    except (json.JSONDecodeError, OSError):
        return chunks, openie_indices

    entries = tl.get("entries", []) if isinstance(tl, dict) else tl if isinstance(tl, list) else []
    if not entries:
        return chunks, openie_indices

    # Categorize
    events: list[dict] = []
    device_by_day: dict[str, list[dict]] = {}
    exam_by_date: dict[str, list[dict]] = {}

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        etype = entry.get("entry_type", "")
        day = entry.get("time", "")[:10]
        if etype == "event":
            events.append(entry)
        elif etype == "device_indicator":
            device_by_day.setdefault(day, []).append(entry)
        elif etype == "exam_indicator":
            exam_by_date.setdefault(day, []).append(entry)

    # --- Event chunks (each independent, with OpenIE) ---
    for entry in events:
        ev = entry.get("event", {})
        text = (
            f"[EVENT] {ev.get('event_name', '?')}\n"
            f"Type: {ev.get('event_type', '?')}\n"
            f"Event ID: {ev.get('event_id', '?')}\n"
            f"Start: {ev.get('start_date', '?')}\n"
            f"Duration: {ev.get('duration_days', '?')} days\n"
            f"Interrupted: {ev.get('interrupted', False)}"
        )
        if ev.get("interrupted") and ev.get("interruption_date"):
            text += f"\nInterruption date: {ev['interruption_date']}"
        chunks.append(text)
        openie_indices.add(len(chunks) - 1)

    # --- Exam chunks (grouped by date, with OpenIE) ---
    for date, exams in sorted(exam_by_date.items()):
        lines = [f"[EXAM] Date: {date}, Location: {exams[0].get('exam_location', '?')}, Type: {exams[0].get('exam_type', '?')}"]
        for e in exams:
            lines.append(f"  {e.get('indicator', '?')} = {e.get('value', '?')} {e.get('unit', '')}")
        chunks.append("\n".join(lines))
        openie_indices.add(len(chunks) - 1)

    # --- Device: Weekly summary chunks (statistical aggregation, with OpenIE) ---
    # Group by ISO week
    from datetime import datetime as _dt
    week_data: dict[str, dict[str, list[float]]] = {}
    for day, indicators in device_by_day.items():
        try:
            dt = _dt.fromisoformat(day)
            week_key = dt.strftime("%Y-W%W")
        except ValueError:
            continue
        week_indicators = week_data.setdefault(week_key, {})
        for ind in indicators:
            name = ind.get("indicator", "?")
            val = ind.get("value")
            if val is not None and val != "None" and isinstance(val, (int, float)):
                week_indicators.setdefault(name, []).append(val)

    for week_key in sorted(week_data.keys()):
        indicators = week_data[week_key]
        lines = [f"[DEVICE SUMMARY] Week: {week_key}"]
        anomalies = []
        for name, values in sorted(indicators.items()):
            if not values:
                continue
            mean_val = sum(values) / len(values)
            min_val = min(values)
            max_val = max(values)
            # Trend: compare mean of first half vs second half
            if len(values) >= 4:
                mid = len(values) // 2
                first_half = sum(values[:mid]) / mid
                second_half = sum(values[mid:]) / (len(values) - mid)
                ratio = (second_half - first_half) / (abs(first_half) + 1e-9)
                if ratio > 0.1:
                    trend = "rising"
                elif ratio < -0.1:
                    trend = "falling"
                else:
                    trend = "stable"
            else:
                trend = "stable"
            lines.append(f"  {name}: mean={mean_val:.2f}, min={min_val:.2f}, max={max_val:.2f}, trend={trend}")
            # Flag anomaly: max exceeds mean by 50%
            if max_val > mean_val * 1.5 and len(values) >= 3:
                anomalies.append(f"{name} peaked at {max_val:.2f} (mean {mean_val:.2f})")
        if anomalies:
            lines.append(f"  Notable: {'; '.join(anomalies)}")
        chunks.append("\n".join(lines))
        openie_indices.add(len(chunks) - 1)

    # --- Device: Raw data chunks (grouped by day, embedding only, no OpenIE) ---
    for day in sorted(device_by_day.keys()):
        indicators = device_by_day[day]
        lines = [f"[DEVICE RAW] Date: {day}"]
        for ind in indicators:
            lines.append(f"  {ind.get('indicator', '?')} = {ind.get('value', '?')} {ind.get('unit', '')}")
        text = "\n".join(lines)
        # Secondary chunking for oversized days
        if len(text) > max_chunk_chars:
            for sub in _chunk_by_chars(text, max_chunk_chars):
                chunks.append(sub)
        else:
            chunks.append(text)
        # Not added to openie_indices

    return chunks, openie_indices


def _split_large_document(text: str, max_chunk_chars: int = 15000) -> list[str]:
    """Large document chunking (replicates run_hipporag.py chunking logic)

    Splits by lines, each chunk no larger than max_chunk_chars.
    If a [PROFILE] section exists, it is prepended to each chunk.
    """
    lines = text.split("\n")

    # Separate [PROFILE] and [DATA] sections
    profile_lines: list[str] = []
    data_lines: list[str] = []
    in_data = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[DATA]"):
            in_data = True
            data_lines.append(line)
            continue
        if stripped == "[PROFILE]":
            in_data = False
            continue
        if in_data:
            data_lines.append(line)
        else:
            profile_lines.append(line)

    profile_text = "[PROFILE]\n" + "\n".join(profile_lines) if profile_lines else ""

    # If no DATA section, chunk by character count directly
    if not data_lines:
        return _chunk_by_chars(text, max_chunk_chars)

    # Chunk DATA lines by character count, prepend [PROFILE] to each chunk
    docs = []
    if profile_text:
        docs.append(profile_text)

    current_chunk: list[str] = []
    current_chars = 0

    for line in data_lines:
        line_chars = len(line) + 1
        if current_chunk and current_chars + line_chars > max_chunk_chars:
            chunk_text = profile_text + "\n" + "\n".join(current_chunk) if profile_text else "\n".join(current_chunk)
            docs.append(chunk_text)
            current_chunk = []
            current_chars = 0
        current_chunk.append(line)
        current_chars += line_chars

    if current_chunk:
        chunk_text = profile_text + "\n" + "\n".join(current_chunk) if profile_text else "\n".join(current_chunk)
        docs.append(chunk_text)

    return docs


def _chunk_by_chars(text: str, max_chars: int, overlap: int = 500) -> list[str]:
    """Chunk by character count with overlap"""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


# ============================================================
# Vector retrieval
# ============================================================


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure Python cosine similarity"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _get_embeddings(texts: list[str], model: str = "text-embedding-3-large") -> list[list[float]]:
    """Call OpenAI Embedding API to get text vectors (same model as remote HippoRAG)"""
    import os

    from openai import AsyncOpenAI

    api_key = os.getenv("OPENAI_API_KEY_EVAL") or os.getenv("OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=api_key, timeout=120.0)

    all_embeddings: list[list[float]] = []
    batch_size = 20
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Truncate overly long text (embedding API limit: 8191 tokens)
        # text-embedding-3-large max 8191 tokens, conservatively truncate at 3 chars/token
        batch = [t[:24000] if len(t) > 24000 else t for t in batch]
        resp = await client.embeddings.create(
            input=batch, model=model, dimensions=EMBEDDING_DIM
        )
        all_embeddings.extend([d.embedding for d in resp.data])

    return all_embeddings


async def _build_or_get_index(
    user_dir: Path,
    data_group: str,
    max_chunk_chars: int,
    embedding_model: str,
) -> tuple[list[str], "np.ndarray", set[int]]:
    """Build or get user's vector index (磁盘持久化，不驻留内存)

    Returns (chunks, embeddings_np, openie_indices)

    embeddings 保持 numpy float32 ndarray —— 不 .tolist()，避免 N×D Python list 的
    8× 对象头膨胀（1172×1536 float32 ≈ 7 MB 的 ndarray，转成 Python list 直奔 55 MB）。
    并发跑多用户时这是 hippo 内存的主要来源，能把单 case 工作集压到 1/8。
    """
    import numpy as np

    lock_key = f"{user_dir}:{max_chunk_chars}:{embedding_model}"
    lock = _INDEX_LOCKS.get(lock_key)
    if lock is None:
        lock = asyncio.Lock()
        _INDEX_LOCKS[lock_key] = lock

    async with lock:
        docs, openie_indices = _load_user_documents(user_dir, max_chunk_chars=max_chunk_chars)
        if not docs:
            raise FileNotFoundError(f"User data directory is empty: {user_dir}")

        # --- Disk cache check ---
        work_dir = _BENCHMARK_DATA_DIR / data_group / ".hippo_work_dirs" / user_dir.name
        meta_file = work_dir / "index_meta.json"
        chunks_file = work_dir / "chunks.json"
        emb_file = work_dir / "chunk_embeddings.npz"

        current_hash = _chunks_hash(docs)

        if meta_file.exists() and chunks_file.exists() and emb_file.exists():
            with open(meta_file, encoding="utf-8") as f:
                meta = json.load(f)
            # meta 中无 dimensions 字段视为 1536（向后兼容旧 cache）
            cached_dim = meta.get("dimensions", EMBEDDING_DIM)
            if (meta.get("chunks_hash") == current_hash
                    and meta.get("embedding_model") == embedding_model
                    and meta.get("max_chunk_chars") == max_chunk_chars
                    and cached_dim == EMBEDDING_DIM):
                logger.info("[HippoRAG] Index disk cache hit for %s", user_dir.name)
                with open(chunks_file, encoding="utf-8") as f:
                    cdata = json.load(f)
                embeddings_np = np.load(emb_file)["embeddings"].astype(np.float32, copy=False)
                return (cdata["chunks"], embeddings_np, set(cdata["openie_indices"]))
            else:
                logger.info("[HippoRAG] Index disk cache invalid (hash/model/dim changed), rebuilding %s", user_dir.name)

        # --- Normal build ---
        logger.info(
            "[HippoRAG] Indexing %d chunks for %s (OpenIE: %d, embedding-only: %d, model=%s, dim=%d)",
            len(docs), user_dir.name, len(openie_indices), len(docs) - len(openie_indices),
            embedding_model, EMBEDDING_DIM,
        )
        embeddings = await _get_embeddings(docs, model=embedding_model)
        embeddings_np = np.array(embeddings, dtype=np.float32)
        del embeddings  # API 返回的 list 立刻释放，只保留 ndarray

        # --- Save to disk ---
        work_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({
                "chunks_hash": current_hash, "embedding_model": embedding_model,
                "max_chunk_chars": max_chunk_chars, "dimensions": EMBEDDING_DIM,
            }, f)
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump({"chunks": docs, "openie_indices": list(openie_indices)}, f, ensure_ascii=False)
        np.savez_compressed(str(emb_file), embeddings=embeddings_np)
        logger.info("[HippoRAG] Index saved to disk: %s", work_dir)

        return (docs, embeddings_np, openie_indices)


async def _retrieve(
    query: str,
    chunks: list[str],
    embeddings,  # np.ndarray (N, D) float32 — 历史接口可能传 list，下面 asarray 兜底
    top_k: int,
    embedding_model: str,
) -> list[str]:
    """Retrieve the top_k most relevant chunks for the query (vectorized cosine)"""
    import numpy as np

    query_emb = np.asarray(
        (await _get_embeddings([query], model=embedding_model))[0], dtype=np.float32
    )
    emb_np = np.asarray(embeddings, dtype=np.float32)

    q_norm = float(np.linalg.norm(query_emb))
    if q_norm == 0 or emb_np.size == 0:
        return [chunks[i] for i in range(min(top_k, len(chunks)))]

    e_norms = np.linalg.norm(emb_np, axis=1)
    e_norms = np.where(e_norms == 0, 1.0, e_norms)
    scores = (emb_np @ query_emb) / (e_norms * q_norm)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_idx if i < len(chunks)]


# ============================================================
# RAG QA Prompt — Structured health data reasoning
# ============================================================

_RAG_SYSTEM_PROMPT = (
    "You are a health data analyst. You answer questions about a user's health records "
    "(events, indicators, exam results, device data) based ONLY on the provided context.\n\n"
    "## Rules\n"
    "1. **Precision first**: Extract exact values, dates, and names from the data. Never guess or approximate.\n"
    "2. **Exhaustive search**: When counting or listing, scan ALL chunks systematically. "
    "Do not stop at the first few matches — the context may contain hundreds of records across multiple chunks.\n"
    "3. **Deduplicate**: When asked for deduplicated counts/lists, remove exact duplicates by name (case-insensitive).\n"
    "4. **Date handling**: Parse date ranges carefully. "
    "Event overlap means their [start, start+duration) intervals intersect. "
    "Use ISO dates (YYYY-MM-DD) for comparison.\n"
    "5. **Format**: Start with \"Thought: \" for step-by-step reasoning, then \"Answer: \" for the final response.\n\n"
    "## Answer type conventions\n"
    "- **Numeric**: Output a single number. Show your counting/calculation work in Thought.\n"
    "- **List**: Output items as a comma-separated list in the format the question specifies "
    "(e.g. \"event name (start date)\"). Sort as requested (by date if unspecified).\n"
    "- **Boolean**: Output Yes or No.\n"
    "- **Text**: Output a concise, direct answer with supporting evidence.\n\n"
    "If the data is insufficient to answer, say so explicitly — do not fabricate information."
)

# Few-shot example: numeric (counting)
_SHOT_NUMERIC_INPUT = (
    "Context:\n"
    "[DATA] Timeline\n"
    "[2024-01-15T00:00:00] Event: Common Cold, Start: 2024-01-15, Duration: 7 days, Interrupted: False\n"
    "[2024-01-15T08:00:00] Device: BodyTemperature = 38.2 °C\n"
    "[2024-01-16T08:00:00] Device: RestingHeartRate = 82 bpm\n"
    "[2024-03-01T00:00:00] Event: Seasonal Allergy, Start: 2024-03-01, Duration: 14 days, Interrupted: False\n"
    "[2024-03-02T09:00:00] Exam: BasophilPercentage-BASO% = 1.2 %\n\n"
    "Question: How many distinct health events are recorded in the timeline?\n"
    "Thought: "
)

_SHOT_NUMERIC_OUTPUT = (
    "I need to count distinct events in the timeline.\n"
    "- Common Cold (2024-01-15)\n"
    "- Seasonal Allergy (2024-03-01)\n"
    "Total: 2 distinct events.\n"
    "Answer: **2**"
)

# Few-shot example: list (time overlap + format)
_SHOT_LIST_INPUT = (
    "Context:\n"
    "[DATA] Timeline\n"
    "[2024-01-01T00:00:00] Event: Gym Training, Start: 2024-01-01, Duration: 90 days, Interrupted: False\n"
    "[2024-02-10T00:00:00] Event: Common Cold, Start: 2024-02-10, Duration: 7 days, Interrupted: False\n"
    "[2024-04-05T00:00:00] Event: Business Dinner, Start: 2024-04-05, Duration: 1 days, Interrupted: False\n\n"
    "Question: Which events overlap in time with Gym Training (2024-01-01~2024-03-31)? "
    "Output \"event name (start date)\".\n"
    "Thought: "
)

_SHOT_LIST_OUTPUT = (
    "Gym Training runs from 2024-01-01 to 2024-03-31 (90 days).\n"
    "- Common Cold: 2024-02-10 to 2024-02-17 → intersects with Gym Training ✓\n"
    "- Business Dinner: 2024-04-05 → after 2024-03-31 → no overlap ✗\n"
    "Answer: Common Cold（2024-02-10）"
)

# Backward compatible: keep old variable names for external references
_ONE_SHOT_INPUT = _SHOT_NUMERIC_INPUT
_ONE_SHOT_OUTPUT = _SHOT_NUMERIC_OUTPUT


def _build_rag_input(query: str, context_chunks: list[str]) -> str:
    """Build RAG user input (replicates HippoRAG prompt concatenation format)"""
    context = "\n\n".join(context_chunks)
    return f"{context}\n\nQuestion: {query}\nThought: "


# ============================================================
# OpenIE: NER + Triple extraction (original HippoRAG OpenIE pipeline)
# ============================================================

_OPENIE_SYSTEM = (
    "Your task is to extract named entities and RDF triples from the given paragraph.\n\n"
    "Respond with a JSON object:\n"
    '{"named_entities": ["entity1", "entity2", ...], "triples": [["subject", "predicate", "object"], ...]}\n\n'
    "Entity extraction:\n"
    "- Extract all meaningful entities: person names, dates, event names, indicator names, "
    "medication names, medical terms, organizations, and numeric values with units.\n\n"
    "Triple extraction:\n"
    "- Each triple should contain at least one of the extracted entities.\n"
    "- Clearly resolve pronouns to their specific names.\n"
    "- For health events, extract relationships like: started_on, duration_days, interrupted.\n"
    "- For device/exam measurements, extract: measured_on, indicator_name, value, unit."
)


async def _extract_ner_and_triples(passage: str, model: str) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Extract named entities and RDF triples in a single LLM call"""
    import re as _re
    result = await do_execute(
        model=model,
        system_prompt=_OPENIE_SYSTEM,
        input=passage[:12000],
        timeout=60,
    )
    entities: list[str] = []
    triples: list[tuple[str, str, str]] = []
    try:
        match = _re.search(r'\{[^{}]*"named_entities"\s*:\s*\[.*?"triples"\s*:\s*\[.*?\]\s*\}', result.content, _re.DOTALL)
        if not match:
            match = _re.search(r'\{[^{}]*"triples"\s*:\s*\[.*?"named_entities"\s*:\s*\[.*?\]\s*\}', result.content, _re.DOTALL)
        if match:
            data = json.loads(match.group())
            entities = list(dict.fromkeys(data.get("named_entities", [])))
            for t in data.get("triples", []):
                if isinstance(t, list) and len(t) == 3:
                    triples.append((str(t[0]).lower().strip(), str(t[1]).lower().strip(), str(t[2]).lower().strip()))
            triples = list(set(triples))
    except (json.JSONDecodeError, KeyError):
        pass
    return entities, triples


def _chunks_hash(chunks: list[str]) -> str:
    """Compute MD5 hash of chunks list for cache validation"""
    import hashlib
    h = hashlib.md5()
    for c in sorted(chunks):
        h.update(c.encode("utf-8"))
    return h.hexdigest()


async def _batch_openie(
    chunks: list[str],
    user_dir: Path,
    model: str,
    openie_indices: set[int] | None = None,
    concurrency: int = 3,
) -> list[dict]:
    """Batch OpenIE (NER + Triple) extraction with disk cache

    Args:
        openie_indices: Set of chunk indices to run OpenIE on, None means all
        concurrency: OpenIE concurrency limit
    """
    import asyncio as _asyncio
    cache_file = user_dir / ".openie_cache.json"
    current_hash = _chunks_hash(chunks)

    if cache_file.exists():
        try:
            with open(cache_file, encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("chunks_hash") == current_hash:
                logger.info("[HippoRAG] OpenIE cache hit for %s (%d chunks)", user_dir.name, len(chunks))
                return cached["openie_results"]
        except (json.JSONDecodeError, KeyError):
            pass

    target_count = len(openie_indices) if openie_indices is not None else len(chunks)
    logger.info(
        "[HippoRAG] Running OpenIE on %d/%d chunks for %s (model=%s, concurrency=%d)",
        target_count, len(chunks), user_dir.name, model, concurrency,
    )

    sem = _asyncio.Semaphore(concurrency)

    async def _do_ner(idx: int, chunk: str) -> dict:
        # Skip chunks that don't need OpenIE
        if openie_indices is not None and idx not in openie_indices:
            return {"chunk_idx": idx, "entities": [], "triples": []}
        max_retries = 2
        for attempt in range(1, max_retries + 1):
            try:
                async with sem:
                    entities, triples = await _extract_ner_and_triples(chunk, model)
                    return {"chunk_idx": idx, "entities": entities, "triples": [list(t) for t in triples]}
            except Exception as e:
                if attempt < max_retries:
                    logger.warning("[HippoRAG] OpenIE chunk %d attempt %d failed (%s), retrying...", idx, attempt, type(e).__name__)
                else:
                    logger.warning("[HippoRAG] OpenIE failed for chunk %d after %d attempts: %s: %s", idx, max_retries, type(e).__name__, e)
        return {"chunk_idx": idx, "entities": [], "triples": []}

    tasks = [_do_ner(i, c) for i, c in enumerate(chunks)]
    results = await _asyncio.gather(*tasks)
    results = sorted(results, key=lambda x: x["chunk_idx"])

    cache_data = {"chunks_hash": current_hash, "openie_results": results}
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        logger.info("[HippoRAG] OpenIE cache saved to %s", cache_file)
    except OSError as e:
        logger.warning("[HippoRAG] Failed to save OpenIE cache: %s", e)

    return results


# Graph build concurrency limit: max 2 users building simultaneously (entity embedding is memory bottleneck)
_GRAPH_BUILD_SEMAPHORE: asyncio.Semaphore | None = None


def _get_graph_build_semaphore() -> asyncio.Semaphore:
    global _GRAPH_BUILD_SEMAPHORE
    if _GRAPH_BUILD_SEMAPHORE is None:
        _GRAPH_BUILD_SEMAPHORE = asyncio.Semaphore(2)
    return _GRAPH_BUILD_SEMAPHORE


async def _build_or_get_graph(
    chunks: list[str],
    embeddings,  # 未使用，签名保留兼容；调用方传 np.ndarray 或 list 都可
    openie_results: list[dict],
    user_dir: Path,
    data_group: str,
    embedding_model: str,
    max_chunk_chars: int,
) -> tuple:
    """Build or get user's knowledge graph (memory cache + disk persistence)

    Returns:
        (graph, entity_list, entity_embeddings_np, chunk_idx_to_node)
    """
    import pickle

    import numpy as np

    lock_key = f"{user_dir}:{max_chunk_chars}:{embedding_model}"
    lock = _GRAPH_LOCKS.get(lock_key)
    if lock is None:
        lock = asyncio.Lock()
        _GRAPH_LOCKS[lock_key] = lock

    async with lock:
        # --- Disk cache check (.index_done exists means both index + graph are complete) ---
        work_dir = _BENCHMARK_DATA_DIR / data_group / ".hippo_work_dirs" / user_dir.name
        index_done = work_dir / ".index_done"
        entity_file = work_dir / "entity_list.json"
        ent_emb_file = work_dir / "entity_embeddings.npz"
        graph_file = work_dir / "graph.pkl"

        if index_done.exists() and entity_file.exists() and ent_emb_file.exists() and graph_file.exists():
            logger.info("[HippoRAG] Graph disk cache hit for %s", user_dir.name)
            with open(entity_file, encoding="utf-8") as f:
                edata = json.load(f)
            ent_emb_np = np.load(ent_emb_file)["embeddings"]
            with open(graph_file, "rb") as f:
                G = pickle.load(f)
            chunk_idx_to_node = {int(k): v for k, v in edata["chunk_idx_to_node"].items()}
            return (G, edata["entity_list"], ent_emb_np, chunk_idx_to_node)

        # --- Serialized graph build (limit concurrent graph builds to prevent entity embedding memory spikes) ---
        async with _get_graph_build_semaphore():
            return await _do_build_graph(
                chunks=chunks,
                openie_results=openie_results,
                user_dir=user_dir,
                embedding_model=embedding_model,
                work_dir=work_dir,
                index_done=index_done,
                entity_file=entity_file,
                ent_emb_file=ent_emb_file,
                graph_file=graph_file,
            )


async def _do_build_graph(
    chunks: list[str],
    openie_results: list[dict],
    user_dir: Path,
    embedding_model: str,
    work_dir: Path,
    index_done: Path,
    entity_file: Path,
    ent_emb_file: Path,
    graph_file: Path,
) -> tuple:
    """Actual graph build (executed under semaphore protection)"""
    import pickle

    import networkx as nx
    import numpy as np

    G = nx.Graph()

    # Collect all entities (deduplicated)
    all_entities: set[str] = set()
    for r in openie_results:
        for e in r["entities"]:
            all_entities.add(e.lower().strip())
        for t in r["triples"]:
            all_entities.add(str(t[0]).lower().strip())
            all_entities.add(str(t[2]).lower().strip())
    all_entities.discard("")
    entity_list = sorted(all_entities)

    if not entity_list:
        logger.warning("[HippoRAG] No entities extracted, will fall back to dense retrieval")
        return (G, [], np.array([]), {})

    # Entity nodes
    for ent in entity_list:
        G.add_node(f"ent:{ent}", type="entity", name=ent)

    # Passage nodes + passage<->entity edges
    chunk_idx_to_node: dict[int, str] = {}
    for r in openie_results:
        idx = r["chunk_idx"]
        pnode = f"passage:{idx}"
        G.add_node(pnode, type="passage", chunk_idx=idx)
        chunk_idx_to_node[idx] = pnode

        seen_ents = set()
        for e in r["entities"]:
            ent_key = e.lower().strip()
            if ent_key and ent_key not in seen_ents:
                G.add_edge(pnode, f"ent:{ent_key}", weight=1.0, type="passage_entity")
                seen_ents.add(ent_key)

    # Fact edges (entity <-> entity)
    for r in openie_results:
        for t in r["triples"]:
            subj = str(t[0]).lower().strip()
            obj = str(t[2]).lower().strip()
            if subj and obj and subj != obj:
                n1, n2 = f"ent:{subj}", f"ent:{obj}"
                if G.has_edge(n1, n2):
                    G[n1][n2]["weight"] += 1.0
                else:
                    G.add_edge(n1, n2, weight=1.0, type="fact")

    # Entity embeddings + synonymy edges
    logger.info("[HippoRAG] Encoding %d entities for synonymy edges", len(entity_list))
    ent_embeddings = await _get_embeddings(entity_list, model=embedding_model)
    ent_emb_np = np.array(ent_embeddings, dtype=np.float32)

    norms = np.linalg.norm(ent_emb_np, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    ent_emb_normed = ent_emb_np / norms
    del ent_emb_np  # 释放未归一化副本

    # 分块计算 synonymy edges，避免分配完整 NxN sim_matrix（N=9000 时 648MB）
    SYN_THRESHOLD = 0.85
    BLOCK_SIZE = 512
    n = len(entity_list)
    syn_count = 0
    for i_start in range(0, n, BLOCK_SIZE):
        i_end = min(i_start + BLOCK_SIZE, n)
        # 只计算上三角：j >= i_start，所以 block @ full.T 然后只取 j > i 的部分
        sim_block = ent_emb_normed[i_start:i_end] @ ent_emb_normed[i_start:].T  # (block, n-i_start) float32
        for bi in range(i_end - i_start):
            i = i_start + bi
            # sim_block[bi] 对应 entity_list[i] vs entity_list[i_start:]
            # 只看 j > i 的部分
            j_offset = i - i_start + 1  # 跳过 j <= i
            row = sim_block[bi, j_offset:]
            hits = np.where(row >= SYN_THRESHOLD)[0]
            for h in hits:
                j = i + 1 + h
                ni, nj = f"ent:{entity_list[i]}", f"ent:{entity_list[j]}"
                if not G.has_edge(ni, nj):
                    G.add_edge(ni, nj, weight=float(row[h]), type="synonymy")
                    syn_count += 1
        del sim_block
    del ent_emb_normed  # graph 构建完毕，释放
    # 重新加载 ent_emb_np 用于后续检索（从 embeddings 列表，不重新归一化）
    ent_emb_np = np.array(ent_embeddings, dtype=np.float32)
    logger.info("[HippoRAG] Found %d synonymy edges (threshold=%.2f)", syn_count, SYN_THRESHOLD)

    logger.info(
        "[HippoRAG] Graph built: %d nodes (%d entities, %d passages), %d edges",
        G.number_of_nodes(), len(entity_list), len(chunk_idx_to_node), G.number_of_edges(),
    )

    # --- Save to disk ---
    work_dir.mkdir(parents=True, exist_ok=True)
    with open(entity_file, "w", encoding="utf-8") as f:
        json.dump({
            "entity_list": entity_list,
            "chunk_idx_to_node": {str(k): v for k, v in chunk_idx_to_node.items()},
        }, f, ensure_ascii=False)
    np.savez_compressed(str(ent_emb_file), embeddings=ent_emb_np.astype(np.float32))
    with open(graph_file, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    index_done.touch()
    logger.info("[HippoRAG] Graph saved to disk: %s", work_dir)

    return (G, entity_list, ent_emb_np, chunk_idx_to_node)


async def _retrieve_with_ppr(
    query: str,
    chunks: list[str],
    chunk_embeddings,  # np.ndarray (N, D) float32（向后兼容 list 输入）
    graph,  # nx.Graph
    entity_list: list[str],
    entity_embeddings,  # np.ndarray
    chunk_idx_to_node: dict[int, str],
    top_k: int,
    embedding_model: str,
    damping: float = 0.5,
    linking_top_k: int = 10,
) -> list[str]:
    """HippoRAG-style graph retrieval: query->entity matching + PPR -> passage ranking

    Automatically falls back to pure cosine retrieval when graph is empty.
    """
    import networkx as nx
    import numpy as np

    # fallback: empty graph
    if graph.number_of_nodes() == 0 or len(entity_list) == 0:
        return await _retrieve(query, chunks, chunk_embeddings, top_k, embedding_model)

    # 1. query embedding
    query_emb = np.array((await _get_embeddings([query], model=embedding_model))[0])
    norms_q = np.linalg.norm(query_emb)
    if norms_q == 0:
        return await _retrieve(query, chunks, chunk_embeddings, top_k, embedding_model)

    # 2. Query-entity similarity -> top linking_top_k seed entities
    ent_emb_np = np.array(entity_embeddings)
    norms_e = np.linalg.norm(ent_emb_np, axis=1, keepdims=True)
    norms_e = np.where(norms_e == 0, 1.0, norms_e)
    entity_scores = (ent_emb_np / norms_e) @ (query_emb / norms_q)
    top_entity_idxs = np.argsort(entity_scores)[::-1][:linking_top_k]

    # 3. PPR personalization vector
    personalization: dict[str, float] = {}
    for idx in top_entity_idxs:
        score = float(entity_scores[idx])
        if score > 0:
            node_name = f"ent:{entity_list[idx]}"
            if node_name in graph:
                personalization[node_name] = score

    # Add passage dense retrieval scores (weak weight)
    chunk_emb_np = np.asarray(chunk_embeddings, dtype=np.float32)
    norms_c = np.linalg.norm(chunk_emb_np, axis=1, keepdims=True)
    norms_c = np.where(norms_c == 0, 1.0, norms_c)
    passage_scores = (chunk_emb_np / norms_c) @ (query_emb / norms_q)

    PASSAGE_WEIGHT = 0.05
    for chunk_idx, pnode in chunk_idx_to_node.items():
        if chunk_idx < len(passage_scores):
            personalization[pnode] = float(passage_scores[chunk_idx]) * PASSAGE_WEIGHT

    if not personalization:
        return await _retrieve(query, chunks, chunk_embeddings, top_k, embedding_model)

    # 4. PPR
    try:
        ppr_scores = nx.pagerank(
            graph,
            alpha=damping,
            personalization=personalization,
            weight="weight",
            max_iter=100,
            tol=1e-6,
        )
    except nx.PowerIterationFailedConvergence:
        logger.warning("[HippoRAG] PPR failed to converge, falling back to dense retrieval")
        return await _retrieve(query, chunks, chunk_embeddings, top_k, embedding_model)

    # 5. Extract and rank passage scores
    passage_ppr: list[tuple[float, int]] = []
    for chunk_idx, pnode in chunk_idx_to_node.items():
        score = ppr_scores.get(pnode, 0.0)
        passage_ppr.append((score, chunk_idx))
    passage_ppr.sort(reverse=True)

    result_chunks: list[str] = []
    for _, chunk_idx in passage_ppr[:top_k]:
        if chunk_idx < len(chunks):
            result_chunks.append(chunks[chunk_idx])

    # Fallback: fill remaining slots with cosine results when PPR returns fewer than top_k
    if len(result_chunks) < top_k:
        cosine_results = await _retrieve(query, chunks, chunk_embeddings, top_k, embedding_model)
        seen = set(result_chunks)
        for c in cosine_results:
            if c not in seen and len(result_chunks) < top_k:
                result_chunks.append(c)
                seen.add(c)

    return result_chunks


# ============================================================
# Config Model & Agent
# ============================================================


class HippoRagApiTargetInfo(BaseModel):
    """HippoRAG target config — retrieval-augmented generation (aligned with HippoRAG params)"""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "_comment": "Default config (aligned with remote HippoRAG)",
                    "type": "hippo_rag_api",
                    "model": "gemini-3-flash-preview",
                    "data_group": "eslbench",
                    "user_email": "user110@demo",
                },
            ],
        },
    )
    type: Literal["hippo_rag_api"] = Field(description="Target type")
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
    ] = Field(description="Generation model name (default gemini-3-flash-preview, aligned with remote HippoRAG)")
    embedding_model: str = Field(
        "text-embedding-3-large",
        description="Embedding model (default text-embedding-3-large, aligned with remote HippoRAG)",
    )
    data_group: str = Field(description="Data directory (benchmark name, e.g. 'eslbench')")
    user_email: Optional[str] = Field(None, description="User email (maps to .data/{user_dir}/)")
    top_k: int = Field(10, description="Number of documents fed to LLM in QA phase (aligned with remote qa_top_k)", ge=1, le=50)
    max_chunk_chars: int = Field(
        15000,
        description="Large document chunking character limit (aligned with remote max_chunk_chars)",
        ge=1000,
        le=50000,
    )
    openie_model: str = Field(
        "gemini-3-flash-preview",
        description="Model for OpenIE (NER + triple extraction), recommend fast/cheap models",
    )
    openie_concurrency: int = Field(6, description="OpenIE concurrency (NER + triple extraction)", ge=1, le=20)
    damping: float = Field(0.5, description="PPR damping factor (0~1, higher favors global distribution)", ge=0.05, le=0.95)
    linking_top_k: int = Field(10, description="PPR initial seed entity count", ge=1, le=50)
    use_graph: bool = Field(True, description="Enable OpenIE + graph retrieval (False for pure vector RAG)")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt (uses HippoRAG rag_qa_musique template if not specified)")


class HippoRagApiTargetAgent(AbstractTargetAgent, name="hippo_rag_api", params_model=HippoRagApiTargetInfo):
    """Retrieval-Augmented Generation (RAG) — replicates HippoRAG evaluation logic"""

    _display_meta = {
        "icon": (
            "M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375"
            " 3.375 0 00-3.375-3.375H8.25m5.231 13.481L15 17.25m-4.5-15H5.625c-.621 0-1.125.504-1.125"
            " 1.125v16.5c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0"
            " 00-9-9zm3.75 11.625a2.625 2.625 0 11-5.25 0 2.625 2.625 0 015.25 0z"
        ),
        "color": "#8b5cf6",
        "features": ["RAG", "Vector Search", "HippoRAG"],
        "icon_url": "/static/images/agents/hipporag.png",
    }
    _cost_meta = {
        "est_input_tokens": 8000,
        "est_output_tokens": 800,
    }

    def __init__(self, target_config: HippoRagApiTargetInfo, history: list[BaseMessage] | None = None):
        super().__init__(target_config, history=history)
        self.config: HippoRagApiTargetInfo = target_config

        # Conversation history
        self._conversation: list[BasicMessage] = []
        for msg in self.history:
            if isinstance(msg, HumanMessage):
                self._conversation.append(BasicMessage(role="user", content=msg.content))
            elif isinstance(msg, AIMessage):
                self._conversation.append(BasicMessage(role="assistant", content=msg.content))

        # User data directory
        self._user_dir: Path | None = None
        if target_config.user_email:
            dir_name = _email_to_dir(target_config.user_email)
            self._user_dir = paths.user_data_dir(target_config.data_group) / dir_name
            logger.info(
                "[HippoRAG] user_dir=%s exists=%s data_group=%s",
                self._user_dir,
                self._user_dir.is_dir(),
                target_config.data_group,
            )
            if not self._user_dir.is_dir():
                logger.warning(
                    "[HippoRAG] error_code=USER_DIR_NOT_FOUND user_dir=%s data_group=%s",
                    self._user_dir,
                    target_config.data_group,
                )

        # Cost tracking
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
                message_list=[{"content": "Hello, how can I help you?"}],
            )

        user_text = self._extract_user_input(test_action)

        if not self._user_dir or not self._user_dir.is_dir():
            return TargetAgentReaction(
                type="message",
                message_list=[{"content": f"Error: User data directory not found ({self._user_dir})"}],
            )

        # 1. Build/get vector index
        chunks, embeddings, openie_indices = await _build_or_get_index(
            self._user_dir,
            data_group=self.config.data_group,
            max_chunk_chars=self.config.max_chunk_chars,
            embedding_model=self.config.embedding_model,
        )

        # 2. Retrieval: OpenIE + graph retrieval or pure vector retrieval
        if self.config.use_graph:
            # 2a. OpenIE triple extraction (only on marked chunks, with disk cache)
            openie_results = await _batch_openie(
                chunks=chunks,
                user_dir=self._user_dir,
                model=self.config.openie_model,
                openie_indices=openie_indices,
                concurrency=self.config.openie_concurrency,
            )

            # 2b. Build knowledge graph
            graph, entity_list, entity_embeddings, chunk_idx_to_node = await _build_or_get_graph(
                chunks=chunks,
                embeddings=embeddings,
                openie_results=openie_results,
                user_dir=self._user_dir,
                data_group=self.config.data_group,
                embedding_model=self.config.embedding_model,
                max_chunk_chars=self.config.max_chunk_chars,
            )

            # 2c. PPR graph retrieval
            context_chunks = await _retrieve_with_ppr(
                query=user_text,
                chunks=chunks,
                chunk_embeddings=embeddings,
                graph=graph,
                entity_list=entity_list,
                entity_embeddings=entity_embeddings,
                chunk_idx_to_node=chunk_idx_to_node,
                top_k=self.config.top_k,
                embedding_model=self.config.embedding_model,
                damping=self.config.damping,
                linking_top_k=self.config.linking_top_k,
            )
        else:
            # Pure vector retrieval (original logic)
            context_chunks = await _retrieve(
                query=user_text,
                chunks=chunks,
                embeddings=embeddings,
                top_k=self.config.top_k,
                embedding_model=self.config.embedding_model,
            )

        # 3. Build RAG prompt (replicates HippoRAG rag_qa_musique template)
        rag_input = _build_rag_input(user_text, context_chunks)
        system_prompt = self.config.system_prompt or _RAG_SYSTEM_PROMPT

        logger.debug(
            "[HippoRAG] Query: %s, Retrieved %d/%d chunks, calling %s",
            user_text[:80],
            len(context_chunks),
            len(chunks),
            self.config.model,
        )

        # 4. Call LLM (with few-shot examples: numeric counting + list time overlap)
        history_with_oneshot = [
            BasicMessage(role="user", content=_SHOT_NUMERIC_INPUT),
            BasicMessage(role="assistant", content=_SHOT_NUMERIC_OUTPUT),
            BasicMessage(role="user", content=_SHOT_LIST_INPUT),
            BasicMessage(role="assistant", content=_SHOT_LIST_OUTPUT),
        ]
        # If there's multi-turn conversation history, append after one-shot
        if self._conversation:
            history_with_oneshot.extend(self._conversation)

        result = await do_execute(
            model=self.config.model,
            system_prompt=system_prompt,
            input=rag_input,
            history_messages=history_with_oneshot,
        )

        assistant_content = result.content
        self._accumulate_cost(result.usage)

        # Append conversation history (using original user_text, without retrieval context)
        self._conversation.append(BasicMessage(role="user", content=user_text))
        self._conversation.append(BasicMessage(role="assistant", content=assistant_content))

        logger.debug("[HippoRAG] Response: %d chars", len(assistant_content))

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
