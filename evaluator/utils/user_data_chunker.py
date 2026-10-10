"""用户数据分块工具 — HippoRAG / mem0 等 RAG agent 共享

仅加载 profile.json + timeline.json，不加载 events.json（含答案，会泄露）。

切分策略:
- profile: 整体 1 chunk, 标记 OpenIE
- event: 每个事件独立 chunk, 标记 OpenIE
- exam_indicator: 按检查日期聚合, 标记 OpenIE
- device_indicator 周摘要: 统计聚合(mean/min/max/trend/异常), 标记 OpenIE
- device_indicator 原始数据: 按天聚合, 只做 embedding
"""

import json
import logging
from datetime import datetime as _dt
from pathlib import Path

logger = logging.getLogger(__name__)

# benchmark/data/ 目录
BENCHMARK_DATA_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "data"


def email_to_dir(email: str) -> str:
    """邮箱 → 目录名: user110@demo → user110_AT_demo"""
    return email.replace("@", "_AT_")


def resolve_user_dir(data_group: str, user_email: str) -> Path:
    """解析用户数据目录路径"""
    return BENCHMARK_DATA_DIR / data_group / ".data" / email_to_dir(user_email)


def chunk_by_chars(text: str, max_chars: int, overlap: int = 500) -> list[str]:
    """按字符数切分，带重叠"""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def load_user_documents(user_dir: Path, max_chunk_chars: int = 15000) -> tuple[list[str], set[int]]:
    """加载用户数据文件，按类型智能切分

    返回 (chunks, openie_indices):
    - chunks: 所有文本块（全部做 embedding）
    - openie_indices: 需要做 OpenIE 的 chunk 索引集合

    注意: 只加载 profile.json + timeline.json，不加载 events.json（含答案）
    """
    chunks: list[str] = []
    openie_indices: set[int] = set()

    # --- 加载 profile ---
    profile_file = user_dir / "profile.json"
    if profile_file.exists():
        try:
            with open(profile_file, encoding="utf-8") as f:
                profile = json.load(f)
            chunks.append(f"[PROFILE]\n{json.dumps(profile, ensure_ascii=False, indent=2)}")
            openie_indices.add(len(chunks) - 1)
        except (json.JSONDecodeError, OSError):
            pass

    # --- 加载 timeline ---
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

    # 分类
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

    # --- 事件 chunks（每个独立, 做 OpenIE）---
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

    # --- 体检 chunks（按日期聚合, 做 OpenIE）---
    for date, exams in sorted(exam_by_date.items()):
        lines = [f"[EXAM] Date: {date}, Location: {exams[0].get('exam_location', '?')}, Type: {exams[0].get('exam_type', '?')}"]
        for e in exams:
            lines.append(f"  {e.get('indicator', '?')} = {e.get('value', '?')} {e.get('unit', '')}")
        chunks.append("\n".join(lines))
        openie_indices.add(len(chunks) - 1)

    # --- Device: 周摘要 chunks（统计聚合, 做 OpenIE）---
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
            if max_val > mean_val * 1.5 and len(values) >= 3:
                anomalies.append(f"{name} peaked at {max_val:.2f} (mean {mean_val:.2f})")
        if anomalies:
            lines.append(f"  Notable: {'; '.join(anomalies)}")
        chunks.append("\n".join(lines))
        openie_indices.add(len(chunks) - 1)

    # --- Device: 原始数据 chunks（按天聚合, 只做 embedding，不做 OpenIE）---
    for day in sorted(device_by_day.keys()):
        indicators = device_by_day[day]
        lines = [f"[DEVICE RAW] Date: {day}"]
        for ind in indicators:
            lines.append(f"  {ind.get('indicator', '?')} = {ind.get('value', '?')} {ind.get('unit', '')}")
        text = "\n".join(lines)
        if len(text) > max_chunk_chars:
            for sub in chunk_by_chars(text, max_chunk_chars):
                chunks.append(sub)
        else:
            chunks.append(text)
        # 不加入 openie_indices

    return chunks, openie_indices
