"""读取和写入 benchmark/report/ 目录下的报告文件"""

from __future__ import annotations

import functools
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluator.core.schema import ReportEntry, TargetInfo
from evaluator.utils import paths

logger = logging.getLogger(__name__)

_REPORT_DIR = paths.report_dir()
_DATA_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "data"
# 匹配: {prefix}_{YYYYMMDD_HHMMSS}.json — prefix 可能含 target label
_REPORT_FILENAME_RE = re.compile(r"^(.+)_(\d{8}_\d{6}|\d{8})\.json$")


def _resolve_dataset_and_target(prefix: str, bench_name: str) -> tuple[str, str]:
    """从文件名前缀中分离 dataset 和 target_label

    新格式: sample_gpt-4.1 → ("sample", "gpt-4.1")
    旧格式: sample → ("sample", "")
    """
    # 先检查整个 prefix 是否就是一个已知 dataset
    data_dir = _DATA_DIR / bench_name
    known = {p.stem for p in data_dir.glob("*.jsonl")} if data_dir.is_dir() else set()

    if prefix in known:
        return prefix, ""

    # 尝试拆分: dataset_target_label（dataset 优先匹配最长的已知名称）
    for ds in sorted(known, key=len, reverse=True):
        if prefix.startswith(ds + "_"):
            return ds, prefix[len(ds) + 1 :]

    # 兜底：整个作为 dataset
    return prefix, ""


def _add_usage(acc: dict[str, dict[str, Any]], usage: dict[str, Any] | None) -> None:
    """Sum {model: {input_tokens, output_tokens, input_token_details.cache_read}} into acc."""
    for model, u in (usage or {}).items():
        if not isinstance(u, dict):
            continue
        a = acc.setdefault(model, {"input_tokens": 0, "output_tokens": 0, "input_token_details": {"cache_read": 0}})
        a["input_tokens"] += u.get("input_tokens") or 0
        a["output_tokens"] += u.get("output_tokens") or 0
        a["input_token_details"]["cache_read"] += (u.get("input_token_details") or {}).get("cache_read") or 0


@functools.lru_cache(maxsize=1024)
def _summarize_report(path: str, mtime_ns: int, size: int) -> dict[str, Any]:
    """Case count, average, ungraded count and summed usage of one report file.

    Keyed on (path, mtime, size) so an unchanged file is parsed once per process. The
    average follows the report page and the CLI: `error` cases have no grade and stay out
    of it. Usage is summed, not priced: the page prices it with the same table it uses
    for a single report, so the list and the report always show the same amount.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data.get("cases") or []
    graded = [
        c["eval"]["score"]
        for c in cases
        if (c.get("eval") or {}).get("result") not in (None, "error") and c["eval"].get("score") is not None
    ]
    usage: dict[str, Any] = {"test": {}, "target": {}, "eval": {}, "target_api_cost": 0.0}
    for c in cases:
        cost = c.get("cost") or {}
        _add_usage(usage["test"], cost.get("test"))
        _add_usage(usage["eval"], cost.get("eval"))
        api_cost = (cost.get("target_detail") or {}).get("total_cost")
        if api_cost is not None:
            usage["target_api_cost"] += api_cost
        else:
            _add_usage(usage["target"], cost.get("target"))
    return {
        "case_count": len(cases),
        "avg_score": sum(graded) / len(graded) if graded else None,
        "error_count": sum(1 for c in cases if (c.get("eval") or {}).get("result") == "error"),
        "usage": usage,
    }


def _report_summary(report_file: Path) -> dict[str, Any]:
    st = report_file.stat()
    try:
        return _summarize_report(str(report_file), st.st_mtime_ns, st.st_size)
    except (OSError, ValueError, KeyError, TypeError) as e:
        logger.warning("[report] cannot summarize %s: %s", report_file, e)
        return {}


def list_reports() -> list[ReportEntry]:
    """列出所有报告文件（按时间倒序，最新在前）"""
    if not _REPORT_DIR.is_dir():
        logger.debug("[report] root=%s does not exist", _REPORT_DIR)
        return []

    result: list[ReportEntry] = []
    bench_count = 0
    for bench_dir in sorted(_REPORT_DIR.iterdir()):
        if not bench_dir.is_dir() or bench_dir.name.startswith((".", "_")):
            continue
        bench_count += 1
        for report_file in bench_dir.glob("*.json"):
            match = _REPORT_FILENAME_RE.match(report_file.name)
            if match:
                prefix, date = match.group(1), match.group(2)
                dataset, target_label = _resolve_dataset_and_target(prefix, bench_dir.name)
                result.append(
                    ReportEntry(
                        benchmark=bench_dir.name,
                        dataset=dataset,
                        date=date,
                        filename=report_file.name,
                        target_label=target_label,
                        **_report_summary(report_file),
                    )
                )

    # 全局按时间倒序（date 格式为 YYYYMMDD_HHMMSS 或 YYYYMMDD，字典序即时间序）
    result.sort(key=lambda r: r.date, reverse=True)
    logger.debug("[report] root=%s found %d reports across %d benchmarks", _REPORT_DIR, len(result), bench_count)
    return result


def get_report_content(benchmark: str, filename: str) -> dict[str, Any]:
    """读取单份报告内容"""
    report_path = (_REPORT_DIR / benchmark / filename).resolve()
    # 安全检查：确保路径在 report 目录内
    if not report_path.is_relative_to(_REPORT_DIR.resolve()):
        raise ValueError("非法路径")
    if not report_path.exists():
        raise FileNotFoundError(f"报告不存在: {benchmark}/{filename}")
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def delete_report(benchmark: str, filename: str) -> None:
    """删除单份报告文件"""
    report_path = (_REPORT_DIR / benchmark / filename).resolve()
    # 安全检查：确保路径在 report 目录内
    if not report_path.is_relative_to(_REPORT_DIR.resolve()):
        raise ValueError("非法路径")
    if not report_path.exists():
        raise FileNotFoundError(f"报告不存在: {benchmark}/{filename}")
    report_path.unlink()
    logger.info("报告已删除: %s", report_path)


def _make_target_label(target: "TargetInfo | None") -> str:
    """从 TargetInfo 生成文件名安全的标签（通用，不感知具体 plugin 类型）

    格式: {type}[_{model}][_{agent}][_k{top_k}]
    - type:  始终包含（区分被测系统类型）
    - model: LLM 模型名（如有）
    - agent: Agent 类型（如有）
    - top_k: 检索数量（如有，RAG 系列）

    示例:
        llm_api + gpt-4.1           → llm_api_gpt-4.1
        mirobody + Deep             → mirobody_Deep
        hippo_rag_api + gemini-3-flash + k10 → hippo_rag_api_gemini-3-flash-preview_k10
        eval-only                   → eval-only
    """
    if target is None:
        return "eval-only"

    parts = [target.type]

    model = getattr(target, "model", None)
    if model:
        parts.append(str(model).replace("/", "-"))

    agent = getattr(target, "agent", None)
    if agent:
        parts.append(str(agent))

    top_k = getattr(target, "top_k", None)
    if top_k is not None:
        parts.append(f"k{top_k}")

    return "_".join(parts)


def save_bench_report(report: "BenchReport", benchmark: str, dataset: str) -> Path:  # noqa: F821
    """保存 BenchReport 到 report/ 目录

    文件名格式: {dataset}_{target_label}_{YYYYMMDD_HHmmss}.json
    target_label = {type}[_{model}][_{agent}][_k{top_k}]
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_label = _make_target_label(report.runtime_target)
    filename = f"{dataset}_{target_label}_{timestamp}.json"
    report_path = _REPORT_DIR / benchmark / filename
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            report.model_dump(mode="json"),
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    size = report_path.stat().st_size
    logger.info("[report] saved %s/%s (%d bytes) -> %s", benchmark, filename, size, report_path)
    return report_path
