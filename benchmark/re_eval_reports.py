"""
Re-evaluate existing reports using the current eval config (no agent re-run).

Reads each saved report, reconstructs (TestCase, memory_list) from the embedded
trace, runs do_batch_eval with the current eval-agent default model, and writes
a new report alongside the original.

Use case: switch the judge LLM (e.g., gpt-4.1 → gpt-5.4) without paying the
agent-side LLM cost again.

Usage:
  uv run python -m benchmark.re_eval_reports \\
    benchmark/report/eslbench/sample50-20260331_llm_api_gpt-5.4_20260506_045743.json \\
    benchmark/report/eslbench_distractor/sample_*_20260506_*.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from evaluator.core.bench_schema import bench_item_to_test_case
from evaluator.core.orchestrator import do_batch_eval
from evaluator.core.schema import (
    SessionInfo,
    TargetAgentReaction,
    TestAgentAction,
    TestAgentMemory,
    TestAgentReaction,
)
from evaluator.utils.benchmark_reader import _read_metadata, load_bench_items

# Trigger plugin registration (sub-packages import their modules)
import evaluator.plugin.eval_agent  # noqa: F401
import evaluator.plugin.target_agent  # noqa: F401
import evaluator.plugin.test_agent  # noqa: F401

logger = logging.getLogger(__name__)


def _bench_index(jsonl: Path) -> dict[str, object]:
    """Map BenchItem.id → BenchItem for the source dataset."""
    items = load_bench_items(jsonl)
    return {it.id: it for it in items}


def _restore_memory(trace: dict) -> list[TestAgentMemory]:
    """Rebuild memory_list from trace.test_memory entries."""
    raw_turns = trace.get("test_memory") or []
    memory: list[TestAgentMemory] = []
    for t in raw_turns:
        tr = t.get("test_reaction") or {}
        action = tr.get("action") or {}
        reaction = TestAgentReaction(
            action=TestAgentAction(**action) if action else TestAgentAction(type="semantic"),
            next_fuzzy_action=tr.get("next_fuzzy_action"),
            reason=tr.get("reason") or "",
            is_finished=tr.get("is_finished", False),
            usage=tr.get("usage"),
        )
        target = t.get("target_response")
        target_response = TargetAgentReaction(**target) if target else None
        memory.append(
            TestAgentMemory(
                test_reaction=reaction,
                test_reaction_time=t.get("test_reaction_time") or datetime.now(),
                target_response=target_response,
                target_response_time=t.get("target_response_time"),
            )
        )
    return memory


_KNOWN_TARGET_TYPES = (
    "hippo_rag_api",
    "dyg_rag_api",
    "naive_rag_api",
    "hermes",
    "evermem",
    "mem0_rag_api",
    "llm_api",
)


def _parse_filename_tokens(report_path: Path) -> list[str]:
    """Strip trailing timestamp + reeval marker, return remaining underscore-split tokens."""
    parts = report_path.stem.split("_")
    if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
        parts = parts[:-2]
    if parts and parts[-1] == "reeval":
        parts = parts[:-1]
    return parts


def _parse_original_target_type(report_path: Path) -> str | None:
    """Recover original target_type from a report filename. Reeval reports save
    target_type='eval_only' on cases, but the filename still encodes the original
    target_label (e.g., full-20260430_llm_api_gpt-5.4_reeval_<ts>.json → 'llm_api')."""
    parts = _parse_filename_tokens(report_path)
    if len(parts) < 2:
        return None
    rest = "_".join(parts[1:])
    for tt in _KNOWN_TARGET_TYPES:
        if rest == tt or rest.startswith(tt + "_"):
            return tt
    return None


def _resolve_dataset(report_path: Path) -> Path:
    """Infer benchmark/dataset jsonl path from report filename.

    Naming: {dataset}_{target_label}_{YYYYMMDD_HHMMSS}.json under benchmark/report/{benchmark}/
    """
    from evaluator.utils.benchmark_reader import resolve_dataset

    parts = _parse_filename_tokens(report_path)
    # Conservative: take the first token as dataset (works for "full-20260430_..." / "sample_...")
    dataset = parts[0] if parts else report_path.stem
    benchmark = report_path.parent.name
    # Through the shared resolver, so re-scoring a report finds a fetched bank the
    # same way the original run did. Resolving it here on its own meant a report
    # produced from a fetched release could not be re-scored at all.
    candidate, _ = resolve_dataset(benchmark, dataset)
    if not candidate.exists():
        raise FileNotFoundError(f"Cannot resolve source dataset for {report_path} (tried {candidate})")
    return candidate


async def re_eval_one(report_path: Path, max_concurrency: int = 8) -> Path:
    report_path = Path(report_path)
    if not report_path.exists():
        raise FileNotFoundError(report_path)

    src_jsonl = _resolve_dataset(report_path)
    original_target_type = _parse_original_target_type(report_path)
    logger.info(
        "[re-eval] %s → using source %s (original target_type=%s)",
        report_path.name, src_jsonl.relative_to(src_jsonl.parents[3]), original_target_type,
    )
    bench_items = _bench_index(src_jsonl)
    # Eval-only doesn't actually call the target — pick the lightest TargetSpec we can build.
    # Prefer llm_api (only `model` is required) when declared, else fall back to first.
    metadata = _read_metadata(src_jsonl.parent)
    targets = (metadata or {}).get("target") or []
    if not targets:
        raise RuntimeError(f"no target spec in {src_jsonl.parent}/metadata.json")
    from evaluator.core.schema import TargetSpec  # local import (registry already loaded)
    spec_dict = next((t for t in targets if t.get("type") == "llm_api"), targets[0])
    target_spec = TargetSpec(**spec_dict)

    report = json.loads(report_path.read_text(encoding="utf-8"))

    # Reconstruct session_info from each case's original target_type so contextual cases
    # are judged with the right has_user_data flag. Only hippo_rag_api / dyg_rag_api are
    # assumed record-aware unconditionally — the other RAG targets fall through to the
    # per-case override lookup below, and get has_user_data=False when the case carries no
    # override key for them. For llm_api, tool_context.user_email counts as user-data
    # access — that's the tool-based retrieval path used when evaluating base LLMs.
    def _build_session_info(orig_target_type: str | None, item: object) -> SessionInfo | None:
        if not orig_target_type:
            return None
        if orig_target_type in ("hippo_rag_api", "dyg_rag_api"):
            return SessionInfo(user_token="", has_user_data=True)
        overrides = getattr(getattr(item, "user", None), "target_overrides", None) or {}
        case_override = overrides.get(orig_target_type) or {}
        tool_ctx = case_override.get("tool_context") or {}
        if tool_ctx.get("user_email") or case_override.get("user_email") or case_override.get("email"):
            return SessionInfo(user_token="", has_user_data=True)
        return SessionInfo(user_token="", has_user_data=False)

    # Build (TestCase, memory_list, session_info) tuples
    eval_items: list = []
    skipped = 0
    for c in report.get("cases", []):
        cid = c.get("id")
        item = bench_items.get(cid)
        if not item:
            logger.warning("[re-eval] skipping %s: not in source jsonl", cid)
            skipped += 1
            continue
        trace = (c.get("eval") or {}).get("trace") or {}
        memory_list = _restore_memory(trace)
        if not memory_list:
            logger.warning("[re-eval] skipping %s: empty memory_list", cid)
            skipped += 1
            continue
        # Convert BenchItem → TestCase using its own params/target_overrides
        try:
            tc = bench_item_to_test_case(item, target_spec)
        except Exception as e:
            logger.warning("[re-eval] %s test_case build failed: %s", cid, e)
            skipped += 1
            continue
        # session_info from the report's recorded target_type (NOT the eval_only used during reeval).
        # When re-eval'ing a previously re-eval'd report, target_type is "eval_only" — fall back
        # to the original target_type encoded in the filename.
        recorded_type = c.get("target_type")
        effective_type = recorded_type if recorded_type and recorded_type != "eval_only" else original_target_type
        session_info = _build_session_info(effective_type, item)
        eval_items.append((tc, memory_list, session_info))

    if skipped:
        logger.warning("[re-eval] skipped %d/%d cases", skipped, len(report.get("cases", [])))

    if not eval_items:
        raise RuntimeError(f"no evaluable cases for {report_path}")

    logger.info("[re-eval] running %d cases (concurrency=%d)", len(eval_items), max_concurrency)
    new_report = await do_batch_eval(eval_items, max_concurrency=max_concurrency)

    # Output: insert "_reeval" before timestamp suffix
    stem = report_path.stem
    parts = stem.split("_")
    if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
        ts = "_".join(parts[-2:])
        prefix = "_".join(parts[:-2])
        new_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{prefix}_reeval_{new_ts}.json"
    else:
        out_name = f"{stem}_reeval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path = report_path.parent / out_name
    out_path.write_text(
        json.dumps(new_report.model_dump(mode="json"), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("[re-eval] wrote %s", out_path)
    return out_path


async def main_async(report_paths: list[str], max_concurrency: int) -> int:
    out_paths: list[Path] = []
    for rp in report_paths:
        try:
            out = await re_eval_one(Path(rp), max_concurrency)
            out_paths.append(out)
        except Exception as e:
            logger.error("[re-eval] %s failed: %s", rp, e)
    print(f"\n[re-eval] done: {len(out_paths)}/{len(report_paths)} reports re-evaluated")
    for p in out_paths:
        print(f"  → {p}")
    return len(out_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate saved reports with current eval config (no agent re-run)")
    parser.add_argument("reports", nargs="+", help="report file paths (glob expanded by shell)")
    parser.add_argument("-p", "--parallel", type=int, default=8, help="max concurrency (default 8)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    n = asyncio.run(main_async(args.reports, args.parallel))
    sys.exit(0 if n == len(args.reports) else 1)


if __name__ == "__main__":
    main()
