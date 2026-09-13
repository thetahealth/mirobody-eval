"""TaskManager — manages benchmark task lifecycle (BenchItem + BenchReport)"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluator.core.bench_schema import (
    ApiCallResult,
    api_result_to_eval_items,
    bench_item_to_test_case,
    build_bench_report,
    resolve_runtime_target,
)
from evaluator.core.orchestrator import BatchSession, CaseContext, do_batch_eval
from evaluator.core.schema import EvalResult, TargetInfo, TargetSpec, TestCost, TestResult
from evaluator.utils.benchmark_reader import _read_metadata, filter_bench_items, load_bench_items, resolve_data_path
from evaluator.utils.checkpoint import CheckpointManager, CheckpointMeta
from evaluator.utils.live_files import cleanup_live_dir, delete_live_file, read_live_file, write_live_file, write_live_init
from evaluator.utils.report_reader import save_bench_report
from evaluator.utils.task_registry import (
    TaskRegistryEntry,
    is_process_alive,
    list_task_entries,
    read_task_entry,
    write_task_entry,
)

logger = logging.getLogger(__name__)


@dataclass
class TaskEntry:
    """Task entry — contains BatchSession and runtime state"""

    task_id: str
    # Checkpoint ID this task writes to / reuses:
    # - New task: same as task_id
    # - Resumed task: reuses old checkpoint session_id (usually different from task_id)
    checkpoint_session_id: str
    benchmark: str
    dataset: str
    runtime_target: TargetInfo | None = None  # no target in eval-only mode
    session: BatchSession | None = None  # no BatchSession in eval-only mode
    asyncio_task: asyncio.Task | None = None
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "running"  # running | completed | cancelled | error
    report_path: str | None = None
    error: str | None = None
    # checkpoint-related
    cli_overrides: dict | None = None
    data_file_hash: str = ""
    resumed_results: list[TestResult] = field(default_factory=list)
    # eval-only progress tracking
    total: int = 0
    completed: int = 0
    eval_results: list[TestResult] = field(default_factory=list)
    # generic task-lifecycle webhook (透明转发, 不感知具体订阅者)
    callback_url: str | None = None
    callback_secret: str | None = None

    def _release_heavy_data(self) -> None:
        """Release heavy objects, keep only lightweight metadata for API queries"""
        self.session = None
        self.eval_results = []
        self.resumed_results = []
        self.asyncio_task = None


class TaskManager:
    """In-memory task manager — based on BenchItem and BenchReport"""

    def __init__(self):
        self._tasks: dict[str, TaskEntry] = {}

    async def create_task(
        self,
        benchmark: str,
        dataset: str,
        spec: TargetSpec,
        cli_overrides: dict | None = None,
        ids: str | None = None,
        limit: int | None = None,
        max_concurrency: int = 3,
    ) -> TaskEntry:
        """Create and start a benchmark task

        Args:
            benchmark: benchmark type (e.g. "healthbench")
            dataset: dataset name (e.g. "sample")
            spec: TargetSpec from metadata.json
            cli_overrides: target override params from frontend (only editable fields apply)
            ids: comma-separated IDs (run only specified cases)
            limit: max number of cases
            max_concurrency: max concurrency

        Returns:
            TaskEntry
        """
        # 1. Load BenchItems (with $ref resolution)
        path = resolve_data_path(benchmark, dataset)
        metadata = _read_metadata(path.parent)
        params = metadata.get("params") or None
        items = load_bench_items(path, params=params)
        items = filter_bench_items(items, ids=ids, limit=limit)

        if not items:
            raise ValueError(f"No runnable cases after filtering ({benchmark}/{dataset})")

        # 2. Build global runtime target (for logging and reports)
        runtime_target = resolve_runtime_target(spec, cli_overrides)
        logger.info(
            "Creating task: %s/%s, target=%s, model=%s, %d cases",
            benchmark,
            dataset,
            runtime_target.type,
            getattr(runtime_target, "model", "N/A"),
            len(items),
        )

        # 3. Convert BenchItem -> TestCase (with three-layer target merge)
        test_cases = []
        for item in items:
            try:
                test_case = bench_item_to_test_case(item, spec, cli_overrides)
                test_cases.append(test_case)
            except Exception as e:
                logger.error("Case %s conversion failed: %s", item.id, e)
                raise ValueError(f"Case {item.id} conversion failed: {e}") from e

        # 4. Create BatchSession
        session = BatchSession(
            cases=test_cases,
            max_concurrency=max_concurrency,
        )

        # 5. Create TaskEntry
        data_file_hash = CheckpointManager.compute_data_hash(path)
        entry = TaskEntry(
            task_id=session.id,
            checkpoint_session_id=session.id,
            benchmark=benchmark,
            dataset=dataset,
            runtime_target=runtime_target,
            session=session,
            cli_overrides=cli_overrides,
            data_file_hash=data_file_hash,
        )
        self._tasks[session.id] = entry

        # 6. Run in background
        entry.asyncio_task = asyncio.create_task(
            self._run_session(entry, max_concurrency)
        )

        logger.info("Task created: %s (%s/%s, %d cases)", session.id, benchmark, dataset, len(test_cases))
        return entry

    def check_eval(
        self,
        benchmark: str,
        dataset: str,
        results: list[ApiCallResult],
    ) -> dict:
        """Validate submitted results against dataset (does not start evaluation)

        Returns:
            validation_info dict
        """
        path = resolve_data_path(benchmark, dataset)
        metadata = _read_metadata(path.parent)
        params = metadata.get("params") or None
        try:
            items = load_bench_items(path, params=params)
        except ValueError:
            logger.exception("数据集加载失败: %s/%s", benchmark, dataset)
            raise ValueError("数据集加载异常，请联系管理员") from None

        if not items:
            raise ValueError(f"Dataset is empty ({benchmark}/{dataset})")

        dataset_ids = {item.id for item in items}
        submitted_ids = {r.id for r in results}
        matched_ids = dataset_ids & submitted_ids
        extra_ids = sorted(submitted_ids - dataset_ids)
        missed_ids = sorted(dataset_ids - submitted_ids)

        # Tag distribution stats (matched cases only)
        matched_items = [item for item in items if item.id in matched_ids]
        tag_counts: dict[str, int] = {}
        for item in matched_items:
            for tag in item.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return {
            "benchmark": benchmark,
            "dataset": dataset,
            "benchmark_description": metadata.get("description", ""),
            "dataset_total": len(dataset_ids),
            "submitted": len(submitted_ids),
            "matched": len(matched_ids),
            "extra_ids": extra_ids[:50],
            "missed_ids": missed_ids[:50],
            "tag_distribution": tag_counts,
        }

    async def create_eval_task(
        self,
        benchmark: str,
        dataset: str,
        results: list[ApiCallResult],
        max_concurrency: int = 5,
        *,
        callback_url: str | None = None,
        callback_secret: str | None = None,
    ) -> tuple["TaskEntry", dict]:
        """Create eval-only task — runs evaluation only, no dialogue loop

        Returns:
            (TaskEntry, validation_info)
        """
        import uuid

        # 1. Reuse check_eval validation
        validation_info = self.check_eval(benchmark, dataset, results)

        matched_ids = set()
        path = resolve_data_path(benchmark, dataset)
        metadata = _read_metadata(path.parent)
        params = metadata.get("params") or None
        try:
            items = load_bench_items(path, params=params)
        except ValueError:
            logger.exception("数据集加载失败: %s/%s", benchmark, dataset)
            raise ValueError("数据集加载异常，请联系管理员") from None
        dataset_ids = {item.id for item in items}
        submitted_ids = {r.id for r in results}
        matched_ids = dataset_ids & submitted_ids

        if not matched_ids:
            logger.warning(
                "No matching cases: %d submitted, %d in dataset, 0 IDs matched — all cases scored as 0",
                len(submitted_ids), len(dataset_ids),
            )

        # 2. Convert (matched only)
        matched_results = [r for r in results if r.id in matched_ids]
        eval_items = api_result_to_eval_items(matched_results, items)

        # Unmatched cases (in dataset but not submitted), scored as 0
        missed_items = [item for item in items if item.id not in matched_ids]

        logger.info(
            "Creating eval-only task: %s/%s, submitted %d, matched %d, missing %d (scored 0)",
            benchmark, dataset, len(results), len(matched_ids), len(missed_items),
        )

        # 3. Create TaskEntry (total = full dataset count)
        task_id = f"eval_{uuid.uuid4().hex[:12]}"
        entry = TaskEntry(
            task_id=task_id,
            checkpoint_session_id=task_id,
            benchmark=benchmark,
            dataset=dataset,
            total=len(items),
            callback_url=callback_url,
            callback_secret=callback_secret,
        )
        self._tasks[task_id] = entry

        # 4. Run in background
        entry.asyncio_task = asyncio.create_task(
            self._run_eval_session(entry, eval_items, max_concurrency, missed_items=missed_items)
        )

        return entry, validation_info

    async def _run_eval_session(
        self,
        entry: TaskEntry,
        eval_items: list,
        max_concurrency: int,
        *,
        missed_items: list | None = None,
    ) -> None:
        """Run eval-only batch evaluation in background and save report

        missed_items: unmatched BenchItems from dataset, scored 0 and merged into final report
        """
        try:
            started_at = datetime.now()

            def _on_eval_progress(result: TestResult) -> None:
                entry.eval_results.append(result)
                entry.completed = len(entry.eval_results)

            report = await do_batch_eval(
                eval_items, max_concurrency=max_concurrency, on_progress=_on_eval_progress,
            )
            finished_at = datetime.now()

            all_results = list(report.cases)

            # Generate zero-score results for unsubmitted cases
            if missed_items:
                for item in missed_items:
                    zero_result = TestResult(
                        id=item.id,
                        title=item.title or item.id,
                        user_type="eval_only",
                        target_type="eval_only",
                        eval_type=item.eval.evaluator if item.eval else "",
                        eval=EvalResult(result="fail", score=0.0, feedback="Not submitted, scored 0"),
                        cost=TestCost(),
                        start=finished_at,
                        end=finished_at,
                        tags=item.tags or [],
                        eval_config=item.eval.model_dump(exclude={"evaluator"}) if item.eval else None,
                    )
                    all_results.append(zero_result)
                logger.info("Added %d unsubmitted cases (scored 0)", len(missed_items))

            entry.eval_results = all_results
            entry.completed = len(all_results)

            # Build and save BenchReport
            bench_report = build_bench_report(
                test_results=all_results,
                benchmark_name=entry.benchmark,
                dataset_name=entry.dataset,
                runtime_target=None,
                max_concurrency=max_concurrency,
                started_at=started_at,
                finished_at=finished_at,
            )

            report_path = save_bench_report(
                bench_report, benchmark=entry.benchmark, dataset=entry.dataset
            )
            entry.report_path = str(report_path)
            entry.status = "completed"

            logger.info(
                "Eval-only task completed: %s (evaluated %d + unsubmitted %d = %d cases, avg_score=%.2f)",
                entry.task_id, len(report.cases), len(missed_items or []),
                len(all_results), bench_report.avg_score,
            )
            # Fire-and-forget webhook (no-op when callback_url 未设置)
            from web.app.services.webhook import fire_task_webhook
            asyncio.create_task(fire_task_webhook(entry))
            entry._release_heavy_data()

        except Exception as e:
            logger.error("Eval-only task failed: %s - %s", entry.task_id, e, exc_info=True)
            entry.status = "error"
            entry.error = str(e)
            from web.app.services.webhook import fire_task_webhook
            asyncio.create_task(fire_task_webhook(entry))
            entry._release_heavy_data()

    async def _run_session(
        self, entry: TaskEntry, max_concurrency: int, *, checkpoint_session_id: str | None = None
    ) -> None:
        """Run BatchSession in background and save BenchReport"""
        try:
            # Create/reuse checkpoint
            is_resume = checkpoint_session_id is not None
            cp_id = checkpoint_session_id or entry.task_id
            mgr = CheckpointManager(cp_id)

            if not is_resume:
                # New task: save checkpoint metadata
                meta = CheckpointMeta(
                    session_id=cp_id,
                    benchmark=entry.benchmark,
                    dataset=entry.dataset,
                    target_type=entry.runtime_target.type,
                    cli_overrides=entry.cli_overrides,
                    runtime_target=entry.runtime_target.model_dump(mode="json"),
                    case_ids=[c.id for c in entry.session.cases],
                    max_concurrency=max_concurrency,
                    started_at=datetime.now().isoformat(),
                    data_file_hash=entry.data_file_hash,
                )
                mgr.save_meta(meta)

            # Set progress callback (append checkpoint + clean up live files)
            original_on_progress = entry.session.on_progress

            def _checkpoint_callback(session: BatchSession, case_id: str) -> None:
                ctx = session.contexts.get(case_id)
                if ctx and ctx.result:
                    mgr.append_result(ctx.result)
                # Case completed, delete corresponding live file
                delete_live_file(entry.task_id, case_id)
                if original_on_progress:
                    original_on_progress(session, case_id)

            entry.session.on_progress = _checkpoint_callback

            # Set per-turn dialogue callback (write live files)
            for ctx in entry.session.contexts.values():
                ctx.on_turn = lambda c, tid=entry.task_id: write_live_file(tid, c)

            # Write task registry (so CLI/Web share the same discovery mechanism)
            import os
            reg = TaskRegistryEntry(
                task_id=entry.task_id,
                checkpoint_session_id=cp_id,
                benchmark=entry.benchmark,
                dataset=entry.dataset,
                runtime_target=entry.runtime_target.model_dump(mode="json") if entry.runtime_target else None,
                status="running",
                created_at=entry.created_at.isoformat(),
                source="web",
                total=len(entry.session.cases),
                max_concurrency=max_concurrency,
                pid=os.getpid(),
            )
            write_task_entry(reg)

            # Write initial live files (so UI shows cases as init immediately)
            write_live_init(entry.task_id, [c.id for c in entry.session.cases])

            started_at = datetime.now()
            test_report = await entry.session.run()
            finished_at = datetime.now()

            # Merge resumed results
            all_results = list(entry.resumed_results) + list(test_report.cases)

            # Build BenchReport
            bench_report = build_bench_report(
                test_results=all_results,
                benchmark_name=entry.benchmark,
                dataset_name=entry.dataset,
                runtime_target=entry.runtime_target,
                max_concurrency=max_concurrency,
                started_at=started_at,
                finished_at=finished_at,
            )

            # Save report
            report_path = save_bench_report(
                bench_report, benchmark=entry.benchmark, dataset=entry.dataset
            )
            entry.report_path = str(report_path)

            # Clean up live directory
            cleanup_live_dir(entry.task_id)

            if entry.session.cancel_event.is_set():
                entry.status = "cancelled"
                reg.status = "cancelled"
                # Keep checkpoint on cancel to allow future resume
            else:
                entry.status = "completed"
                reg.status = "completed"
                mgr.cleanup()
            reg.report_path = entry.report_path
            write_task_entry(reg)
            entry._release_heavy_data()

        except Exception as e:
            # Keep checkpoint on error to allow future resume
            cleanup_live_dir(entry.task_id)
            logger.error("Task execution failed: %s - %s (checkpoint preserved)", entry.task_id, e, exc_info=True)
            entry.status = "error"
            entry.error = str(e)
            reg.status = "error"
            reg.error = str(e)
            write_task_entry(reg)
            entry._release_heavy_data()

    def cancel_task(self, task_id: str) -> None:
        """Cancel specified task"""
        entry = self._tasks.get(task_id)
        if not entry:
            raise KeyError(f"Task not found: {task_id}")
        if entry.session:
            entry.session.cancel()
        logger.info("Task cancellation requested: %s", task_id)

    def get_task(self, task_id: str) -> TaskEntry | None:
        """Get task entry"""
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[TaskEntry]:
        """List all tasks"""
        return list(self._tasks.values())

    def get_snapshot(self, task_id: str) -> dict[str, Any]:
        """Get task snapshot (JSON serializable)"""
        entry = self._tasks.get(task_id)
        if not entry:
            raise KeyError(f"Task not found: {task_id}")

        # Session already released (completed/error/cancelled or eval-only)
        if entry.session is None:
            # Rebuild cases summary from report file (no cases in memory after session release)
            cases, max_concurrency, runtime_target = _load_report_cases(entry.report_path)
            snap: dict[str, Any] = {
                "id": entry.task_id,
                "status": entry.status,
                "benchmark": entry.benchmark,
                "dataset": entry.dataset,
                "runtime_target": runtime_target,
                "created_at": entry.created_at.isoformat(),
                "report_path": entry.report_path,
                "total": entry.total or len(cases),
                "completed": entry.completed or len(cases),
                "cancelled": False,
                "cases": cases,
                "error": entry.error,
                "max_concurrency": max_concurrency,
            }
            # Stats grouped by tag (eval-only tasks computed from eval_results)
            if entry.eval_results:
                sbt = _compute_test_result_tag_stats(entry.eval_results)
                snap["stats_by_tag"] = sbt
                snap["report_summary"] = _build_report_summary(entry.eval_results, sbt)
            elif entry.report_path:
                stats_by_tag, report_summary = _load_report_summary(entry.report_path)
                snap["stats_by_tag"] = stats_by_tag
                if report_summary:
                    snap["report_summary"] = report_summary
            else:
                snap["stats_by_tag"] = {}
            return snap

        snapshot = entry.session.snapshot()
        snapshot["status"] = entry.status
        snapshot["benchmark"] = entry.benchmark
        snapshot["dataset"] = entry.dataset
        snapshot["runtime_target"] = entry.runtime_target.model_dump(mode="json") if entry.runtime_target else None
        snapshot["created_at"] = entry.created_at.isoformat()
        snapshot["report_path"] = entry.report_path
        snapshot["max_concurrency"] = entry.session.max_concurrency or None
        if entry.error:
            snapshot["error"] = entry.error
        # Resumed task: include previously completed count
        if entry.resumed_results:
            snapshot["resumed_count"] = len(entry.resumed_results)

        # Real-time tag-grouped stats (computed from completed cases)
        snapshot["stats_by_tag"] = _compute_snapshot_tag_stats(snapshot.get("cases", {}))

        return snapshot

    def discover_external_tasks(self) -> list[TaskRegistryEntry]:
        """Discover tasks not managed by this TaskManager (from .tasks/ registry)

        Includes both CLI-started tasks and orphaned web tasks (e.g. after web server restart).
        """
        entries = list_task_entries()  # All sources
        result = []
        for entry in entries:
            # Skip tasks already managed in memory
            if entry.task_id in self._tasks:
                continue
            # Check liveness: if status is "running" but process is dead, mark as error
            if entry.status == "running" and entry.pid and not is_process_alive(entry.pid):
                entry.status = "error"
                entry.error = "进程已退出（PID 不存在）"
                write_task_entry(entry)
            result.append(entry)
        return result

    def get_cli_snapshot(self, task_id: str) -> dict[str, Any] | None:
        """Build a task snapshot for a CLI-started task from disk files

        Returns snapshot dict compatible with BatchSession.snapshot() format, or None.
        """
        reg = read_task_entry(task_id)
        if not reg:
            return None

        # Check liveness
        if reg.status == "running" and reg.pid and not is_process_alive(reg.pid):
            reg.status = "error"
            reg.error = "CLI 进程已退出（PID 不存在）"
            write_task_entry(reg)

        # Build cases dict from checkpoint + live files
        cases: dict[str, dict] = {}
        completed_count = 0
        cancelled_count = 0

        # Try loading checkpoint meta for case_ids
        case_ids: list[str] = []
        completed_results: list[TestResult] = []
        try:
            meta, completed_results = CheckpointManager.load(reg.checkpoint_session_id)
            case_ids = meta.case_ids
        except FileNotFoundError:
            pass

        # Fallback: if no checkpoint but task is running/error, load case_ids from data file
        if not case_ids and reg.status in ("running", "error"):
            try:
                path = resolve_data_path(reg.benchmark, reg.dataset)
                items = load_bench_items(path)
                case_ids = [item.id for item in items]
                # If we have a limit in registry, only take first N
                if reg.total and len(case_ids) > reg.total:
                    case_ids = case_ids[:reg.total]
            except Exception:
                pass

        # Fallback: if no checkpoint and completed, load from report
        if not case_ids and reg.status == "completed" and reg.report_path:
            report_cases, max_conc, rt = _load_report_cases(reg.report_path)
            if report_cases:
                cases = report_cases
                completed_count = len(report_cases)

        # Initialize all cases as pending (if not already loaded from report)
        if not cases:
            for cid in case_ids:
                cases[cid] = {"status": "pending", "turn": 0, "title": cid, "tags": []}

        # Mark completed cases from checkpoint results
        for r in completed_results:
            cases[r.id] = {
                "status": "completed",
                "turn": 0,
                "title": r.title or r.id,
                "user_type": r.user_type,
                "target_type": r.target_type,
                "eval_type": r.eval_type,
                "score": r.eval.score,
                "eval_result": r.eval.result,
                "cost": r.cost.model_dump(mode="json") if r.cost else None,
                "tags": r.tags,
            }
            if r.eval.feedback == "用例被取消":
                cancelled_count += 1
            else:
                completed_count += 1

        # Check live files for in-progress cases
        from evaluator.utils.live_files import LIVE_DIR, read_live_file
        live_dir = LIVE_DIR / task_id
        if live_dir.is_dir():
            for live_file in live_dir.glob("*.json"):
                cid = live_file.stem
                # Distinguish init-only files (pending) from real in-progress files
                live_data = read_live_file(task_id, cid)
                is_init_only = live_data.get("_init", False) if live_data else False
                if is_init_only:
                    continue  # Still pending, not yet started
                if cid in cases and cases[cid]["status"] == "pending":
                    cases[cid]["status"] = "dialogue"
                elif cid not in cases:
                    cases[cid] = {"status": "dialogue", "turn": 0, "title": cid, "tags": []}

        snap: dict[str, Any] = {
            "id": reg.task_id,
            "status": reg.status,
            "benchmark": reg.benchmark,
            "dataset": reg.dataset,
            "runtime_target": reg.runtime_target,
            "created_at": reg.created_at,
            "report_path": reg.report_path,
            "total": reg.total or len(cases),
            "completed": completed_count,
            "cancelled": cancelled_count > 0,
            "cases": cases,
            "error": reg.error,
            "max_concurrency": reg.max_concurrency,
            "source": "cli",
        }

        # Stats from report if completed
        if reg.status == "completed" and reg.report_path:
            stats_by_tag, report_summary = _load_report_summary(reg.report_path)
            snap["stats_by_tag"] = stats_by_tag
            if report_summary:
                snap["report_summary"] = report_summary
        else:
            snap["stats_by_tag"] = _compute_snapshot_tag_stats(cases)

        return snap

    def cancel_all(self) -> None:
        """Cancel all active tasks on shutdown"""
        for entry in self._tasks.values():
            if entry.status == "running" and entry.session:
                entry.session.cancel()


def _compute_snapshot_tag_stats(cases: dict[str, dict]) -> dict[str, dict]:
    """Compute tag-grouped statistics from snapshot cases"""
    tag_data: dict[str, list[dict]] = {}
    for info in cases.values():
        if info.get("eval_result") is None:
            continue
        for tag in info.get("tags", []):
            tag_data.setdefault(tag, []).append(info)

    stats: dict[str, dict] = {}
    for tag, items in tag_data.items():
        tag_pass = sum(1 for i in items if i.get("eval_result") == "pass")
        tag_fail = sum(1 for i in items if i.get("eval_result") == "fail")
        tag_error = sum(1 for i in items if i.get("eval_result") == "error")
        tag_judged = tag_pass + tag_fail
        # 与 bench_schema.compute_stats_by_tag 同一口径:没判成分的用例不进平均分。
        # 这里输入是快照 dict 而不是 TestResult，所以只能重述规则、不能直接调它 ——
        # 但进行中的数字和跑完后的报告必须一致，否则同一个批次会在跑完那一刻跳变。
        scores = [i["score"] for i in items if i.get("score") is not None and i.get("eval_result") != "error"]
        stats[tag] = {
            "total": len(items),
            "pass_count": tag_pass,
            "fail_count": tag_fail,
            "error_count": tag_error,
            "pass_rate": tag_pass / tag_judged if tag_judged else 0.0,
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
        }
    return stats


def _compute_test_result_tag_stats(results: list[TestResult]) -> dict[str, dict]:
    """Compute tag-grouped statistics from TestResult list.

    Delegates to `evaluator.core.bench_schema.compute_stats_by_tag`: the same
    numbers are shown on a report page and printed by the CLI, and they came
    from two separate implementations of the same loop. A tag's average score
    must not depend on which one you are looking at.
    """
    from evaluator.core.bench_schema import compute_stats_by_tag

    return compute_stats_by_tag(results)


def _load_report_cases(report_path: str | None) -> tuple[dict[str, dict], int | None, dict | None]:
    """Rebuild cases summary dict from report file, used for snapshots after session release.

    Returns: (cases_dict, max_concurrency, runtime_target)
    """
    if not report_path:
        return {}, None, None
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}, None, None

    cases: dict[str, dict] = {}
    for c in report_data.get("cases", []):
        cid = c.get("id", "")
        ev = c.get("eval") or {}
        cases[cid] = {
            "status": "completed",
            "title": c.get("title", ""),
            "user_type": c.get("user_type", ""),
            "target_type": c.get("target_type", ""),
            "eval_type": c.get("eval_type", ""),
            "score": ev.get("score"),
            "eval_result": ev.get("result"),
            "cost": c.get("cost"),
            "tags": c.get("tags", []),
            "turn": 0,
        }
    max_concurrency = report_data.get("max_concurrency")
    runtime_target = report_data.get("runtime_target")
    return cases, max_concurrency, runtime_target


def _build_report_summary(results: list[TestResult], stats_by_tag: dict[str, dict]) -> dict[str, Any]:
    """Use the same score denominator for in-memory and saved task summaries."""
    graded = [r for r in results if r.eval.result != "error"]
    return {
        "avg_score": sum(r.eval.score for r in graded) / len(graded) if graded else 0.0,
        "graded_count": len(graded),
        "error_count": len(results) - len(graded),
        "stats_by_tag": {
            tag: {
                "avg_score": value["avg_score"],
                "count": value["total"],
                "graded_count": value["total"] - value["error_count"],
                "error_count": value["error_count"],
            }
            for tag, value in stats_by_tag.items()
        },
    }


def _load_report_summary(report_path: str | None) -> tuple[dict[str, dict], dict[str, Any] | None]:
    """Read stats_by_tag / report_summary from report file as fallback when not in memory"""
    if not report_path:
        return {}, None

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)
        results = [TestResult.model_validate(case) for case in report_data["cases"]]
    except (OSError, ValueError, KeyError, TypeError):
        return {}, None

    # Case results also let reports saved before error_count existed use the
    # current denominator, without rewriting the original report file.
    stats_by_tag = _compute_test_result_tag_stats(results)
    return stats_by_tag, _build_report_summary(results, stats_by_tag)


    # Note: live file functions have been moved to evaluator/utils/live_files.py
    # and imported at the top of this module: write_live_file, read_live_file, delete_live_file, cleanup_live_dir


# Global singleton
task_manager = TaskManager()
