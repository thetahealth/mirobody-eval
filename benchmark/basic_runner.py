"""
basic_runner — 跑分执行器（基于 BenchItem + BenchReport）

职责：
1. 加载 data/ 中的 BenchItem JSONL（通过 benchmark_reader）
2. 根据 metadata TargetSpec（字段级默认值 + 编辑权限）+ CLI 覆盖构建 target
3. 调用 evaluator 的 do_batch_test 并发执行
4. 生成 BenchReport 并写入 report/
5. 断点续跑 — 每完成一个用例实时写入检查点，中断后可 --resume 恢复

target 合并优先级（高→低）:
    CLI/UI 覆盖（仅 editable 字段） > JSONL target_overrides（per-case） > metadata 默认值

用法:
    # editable 字段可通过 CLI 覆盖
    python -m benchmark.basic_runner healthbench sample --target-model gpt-4.1
    python -m benchmark.basic_runner healthbench sample --target-model gemini-3-pro-preview

    # 所有字段 editable=false 时 — 直接使用 metadata 默认值
    python -m benchmark.basic_runner medcalc sample

    # 断点续跑 — 恢复上次未完成的评测
    python -m benchmark.basic_runner healthbench sample --resume

报告输出:
    benchmark/report/healthbench/sample_gpt-4.1_20260213_143012.json
"""

import argparse
import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List
from uuid import uuid4

# 导入插件包，触发 __init_subclass__ 注册
import evaluator.plugin.eval_agent  # noqa: F401
import evaluator.plugin.target_agent  # noqa: F401
import evaluator.plugin.test_agent  # noqa: F401

from evaluator.core.bench_schema import (
    BenchReport,
    bench_item_to_test_case,
    build_bench_report,
    find_target_spec,
    resolve_runtime_target,
)
from evaluator.core.orchestrator import BatchSession
from evaluator.core.schema import TestCase
from evaluator.utils.benchmark_reader import filter_bench_items, load_benchmark, resolve_data_path
from evaluator.utils.checkpoint import CheckpointManager, CheckpointMeta
from evaluator.utils.live_files import cleanup_live_dir, delete_live_file, write_live_file, write_live_init
from evaluator.utils.report_reader import save_bench_report
from evaluator.utils.task_registry import TaskRegistryEntry, cleanup_task_entry, write_task_entry

logger = logging.getLogger(__name__)


# ============================================================
# 核心函数
# ============================================================


async def run_benchmark(
    benchmark: str,
    dataset: str,
    cli_overrides: dict | None = None,
    target_type: str | None = None,
    ids: str | None = None,
    limit: int | None = None,
    max_concurrency: int = 0,
    resume: bool = False,
) -> BenchReport:
    """执行跑分并将报告写入 report/ 目录

    Args:
        benchmark:       评测类型（如 "healthbench"）
        dataset:         数据集名称（如 "sample"）
        cli_overrides:   CLI/UI 传入的 target 覆盖参数（仅 editable 字段生效）
        target_type:     目标系统类型（如 "llm_api"）；仅有一个候选时可省略
        ids:             逗号分隔的 ID（仅运行指定用例）
        limit:           最大用例数
        max_concurrency: 最大并发数（0 = 不限制）
        resume:          恢复上次未完成的评测

    Returns:
        BenchReport
    """
    if resume:
        return await _resume_benchmark(benchmark, dataset, max_concurrency)

    # 1. 加载 BenchMark（含 metadata + items）
    bench = load_benchmark(benchmark, dataset)
    if max_concurrency == 0 and bench.max_concurrency > 0:
        max_concurrency = bench.max_concurrency
        logger.info("使用 metadata 默认并发数: %d", max_concurrency)
    items = filter_bench_items(bench.items, ids=ids, limit=limit)

    if not items:
        raise ValueError(f"过滤后无可执行用例 ({benchmark}/{dataset}, ids={ids}, limit={limit})")

    # 2. 从 TargetSpec 列表中选择目标 spec
    spec = find_target_spec(bench.target, target_type)

    # 3. 构建全局 runtime target（spec defaults + CLI editable overrides，用于日志和报告）
    runtime_target = resolve_runtime_target(spec, cli_overrides)
    logger.info(
        "运行时 target: type=%s, model=%s",
        runtime_target.type,
        getattr(runtime_target, "model", "N/A"),
    )

    # 4. 转换 BenchItem → TestCase（含三层 target 合并）
    test_cases: List[TestCase] = []
    for item in items:
        try:
            test_case = bench_item_to_test_case(item, spec, cli_overrides)
            test_cases.append(test_case)
        except Exception as e:
            logger.error("用例 %s 转换失败: %s", item.id, e, exc_info=True)
            raise ValueError(f"用例 {item.id} 转换失败: {e}") from e

    # 5. 创建检查点
    data_path = resolve_data_path(benchmark, dataset)
    checkpoint_id = uuid4().hex[:8]
    mgr = CheckpointManager(checkpoint_id)
    meta = CheckpointMeta(
        session_id=checkpoint_id,
        benchmark=benchmark,
        dataset=dataset,
        target_type=spec.type,
        cli_overrides=cli_overrides,
        runtime_target=runtime_target.model_dump(mode="json"),
        case_ids=[c.id for c in test_cases],
        max_concurrency=max_concurrency,
        started_at=datetime.now().isoformat(),
        data_file_hash=CheckpointManager.compute_data_hash(data_path),
    )
    mgr.save_meta(meta)

    logger.info(
        "开始跑分: %s/%s (%d 条用例, target=%s, concurrency=%s, checkpoint=%s)",
        benchmark,
        dataset,
        len(test_cases),
        runtime_target.type,
        max_concurrency or "unlimited",
        checkpoint_id,
    )

    # 6. 创建 BatchSession（直接使用，支持 live file 和 task registry）
    session = BatchSession(cases=test_cases, max_concurrency=max_concurrency)
    task_id = session.id

    # 写入 task registry（Web 可发现）
    registry_entry = TaskRegistryEntry(
        task_id=task_id,
        checkpoint_session_id=checkpoint_id,
        benchmark=benchmark,
        dataset=dataset,
        runtime_target=runtime_target.model_dump(mode="json"),
        status="running",
        created_at=datetime.now().isoformat(),
        source="cli",
        total=len(test_cases),
        max_concurrency=max_concurrency,
        pid=os.getpid(),
    )
    write_task_entry(registry_entry)

    # 设置 checkpoint + live file 回调
    def _on_progress(sess: BatchSession, case_id: str) -> None:
        ctx = sess.contexts.get(case_id)
        if ctx and ctx.result:
            mgr.append_result(ctx.result)
        delete_live_file(task_id, case_id)

    session.on_progress = _on_progress

    # 设置每轮对话 live file 回调
    for ctx in session.contexts.values():
        ctx.on_turn = lambda c, tid=task_id: write_live_file(tid, c)

    # 写入初始 live file（让 Web UI 立即看到用例状态为 init）
    write_live_init(task_id, [c.id for c in test_cases])

    started_at = datetime.now()
    try:
        test_report = await session.run()
    except Exception:
        logger.error("评测执行失败（检查点已保留: %s，可使用 --resume 恢复）", checkpoint_id)
        cleanup_live_dir(task_id)
        registry_entry.status = "error"
        registry_entry.error = "评测执行异常"
        write_task_entry(registry_entry)
        raise
    finished_at = datetime.now()

    # 7. 构建 BenchReport
    bench_report = build_bench_report(
        test_results=test_report.cases,
        benchmark_name=benchmark,
        dataset_name=dataset,
        runtime_target=runtime_target,
        max_concurrency=max_concurrency,
        started_at=started_at,
        finished_at=finished_at,
    )

    # 8. 保存报告 + 清理检查点 + 更新 registry
    report_path = save_bench_report(bench_report, benchmark=benchmark, dataset=dataset)
    mgr.cleanup()
    cleanup_live_dir(task_id)

    registry_entry.status = "completed"
    registry_entry.report_path = str(report_path)
    write_task_entry(registry_entry)

    _print_summary(bench_report, report_path)

    return bench_report


async def _resume_benchmark(
    benchmark: str,
    dataset: str,
    max_concurrency: int = 0,
) -> BenchReport:
    """恢复上次未完成的评测"""
    # 1. 查找检查点
    checkpoints = CheckpointManager.find_checkpoints(benchmark=benchmark, dataset=dataset)
    if not checkpoints:
        raise ValueError(f"未找到 {benchmark}/{dataset} 的检查点，无法恢复")

    cp_meta = checkpoints[0]  # 最近的检查点
    meta, completed_results = CheckpointManager.load(cp_meta.session_id)
    mgr = CheckpointManager(meta.session_id)

    # 2. 验证数据文件哈希
    data_path = resolve_data_path(benchmark, dataset)
    current_hash = CheckpointManager.compute_data_hash(data_path)
    if meta.data_file_hash and current_hash != meta.data_file_hash:
        logger.warning("数据文件已变更（hash: %s → %s），已完成的结果不受影响", meta.data_file_hash, current_hash)

    # 3. 过滤已完成用例（被取消的用例会重跑）
    cancelled_ids = {r.id for r in completed_results if r.eval.feedback == "用例被取消"}
    completed_ids = {r.id for r in completed_results} - cancelled_ids
    completed_results = [r for r in completed_results if r.id not in cancelled_ids]

    remaining_ids = set(meta.case_ids) - completed_ids
    logger.info(
        "恢复检查点 %s: %s/%s（已完成 %d/%d，剩余 %d）",
        meta.session_id, benchmark, dataset,
        len(completed_ids), len(meta.case_ids), len(remaining_ids),
    )

    # 4. 加载并转换剩余用例
    bench = load_benchmark(benchmark, dataset)
    spec = find_target_spec(bench.target, meta.target_type)
    runtime_target = resolve_runtime_target(spec, meta.cli_overrides)
    concurrency = max_concurrency if max_concurrency > 0 else meta.max_concurrency

    remaining_items = [item for item in bench.items if item.id in remaining_ids]
    registry_entry: TaskRegistryEntry | None = None
    if not remaining_items:
        logger.info("所有用例已完成，直接生成报告")
        all_results = completed_results
    else:
        test_cases = []
        for item in remaining_items:
            try:
                test_cases.append(bench_item_to_test_case(item, spec, meta.cli_overrides))
            except Exception as e:
                logger.error("用例 %s 转换失败: %s", item.id, e, exc_info=True)
                raise ValueError(f"用例 {item.id} 转换失败: {e}") from e

        logger.info("继续执行 %d 条用例 (concurrency=%s)", len(test_cases), concurrency or "unlimited")

        # 创建 BatchSession + task registry + live files
        session = BatchSession(cases=test_cases, max_concurrency=concurrency)
        task_id = session.id

        registry_entry = TaskRegistryEntry(
            task_id=task_id,
            checkpoint_session_id=meta.session_id,
            benchmark=benchmark,
            dataset=dataset,
            runtime_target=meta.runtime_target,
            status="running",
            created_at=datetime.now().isoformat(),
            source="cli",
            total=len(meta.case_ids),
            max_concurrency=concurrency,
            pid=os.getpid(),
        )
        write_task_entry(registry_entry)

        def _on_progress(sess: BatchSession, case_id: str) -> None:
            ctx = sess.contexts.get(case_id)
            if ctx and ctx.result:
                mgr.append_result(ctx.result)
            delete_live_file(task_id, case_id)

        session.on_progress = _on_progress
        for ctx in session.contexts.values():
            ctx.on_turn = lambda c, tid=task_id: write_live_file(tid, c)

        write_live_init(task_id, [c.id for c in test_cases])

        try:
            test_report = await session.run()
        except Exception:
            logger.error("恢复执行失败（检查点已保留: %s，可再次 --resume）", meta.session_id)
            cleanup_live_dir(task_id)
            registry_entry.status = "error"
            registry_entry.error = "恢复执行异常"
            write_task_entry(registry_entry)
            raise

        cleanup_live_dir(task_id)
        all_results = completed_results + list(test_report.cases)

    # 5. 按原始顺序排列结果
    order = {cid: i for i, cid in enumerate(meta.case_ids)}
    all_results.sort(key=lambda r: order.get(r.id, len(order)))

    # 6. 构建 BenchReport
    bench_report = build_bench_report(
        test_results=all_results,
        benchmark_name=benchmark,
        dataset_name=dataset,
        runtime_target=runtime_target,
        max_concurrency=concurrency,
        started_at=datetime.fromisoformat(meta.started_at),
        finished_at=datetime.now(),
    )

    # 7. 保存报告 + 清理检查点 + 更新 registry
    report_path = save_bench_report(bench_report, benchmark=benchmark, dataset=dataset)
    mgr.cleanup()

    if registry_entry is not None:
        registry_entry.status = "completed"
        registry_entry.report_path = str(report_path)
        write_task_entry(registry_entry)

    _print_summary(bench_report, report_path)

    return bench_report


# ============================================================
# 摘要输出
# ============================================================


def _avg_cell(stats: dict, precision: int = 2) -> str:
    """Show the grading denominator; an empty graded subset has no average."""
    total = stats["total"]
    graded = total - stats.get("error_count", 0)
    average = f"{stats['avg_score']:.{precision}f}" if graded else "—"
    return f"{average} （{graded}/{total} 条已评分）"


def _print_grouped_stats(stats_by_tag: dict, prefix: str, title: str) -> None:
    """打印特定 tag 前缀（如 hallu_type:）的分组统计，按 avg_score 降序。"""
    rows = [
        (tag[len(prefix):], s)
        for tag, s in stats_by_tag.items()
        if tag.startswith(prefix)
    ]
    if not rows:
        return
    rows.sort(key=lambda x: -x[1]["avg_score"])
    print(f"\n按{title}统计:")
    print(f"  {'类别':<14} {'条数':>6}  {'通过率':>8}  {'未判成':>6}  平均分（已评分/总数）")
    for name, s in rows:
        print(
            f"  {name:<14} {s['total']:>6}  {s['pass_rate']:>7.1%}"
            f"  {s.get('error_count', 0):>6}  {_avg_cell(s, precision=3)}"
        )


def _print_summary(report: BenchReport, report_path: Path) -> None:
    """打印跑分摘要到 stdout"""
    sep = "=" * 60

    print(f"\n{sep}")
    print("跑分完成")
    print(sep)
    print(f"  数据集:    {report.benchmark_name}/{report.dataset_name}")
    target = report.runtime_target
    print(f"  被测系统:  {getattr(target, 'type', 'N/A')} ({getattr(target, 'model', 'N/A')})")
    print(f"  总用例:    {len(report.cases)}")
    print(f"  通过:      {report.pass_count}")
    print(f"  失败:      {report.fail_count}")
    print(f"  未判成:    {report.error_count}")
    print(f"  通过率:    {report.pass_rate:.1%}")
    summary = {"total": len(report.cases), "error_count": report.error_count, "avg_score": report.avg_score}
    print(f"  平均得分:  {_avg_cell(summary)}")
    print(f"  总耗时:    {report.total_duration_seconds:.1f}s")

    # 按主类别分组（hallu_type:* 或 dim:* 等命名空间的 tag 优先展示为独立分组）
    _print_grouped_stats(report.stats_by_tag, prefix="hallu_type:", title="幻觉类型")
    _print_grouped_stats(report.stats_by_tag, prefix="dim:", title="评测维度")

    # 按 tag 统计（完整）
    if report.stats_by_tag:
        print("\n按标签统计（全部）:")
        for tag, stats in sorted(report.stats_by_tag.items()):
            print(
                f"  [{tag}] {stats['total']} 条, 通过率 {stats['pass_rate']:.1%},"
                f" 未判成 {stats.get('error_count', 0)} 条, 平均分 {_avg_cell(stats)}"
            )

    # 失败用例
    failed = [c for c in report.cases if c.eval.result == "fail"]
    if failed:
        print(f"\n失败用例 ({len(failed)} 条):")
        for c in failed[:20]:
            feedback = (c.eval.feedback or "")[:80]
            print(f"  - {c.id}: score={c.eval.score:.2f}, {feedback}")
        if len(failed) > 20:
            print(f"  ... 还有 {len(failed) - 20} 条")

    errors = [c for c in report.cases if c.eval.result == "error"]
    if errors:
        print(f"\n未判成的用例 ({len(errors)} 条):")
        for c in errors[:20]:
            lines = (c.eval.feedback or "").strip().splitlines()
            note = lines[0][:160] if lines else "No feedback provided"
            detail = (c.eval.trace.eval_detail or {}) if c.eval.trace else {}
            exception_type = detail.get("exception_type")
            label = f"[{exception_type}] " if exception_type else ""
            print(f"  - {c.id}: {label}{note}")
        if len(errors) > 20:
            print(f"  ... 还有 {len(errors) - 20} 条")

    print(f"\n报告: {report_path}")
    print(sep)


# ============================================================
# CLI 入口
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="跑分执行器 — 加载 BenchItem 数据集并执行批量评测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  # 覆盖 editable 字段\n"
            "  python -m benchmark.basic_runner healthbench sample --target-model gpt-4.1\n"
            "  python -m benchmark.basic_runner healthbench sample --target-model gemini-3-pro-preview\n"
            "\n"
            "  # 所有字段锁定的 benchmark — 直接使用 metadata 默认值\n"
            "  python -m benchmark.basic_runner medcalc sample\n"
            "\n"
            "  # 过滤和并发选项\n"
            "  python -m benchmark.basic_runner healthbench sample --target-model gpt-4.1 --ids hb_abc,hb_def\n"
            "  python -m benchmark.basic_runner healthbench sample --target-model gpt-4.1 --limit 10 -p 3 -v\n"
            "\n"
            "  # 断点续跑 — 恢复上次中断的评测\n"
            "  python -m benchmark.basic_runner healthbench sample --resume\n"
        ),
    )
    parser.add_argument("benchmark", help="评测类型（对应 data/ 下的子目录，如 healthbench）")
    parser.add_argument("dataset", help="数据集名称（对应子目录下的 JSONL 文件，如 sample、full、hard）")
    parser.add_argument(
        "--target-type",
        type=str,
        default=None,
        help="目标系统类型（如 llm_api、hermes，多 target 时指定；单 target 时可省略）",
    )
    parser.add_argument(
        "--target-model",
        type=str,
        default=None,
        help="被测系统模型（如 gpt-4.1、gemini-3-pro-preview，覆盖 metadata 默认值，需 editable=true）",
    )
    parser.add_argument(
        "--user-model",
        type=str,
        default=None,
        help="扮演虚拟用户的模型，仅 auto 模式数据集有效（如 anthropic/claude-sonnet-4.6）。"
        "不传则用 AutoTestAgent.DEFAULT_MODEL",
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        default=None,
        help="判分器模型（覆盖用例 eval 配置里的 model / judge_model / extractor_model）。"
        "纯规则判分的题型不受影响",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="系统提示词（覆盖 metadata 默认值，需 editable=true）",
    )
    parser.add_argument(
        "--target-override", nargs="*", default=[], metavar="KEY=VALUE",
        help="覆盖 target 字段（如 agent=expert），仅 editable=true 的字段生效",
    )
    parser.add_argument("--ids", type=str, default=None, help="仅运行指定 ID 的用例（逗号分隔）")
    parser.add_argument("--limit", type=int, default=None, help="仅运行前 N 条用例")
    parser.add_argument("--parallel", "-p", type=int, default=0, help="最大并发数（默认不限制）")
    parser.add_argument("--resume", action="store_true", default=False, help="恢复上次未完成的评测（自动检测最近的检查点）")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出详细日志")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 将 CLI target 参数组装为 dict（仅 editable 字段会被 resolve_effective_target 接受）
    cli_overrides: dict[str, str] = {}
    if args.target_type:
        cli_overrides["target_type"] = args.target_type  # 内部路由用，会在 run_benchmark 中 pop
    if args.target_model:
        cli_overrides["model"] = args.target_model
    if args.user_model:
        # 保留键：给虚拟用户用，bench_item_to_test_case 会取走，不参与 target 解析
        cli_overrides["user_model"] = args.user_model
    if args.eval_model:
        cli_overrides["eval_model"] = args.eval_model  # 同上，给判分器用
    if args.system_prompt:
        cli_overrides["system_prompt"] = args.system_prompt
    for kv in args.target_override:
        k, _, v = kv.partition("=")
        if k and v:
            cli_overrides[k] = v

    asyncio.run(
        run_benchmark(
            benchmark=args.benchmark,
            dataset=args.dataset,
            cli_overrides=cli_overrides or None,
            target_type=args.target_type,
            ids=args.ids,
            limit=args.limit,
            max_concurrency=args.parallel,
            resume=args.resume,
        )
    )


if __name__ == "__main__":
    main()
