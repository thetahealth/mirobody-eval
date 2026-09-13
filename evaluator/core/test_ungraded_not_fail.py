"""没跑完的用例不许算成「答错了」。

纯逻辑，不连网、不需要部署。

    pytest evaluator/core/test_ungraded_not_fail.py -v

锁的是一个实测出来的事故形状：跨全量统计的题在超时预算不足时抛
`TimeoutError`，`do_single_test` 的兜底把它记成 `fail` / `score=0.0`，报告里
与真答错的题并排显示。而同一道题在预算够时得 **0.89** —— 也就是说那个 0.00
是评测方的 `MIROBODY_TIMEOUT` 设出来的，不是被测系统的属性。

这里锁三件事，缺任何一件这个修复都是假的：

  1. 抛异常的用例 result 是 `error`，不是 `fail`
  2. `error` 不进 `avg_score` —— 这一条此前独立地漏了:`pass_rate` 的分母早就
     排除了 error，`avg_score` 没有，所以连判分器不可用那条本来做对了的路
     也一直在按 0 分拉低成绩
  3. `error_count` 必须被发布出来 —— 把它们从分母里剔掉却不报数量，是把
     「假 0 分」换成更隐蔽的「假高分」
"""

from __future__ import annotations

from datetime import datetime

import pytest

from evaluator.core.bench_schema import build_bench_report, compute_stats_by_tag
from evaluator.core.orchestrator import do_single_test
from evaluator.core.schema import EvalResult, TestCase, TestCost, TestReport, TestResult


def _result(case_id: str, verdict: str, score: float, tags: list[str] | None = None) -> TestResult:
    now = datetime.now()
    return TestResult(
        id=case_id,
        eval=EvalResult(result=verdict, score=score),
        cost=TestCost(),
        start=now,
        end=now,
        tags=tags or [],
    )


# ---------------------------------------------------------------------------
# 1. 抛异常 → error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_timeout_is_reported_as_error_not_a_zero_score(monkeypatch):
    """超时的用例不是 0 分,是没有分。

    让被测方在对话轮里抛 `TimeoutError` —— 这正是超时预算不足时 aiohttp 抛的
    东西,走的是 `do_single_test` 的兜底 handler。不连网:target 被换掉了。
    """
    # 必须导到子模块:`evaluator.plugin` 自己是个空的 __init__，注册发生在
    # 各 agent 模块被导入时。只导父包会让 TestCase 校验拒绝一切类型。
    import evaluator.plugin.eval_agent  # noqa: F401
    import evaluator.plugin.target_agent  # noqa: F401
    import evaluator.plugin.test_agent  # noqa: F401
    from evaluator.core.interfaces.abstract_target_agent import AbstractTargetAgent

    case = TestCase.model_validate(
        {
            "id": "probe_timeout",
            "title": "一道跑不完的题",
            "user": {"type": "manual", "strict_inputs": ["算一下全部指标的趋势"]},
            "target": {"type": "llm_api", "model": "gpt-5.4-mini"},
            # 判分器配置留最简 —— 它压根跑不到:对话轮先抛异常。
            "eval": {"evaluator": "semantic"},
        }
    )

    async def _times_out(self, test_action):
        raise TimeoutError("模拟单轮超时（真实场景由 aiohttp 的 ClientTimeout 抛出）")

    monkeypatch.setattr(AbstractTargetAgent.get("llm_api"), "_generate_next_reaction", _times_out, raising=True)

    result = await do_single_test(case)

    assert result.eval.result == "error", (
        "抛异常意味着判分没发生。写成 fail 会让「没跑完」和「答错了」进同一栏，"
        "而这两者的处置完全不同 —— 前者调预算或修部署，后者才是能力问题"
    )
    assert result.eval.score == 0.0
    detail = (result.eval.trace.eval_detail or {}) if result.eval.trace else {}
    assert detail.get("exception_type"), (
        "异常类型必须落 trace:一屏 TimeoutError 是调预算，一屏 JudgeUnavailable 是配 key,报表上分不开就没法处置"
    )


# ---------------------------------------------------------------------------
# 2. error 不进平均分
# ---------------------------------------------------------------------------


def test_an_ungraded_case_does_not_drag_the_average_to_zero():
    """两条判成分的题各 1.0 和 0.8，外加一条超时 —— 平均分必须是 0.9。

    旧口径给 0.6（把 0.0 算进 3 条的分母）。那个数会随评测方的超时设置变化,
    所以它不是被测系统的成绩。
    """
    report = build_bench_report(
        [
            _result("a", "scored", 1.0),
            _result("b", "scored", 0.8),
            _result("c", "error", 0.0),
        ],
        benchmark_name="probe",
        dataset_name="probe",
    )

    assert report.avg_score == pytest.approx(0.9)
    assert report.error_count == 1


def test_the_two_average_implementations_agree():
    """`BenchReport` 和 `TestReport` 各有一份 avg_score。

    它们算的是同一个东西,一处修了另一处没修,同一批结果就会因为你从 CLI 还是
    从 Web 看而给出不同的分数。
    """
    results = [_result("a", "scored", 1.0), _result("b", "error", 0.0)]

    bench = build_bench_report(results, benchmark_name="probe", dataset_name="probe")
    plain = TestReport(cases=results)

    assert bench.avg_score == pytest.approx(plain.avg_score) == pytest.approx(1.0)
    assert bench.error_count == plain.error_count == 1


def test_a_run_that_graded_nothing_reports_zero_rather_than_dividing_by_zero():
    """全部超时 —— 没有可算平均的用例，不能崩，也不能凭空造一个分数。"""
    report = build_bench_report(
        [_result("a", "error", 0.0), _result("b", "error", 0.0)],
        benchmark_name="probe",
        dataset_name="probe",
    )

    assert report.avg_score == 0.0
    assert report.error_count == 2, "这份报告唯一诚实的读法是「两条都没跑成」"


# ---------------------------------------------------------------------------
# 3. 剔出分母就必须公布数量
# ---------------------------------------------------------------------------


def test_tag_stats_carry_the_error_count_beside_the_average():
    """按维度看分时同理:一个维度的题大面积超时，不能显示成这个维度得分很高。"""
    stats = compute_stats_by_tag(
        [
            _result("a", "scored", 1.0, tags=["difficulty:Trend"]),
            _result("b", "error", 0.0, tags=["difficulty:Trend"]),
        ]
    )

    row = stats["difficulty:Trend"]
    assert row["avg_score"] == pytest.approx(1.0)
    assert row["error_count"] == 1
    assert row["total"] == 2, "total 仍是全部用例 —— 分子分母的差额就是靠它才看得出来"


def test_the_cli_prints_the_error_count():
    """报告里有这个字段、但汇总不打印，等于没有。

    源码级断言:行为级测试要跑完一整个 batch,而这一条锁的只是「那一栏还在」。
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "benchmark" / "basic_runner.py").read_text(encoding="utf-8")

    assert "report.error_count" in src, "CLI 汇总必须打印未判成的数量，否则平均分没有分母"
    assert 'result == "error"' in src, "未判成的用例要单独列，不能混进失败用例那张表"


@pytest.mark.parametrize("feedback", [None, "", "\n", "timeout\nadditional details"])
def test_cli_summary_accepts_error_results_with_optional_feedback(capsys, tmp_path, feedback):
    from benchmark.basic_runner import _print_summary
    from evaluator.plugin.target_agent.llm_api_target_agent import LlmApiTargetInfo

    result = _result("unfinished-case", "error", 0.0)
    result.eval.feedback = feedback
    report = build_bench_report(
        [result], benchmark_name="probe", dataset_name="probe",
        runtime_target=LlmApiTargetInfo(type="llm_api", model="gpt-5.4-mini"),
    )
    path = tmp_path / "report.json"
    _print_summary(report, path)
    output = capsys.readouterr().out
    assert "unfinished-case" in output
    assert str(path) in output
    assert "平均得分:  —" in output
    assert "IndexError" not in output
    if not feedback:
        assert "No feedback provided" in output
    if feedback and feedback.startswith("timeout"):
        assert "timeout" in output
        assert "additional details" not in output


def test_cli_summary_shows_coverage_for_mixed_results_and_tags(capsys, tmp_path):
    from benchmark.basic_runner import _print_summary

    report = build_bench_report(
        [
            _result("completed", "scored", 1.0, ["dim:completed"]),
            _result("unfinished", "error", 0.0, ["dim:unfinished"]),
        ],
        benchmark_name="probe", dataset_name="probe",
    )
    _print_summary(report, tmp_path / "report.json")
    output = capsys.readouterr().out
    assert "未判成:    1" in output
    assert "平均得分:  1.00 （1/2 条已评分）" in output
    assert "[dim:unfinished] 1 条, 通过率 0.0%, 未判成 1 条, 平均分 — （0/1 条已评分）" in output
    grouped = output.split("按标签统计（全部）:", 1)[0]
    assert "— （0/1 条已评分）" in grouped
    assert "0.000" not in grouped


@pytest.mark.parametrize("verdict", ["pass", "fail", "scored"])
def test_cli_displays_a_real_zero_grade_as_zero(capsys, tmp_path, verdict):
    from benchmark.basic_runner import _print_summary

    report = build_bench_report(
        [_result("zero", verdict, 0.0)], benchmark_name="probe", dataset_name="probe",
    )
    _print_summary(report, tmp_path / "report.json")
    output = capsys.readouterr().out
    assert "平均得分:  0.00 （1/1 条已评分）" in output
    assert "未判成:    0" in output


def test_cli_displays_no_average_for_an_empty_run(capsys, tmp_path):
    from benchmark.basic_runner import _print_summary

    report = build_bench_report([], benchmark_name="probe", dataset_name="probe")
    _print_summary(report, tmp_path / "report.json")
    assert "平均得分:  — （0/0 条已评分）" in capsys.readouterr().out


@pytest.mark.parametrize("verdicts", [["scored", "error"], ["error", "error"], ["scored", "fail"], []])
def test_task_summary_survives_releasing_results_and_reloading_from_disk(tmp_path, verdicts):
    from web.app.services.task_manager import TaskEntry, TaskManager

    results = [
        _result(str(i), verdict, 1.0 if verdict == "scored" else 0.0, tags=["difficulty:Trend"])
        for i, verdict in enumerate(verdicts)
    ]
    report = build_bench_report(results, benchmark_name="probe", dataset_name="probe")
    path = tmp_path / "report.json"
    path.write_text(report.model_dump_json(), encoding="utf-8")
    manager = TaskManager()
    entry = TaskEntry(
        task_id="review-probe", checkpoint_session_id="review-probe",
        benchmark="probe", dataset="probe", status="completed",
        report_path=str(path), eval_results=results, total=len(results), completed=len(results),
    )
    manager._tasks[entry.task_id] = entry
    before = manager.get_snapshot(entry.task_id)
    entry._release_heavy_data()
    after = manager.get_snapshot(entry.task_id)

    assert after["report_summary"] == before["report_summary"]
    assert after["stats_by_tag"] == before["stats_by_tag"]
    summary = after["report_summary"]
    errors = verdicts.count("error")
    assert summary["error_count"] == errors
    assert summary["graded_count"] == len(results) - errors
    assert summary["avg_score"] == report.avg_score
    if results:
        tag = summary["stats_by_tag"]["difficulty:Trend"]
        assert tag["error_count"] == errors
        assert tag["graded_count"] == len(results) - errors
        assert tag["count"] == len(results)


def test_saved_case_results_define_the_denominator_for_older_reports(tmp_path):
    import json

    from web.app.services.task_manager import _load_report_summary

    report = build_bench_report(
        [_result("graded", "scored", 1.0, ["dim:trend"]), _result("timeout", "error", 0.0, ["dim:trend"])],
        benchmark_name="probe", dataset_name="probe",
    ).model_dump(mode="json")
    # Old summaries averaged in errors and did not store a separate count.
    report.pop("error_count")
    report["avg_score"] = 0.5
    report["stats_by_tag"]["dim:trend"].pop("error_count")
    report["stats_by_tag"]["dim:trend"]["avg_score"] = 0.5
    path = tmp_path / "legacy.json"
    original = json.dumps(report)
    path.write_text(original, encoding="utf-8")

    stats, summary = _load_report_summary(str(path))
    assert summary["avg_score"] == 1.0
    assert summary["graded_count"] == 1
    assert summary["error_count"] == 1
    assert stats["dim:trend"]["avg_score"] == 1.0
    assert stats["dim:trend"]["error_count"] == 1
    assert path.read_text(encoding="utf-8") == original
