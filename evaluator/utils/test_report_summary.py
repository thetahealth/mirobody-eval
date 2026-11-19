"""The report list carries a summary of each report, computed the way the report page computes it.

Pure logic: throwaway report files, no network.

    pytest evaluator/utils/test_report_summary.py -v
"""

from __future__ import annotations

import json
import os

from evaluator.utils import report_reader


def _case(result: str, score: float | None, cost: dict | None = None) -> dict:
    return {"id": f"c{result}{score}", "eval": {"result": result, "score": score}, "cost": cost or {}}


GPT = {"gpt-4.1": {"input_tokens": 1000, "output_tokens": 100, "input_token_details": {"cache_read": 200}}}


def _write_report(dir_, name: str, cases: list[dict]) -> None:
    bench = dir_ / "eslbench"
    bench.mkdir(exist_ok=True)
    (bench / name).write_text(json.dumps({"cases": cases}), encoding="utf-8")


def _entries(tmp_path, monkeypatch):
    monkeypatch.setattr(report_reader, "_REPORT_DIR", tmp_path)
    return {e.filename: e for e in report_reader.list_reports()}


def test_average_leaves_out_ungraded_cases(tmp_path, monkeypatch):
    """`error` carries score 0.0, but that 0 is not a grade — same rule as the report page."""
    _write_report(
        tmp_path,
        "sample_x_20260101_000000.json",
        [
            _case("scored", 1.0),
            _case("scored", 0.5),
            _case("error", 0.0),
            _case("fail", 0.0),
        ],
    )

    e = _entries(tmp_path, monkeypatch)["sample_x_20260101_000000.json"]

    assert e.case_count == 4
    assert e.error_count == 1
    assert e.avg_score == 0.5  # (1.0 + 0.5 + 0.0) / 3


def test_api_reported_target_cost_is_not_counted_twice(tmp_path, monkeypatch):
    """A target that reports its own USD has its tokens left out of the usage the page prices."""
    _write_report(
        tmp_path,
        "sample_x_20260101_000000.json",
        [
            _case("scored", 1.0, {"target": GPT, "target_detail": {"total_cost": 0.25}, "eval": GPT}),
            _case("scored", 1.0, {"target": GPT, "test": GPT}),
        ],
    )

    u = _entries(tmp_path, monkeypatch)["sample_x_20260101_000000.json"].usage

    assert u["target_api_cost"] == 0.25
    assert u["target"]["gpt-4.1"]["input_tokens"] == 1000  # only the case without total_cost
    assert u["eval"]["gpt-4.1"]["input_token_details"]["cache_read"] == 200
    assert u["test"]["gpt-4.1"]["output_tokens"] == 100


def test_a_changed_file_is_summarized_again(tmp_path, monkeypatch):
    name = "sample_x_20260101_000000.json"
    _write_report(tmp_path, name, [_case("scored", 1.0)])
    assert _entries(tmp_path, monkeypatch)[name].case_count == 1

    _write_report(tmp_path, name, [_case("scored", 1.0), _case("scored", 0.0)])
    path = tmp_path / "eslbench" / name
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

    assert _entries(tmp_path, monkeypatch)[name].case_count == 2


def test_the_target_comes_from_the_run_not_the_filename(tmp_path, monkeypatch):
    """`llm_api` itself has an underscore, so the filename's label cannot be split back apart."""
    bench = tmp_path / "eslbench"
    bench.mkdir()
    (bench / "sample_llm_api_deepseek:deepseek-flash_20260101_000000.json").write_text(
        json.dumps({"runtime_target": {"type": "llm_api", "model": "deepseek:deepseek-flash"}, "cases": []}),
        encoding="utf-8",
    )

    e = _entries(tmp_path, monkeypatch)["sample_llm_api_deepseek:deepseek-flash_20260101_000000.json"]

    assert (e.target_type, e.target_model) == ("llm_api", "deepseek:deepseek-flash")


def test_an_unreadable_report_is_still_listed_without_a_summary(tmp_path, monkeypatch):
    bench = tmp_path / "eslbench"
    bench.mkdir()
    (bench / "sample_x_20260101_000000.json").write_text("{not json", encoding="utf-8")

    e = _entries(tmp_path, monkeypatch)["sample_x_20260101_000000.json"]

    assert e.case_count is None and e.avg_score is None and e.usage is None
