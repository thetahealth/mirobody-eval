"""weeks_days ground truths in the shipped MedCalc data are tuple strings, and they must parse.

Pure logic: no LLM, no network.

    pytest evaluator/plugin/eval_agent/test_medcalc_weeks_days.py -v

MedCalc-Bench stores the gestational-age answer as "('35 weeks', '4 days')". The normalizer only
allowed commas and spaces between the two numbers, so every weeks_days case failed with
"Standard answer weeks/days format error", whatever the target answered.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluator.plugin.eval_agent.medcalc_eval_agent import MedCalcEvalAgent, MedCalcEvalInfo

_DATA = Path(__file__).resolve().parents[3] / "benchmark" / "data" / "medcalc"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("('35 weeks', '4 days')", (35, 4)),
        ("('0 weeks', '6 days')", (0, 6)),
        ("35 weeks, 4 days", (35, 4)),
        ("17 weeks 4 days", (17, 4)),
        ("1 week, 1 day", (1, 1)),
        ("no answer", None),
    ],
)
def test_normalize_weeks_days(text, expected):
    assert MedCalcEvalAgent._normalize_weeks_days(text) == expected


def _weeks_days_ground_truths() -> list[str]:
    truths = []
    for line in (_DATA / "full.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        cfg = json.loads(line)["eval"]
        if cfg.get("output_type") == "weeks_days":
            truths.append(cfg["ground_truth"])
    return truths


def test_shipped_weeks_days_ground_truths_parse():
    truths = _weeks_days_ground_truths()
    assert len(truths) == 20
    assert all(MedCalcEvalAgent._normalize_weeks_days(t) is not None for t in truths)


def test_matching_answer_passes_against_shipped_format():
    agent = MedCalcEvalAgent(
        MedCalcEvalInfo(evaluator="medcalc", ground_truth="('35 weeks', '4 days')", output_type="weeks_days")
    )
    assert agent._match_answer("35 weeks, 4 days")[0] is True
    assert agent._match_answer("35 weeks, 5 days")[0] is False
