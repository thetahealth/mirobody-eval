"""The answer that is graded is the last ANSWER: line the response commits to.

Pure logic: no LLM, no network.

    pytest evaluator/plugin/eval_agent/test_kg_qa_answer_line.py -v

A response may draft an answer, check its work and correct itself. The parser used to take the
first ANSWER: line, so the corrected answer was never graded, and an empty `ANSWER:` borrowed the
next line of text as its value.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluator.core.schema import SessionInfo
from evaluator.plugin.eval_agent.kg_qa_eval_agent import KgQaEvalAgent, KgQaEvalInfo, _extract_answer_line

_BANK = Path(__file__).resolve().parents[3] / "benchmark" / "data" / "eslbench" / "sample200-20260430.jsonl"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("No answer line here.", None),
        ("Reasoning.\nANSWER: 36", "36"),
        ("ANSWER: 30\nWait, window B was misread.\nANSWER: 36", "36"),
        ("Reasoning.\nAnswer: 36", "36"),
        ("  ANSWER :  36  ", "36"),
        ("ANSWER: 30\nOn reflection there is no answer.\nANSWER:", ""),
        ("ANSWER:\n36", ""),
        ("The ANSWER: field is described below.", None),
    ],
)
def test_extract_answer_line(text, expected):
    assert _extract_answer_line(text) == expected


def _numeric_case() -> tuple[KgQaEvalInfo, float]:
    """The first shipped numeric question whose gold is a single non-zero number."""
    if not _BANK.is_file():
        pytest.skip(f"{_BANK.name} is not shipped")
    with _BANK.open(encoding="utf-8") as fh:
        for line in fh:
            cfg = (json.loads(line) if line.strip() else {}).get("eval") or {}
            if cfg.get("answer_type") != "numeric_value":
                continue
            try:
                gold = float(str(cfg["expected_value"]))
            except (KeyError, ValueError):
                continue
            if gold:
                return KgQaEvalInfo(**cfg), gold
    pytest.skip("no single-number numeric question in the bank")


async def _score(cfg: KgQaEvalInfo, response: str) -> float:
    agent = KgQaEvalAgent(cfg)
    agent._extract_response = staticmethod(lambda _memory: response)
    result = await agent.run([], SessionInfo(user_id="u", user_token="t", has_user_data=True))
    return result.score


async def test_a_corrected_answer_is_graded_as_corrected():
    cfg, gold = _numeric_case()
    wrong = gold * 10

    corrected = await _score(cfg, f"ANSWER: {wrong}\nI misread a window; recomputing.\nANSWER: {gold}")
    reverted = await _score(cfg, f"ANSWER: {gold}\nI misread a window; recomputing.\nANSWER: {wrong}")

    assert corrected == 1.0
    assert reverted < 1.0
