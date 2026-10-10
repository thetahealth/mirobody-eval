"""判分器跑不起来的时候，不许返回一个看着像分数的数字。

纯逻辑：把判分器要调的 LLM 换成必然失败的假函数，不连网。

    pytest evaluator/plugin/eval_agent/test_kg_qa_judge_failure.py -v

这组测试锁的是一个真实事故的形状：`_eval_text` 原先在三次重试全失败后，退化成
数「标准答案里的词被蹭中了几个」，并把那个比率当分数返回。于是一次没有可用 key
的运行会产出一整份看似正常的报告 —— 实测过一次平均 0.51，每条文字题的
feedback 里写着 "retries exhausted, fallback"，而那个字段没人会展开看。

规则判分的题型（数字/是否/列表）不碰 LLM，必须完全不受影响 —— 这是「没 key 也能
跑 106 道题」的依据，所以一并锁住。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluator.core.schema import SessionInfo
from evaluator.plugin.eval_agent import kg_qa_eval_agent as mod
from evaluator.plugin.eval_agent.kg_qa_eval_agent import JudgeUnavailable, KgQaEvalAgent, KgQaEvalInfo

_DATA = Path(__file__).resolve().parents[3] / "benchmark" / "data"
# behavioral 题只在 distractor 语料里，所以两份都扫 —— 否则那条测试会永远 skip，
# 而它锁的正是本轮改动里最容易被误读的一处（判分器故障被记成被考方失败）。
DATASETS = (
    _DATA / "eslbench" / "sample200-20260430.jsonl",
    _DATA / "eslbench_distractor" / "sample.jsonl",
)


def _config(answer_type: str) -> tuple[KgQaEvalInfo, str]:
    """题库里第一道该类型的题，连同它的标准答案。"""
    for path in DATASETS:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                cfg = json.loads(line).get("eval") or {}
                if cfg.get("answer_type") == answer_type:
                    return KgQaEvalInfo(**cfg), str(cfg["expected_value"])
    pytest.skip(f"随附语料里没有 {answer_type} 类型的题")


def _agent(cfg: KgQaEvalInfo, response: str) -> KgQaEvalAgent:
    agent = KgQaEvalAgent(cfg)
    agent._extract_response = staticmethod(lambda _memory: response)
    return agent


@pytest.fixture
def judge_is_down(monkeypatch):
    """让判分器要调的 LLM 必然失败，模拟「没有可用 key」。"""
    async def _boom(*_args, **_kwargs):
        raise RuntimeError("no api key configured")

    monkeypatch.setattr(mod, "do_execute", _boom)


async def test_text_reports_an_error_instead_of_a_keyword_ratio(judge_is_down):
    cfg, expected = _config("text")

    result = await _agent(cfg, expected).run([], SessionInfo(user_id="u", user_token="t", has_user_data=True))

    assert result.result == "error"
    assert "judge_error" in result.feedback
    # 关键：不是「分低」，是「没有分」。喂的是标准答案，关键词退路会给出接近满分
    # 的比率 —— 那正是必须消失的东西。
    assert result.score == 0.0
    assert "fallback" not in result.feedback


async def test_the_error_names_the_model_it_tried(judge_is_down):
    """报错要能让人查下去：用的是哪个模型。"""
    cfg, expected = _config("text")

    result = await _agent(cfg, expected).run([], SessionInfo(user_id="u", user_token="t", has_user_data=True))

    assert (result.trace.eval_detail or {}).get("judge_model") == cfg.model


async def test_behavioral_does_not_blame_the_agent_for_a_judge_outage(judge_is_down):
    """behavioral 原先返回 0.0「cannot confirm resistance」。

    那个 0 读起来像「被考方没能抵抗错误前提」—— 一句关于被考方的判断。而实际
    发生的事情是判分器根本没跑。
    """
    cfg, expected = _config("behavioral")

    result = await _agent(cfg, expected).run([], SessionInfo(user_id="u", user_token="t", has_user_data=True))

    assert result.result == "error"
    assert "cannot confirm resistance" not in result.feedback


@pytest.mark.parametrize("answer_type", ["numeric_value", "list", "boolean"])
async def test_rule_scored_types_are_untouched_by_a_dead_judge(judge_is_down, answer_type):
    """这三种题型不调 LLM，判分器挂了也必须照常给分。

    106 道题「零 key 可跑」就是靠这条。
    """
    cfg, expected = _config(answer_type)

    result = await _agent(cfg, expected).run([], SessionInfo(user_id="u", user_token="t", has_user_data=True))

    assert result.result == "scored", f"{answer_type} 不该受判分器影响"


async def test_judge_unavailable_never_escapes_to_the_framework(judge_is_down):
    """异常必须在 `run()` 里被接住 —— 漏出去会变成 orchestrator 的「测试执行异常」，
    和「被测系统崩了」混为一谈。"""
    cfg, expected = _config("text")

    try:
        result = await _agent(cfg, expected).run([], SessionInfo(user_id="u", user_token="t", has_user_data=True))
    except JudgeUnavailable as e:  # pragma: no cover - 这就是要防的回归
        pytest.fail(f"JudgeUnavailable 漏到了 run() 外面: {e}")

    assert result.result == "error"
