"""一次评测里有三个 LLM 角色，CLI 得能分别指定它们。

纯逻辑：不连网、不建库、不调模型。

    pytest evaluator/core/test_model_overrides.py -v

为什么值得单测：这三个角色的模型解析各自走不同的路径 —— target 走
``resolve_effective_target`` 的三层合并，虚拟用户走 ``AutoUserInfo``，判分器走
用例自己的 eval 配置，而且不同判分器的字段名还不一样。回归的方式很安静：把
``--user-model`` 接错地方，跑起来不会报错，只是**默认模型照旧**，非 OpenAI 环境
里全线失败而看不出原因 —— 这正是修好之前的状态。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import evaluator.plugin.eval_agent  # noqa: F401  触发判分器注册
import evaluator.plugin.target_agent  # noqa: F401
import evaluator.plugin.test_agent  # noqa: F401
from evaluator.core.bench_schema import BenchItem, bench_item_to_test_case
from evaluator.utils.benchmark_reader import _parse_target_specs

DATA = Path(__file__).resolve().parents[2] / "benchmark" / "data"

# (数据集, jsonl, 该数据集里 llm_api 那条 spec 的下标, 判分器声明的模型字段名)
CASES = [
    ("virtual_user", "round1.jsonl", "llm_api", "model"),
    ("eslbench", "sample200-20260430.jsonl", "llm_api", "model"),
    ("medcalc", "sample.jsonl", "llm_api", "extractor_model"),
    ("healthbench", "sample.jsonl", "llm_api", "model"),
]


def _first(dataset: str, fname: str, target_type: str):
    path = DATA / dataset / fname
    if not path.is_file():
        pytest.skip(f"{path} 不在仓库里")
    with path.open(encoding="utf-8") as fh:
        item = BenchItem.model_validate(json.loads(next(line for line in fh if line.strip())))
    meta = json.loads((DATA / dataset / "metadata.json").read_text(encoding="utf-8"))
    specs = [s for s in _parse_target_specs(meta["target"]) if s.type == target_type]
    if not specs:
        pytest.skip(f"{dataset} 没有 {target_type} target")
    return item, specs[0]


@pytest.mark.parametrize("dataset,fname,target_type,eval_field", CASES)
def test_eval_model_lands_on_whichever_field_the_evaluator_declares(dataset, fname, target_type, eval_field):
    """判分器们的模型字段名不统一 —— `model` / `judge_model` / `extractor_model`。

    只认一个名字的实现会在其他判分器上静默失效：命令接受了，判分器却仍用默认
    模型。所以断言的是「落到了这个判分器实际声明的那个字段」。
    """
    item, spec = _first(dataset, fname, target_type)

    case = bench_item_to_test_case(item, spec, {"eval_model": "SENTINEL-EVAL"})

    assert hasattr(case.eval, eval_field), f"{dataset} 的判分器没有 {eval_field} 字段，用例表需要更新"
    assert getattr(case.eval, eval_field) == "SENTINEL-EVAL"


@pytest.mark.parametrize("dataset,fname,target_type,_eval_field", CASES)
def test_reserved_keys_do_not_warn_about_the_target(caplog, dataset, fname, target_type, _eval_field):
    """`user_model` / `eval_model` / `target_type` 说的都不是被测目标。

    断言在**日志**上而不是在 target 上，因为 ``resolve_effective_target`` 对不可
    编辑的键本来就只是忽略 —— target 无论如何都干净，断言它等于什么都没测。真正
    的症状是每条用例刷一行「字段不可修改，已忽略」，`target_type` 原先就是这么
    刷的：一次 200 条的运行会淹掉 200 行。
    """
    item, spec = _first(dataset, fname, target_type)

    with caplog.at_level("WARNING", logger="evaluator.core.bench_schema"):
        case = bench_item_to_test_case(
            item, spec,
            {"target_type": target_type, "user_model": "SENTINEL-USER", "eval_model": "SENTINEL-EVAL"},
        )

    noisy = [r.getMessage() for r in caplog.records
             if any(k in r.getMessage() for k in ("user_model", "eval_model", "target_type"))]
    assert not noisy, f"保留键漏进了 target 解析: {noisy}"
    # 顺带确认它们也没被写进 target
    assert not {"user_model", "eval_model", "target_type"} & set(case.target.model_dump())


def test_user_model_reaches_an_auto_mode_virtual_user():
    """auto 模式下扮演用户的那个 LLM 必须可指定。

    它原先是 ``AutoTestAgent.__init__`` 的一个默认参数，而 orchestrator 建实例时
    只传 ``(user, history=...)`` —— 那个默认值从数据和 CLI 都碰不到，于是所有
    auto 数据集被钉死在一个 provider 上。
    """
    item, spec = _first("virtual_user", "round1.jsonl", "llm_api")
    assert item.user.type == "auto", "这条用例本该是 auto 模式"

    case = bench_item_to_test_case(item, spec, {"user_model": "SENTINEL-USER"})

    assert case.user.model == "SENTINEL-USER"


def test_the_agent_actually_uses_the_configured_model():
    """光把值放进配置不算 —— agent 得真的读它。"""
    from evaluator.core.schema import AutoUserInfo
    from evaluator.plugin.test_agent.auto_test_agent import AutoTestAgent

    configured = AutoTestAgent(AutoUserInfo(goal="g", model="SENTINEL-USER"))
    assert configured.model == "SENTINEL-USER"

    fallback = AutoTestAgent(AutoUserInfo(goal="g"))
    assert fallback.model == AutoTestAgent.DEFAULT_MODEL


def test_manual_mode_has_no_virtual_user_to_configure():
    """manual 模式照剧本念，没有 LLM 扮演用户 —— 传了也不该凭空长出字段。"""
    item, spec = _first("eslbench", "sample200-20260430.jsonl", "llm_api")
    assert item.user.type == "manual"

    case = bench_item_to_test_case(item, spec, {"user_model": "SENTINEL-USER"})

    assert "model" not in case.user.model_dump()


def test_no_overrides_leaves_everything_at_its_default():
    """不传任何覆盖时，行为必须和改动前一致。"""
    item, spec = _first("virtual_user", "round1.jsonl", "llm_api")

    case = bench_item_to_test_case(item, spec, None)

    assert case.user.model is None          # 交给 AutoTestAgent.DEFAULT_MODEL
    assert case.eval.model is None          # 交给判分器自己的默认值
