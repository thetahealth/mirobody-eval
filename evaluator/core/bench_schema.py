"""Benchmark 层数据结构 — 面向数据集管理和批量评测

与 core schema 的关系:
- BenchItem (数据集用例) → TestCase (执行单元)
- BenchUserInfo (可配置用户) → UserInfo (运行时用户)
- BenchMark (数据集) → List[TestCase] (执行队列)

核心差异:
1. BenchUserInfo 包含 target_overrides，用于在不同被测系统下覆盖参数
2. BenchItem 不包含 target 字段，由运行时 + target_overrides 决定
3. BenchMark 包含数据集元信息（描述、默认配置等）

转换流程:
    BenchItem + runtime_target → merge target_overrides → TestCase
"""

import logging
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Discriminator, Field, TypeAdapter, model_validator

from langchain_core.messages import AIMessage, HumanMessage

from evaluator.core.schema import (
    AutoUserInfo,
    ChatMessage,
    EvalInfo,
    ManualUserInfo,
    TargetAgentReaction,
    TargetInfo,
    TargetSpec,
    TestAgentAction,
    TestAgentMemory,
    TestAgentReaction,
    TestCase,
    TestCost,
    TestResult,
    UserInfo,
)

# TypeAdapter 用于将 dict → 正确的子类型（根据 discriminator 字段自动路由）
_TARGET_ADAPTER = TypeAdapter(TargetInfo)
_USER_ADAPTER = TypeAdapter(UserInfo)

def _normalize_target_overrides(v: Any) -> Dict[str, Dict[str, Any]]:
    """兼容旧 list 格式，统一转为 dict（key = target_type）"""
    if isinstance(v, dict):
        return v
    if isinstance(v, list):
        result: Dict[str, Dict[str, Any]] = {}
        for entry in v:
            if isinstance(entry, dict) and "type" in entry:
                entry = dict(entry)  # 避免修改原始数据
                t = entry.pop("type")
                result[t] = entry
        return result
    return {}


_TargetOverrides = Annotated[Dict[str, Dict[str, Any]], BeforeValidator(_normalize_target_overrides)]

_OVERRIDES_FIELD = Field(
    default_factory=dict,
    description=(
        "跨被测系统的参数覆盖映射（key = target_type，value = 覆盖参数 dict）。\n"
        "合并规则：override 优先（用例级定制），runtime 补充缺失字段。"
    ),
)


# ============================================================
# Benchmark 用户配置 — 支持 target_overrides（Discriminated Union）
# ============================================================


class BenchAutoUserInfo(AutoUserInfo):
    """Benchmark 自动模式用户配置 — 扩展 AutoUserInfo，支持 target_overrides"""

    model_config = ConfigDict(extra="forbid")
    target_overrides: _TargetOverrides = _OVERRIDES_FIELD


class BenchManualUserInfo(ManualUserInfo):
    """Benchmark 手动模式用户配置 — 扩展 ManualUserInfo，支持 target_overrides"""

    model_config = ConfigDict(extra="ignore")  # 忽略旧格式中的 goal / max_turns 字段
    target_overrides: _TargetOverrides = _OVERRIDES_FIELD


BenchUserInfo = Annotated[
    BenchAutoUserInfo | BenchManualUserInfo,
    Discriminator("type"),
]


# ============================================================
# Benchmark 用例定义 — 不含 target 字段
# ============================================================


class BenchItem(BaseModel):
    """Benchmark 用例定义 — 面向数据集管理

    与 TestCase 的差异:
    1. user 类型为 BenchUserInfo（支持 target_overrides）
    2. 没有 target 字段（由运行时 + target_overrides 决定）
    3. description 为可选字段（用于补充说明）

    转换为 TestCase:
        bench_item_to_test_case(item, runtime_target) → TestCase
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="用例唯一标识（如 hb_headache_001）")
    title: str = Field(description="用例标题（一句话概括场景）")
    description: Optional[str] = Field(None, description="用例描述（补充说明）")
    user: BenchUserInfo = Field(description="虚拟用户配置（包含 target_overrides）")
    eval: EvalInfo = Field(description="评估配置")
    history: List[Dict[str, str]] = Field(
        default_factory=list,
        description=(
            "评测前的历史对话记录（dict 格式，转换为 TestCase 时自动归一化为 BaseMessage）。\n"
            "格式: [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]\n"
            "示例场景：HealthBench 多轮 prompt 中最后一条 user message 之前的对话历史。"
        ),
    )
    tags: List[str] = Field(default_factory=list, description="标签（用于分类和筛选）")


# ============================================================
# Benchmark 数据集定义
# ============================================================


class BenchMark(BaseModel):
    """Benchmark 数据集定义 — 包含元信息和用例列表"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="数据集名称（如 healthbench/sample）")
    description: str = Field(default="", description="数据集描述（来自 metadata.json，支持 Markdown）")
    target: List[TargetSpec] = Field(default_factory=list, description="Target 配置规格列表（支持多 target 类型）")
    items: List[BenchItem] = Field(description="用例列表")
    total_count: int = Field(default=0, description="总用例数（自动计算）")
    max_concurrency: int = Field(default=0, description="默认并发数（0=不限制），可被 CLI --parallel 覆盖")

    @model_validator(mode="after")
    def _compute_total_count(self) -> "BenchMark":
        if self.total_count == 0:
            self.total_count = len(self.items)
        return self


# ============================================================
# Benchmark 报告定义
# ============================================================


class BenchReport(BaseModel):
    """Benchmark 评测报告"""

    model_config = ConfigDict(extra="forbid")

    benchmark_name: str = Field(description="评测类型（如 healthbench）")
    dataset_name: str = Field(description="数据集名称（如 sample）")
    dataset_provenance: dict = Field(
        default_factory=dict,
        description=(
            "这份分数对应的数据集版本。`source=fetched` 时带 batch / manifest_version / "
            "batch_checksum —— 有了它，一个分数才能指回一个确定的发布；`source=vendored` "
            "是仓库内自带的副本，能离线跑，但无法说明对应哪一期。缺这个字段的报告是本字段"
            "加入之前产生的。"
        ),
    )
    runtime_target: Optional[TargetInfo] = Field(None, description="运行时使用的被测系统配置（eval-only 模式为 None）")
    max_concurrency: int = Field(default=0, description="并发数")
    cases: List[TestResult] = Field(description="用例结果列表")
    pass_count: int = Field(default=0)
    fail_count: int = Field(default=0)
    error_count: int = Field(
        default=0,
        description="没判成分的用例数（超时/连接失败/判分器不可用/被取消）。"
        "不计入 pass/fail，也不计入 avg_score —— 所以读 avg_score 必须同时读它，"
        "否则一个大面积失败的跑批会显示成漂亮的高分",
    )
    pass_rate: float = Field(default=0.0)
    avg_score: float = Field(default=0.0, description="平均分，分母只含判成了分的用例（不含 error）")
    total_duration_seconds: float = Field(default=0.0)
    stats_by_tag: dict[str, dict] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=datetime.now)
    finished_at: Optional[datetime] = Field(None)


# ============================================================
# Eval-Only 外部输入
# ============================================================


class ApiCallResult(BaseModel):
    """外部调用结果 — eval-only 模式的输入格式

    每条对应一个用例的推理结果，通过 id 关联 BenchItem 获取 eval 配置。
    端点内部负责从 BenchItem 获取问题，与 answer 拼接为对话 memory。

    JSONL 格式::

        {"id": "user101_AT_demo_Q065", "answer": "根据您的血压..."}
        {"id": "user101_AT_demo_Q065", "answer": "...", "cost": {"gpt-5.2": {"input_tokens": 3791, "output_tokens": 68, "total_tokens": 3859}}}
    """

    id: str = Field(description="用例 ID（与 BenchItem.id 一一对应）")
    answer: str = Field(min_length=1, description="Agent 的回答内容")
    start: Optional[datetime] = Field(None, description="开始处理时间")
    end: Optional[datetime] = Field(None, description="处理完成时间")
    cost: Optional[Dict[str, Any]] = Field(None, description="Token 消耗，key 为模型名，value 含 input/output/total_tokens")


# ============================================================
# Eval-Only 转换
# ============================================================


def messages_to_memory(messages: list) -> list[TestAgentMemory]:
    """将 BaseMessage 列表转换为 TestAgentMemory 列表

    按 (user, assistant) 配对分组：
    - HumanMessage → TestAgentReaction (semantic_content)
    - AIMessage → TargetAgentReaction (message_list)
    - 末尾只有 user 无 assistant 时 target_response 为 None
    """
    memory_list: list[TestAgentMemory] = []
    now = datetime.now()
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.type == "human":
            action = TestAgentAction(type="semantic", semantic_content=str(msg.content))
            reaction = TestAgentReaction(action=action, is_finished=False)
            target = None
            if i + 1 < len(messages) and messages[i + 1].type == "ai":
                target = TargetAgentReaction(type="message", message_list=[{"content": str(messages[i + 1].content)}])
                i += 2
            else:
                i += 1
            memory_list.append(TestAgentMemory(
                test_reaction=reaction, test_reaction_time=now,
                target_response=target, target_response_time=now if target else None,
            ))
        else:
            i += 1
    return memory_list


def api_result_to_eval_items(
    results: list["ApiCallResult"],
    items: list["BenchItem"],
) -> list[tuple[TestCase, list[TestAgentMemory]]]:
    """将 ApiCallResult 列表 + BenchItem 列表转为 do_batch_eval 所需的参数

    对每个 result：
    1. 通过 id 匹配 BenchItem 获取 question（strict_inputs）和 eval 配置
    2. 构建对话 messages = history + [user question, assistant answer]
    3. 转为 (TestCase, memory_list) 元组

    Returns:
        [(TestCase, memory_list)] — 直接传给 do_batch_eval()
    """
    item_map: dict[str, "BenchItem"] = {item.id: item for item in items}
    eval_items: list[tuple[TestCase, list[TestAgentMemory]]] = []

    for r in results:
        item = item_map.get(r.id)
        if not item:
            _logger.warning("ApiCallResult id=%s 无匹配 BenchItem，跳过", r.id)
            continue

        # 从 BenchItem 获取 question
        question = ""
        if item.user and hasattr(item.user, "strict_inputs") and item.user.strict_inputs:
            question = item.user.strict_inputs[-1]

        # 构建 messages: history + [question, answer]
        messages = []
        if item.history:
            from evaluator.core.schema import _to_base_message
            messages.extend(_to_base_message(h) for h in item.history)
        if question:
            messages.append(HumanMessage(content=question))
        messages.append(AIMessage(content=r.answer))

        # 转为 memory_list
        memory_list = messages_to_memory(messages)

        # 构建 eval-only TestCase（仅需 eval 配置，不需要 target）
        test_case = TestCase(
            id=item.id,
            title=item.title or item.id,
            description=item.description,
            user={"type": "manual", "strict_inputs": ["(eval-only)"]},
            eval=item.eval,
            tags=item.tags or [],
            history=item.history or [],
        )

        eval_items.append((test_case, memory_list))

    return eval_items


# ============================================================
# 转换函数
# ============================================================

_logger = logging.getLogger(__name__)


def find_target_spec(specs: List[TargetSpec], target_type: str | None = None) -> TargetSpec:
    """从 TargetSpec 列表中查找指定类型的 spec

    Args:
        specs:       TargetSpec 列表（来自 metadata.json）
        target_type: 目标类型（如 "llm_api"）。只有一个候选时可以不给

    Returns:
        匹配的 TargetSpec

    Raises:
        ValueError: 列表为空、有多个候选却没指定、或 target_type 不匹配
    """
    if not specs:
        raise ValueError("metadata.json 中未定义 target 配置")
    if target_type is None:
        if len(specs) == 1:
            return specs[0]
        # 多个候选时不猜。以前是无条件返回 specs[0] —— 而 eslbench 的 metadata 把
        # `mirobody` 排在第一位,于是 README 里少写了 `--target-type` 的那条命令会
        # 静默选中 mirobody、跑到连数据库时才失败,报错说的是「数据库 admin 不存在」,
        # 跟真实原因（没说要测哪个系统）毫无关系。
        raise ValueError(
            f"该数据集有 {len(specs)} 个可测系统，必须用 --target-type 指定一个: "
            f"{[s.type for s in specs]}"
        )
    for spec in specs:
        if spec.type == target_type:
            return spec
    available = [s.type for s in specs]
    raise ValueError(f"未找到 target type '{target_type}'，可用类型: {available}")


def resolve_effective_target(
    spec: TargetSpec,
    cli_overrides: Dict[str, Any] | None = None,
    case_overrides: Dict[str, Any] | None = None,
) -> TargetInfo:
    """三层合并构建最终 TargetInfo

    优先级（高→低）:
        1. cli_overrides（CLI / Web UI 用户输入，仅 editable 字段生效）
        2. case_overrides（JSONL target_overrides，数据集作者控制，不受 editable 限制）
        3. spec defaults（metadata.json 默认值）

    Args:
        spec:           metadata.json 中的 TargetSpec
        cli_overrides:  CLI / Web UI 传入的覆盖参数（dict，如 {"model": "gpt-4.1"}）
        case_overrides: BenchItem.user.target_overrides 中匹配的覆盖条目（dict）

    Raises:
        ValueError: 必填字段缺少值
    """
    # Layer 1: spec defaults
    merged = spec.build_defaults()

    # Layer 2: case-level overrides（数据集作者控制，任意字段）
    if case_overrides:
        for k, v in case_overrides.items():
            if v is not None:
                merged[k] = v

    # Layer 3: CLI/UI overrides（仅 editable 字段）
    if cli_overrides:
        editable = spec.editable_field_names()
        for k, v in cli_overrides.items():
            if k == "type":
                continue
            if k in editable and v is not None:
                merged[k] = v
            elif k not in editable and v is not None:
                _logger.warning("字段 '%s' 不可修改（editable=false），已忽略", k)

    # 校验 required 字段
    for name, field in spec.fields.items():
        if field.required and name not in merged:
            raise ValueError(f"必填字段 '{name}' 缺少值（需通过 CLI/UI 提供或在 metadata 中设置默认值）")

    return _TARGET_ADAPTER.validate_python(merged)


def _get_target_type_map() -> Dict[str, type]:
    """从 AbstractTargetAgent._params_registry 获取 type → config model 映射"""
    from evaluator.core.interfaces.abstract_target_agent import AbstractTargetAgent

    return dict(AbstractTargetAgent._params_registry)


def resolve_runtime_target(
    spec: TargetSpec,
    cli_overrides: Dict[str, Any] | None = None,
) -> TargetInfo:
    """从 spec + CLI 覆盖构建全局 runtime target（用于日志和报告，不含 case-level overrides）

    与 resolve_effective_target 的区别：不含 case-level overrides，
    且使用 model_construct() 跳过 Pydantic 校验 — 允许缺少仅由
    target_overrides 提供的字段（如 extraction 的 email）。
    """
    merged = spec.build_defaults()
    if cli_overrides:
        editable = spec.editable_field_names()
        for k, v in cli_overrides.items():
            if k == "type":
                continue
            if k in editable and v is not None:
                merged[k] = v

    # 校验 editable + required 字段（必须由 CLI/UI 提供）
    for name, field_spec in spec.fields.items():
        if field_spec.required and field_spec.editable and name not in merged:
            raise ValueError(f"必填字段 '{name}' 缺少值（需通过 CLI/UI 提供或在 metadata 中设置默认值）")

    target_map = _get_target_type_map()
    cls = target_map.get(spec.type)
    if cls is None:
        raise ValueError(f"未知的 target type: {spec.type}，已注册: {sorted(target_map.keys())}")
    return cls.model_construct(**merged)


def bench_item_to_test_case(
    item: BenchItem,
    spec: TargetSpec,
    cli_overrides: Dict[str, Any] | None = None,
) -> TestCase:
    """将 BenchItem 转换为 TestCase（含三层 target 合并）

    Args:
        item:           BenchItem 数据
        spec:           metadata.json 中的 TargetSpec
        cli_overrides:  CLI / Web UI 传入的覆盖参数
    """
    # 按 target_type 查找 case-level override（支持同族 fallback）
    matched_override = item.user.target_overrides.get(spec.type)
    if matched_override is None:
        # 同族 fallback: 共享相同字段结构的 target type 可互相复用 override
        _OVERRIDE_FALLBACKS: Dict[str, list[str]] = {
            "dyg_rag_api": ["hippo_rag_api"],
            "hippo_rag_api": ["dyg_rag_api"],
        }
        for fallback_type in _OVERRIDE_FALLBACKS.get(spec.type, []):
            matched_override = item.user.target_overrides.get(fallback_type)
            if matched_override is not None:
                break

    # 跨族 fallback: 只认身份的 target 从 llm_api.tool_context.user_email 取 user_email。
    #
    # 这些 target 的 override 里唯一有用的字段就是「问的是谁」，而每条用例都要在
    # llm_api 那侧写一遍同一个邮箱，所以复用它，而不是要求语料为每个 target 各写一份。
    # 随附语料实测：35,535 条带 override 的用例，全部有这个字段，缺失 0 条。
    #
    # 这里原先读的是另一个 target 的键名，而那个 target 并不在本仓库里 —— 一个不存在的
    # target 的键成了几个公开 target 的唯一身份来源，等于把它的形状焊死在语料里。改读
    # llm_api 之后语料里就不必再留那个键，35,535 条用例的 override 也随之清干净。
    _IDENTITY_ONLY_TARGETS = (
        "hippo_rag_api", "dyg_rag_api", "naive_rag_api", "evermem", "mem0_rag_api", "hermes",
    )
    if matched_override is None and spec.type in _IDENTITY_ONLY_TARGETS:
        llm_override = item.user.target_overrides.get("llm_api")
        if isinstance(llm_override, dict):
            tool_ctx = llm_override.get("tool_context")
            if isinstance(tool_ctx, dict) and tool_ctx.get("user_email"):
                matched_override = {"user_email": tool_ctx["user_email"]}

    # A run has three LLM-driven actors — the virtual user, the target, and the
    # judge — and `cli_overrides` reaches only the target. The other two are
    # addressed by reserved keys, split off here: left in, each would trip the
    # "field is not editable, ignored" warning on every single case. `target_type`
    # is the pre-existing member of that family and did exactly that.
    target_overrides_from_cli = dict(cli_overrides or {})
    user_model = target_overrides_from_cli.pop("user_model", None)
    eval_model = target_overrides_from_cli.pop("eval_model", None)
    target_overrides_from_cli.pop("target_type", None)

    effective_target = resolve_effective_target(spec, target_overrides_from_cli or None, matched_override)
    user_dict = item.user.model_dump(exclude={"target_overrides"})
    if user_model and user_dict.get("type") == "auto":
        user_dict["model"] = user_model
    user_info = _USER_ADAPTER.validate_python(user_dict)

    eval_config = item.eval
    if eval_model:
        # Evaluators disagree on the field name (`model`, `judge_model`,
        # `extractor_model`), so set whichever this one declares rather than
        # guessing one and silently doing nothing on the others.
        for field in ("model", "judge_model", "extractor_model"):
            if field in type(eval_config).model_fields:
                eval_config = eval_config.model_copy(update={field: eval_model})
                break
        else:
            _logger.warning(
                "评测器 %s 没有可覆盖的模型字段，--eval-model 对它无效",
                type(eval_config).__name__,
            )

    # answer_format_hint: 如果 eval config 定义了格式提示，追加到最后一条 strict_inputs
    hint = getattr(item.eval, "answer_format_hint", None)
    if hint and user_info.strict_inputs:
        user_info.strict_inputs[-1] += hint

    return TestCase(
        id=item.id,
        title=item.title,
        description=item.description,
        user=user_info,
        target=effective_target,
        eval=eval_config,
        history=item.history,  # dict 列表 → TestCase field_validator 自动归一化为 BaseMessage
        tags=item.tags,
    )


def compute_stats_by_tag(test_results: list[TestResult]) -> dict[str, dict]:
    """按 tag 分组统计 —— CLI 报告和 Web UI 共用这一份。

    以前是两份:`build_bench_report` 一份、`web/app/services/task_manager.py`
    一份。两份算的是同一个东西，但「平均分要不要排除没判成分的用例」这条规则
    只在一份里修，另一份就继续用旧口径 —— 一个 tag 的分数会因为你从 CLI 还是
    从网页看而不同。规则只写一次，就不会有那种分歧。
    """
    tag_results: dict[str, list[TestResult]] = {}
    for result in test_results:
        for tag in result.tags:
            tag_results.setdefault(tag, []).append(result)

    stats: dict[str, dict] = {}
    for tag, results in tag_results.items():
        tag_pass = sum(1 for r in results if r.eval.result == "pass")
        tag_fail = sum(1 for r in results if r.eval.result == "fail")
        tag_error = sum(1 for r in results if r.eval.result == "error")
        tag_judged = tag_pass + tag_fail
        tag_scored = [r for r in results if r.eval.result != "error"]
        stats[tag] = {
            "total": len(results),
            "pass_count": tag_pass,
            "fail_count": tag_fail,
            "error_count": tag_error,
            "pass_rate": tag_pass / tag_judged if tag_judged else 0.0,
            "avg_score": (sum(r.eval.score for r in tag_scored) / len(tag_scored)) if tag_scored else 0.0,
        }
    return stats


def build_bench_report(
    test_results: list[TestResult],
    benchmark_name: str,
    dataset_name: str,
    runtime_target: TargetInfo | None = None,
    max_concurrency: int = 0,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
) -> BenchReport:
    """从 TestResult 列表构建 BenchReport"""
    pass_count = sum(1 for r in test_results if r.eval.result == "pass")
    fail_count = sum(1 for r in test_results if r.eval.result == "fail")
    error_count = sum(1 for r in test_results if r.eval.result == "error")
    judged = pass_count + fail_count  # 有 pass/fail 判定的用例数（不含 scored）
    pass_rate = pass_count / judged if judged else 0.0
    # 平均分只在「判成了分」的用例上算。error 的 score 恒为 0.0，但那个 0 不是
    # 成绩 —— 把它算进分母和分子，等于让没跑完的用例按 0 分拉低被测系统的成绩。
    # 这一条此前漏了:`pass_rate` 的分母早就排除了 error，`avg_score` 没有，所以
    # 连判分器不可用那条本来做对了的路（`JudgeUnavailable` → error）也一直在
    # 被算作 0 分。`error_count` 与之配套发布 —— 排除了就必须让分母可见,否则
    # 一个大面积失败的跑批会显示成漂亮的高分。
    scored_results = [r for r in test_results if r.eval.result != "error"]
    avg_score = sum(r.eval.score for r in scored_results) / len(scored_results) if scored_results else 0.0
    total_duration = sum((r.end - r.start).total_seconds() for r in test_results)

    stats_by_tag = compute_stats_by_tag(test_results)

    # 读一次数据集来源，写进报告：一个分数必须能指回它跑的是哪一期。
    # 解析失败不该让一份跑完的报告丢掉 —— 来源缺失只是少了追溯信息。
    try:
        from evaluator.utils.benchmark_reader import resolve_dataset

        _, provenance = resolve_dataset(benchmark_name, dataset_name)
    except Exception:  # pragma: no cover - 来源信息是附加项，不能反过来毁掉报告
        provenance = {}

    return BenchReport(
        benchmark_name=benchmark_name,
        dataset_name=dataset_name,
        dataset_provenance=provenance,
        runtime_target=runtime_target,
        max_concurrency=max_concurrency,
        cases=test_results,
        pass_count=pass_count,
        fail_count=fail_count,
        error_count=error_count,
        pass_rate=pass_rate,
        avg_score=avg_score,
        total_duration_seconds=total_duration,
        stats_by_tag=stats_by_tag,
        started_at=started_at or datetime.now(),
        finished_at=finished_at,
    )
