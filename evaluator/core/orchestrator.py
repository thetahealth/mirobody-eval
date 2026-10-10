"""
编排器 — 评测框架唯一入口

核心函数 do_single_test(TestCase) -> TestResult：
  1. 初始化   — 通过基类注册表创建 TestAgent / TargetAgent / EvalAgent
  2. 对话循环 — TestAgent.do_generate ↔ TargetAgent.do_generate 交替执行
  3. 评估     — EvalAgent.run(memory_list, session_info)
  4. 汇总     — 封装 TestResult 返回

批量执行:
  do_batch_test  — 向后兼容的简便接口
  BatchSession   — 可观测、可取消的批量执行会话（适用于 server 场景）

插件注册（基于 __init_subclass__）:
  class AutoTestAgent(AbstractTestAgent, name="auto"):       # LLM 驱动
  class ManualTestAgent(AbstractTestAgent, name="manual"):   # 脚本驱动
  class LlmApiTargetAgent(AbstractTargetAgent, name="llm_api"):    # 基于 do_execute
  class HermesTargetAgent(AbstractTargetAgent, name="hermes"):     # 外部 HTTP 服务
  class SemanticEvalAgent(AbstractEvalAgent, name="semantic"):

插件导入（触发注册）:
  import evaluator.plugin.test_agent    # AutoTestAgent / ManualTestAgent
  import evaluator.plugin.target_agent  # LlmApiTargetAgent / HermesTargetAgent / ...
  import evaluator.plugin.eval_agent    # SemanticEvalAgent / RubricEvalAgent / ...

设计原则:
- core 不依赖任何具体实现，仅依赖抽象接口
- 插件通过 import 触发 __init_subclass__ 自动注册，无需手动调用工厂方法
- 所有调用方式（API / CLI / Scheduler）最终都调用 do_single_test
- 批量执行在上层循环调用此函数

"""

import asyncio
import enum
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List
from uuid import uuid4

from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent
from evaluator.core.interfaces.abstract_target_agent import AbstractTargetAgent
from evaluator.core.interfaces.abstract_test_agent import AbstractTestAgent
from langchain_core.messages.ai import UsageMetadata

from evaluator.core.schema import (
    EvalResult,
    EvalTrace,
    TargetInfo,
    TestAgentMemory,
    TestCase,
    TestCost,
    TestReport,
    TestResult,
)

logger = logging.getLogger(__name__)

_SEPARATOR_HEAVY = "=" * 60
_SEPARATOR_LIGHT = "-" * 60


# ============================================================
# 内部工厂方法
# ============================================================


def _create_test_agent(test_case: TestCase) -> AbstractTestAgent:
    """根据 UserInfo.type 从基类注册表创建 TestAgent 实例"""
    cls = AbstractTestAgent.get(test_case.user.type)
    return cls(test_case.user, history=test_case.history)


def _create_target_agent(test_case: TestCase, default_target: TargetInfo | None = None) -> AbstractTargetAgent:
    """根据 TargetInfo.type 从基类注册表创建 TargetAgent 实例

    Args:
        test_case: 测试用例
        default_target: 全局默认 target 配置（可选）

    优先级: test_case.target > default_target > 抛出异常
    """
    target_info = test_case.target or default_target
    if target_info is None:
        raise ValueError(
            f"用例 {test_case.id} 未指定 target 配置，且未提供全局默认配置。"
            "请在 TestCase 中指定 target 字段，或通过 do_single_test(default_target=...) 提供全局配置。"
        )

    cls = AbstractTargetAgent.get(target_info.type)
    return cls(target_info, history=test_case.history)


def _create_eval_agent(test_case: TestCase) -> AbstractEvalAgent:
    """根据 EvalInfo.evaluator 从基类注册表创建 EvalAgent 实例

    model 等配置由各 EvalAgent 自行从 eval_config 中读取，orchestrator 不做额外注入。
    """
    eval_name = test_case.eval.evaluator
    cls = AbstractEvalAgent.get(eval_name)
    return cls(test_case.eval, history=test_case.history, user_info=test_case.user, case_id=test_case.id)


async def _do_eval(
    test_case: TestCase,
    memory_list: list[TestAgentMemory],
    *,
    meta: dict,
    start_time: datetime,
    session_info: Any = None,
    extra_cost: TestCost | None = None,
    target_memory: list | None = None,
) -> TestResult:
    """创建 EvalAgent + 执行评估 + 组装 TestResult

    do_single_test 和 do_eval_only 共享的评估 + 结果组装逻辑。
    """
    case_id = test_case.id
    eval_agent = _create_eval_agent(test_case)
    eval_model = getattr(eval_agent, "model", "unknown")

    logger.info("[%s] 开始评估 (evaluator=%s, model=%s)", case_id, test_case.eval.evaluator, eval_model)
    eval_result: EvalResult = await eval_agent.run(memory_list, session_info)

    logger.info("[%s] 评估完成 — result=%s, score=%.2f", case_id, eval_result.result.upper(), eval_result.score)
    if eval_result.feedback:
        logger.info("[%s] 反馈: %s", case_id, _truncate(eval_result.feedback, 300))

    # 收集 EvalAgent 成本
    eval_cost_dict: dict[str, UsageMetadata] = {}
    if hasattr(eval_agent, "cost") and hasattr(eval_agent, "model"):
        eval_cost: UsageMetadata = eval_agent.cost
        if eval_cost["total_tokens"] > 0:
            eval_cost_dict[eval_agent.model] = eval_cost

    # 组装 TestResult
    end_time = datetime.now()
    cost = TestCost(
        test=extra_cost.test if extra_cost else {},
        eval=eval_cost_dict,
        target=extra_cost.target if extra_cost else None,
        target_detail=extra_cost.target_detail if extra_cost else None,
    )

    return TestResult(
        id=case_id,
        **meta,
        eval=EvalResult(
            result=eval_result.result,
            score=eval_result.score,
            feedback=eval_result.feedback,
            trace=EvalTrace(
                history=test_case.history,
                test_memory=memory_list,
                target_memory=target_memory or [],
                eval_detail=eval_result.trace.eval_detail if eval_result.trace else None,
            ),
        ),
        cost=cost,
        start=start_time,
        end=end_time,
    )


# ============================================================
# 执行上下文
# ============================================================


class CaseStatus(str, enum.Enum):
    """用例执行状态"""

    PENDING = "pending"
    INIT = "init"
    DIALOGUE = "dialogue"
    EVAL = "eval"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class CaseContext:
    """单个用例的运行时上下文 — 可变状态容器

    由 BatchSession 或外部调用方创建，传入 do_single_test。
    do_single_test 在各阶段转换时更新 status 和 turn。
    外部可通过读取 test_agent / target_agent 的 memory_list 获取实时对话记忆。

    状态流转: pending → init → dialogue → eval → completed
                                                → error（异常）
                                                → cancelled（取消）
    """

    case_id: str
    status: CaseStatus = CaseStatus.PENDING
    turn: int = 0
    error: str | None = None
    test_agent: AbstractTestAgent | None = field(default=None, repr=False)
    target_agent: AbstractTargetAgent | None = field(default=None, repr=False)
    result: TestResult | None = field(default=None, repr=False)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    on_turn: Any = field(default=None, repr=False)  # Callable[[CaseContext], None] | None — 每轮对话完成后回调

    @property
    def is_cancel_requested(self) -> bool:
        """检查是否有取消请求"""
        return self.cancel_event.is_set()

    def request_cancel(self) -> None:
        """请求取消此用例"""
        self.cancel_event.set()


class _CancellationError(Exception):
    """内部异常 — 仅用于取消用例时跳出执行流"""


# ============================================================
# 核心编排函数
# ============================================================


async def do_single_test(
    test_case: TestCase,
    *,
    context: CaseContext | None = None,
    default_target: TargetInfo | None = None,
) -> TestResult:
    """
    执行单个测试用例 — 框架唯一入口

    执行流程:
      1. 初始化   — 通过基类注册表创建 TestAgent / TargetAgent / EvalAgent
      2. 对话循环 — TestAgent ↔ TargetAgent 交替，直到 is_finished 或 max_turns
      3. 评估     — EvalAgent.run(memory_list, session_info) -> EvalResult
      4. 汇总     — 封装 TestResult 返回

    异常处理:
      任何阶段抛出异常，均捕获并返回 result="error" 的 TestResult，异常类型与
      信息写入 feedback 和 trace.eval_detail，确保批量执行不会因单用例中断。
      是 error 而不是 fail —— fail 的含义是「判过分、分低」，而抛异常意味着
      没判成，两者的处置完全不同。error 不计入 pass/fail，也不计入 avg_score。

    Args:
        test_case: 测试用例
        context:   可选的执行上下文，用于实时状态跟踪和取消控制。
                   传入 CaseContext 后，do_single_test 会在各阶段更新其 status/turn/agent 引用，
                   外部可随时读取 context.test_agent.memory_list 等实时数据。
        default_target: 全局默认 target 配置（可选）。
                        当 test_case.target 为 None 时使用此配置。
                        优先级: test_case.target > default_target > 错误

    Returns:
        TestResult
    """
    start_time = datetime.now()
    case_id = test_case.id

    # ---- 上下文辅助（context 为 None 时全部 no-op）----
    ctx = context

    def _update_ctx(status: CaseStatus | None = None, **kwargs: object) -> None:
        if ctx is None:
            return
        if status is not None:
            ctx.status = status
        for k, v in kwargs.items():
            setattr(ctx, k, v)

    def _check_cancel() -> None:
        if ctx is not None and ctx.is_cancel_requested:
            raise _CancellationError(f"用例 {case_id} 被取消")

    # ---- 测试用例概览 ----
    logger.info("[%s] %s", case_id, _SEPARATOR_HEAVY)
    logger.info("[%s] 测试用例: %s", case_id, test_case.title)
    if test_case.description:
        logger.info("[%s] 描述: %s", case_id, test_case.description)
    if hasattr(test_case.user, "goal") and test_case.user.goal:
        logger.info("[%s] 用户目标: %s", case_id, test_case.user.goal)
    if hasattr(test_case.user, "context") and test_case.user.context:
        logger.info("[%s] 用户背景: %s", case_id, _truncate(test_case.user.context, 200))
    if test_case.user.strict_inputs:
        logger.info("[%s] 强制输入: %d 条", case_id, len(test_case.user.strict_inputs))
    if test_case.history:
        logger.info("[%s] 历史对话: %d 条消息", case_id, len(test_case.history))
    logger.info("[%s] 最大轮次: %s", case_id, getattr(test_case.user, "max_turns", "auto"))

    # 用例元数据（写入 TestResult 供报告展示）
    # 注意：target_type 需要从 effective target 获取（因为 test_case.target 可能为 None）
    effective_target = test_case.target or default_target
    _meta = {
        "title": test_case.title,
        "user_type": test_case.user.type,
        "target_type": effective_target.type if effective_target else "",
        "eval_type": test_case.eval.evaluator,
        "tags": test_case.tags,
        "target_config": effective_target.model_dump(exclude={"type"}) if effective_target else None,
        "eval_config": test_case.eval.model_dump(exclude={"evaluator"}),
    }

    try:
        # ---- 1. 初始化阶段 ----
        test_agent = _create_test_agent(test_case)
        target_agent = _create_target_agent(test_case, default_target=default_target)

        _update_ctx(CaseStatus.INIT, test_agent=test_agent, target_agent=target_agent)
        _check_cancel()

        test_model = getattr(test_agent, "model", "unknown")

        logger.info(
            "[%s] Agent 初始化完成 — TestAgent=%s(model=%s), TargetAgent=%s",
            case_id,
            type(test_agent).__name__, test_model,
            type(target_agent).__name__,
        )
        logger.info("[%s] %s", case_id, _SEPARATOR_LIGHT)

        # ---- 2. 对话循环阶段 ----
        _update_ctx(CaseStatus.DIALOGUE)
        target_reaction = None
        turn = 0

        while True:
            turn += 1
            _update_ctx(turn=turn)
            _check_cancel()

            # TestAgent 生成用户动作
            test_reaction = await test_agent.do_generate(target_reaction)

            # 输出 TestAgent 这一轮的行为
            user_text = test_reaction.action.semantic_content or ""
            if test_reaction.is_finished:
                # is_finished 轮次未发送给 target，不计入对话轮次
                turn -= 1
                _update_ctx(turn=turn)
                logger.info("[%s] 虚拟用户标记对话结束 (reason: %s)", case_id, test_reaction.reason)
                break

            logger.info("[%s] Turn %d", case_id, turn)
            logger.info(
                "[%s]   用户: %s",
                case_id, _truncate(user_text, 200),
            )
            if test_reaction.reason:
                logger.debug("[%s]         reason: %s", case_id, test_reaction.reason)
            if test_reaction.next_fuzzy_action:
                logger.debug("[%s]         next_fuzzy: %s", case_id, test_reaction.next_fuzzy_action)

            # TargetAgent 生成被测系统响应
            target_reaction = await target_agent.do_generate(test_reaction.action)

            # 输出 TargetAgent 这一轮的响应
            target_text = target_reaction.extract_text()
            logger.info(
                "[%s]   AI助手: %s",
                case_id, _truncate(target_text, 200),
            )

            # 每轮对话完成后触发回调（用于持久化 live 数据）
            if context and context.on_turn:
                try:
                    context.on_turn(context)
                except Exception:
                    logger.warning("[%s] on_turn 回调异常", case_id, exc_info=True)

        logger.info("[%s] %s", case_id, _SEPARATOR_LIGHT)
        logger.info("[%s] 对话完成，共 %d 轮", case_id, turn)

        # 对话结束后写一次完整的 live 数据（此时 target_response 已全部回填）
        if context and context.on_turn:
            try:
                context.on_turn(context)
            except Exception:
                logger.warning("[%s] on_turn 回调异常（对话结束）", case_id, exc_info=True)

        # ---- 3. 清理资源 ----
        if hasattr(target_agent, "cleanup"):
            await target_agent.cleanup()

        _check_cancel()

        # ---- 4. 评估 + 结果汇总 ----
        _update_ctx(CaseStatus.EVAL)
        logger.info("[%s] %s", case_id, _SEPARATOR_LIGHT)

        # 收集对话阶段成本
        dialogue_cost = TestCost()
        if hasattr(test_agent, "cost") and hasattr(test_agent, "model"):
            if test_agent.cost["total_tokens"] > 0:
                dialogue_cost.test[test_agent.model] = test_agent.cost
        if hasattr(target_agent, "cost") and hasattr(target_agent, "model"):
            if target_agent.cost["total_tokens"] > 0:
                dialogue_cost.target = {target_agent.model: target_agent.cost}
        # Attach raw cost detail if target agent provides it (e.g. target agents that report a per-call cost breakdown)
        if hasattr(target_agent, "cost_detail") and target_agent.cost_detail:
            dialogue_cost.target_detail = target_agent.cost_detail

        result = await _do_eval(
            test_case,
            list(test_agent.memory_list),
            meta=_meta,
            start_time=start_time,
            session_info=target_agent.get_session_info(),
            extra_cost=dialogue_cost,
            target_memory=list(target_agent.memory_list),
        )

        _update_ctx(CaseStatus.COMPLETED, result=result)

        elapsed = (result.end - start_time).total_seconds()
        logger.info(
            "[%s] 测试完成 — %s (score=%.2f, 耗时=%.1fs)",
            case_id, result.eval.result.upper(), result.eval.score, elapsed,
        )
        logger.info("[%s] %s", case_id, _SEPARATOR_HEAVY)
        return result

    except _CancellationError:
        # 取消中断 — 生成 cancelled 结果（部分对话数据保留在 agent memory 中）
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        logger.info("[%s] 用例被取消 (耗时=%.1fs)", case_id, elapsed)

        result = _make_cancelled_result(case_id, start_time, end_time, **_meta)
        _update_ctx(CaseStatus.CANCELLED, result=result, error="用例被取消")
        return result

    except Exception as e:
        # 任何异常均捕获，不中断批量执行 —— 但结果是 error，不是 fail。
        #
        # `fail` 的定义（见 schema.EvalResult）是「被判过分、分低于 threshold」。
        # 异常意味着压根没判成,两者不是一回事。写成 fail 会把「没跑完」和
        # 「答错了」混进同一栏,而它们的处置完全不同 —— 前者要调预算或修部署,
        # 后者才是被测系统的能力问题。
        #
        # 具体踩到的:跨全量统计的题在超时预算不足时抛 TimeoutError,报告里显示
        # `fail 0.00`,与真答错无法区分;而同一道题预算够时得 0.89。一个随评测方
        # 配置变化的数不能算分数。
        #
        # 不按异常类型分流:要维护一张异常清单是过度设计,而「没判成分」这个
        # 事实与异常类型无关。类型落 eval_detail,超时/连接失败/本框架自身的
        # bug 照样分得开。
        #
        # 注意此处 `_update_ctx` 一直传的就是 `CaseStatus.ERROR` —— 内部状态机
        # 早就认定这是 error,只有 EvalResult 写着 fail。这次是把两者对齐。
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        error_detail = traceback.format_exc()

        logger.error("[%s] 测试执行异常 (耗时=%.1fs):\n%s", case_id, elapsed, error_detail)

        result = TestResult(
            id=case_id,
            **_meta,
            eval=EvalResult(
                result="error",
                score=0.0,
                feedback=f"测试执行异常 [{type(e).__name__}]:\n{error_detail}",
                trace=EvalTrace(
                    eval_detail={
                        "exception_type": type(e).__name__,
                        "exception": str(e),
                        "elapsed_seconds": round(elapsed, 1),
                    }
                ),
            ),
            cost=TestCost(),
            start=start_time,
            end=end_time,
        )
        _update_ctx(CaseStatus.ERROR, result=result, error=error_detail)
        return result


# ============================================================
# 工具函数
# ============================================================


def _make_cancelled_result(
    case_id: str,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    **meta: Any,
) -> TestResult:
    """创建取消用例的 TestResult

    `error` 而非 `fail`:取消的用例没被判分,和超时同理。留成 fail 会让「中途
    取消一个批次」把一串假 0 分算进平均分 —— 一份跑了一半就停的报告,应当报
    实际跑完那些的平均分 + 一个显式的 error 计数,而不是被没跑的部分拉低。
    """
    now = datetime.now()
    return TestResult(
        id=case_id,
        **meta,
        eval=EvalResult(
            result="error",
            score=0.0,
            feedback="用例被取消",
            trace=EvalTrace(eval_detail={"exception_type": "Cancelled"}),
        ),
        cost=TestCost(),
        start=start_time or now,
        end=end_time or now,
    )


def _build_report(results: List[TestResult]) -> TestReport:
    """从 TestResult 列表构建 TestReport — 合并成本统计"""
    total_cost: dict[str, UsageMetadata] = {}
    for r in results:
        for cost_dict in (r.cost.test, r.cost.eval, r.cost.target):
            if not cost_dict:
                continue
            for model_name, usage in cost_dict.items():
                if model_name in total_cost:
                    existing = total_cost[model_name]
                    total_cost[model_name] = UsageMetadata(
                        input_tokens=existing["input_tokens"] + usage["input_tokens"],
                        output_tokens=existing["output_tokens"] + usage["output_tokens"],
                        total_tokens=existing["total_tokens"] + usage["total_tokens"],
                    )
                else:
                    total_cost[model_name] = usage

    return TestReport(cases=results, total_cost=total_cost)


def _truncate(text: str | None, max_len: int = 80) -> str:
    """截断文本用于日志输出"""
    if text is None:
        return "(None)"
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


# ============================================================
# 批量执行会话
# ============================================================


class BatchSession:
    """可观测、可取消的批量执行会话

    提供进度跟踪、实时状态查询和取消功能。
    适用于 server 场景，支持从外部协程查询进度和取消执行。

    用法:
        session = BatchSession(cases, max_concurrency=5)
        report = await session.run()

        # 从另一个协程:
        session.snapshot()    # 查看进度
        session.cancel()      # 取消执行
        session.contexts["hb_xxx"].test_agent.memory_list  # 实时对话数据
    """

    def __init__(
        self,
        cases: List[TestCase],
        max_concurrency: int = 0,
        on_progress: Callable[["BatchSession", str], None] | None = None,
        cancel_event: asyncio.Event | None = None,
        default_target: TargetInfo | None = None,
    ):
        self.id = uuid4().hex[:8]
        self.cases = cases
        self.max_concurrency = max_concurrency
        self.on_progress = on_progress
        self.cancel_event = cancel_event or asyncio.Event()
        self.default_target = default_target
        self.completed = 0

        # 为每个 case 创建上下文（共享全局 cancel_event）
        self._cases_map: Dict[str, TestCase] = {}
        self.contexts: Dict[str, CaseContext] = {}
        for case in cases:
            self._cases_map[case.id] = case
            self.contexts[case.id] = CaseContext(case_id=case.id, cancel_event=self.cancel_event)

    @property
    def total(self) -> int:
        return len(self.cases)

    def cancel(self) -> None:
        """请求取消整个批次 — 已运行的用例跑完当前轮次，pending 的跳过"""
        self.cancel_event.set()

    def snapshot(self) -> dict:
        """返回当前执行状态快照（JSON 可序列化）"""
        cases_info: dict = {}
        for cid, ctx in self.contexts.items():
            case = self._cases_map[cid]
            effective_target = case.target or self.default_target
            info: dict = {
                "status": ctx.status.value,
                "turn": ctx.turn,
                "title": case.title,
                "user_type": case.user.type,
                "target_type": effective_target.type if effective_target else "",
                "eval_type": case.eval.evaluator,
                "tags": case.tags,
            }
            if ctx.test_agent:
                info["memory_count"] = len(ctx.test_agent.memory_list)
            if ctx.result:
                info["score"] = ctx.result.eval.score
                info["eval_result"] = ctx.result.eval.result
                info["cost"] = ctx.result.cost.model_dump(mode="json")
            if ctx.error:
                info["error"] = ctx.error
            cases_info[cid] = info

        return {
            "id": self.id,
            "total": self.total,
            "completed": self.completed,
            "cancelled": self.cancel_event.is_set(),
            "cases": cases_info,
        }

    async def run(self) -> TestReport:
        """执行所有用例，返回评测报告"""
        logger.info(
            "BatchSession[%s] 开始: %d 个用例 (max_concurrency=%s)",
            self.id, self.total, self.max_concurrency or "unlimited",
        )

        if self.max_concurrency > 0:
            semaphore = asyncio.Semaphore(self.max_concurrency)

            async def _run_with_sem(case: TestCase) -> TestResult:
                ctx = self.contexts[case.id]
                # 获取信号量前检查取消
                if self.cancel_event.is_set() and ctx.status == CaseStatus.PENDING:
                    return self._handle_cancelled(case.id)
                async with semaphore:
                    # 获取信号量后再次检查（可能在等待期间被取消）
                    if self.cancel_event.is_set() and ctx.status == CaseStatus.PENDING:
                        return self._handle_cancelled(case.id)
                    return await self._run_case(case)

            results: List[TestResult] = list(
                await asyncio.gather(*[_run_with_sem(c) for c in self.cases])
            )
        else:
            results = list(
                await asyncio.gather(*[self._run_case(c) for c in self.cases])
            )

        report = _build_report(results)

        judged = report.pass_count + report.fail_count
        scored = len(report.cases) - judged
        if judged > 0:
            logger.info(
                "BatchSession[%s] 完成: %d pass / %d fail (pass_rate=%.1f%%, avg_score=%.2f, 总耗时=%.1fs)",
                self.id, report.pass_count, report.fail_count,
                report.pass_rate * 100, report.avg_score, report.total_duration_seconds,
            )
        else:
            logger.info(
                "BatchSession[%s] 完成: %d 条仅评分 (avg_score=%.2f, 总耗时=%.1fs)",
                self.id, scored, report.avg_score, report.total_duration_seconds,
            )

        return report

    async def _run_case(self, case: TestCase) -> TestResult:
        """执行单个用例（含进度回调）"""
        ctx = self.contexts[case.id]

        # 无信号量时也需检查取消
        if self.cancel_event.is_set() and ctx.status == CaseStatus.PENDING:
            return self._handle_cancelled(case.id)

        result = await do_single_test(case, context=ctx, default_target=self.default_target)
        self.completed += 1
        self._fire_progress(case.id)
        return result

    def _handle_cancelled(self, case_id: str) -> TestResult:
        """处理被取消的用例"""
        ctx = self.contexts[case_id]
        case = self._cases_map[case_id]
        effective_target = case.target or self.default_target
        result = _make_cancelled_result(
            case_id,
            title=case.title,
            user_type=case.user.type,
            target_type=effective_target.type if effective_target else "",
            eval_type=case.eval.evaluator,
            tags=case.tags,
            target_config=effective_target.model_dump(exclude={"type"}) if effective_target else None,
            eval_config=case.eval.model_dump(exclude={"evaluator"}),
        )
        ctx.status = CaseStatus.CANCELLED
        ctx.result = result
        ctx.error = "用例被取消"
        self.completed += 1
        self._fire_progress(case_id)
        return result

    def _fire_progress(self, case_id: str) -> None:
        """触发进度回调"""
        if self.on_progress:
            try:
                self.on_progress(self, case_id)
            except Exception:
                logger.warning("on_progress 回调异常", exc_info=True)


# ============================================================
# 批量并发执行（向后兼容接口）
# ============================================================


async def do_batch_test(
    cases: List[TestCase],
    max_concurrency: int = 0,
    *,
    on_progress: Callable[[BatchSession, str], None] | None = None,
    cancel_event: asyncio.Event | None = None,
    default_target: TargetInfo | None = None,
) -> TestReport:
    """
    并发执行多个测试用例，返回完整评测报告

    向后兼容的简便接口。需要完整控制（进度查询、取消）请直接使用 BatchSession。

    Args:
        cases:           测试用例列表
        max_concurrency: 最大并发数，0 或负数表示不限制
        on_progress:     进度回调，每完成一个用例调用 fn(session, case_id)
        cancel_event:    取消事件，set() 后跳过 pending 用例
        default_target:  全局默认 target 配置（可选）

    所有用例通过 asyncio.gather 并发执行。
    do_single_test 内部已有 try/except，单个用例失败不会中断其他用例。
    """
    session = BatchSession(
        cases=cases,
        max_concurrency=max_concurrency,
        on_progress=on_progress,
        cancel_event=cancel_event,
        default_target=default_target,
    )
    return await session.run()


# ============================================================
# Eval-Only 模式 — 跳过对话循环，直接评测已有对话
# ============================================================


async def do_eval_only(
    test_case: TestCase,
    memory_list: list[TestAgentMemory],
    session_info: Any = None,
) -> TestResult:
    """仅评测 — 跳过对话循环，直接对已有对话执行评测

    与 do_single_test 共享 _do_eval 逻辑，入参完全对齐。
    messages → memory_list 的转换由上层调用方负责。

    Args:
        test_case:    测试用例（提供 eval/history/user 配置）
        memory_list:  对话记忆列表（TestAgentMemory）
        session_info: 目标系统会话信息（可选）
    """
    start_time = datetime.now()
    case_id = test_case.id

    logger.info("[%s] %s", case_id, _SEPARATOR_HEAVY)
    logger.info("[%s] Eval-Only: %s (%d 轮, evaluator=%s)",
                case_id, test_case.title or case_id, len(memory_list), test_case.eval.evaluator)

    _meta = {
        "title": test_case.title,
        "user_type": "eval_only",
        "target_type": "eval_only",
        "eval_type": test_case.eval.evaluator,
        "tags": test_case.tags,
        "target_config": None,
        "eval_config": test_case.eval.model_dump(exclude={"evaluator"}),
    }

    try:
        result = await _do_eval(test_case, memory_list, meta=_meta, start_time=start_time, session_info=session_info)

        elapsed = (result.end - start_time).total_seconds()
        logger.info("[%s] Eval-Only 完成 — %s (score=%.2f, 耗时=%.1fs)",
                    case_id, result.eval.result.upper(), result.eval.score, elapsed)
        logger.info("[%s] %s", case_id, _SEPARATOR_HEAVY)
        return result

    except Exception as e:
        # 同 do_single_test 的兜底:没判成分就不是 fail。见那里的完整说明。
        end_time = datetime.now()
        error_detail = traceback.format_exc()
        logger.error("[%s] Eval-Only 异常:\n%s", case_id, error_detail)
        return TestResult(
            id=case_id, **_meta,
            eval=EvalResult(
                result="error",
                score=0.0,
                feedback=f"评测执行异常 [{type(e).__name__}]:\n{error_detail}",
                trace=EvalTrace(
                    eval_detail={"exception_type": type(e).__name__, "exception": str(e)}
                ),
            ),
            cost=TestCost(), start=start_time, end=end_time,
        )


async def do_batch_eval(
    items: List[tuple[TestCase, list[TestAgentMemory]]] | List[tuple[TestCase, list[TestAgentMemory], Any]],
    max_concurrency: int = 0,
    on_progress: Callable[[TestResult], None] | None = None,
) -> TestReport:
    """批量仅评测 — 并发执行多条对话的评测

    Args:
        items:           [(TestCase, memory_list)] 或 [(TestCase, memory_list, session_info)] 列表
                         session_info 用于让 judge 区分「能读到用户数据的 target」与基模（has_user_data 标志）;
                         省略时一律按基模评测。
        max_concurrency: 最大并发数，0 表示不限制
        on_progress:     每条评测完成时的回调（用于实时进度跟踪）
    """
    total = len(items)
    logger.info("do_batch_eval 开始: %d 条对话 (max_concurrency=%s)", total, max_concurrency or "unlimited")

    def _unpack(tup):
        if len(tup) == 3:
            return tup[0], tup[1], tup[2]
        return tup[0], tup[1], None

    async def _run_one(tup) -> TestResult:
        tc, mem, session_info = _unpack(tup)
        result = await do_eval_only(tc, mem, session_info=session_info)
        if on_progress:
            on_progress(result)
        return result

    if max_concurrency > 0:
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _run_with_sem(tup) -> TestResult:
            async with semaphore:
                return await _run_one(tup)

        results = list(await asyncio.gather(*[_run_with_sem(t) for t in items]))
    else:
        results = list(await asyncio.gather(*[_run_one(t) for t in items]))

    report = _build_report(results)
    logger.info(
        "do_batch_eval 完成: %d 条, avg_score=%.2f, 总耗时=%.1fs",
        total, report.avg_score, report.total_duration_seconds,
    )
    return report
