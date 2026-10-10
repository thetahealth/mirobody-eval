"""
RubricEvalAgent — general-purpose rubric-based scoring evaluator.

Registered name: "rubric".

Each case declares per-turn and/or case-level `Criterion` items. Each criterion
is one of two kinds:
  - llm_rubric:   natural-language description, scored 0.0–1.0 by a judge LLM
  - signal_check: local deterministic assertion against a registered signal
                  (e.g. `has_chart == True`, `char_count <= 3000`). Scored 1.0
                  or 0.0.

Per turn, `LatencyBudget` (total seconds, TTFT seconds) is scored via a linear
piecewise function: within budget → 1.0, at 2× budget → 0.0. Missing TTFT data
is skipped (not penalized) so targets that don't surface TTFT degrade gracefully.

Output: always `result="scored"` with continuous `score`; does not emit pass/fail.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages.ai import UsageMetadata
from pydantic import BaseModel, ConfigDict, Field, model_validator

from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent
from evaluator.core.schema import (
    EvalResult,
    EvalTrace,
    SessionInfo,
    TestAgentMemory,
)
from evaluator.utils.llm import accumulate_usage, do_execute
from evaluator.utils.signals import compute_signal, has_signal

# Importing the signals package triggers built-in @register_signal decorators.
from evaluator.plugin import signals as _signals_pkg  # noqa: F401

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "gpt-5.4-mini"


# ============================================================
# Config schema
# ============================================================


class SignalCheck(BaseModel):
    """Local deterministic assertion against a registered signal."""

    model_config = ConfigDict(extra="forbid")

    signal: str = Field(description="Registered signal name (see evaluator.utils.signals)")
    op: Literal["==", "!=", ">=", ">", "<=", "<", "in", "not_in"] = Field(
        "==", description="Comparison operator"
    )
    value: Any = Field(True, description="Expected value (any JSON-serialisable type)")


class Criterion(BaseModel):
    """One scoring item. Exactly one of llm_rubric / signal_check must be set."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Short label used in trace/UI")
    weight: float = Field(1.0, ge=0.0, description="Relative weight within the criteria list")
    llm_rubric: Optional[str] = Field(
        None, description="Natural-language rubric; judge LLM scores 0.0–1.0"
    )
    signal_check: Optional[SignalCheck] = Field(
        None, description="Local signal assertion; True → 1.0, False → 0.0"
    )

    @model_validator(mode="after")
    def _exactly_one_kind(self) -> "Criterion":
        count = int(self.llm_rubric is not None) + int(self.signal_check is not None)
        if count != 1:
            raise ValueError(
                f"Criterion {self.name!r}: must set exactly one of llm_rubric / signal_check"
            )
        return self


class LatencyBudget(BaseModel):
    """Per-turn latency budgets. Missing thresholds are simply skipped."""

    model_config = ConfigDict(extra="forbid")

    total_seconds: Optional[float] = Field(None, gt=0)
    ttft_seconds: Optional[float] = Field(None, gt=0)
    weight: float = Field(1.0, ge=0.0, description="Weight of latency dimension within the turn")


class TurnRubric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn: int = Field(ge=1, description="1-based turn index into memory_list")
    latency: Optional[LatencyBudget] = None
    criteria: List[Criterion] = Field(default_factory=list)


class RubricEvalInfo(BaseModel):
    """Generic rubric evaluator — llm-scored criteria + signal checks + latency budgets.

    Always returns `result="scored"`; consumers should key off `score` not pass/fail.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "evaluator": "rubric",
                    "judge_model": "gpt-5.4-mini",
                    "turns": [
                        {
                            "turn": 1,
                            "latency": {"total_seconds": 8, "ttft_seconds": 3},
                            "criteria": [
                                {
                                    "name": "identify_side",
                                    "llm_rubric": "必须明确指出骨折在左手/腕；只说'你的手'得 0 分",
                                },
                                {
                                    "name": "render_chart",
                                    "signal_check": {"signal": "has_chart", "op": "==", "value": True},
                                },
                            ],
                        }
                    ],
                }
            ]
        },
    )

    evaluator: Literal["rubric"] = Field(description="Evaluator type")
    turns: List[TurnRubric] = Field(default_factory=list, description="Per-turn criteria")
    case_criteria: List[Criterion] = Field(
        default_factory=list, description="Case-level (cross-turn) criteria"
    )
    case_weight: float = Field(
        1.0, ge=0.0, description="Weight of case_criteria score vs mean-of-turns score"
    )
    judge_model: str = Field(DEFAULT_JUDGE_MODEL, description="Judge model for llm_rubric items")

    @model_validator(mode="after")
    def _at_least_one_check(self) -> "RubricEvalInfo":
        if not self.turns and not self.case_criteria:
            raise ValueError("RubricEvalInfo: must declare at least one of turns / case_criteria")
        return self


# ============================================================
# Judge prompts
# ============================================================


_TURN_JUDGE_PROMPT = """你是一个严格的对话质量评分专家。请对以下对话的指定一轮（Turn {turn}）按给定的多条 rubric 分别打分。

## 完整对话上下文
{conversation}

## 本次评分的轮次
Turn {turn}:
User: {user_input}{attachments_note}
AI reply: {ai_reply}

## 该轮的结构事实（已由系统自动检测，直接以此为准，不要怀疑）
{signal_facts}

## Rubrics（每条独立评分，0.0 ~ 1.0）
{rubric_block}

## 评分要求
- 对每条 rubric 输出一个 0.0 ~ 1.0 的小数（允许 0.25 / 0.5 / 0.75 等中间值，不限于 0/1）
- 严格而非仁慈：rubric 明确要求的内容若缺失或违反，就打低分
- 当 rubric 含"= 0"或"≤ 0.3"这种强约束时，必须严格遵守
- 若 rubric 提及的结构事实（如 has_chart）已在上面给出，以事实为准，不要再从文本推断

## 输出格式（严格 JSON，不要多余文字）
```json
{{
  "scores": {{"<rubric_name>": <float 0~1>, ...}},
  "comments": {{"<rubric_name>": "<一句话说明>", ...}}
}}
```"""


_CASE_JUDGE_PROMPT = """你是一个严格的对话质量评分专家。请基于整段对话对给定的 case 级 rubric 分别打分。

## 完整对话
{conversation}

## 结构事实（全局聚合，已由系统自动检测）
{signal_facts}

## Case-level Rubrics（每条独立评分，0.0 ~ 1.0）
{rubric_block}

## 评分要求同上：严格打分；引用事实；允许中间值。

## 输出格式（严格 JSON，不要多余文字）
```json
{{
  "scores": {{"<rubric_name>": <float 0~1>, ...}},
  "comments": {{"<rubric_name>": "<一句话说明>", ...}}
}}
```"""


_SIGNAL_FACT_KEYS = (
    "has_chart",
    "has_table",
    "char_count",
    "md_image_count",
    "tool_calls_count",
    "has_annotation",
)


# ============================================================
# Agent implementation
# ============================================================


class RubricEvalAgent(AbstractEvalAgent, name="rubric", params_model=RubricEvalInfo):
    """General rubric-based scoring evaluator."""

    _display_meta = {
        "icon": "M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z",
        "color": "#10b981",
        "features": ["Scored", "Rubric + Signal", "Latency + TTFT", "Pluggable Signals"],
    }
    _cost_meta = {"est_cost_per_case": 0.01}

    def __init__(self, eval_config: RubricEvalInfo, **kwargs):
        super().__init__(eval_config, **kwargs)
        self.eval_config: RubricEvalInfo = eval_config
        self.model = eval_config.judge_model or DEFAULT_JUDGE_MODEL
        self._cost = UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)

    @property
    def cost(self) -> UsageMetadata:
        return self._cost

    def _accumulate_cost(self, usage: Optional[Dict[str, UsageMetadata]]) -> None:
        self._cost = accumulate_usage(self._cost, usage)

    # ------------------------------------------------------------------
    # AbstractEvalAgent interface
    # ------------------------------------------------------------------

    async def run(
        self,
        memory_list: List[TestAgentMemory],
        session_info: Optional[SessionInfo] = None,
    ) -> EvalResult:
        try:
            turns_data = self._extract_turns(memory_list)
            conversation_str = self._format_conversation(turns_data)

            # Per-turn evaluations in parallel (each turn = 0 or 1 judge call).
            turn_results: List[Dict[str, Any]] = []
            if self.eval_config.turns:
                turn_results = list(
                    await asyncio.gather(
                        *[
                            self._eval_turn(turn_cfg, turns_data, conversation_str)
                            for turn_cfg in self.eval_config.turns
                        ]
                    )
                )

            case_result: Optional[Dict[str, Any]] = None
            if self.eval_config.case_criteria:
                case_result = await self._eval_case(
                    self.eval_config.case_criteria, turns_data, conversation_str
                )

            turn_scores = [r["score"] for r in turn_results if r["score"] is not None]
            mean_turn_score = sum(turn_scores) / len(turn_scores) if turn_scores else None
            case_score = (
                case_result["score"] if case_result and case_result["score"] is not None else None
            )
            final_score = self._combine(mean_turn_score, case_score)
            final_score = max(0.0, min(1.0, final_score))

            feedback = self._make_feedback(final_score, turn_results, case_result)
            detail = {
                "turns": turn_results,
                "case_criteria": case_result,
                "final_score": final_score,
                "judge_model": self.model,
            }
            logger.info(
                "[RubricEval] case=%s score=%.3f (turns=%d, case_crits=%d, judge=%s)",
                self.case_id or "?",
                final_score,
                len(turn_results),
                len(self.eval_config.case_criteria),
                self.model,
            )
            return EvalResult(
                result="scored",
                score=final_score,
                feedback=feedback,
                trace=EvalTrace(eval_detail=detail),
            )
        except Exception as e:
            logger.error("[RubricEval] evaluation failed: %s", e, exc_info=True)
            return EvalResult(result="error", score=0.0, feedback=f"rubric eval error: {e}")

    # ------------------------------------------------------------------
    # Turn-level evaluation
    # ------------------------------------------------------------------

    async def _eval_turn(
        self,
        turn_cfg: TurnRubric,
        turns_data: List[Dict[str, Any]],
        conversation_str: str,
    ) -> Dict[str, Any]:
        turn_idx = turn_cfg.turn - 1
        if turn_idx < 0 or turn_idx >= len(turns_data):
            return {
                "turn": turn_cfg.turn,
                "score": None,
                "latency": None,
                "criteria": [],
                "error": (
                    f"turn {turn_cfg.turn} not present in memory_list "
                    f"(got {len(turns_data)} turns)"
                ),
            }

        turn = turns_data[turn_idx]
        response = turn["ai_reply"]
        reaction = turn["reaction"]

        latency_info = self._score_latency(turn_cfg.latency, turn)

        signal_items = [c for c in turn_cfg.criteria if c.signal_check is not None]
        llm_items = [c for c in turn_cfg.criteria if c.llm_rubric is not None]

        signal_results = [self._score_signal(c, response, reaction) for c in signal_items]

        llm_results: List[Dict[str, Any]] = []
        if llm_items:
            llm_results = await self._score_llm_rubrics(
                llm_items,
                conversation_str=conversation_str,
                turn_num=turn_cfg.turn,
                user_input=turn["user_input"],
                attachments_note=turn["attachments_note"],
                ai_reply=response,
                reaction=reaction,
                scope="turn",
            )

        criteria_results = signal_results + llm_results
        turn_score = self._combine_turn_score(turn_cfg, latency_info, criteria_results)

        return {
            "turn": turn_cfg.turn,
            "score": turn_score,
            "latency": latency_info,
            "criteria": criteria_results,
        }

    @staticmethod
    def _combine_turn_score(
        turn_cfg: TurnRubric,
        latency_info: Optional[Dict[str, Any]],
        criteria_results: List[Dict[str, Any]],
    ) -> Optional[float]:
        weighted: List[tuple[float, float]] = []
        if (
            turn_cfg.latency is not None
            and latency_info is not None
            and latency_info.get("score") is not None
        ):
            weighted.append((turn_cfg.latency.weight, float(latency_info["score"])))
        for r in criteria_results:
            score = r.get("score")
            if score is None:
                continue
            weighted.append((float(r.get("weight", 1.0)), float(score)))
        total_w = sum(w for w, _ in weighted)
        if total_w <= 0:
            return None
        return sum(w * s for w, s in weighted) / total_w

    # ------------------------------------------------------------------
    # Case-level evaluation
    # ------------------------------------------------------------------

    async def _eval_case(
        self,
        criteria: List[Criterion],
        turns_data: List[Dict[str, Any]],
        conversation_str: str,
    ) -> Dict[str, Any]:
        joined_reply = "\n\n".join(t["ai_reply"] for t in turns_data if t["ai_reply"])
        last_reaction = turns_data[-1]["reaction"] if turns_data else None

        signal_items = [c for c in criteria if c.signal_check is not None]
        llm_items = [c for c in criteria if c.llm_rubric is not None]

        signal_results = [self._score_signal(c, joined_reply, last_reaction) for c in signal_items]

        llm_results: List[Dict[str, Any]] = []
        if llm_items:
            llm_results = await self._score_llm_rubrics(
                llm_items,
                conversation_str=conversation_str,
                turn_num=0,
                user_input="",
                attachments_note="",
                ai_reply=joined_reply,
                reaction=last_reaction,
                scope="case",
            )

        all_results = signal_results + llm_results
        weighted = [
            (float(r.get("weight", 1.0)), float(r["score"]))
            for r in all_results
            if r.get("score") is not None
        ]
        total_w = sum(w for w, _ in weighted)
        score = sum(w * s for w, s in weighted) / total_w if total_w > 0 else None
        return {"score": score, "criteria": all_results}

    # ------------------------------------------------------------------
    # LLM rubric scoring (batched per turn / case)
    # ------------------------------------------------------------------

    async def _score_llm_rubrics(
        self,
        items: List[Criterion],
        *,
        conversation_str: str,
        turn_num: int,
        user_input: str,
        attachments_note: str,
        ai_reply: str,
        reaction: Any,
        scope: Literal["turn", "case"],
    ) -> List[Dict[str, Any]]:
        facts = self._collect_known_signals(ai_reply, reaction)
        facts_str = (
            "\n".join(f"  - {k}: {v}" for k, v in facts.items()) if facts else "  (none)"
        )
        rubric_block = "\n".join(f"  - {c.name}: {c.llm_rubric}" for c in items)

        if scope == "turn":
            prompt = _TURN_JUDGE_PROMPT.format(
                turn=turn_num,
                conversation=conversation_str,
                user_input=user_input or "(empty)",
                attachments_note=f"\nAttachments: {attachments_note}" if attachments_note else "",
                ai_reply=ai_reply or "(empty)",
                signal_facts=facts_str,
                rubric_block=rubric_block,
            )
        else:
            prompt = _CASE_JUDGE_PROMPT.format(
                conversation=conversation_str,
                signal_facts=facts_str,
                rubric_block=rubric_block,
            )

        parsed = await self._call_judge(prompt)
        scores = parsed.get("scores", {}) if parsed else {}
        comments = parsed.get("comments", {}) if parsed else {}

        results: List[Dict[str, Any]] = []
        for c in items:
            score = self._coerce_score(scores.get(c.name))
            results.append(
                {
                    "name": c.name,
                    "kind": "llm",
                    "rubric": c.llm_rubric,
                    "weight": c.weight,
                    "score": score if score is not None else 0.0,
                    "score_missing": score is None,
                    "comment": comments.get(c.name, "") if isinstance(comments, dict) else "",
                }
            )
        return results

    async def _call_judge(self, prompt: str) -> Optional[dict]:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await do_execute(
                    model=self.model,
                    system_prompt="You are a strict rubric-based scoring judge for dialogue quality.",
                    input=prompt,
                    max_tokens=1500,
                )
                self._accumulate_cost(result.usage)
                parsed = self._parse_json(result.content)
                if parsed and isinstance(parsed.get("scores"), dict):
                    return parsed
                logger.warning(
                    "[RubricEval] judge output format error (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    (result.content or "")[:300],
                )
            except Exception as e:
                logger.warning(
                    "[RubricEval] judge call failed (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    e,
                )
        logger.error("[RubricEval] judge retries exhausted, all rubrics default to 0")
        return None

    # ------------------------------------------------------------------
    # Signal scoring (local)
    # ------------------------------------------------------------------

    @staticmethod
    def _score_signal(c: Criterion, response: str, reaction: Any) -> Dict[str, Any]:
        chk = c.signal_check
        assert chk is not None  # guaranteed by Criterion validator
        if not has_signal(chk.signal):
            return {
                "name": c.name,
                "kind": "signal",
                "signal": chk.signal,
                "op": chk.op,
                "expected": chk.value,
                "weight": c.weight,
                "score": 0.0,
                "error": f"signal {chk.signal!r} not registered",
            }
        try:
            actual = compute_signal(chk.signal, response, reaction)
        except Exception as e:
            return {
                "name": c.name,
                "kind": "signal",
                "signal": chk.signal,
                "op": chk.op,
                "expected": chk.value,
                "weight": c.weight,
                "score": 0.0,
                "error": f"signal compute error: {e}",
            }
        ok = _apply_op(actual, chk.op, chk.value)
        return {
            "name": c.name,
            "kind": "signal",
            "signal": chk.signal,
            "op": chk.op,
            "expected": chk.value,
            "actual": actual,
            "weight": c.weight,
            "score": 1.0 if ok else 0.0,
        }

    # ------------------------------------------------------------------
    # Latency scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _score_latency(
        budget: Optional[LatencyBudget], turn: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if budget is None:
            return None
        total_s = turn.get("total_seconds")
        ttft_s = turn.get("ttft_seconds")

        parts: List[Dict[str, Any]] = []
        if budget.total_seconds is not None:
            if total_s is None:
                parts.append({"dim": "total", "score": None, "note": "total latency not measured"})
            else:
                parts.append(
                    {
                        "dim": "total",
                        "actual_s": round(float(total_s), 2),
                        "budget_s": budget.total_seconds,
                        "score": _latency_score(float(total_s), budget.total_seconds),
                    }
                )
        if budget.ttft_seconds is not None:
            if ttft_s is None:
                parts.append(
                    {
                        "dim": "ttft",
                        "score": None,
                        "note": "ttft not reported by target; skipped",
                    }
                )
            else:
                parts.append(
                    {
                        "dim": "ttft",
                        "actual_s": round(float(ttft_s), 2),
                        "budget_s": budget.ttft_seconds,
                        "score": _latency_score(float(ttft_s), budget.ttft_seconds),
                    }
                )
        scored = [p["score"] for p in parts if p.get("score") is not None]
        score = sum(scored) / len(scored) if scored else None
        return {"score": score, "parts": parts}

    # ------------------------------------------------------------------
    # Memory extraction & prompt helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_turns(memory_list: List[TestAgentMemory]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for mem in memory_list:
            if mem.test_reaction.is_finished and mem.target_response is None:
                continue
            user_input = mem.test_reaction.action.semantic_content or ""
            reaction = mem.target_response
            ai_reply = reaction.extract_text() if reaction else ""

            total_s: Optional[float] = None
            if mem.test_reaction_time and mem.target_response_time:
                total_s = (mem.target_response_time - mem.test_reaction_time).total_seconds()

            ttft_s = _extract_ttft_seconds(reaction)

            attachments = mem.test_reaction.action.attachments or []
            attachments_note = ", ".join(
                f"{a.file_name} ({a.file_type}, {a.file_size}B)" for a in attachments
            )

            out.append(
                {
                    "user_input": user_input,
                    "ai_reply": ai_reply,
                    "reaction": reaction,
                    "total_seconds": total_s,
                    "ttft_seconds": ttft_s,
                    "attachments_note": attachments_note,
                }
            )
        return out

    @staticmethod
    def _format_conversation(turns_data: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for i, t in enumerate(turns_data, 1):
            lines.append(f"Turn {i} — User: {t['user_input'] or '(empty)'}")
            if t["attachments_note"]:
                lines.append(f"  [Attachments: {t['attachments_note']}]")
            ai = t["ai_reply"] or "(empty)"
            if len(ai) > 2000:
                ai = ai[:2000] + "...(truncated)"
            lines.append(f"Turn {i} — AI:   {ai}")
        return "\n".join(lines)

    @staticmethod
    def _collect_known_signals(response: str, reaction: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k in _SIGNAL_FACT_KEYS:
            if has_signal(k):
                try:
                    out[k] = compute_signal(k, response, reaction)
                except Exception:
                    pass
        return out

    # ------------------------------------------------------------------
    # Aggregation & formatting
    # ------------------------------------------------------------------

    def _combine(
        self, turn_score: Optional[float], case_score: Optional[float]
    ) -> float:
        if turn_score is None and case_score is None:
            return 0.0
        if case_score is None:
            return float(turn_score)
        if turn_score is None:
            return float(case_score)
        cw = self.eval_config.case_weight
        return (turn_score + cw * case_score) / (1.0 + cw)

    @staticmethod
    def _make_feedback(
        final_score: float,
        turn_results: List[Dict[str, Any]],
        case_result: Optional[Dict[str, Any]],
    ) -> str:
        parts = [f"score={final_score:.3f}"]
        if turn_results:
            turn_bits = []
            for r in turn_results:
                s = r.get("score")
                turn_bits.append(f"T{r['turn']}={s:.2f}" if s is not None else f"T{r['turn']}=N/A")
            parts.append("turns[" + ", ".join(turn_bits) + "]")
        if case_result is not None:
            s = case_result.get("score")
            parts.append(f"case={s:.2f}" if s is not None else "case=N/A")
        return " | ".join(parts)

    # ------------------------------------------------------------------
    # JSON parsing for judge output
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_score(v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(1.0, f))

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        if not text:
            return None
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.IGNORECASE
        )
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


# ============================================================
# Module-level helpers (exported for reuse and testing)
# ============================================================


def _apply_op(actual: Any, op: str, expected: Any) -> bool:
    try:
        if op == "==":
            return actual == expected
        if op == "!=":
            return actual != expected
        if op == ">=":
            return actual >= expected
        if op == ">":
            return actual > expected
        if op == "<=":
            return actual <= expected
        if op == "<":
            return actual < expected
        if op == "in":
            return actual in expected
        if op == "not_in":
            return actual not in expected
    except TypeError:
        return False
    return False


def _latency_score(actual_s: float, budget_s: float) -> float:
    """Linear piecewise: within budget → 1.0, at 2× budget → 0.0, monotonic in between."""
    if actual_s <= budget_s:
        return 1.0
    if actual_s >= 2.0 * budget_s:
        return 0.0
    return max(0.0, 1.0 - (actual_s - budget_s) / budget_s)


def _extract_ttft_seconds(reaction: Any) -> Optional[float]:
    """Find a ``{"type": "latency", "content": {"first_token_ms": N}}`` meta chunk.

    Emitted by target agents that surface first-chunk timing. Missing → returns
    None so the evaluator skips the TTFT dimension instead of penalising targets
    that do not report it.
    """
    if reaction is None:
        return None
    items = getattr(reaction, "message_list", None) or []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "latency":
            continue
        content = item.get("content")
        if not isinstance(content, dict):
            continue
        raw = content.get("first_token_ms")
        if raw is None:
            continue
        try:
            return float(raw) / 1000.0
        except (TypeError, ValueError):
            return None
    return None
