"""
RecordRetrievalEvalAgent — Per-turn checkpoint evaluator for RECORD + RETRIEVAL dialogues

Registered name: "record_retrieval"

Evaluates multi-turn health management dialogues via per-turn checkpoints:
  - record_ack:     RECORD turn — brevity, no unsolicited advice, echo key data, response time
  - retrieval_data: RETRIEVAL turn — response contains expected data values
  - chart_rendered: turn produced a chart — markdown image in reply, or an
                    annotation / tool_call payload indicating chart rendering
  - skip:           No evaluation for this turn

Zero LLM calls. Pure rule-based. Deterministic.

Config example:
{
    "evaluator": "record_retrieval",
    "checkpoints": [
        {"turn": 1, "type": "record_ack", "echo_keywords": ["138/92"], "max_chars": 80, "max_seconds": 5.0},
        {"turn": 2, "type": "record_ack", "echo_keywords": ["牛肉面", "豆浆"]},
        {"turn": 3, "type": "retrieval_data", "must_contain": ["138/92"]},
        {"turn": 4, "type": "retrieval_data", "must_contain": ["牛肉面", "豆浆"]}
    ],
    "threshold": 0.8
}
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent
from evaluator.core.schema import (
    EvalResult,
    EvalTrace,
    SessionInfo,
    TestAgentMemory,
)

logger = logging.getLogger(__name__)

_ADVICE_KEYWORDS_ZH = ["建议", "注意", "提醒", "应该", "需要注意", "小贴士", "温馨提示"]
_ADVICE_KEYWORDS_EN = ["recommend", "suggest", "should", "advice", "tips", "reminder"]
_ADVICE_KEYWORDS = _ADVICE_KEYWORDS_ZH + _ADVICE_KEYWORDS_EN

# Chart detection heuristics — the backend can surface a rendered chart via
# any of: markdown image in the reply, an annotation chunk whose type hints
# at "chart" / "sparkline", or a tool call named like a charting tool.
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
_CHART_HINT_KEYS = ("chart", "sparkline", "plot", "graph")


class Checkpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    turn: int = Field(ge=1, description="Which dialogue turn to check (1-indexed)")
    type: Literal["record_ack", "retrieval_data", "chart_rendered", "skip"] = Field(description="Checkpoint type")
    echo_keywords: List[str] = Field(default_factory=list, description="Keywords the AI should echo back (record_ack)")
    max_chars: int = Field(default=80, description="Max response length in characters (record_ack)")
    max_seconds: float = Field(default=5.0, description="Max response time in seconds (record_ack)")
    must_contain: List[str] = Field(default_factory=list, description="Strings that must appear in AI response (retrieval_data / chart_rendered)")


class RecordRetrievalEvalInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evaluator: Literal["record_retrieval"] = Field(default="record_retrieval", description="Evaluator type")
    checkpoints: List[Checkpoint] = Field(description="Per-turn evaluation checkpoints")
    threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Pass threshold (fraction of checkpoints passed)")


class RecordRetrievalEvalAgent(
    AbstractEvalAgent,
    name="record_retrieval",
    params_model=RecordRetrievalEvalInfo,
):
    _display_meta = {
        "icon": "M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 002.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 00-1.123-.08",
        "color": "#10b981",
        "features": ["Zero LLM", "Per-turn checkpoints", "RECORD + RETRIEVAL"],
    }
    _cost_meta = {"est_cost_per_case": 0}

    def __init__(self, eval_config: RecordRetrievalEvalInfo, **kwargs):
        super().__init__(eval_config, **kwargs)

    async def run(self, memory_list: List[TestAgentMemory], session_info: SessionInfo | None = None) -> EvalResult:
        cfg: RecordRetrievalEvalInfo = self.eval_config
        checkpoints = cfg.checkpoints
        if not checkpoints:
            return EvalResult(result="fail", score=0.0, feedback="No checkpoints defined")
        turn_responses, turn_latencies = self._extract_turn_data(memory_list)
        turn_reactions = self._extract_turn_reactions(memory_list)
        results: List[Dict[str, Any]] = []
        for cp in checkpoints:
            if cp.type == "skip":
                continue
            turn_idx = cp.turn - 1
            ai_response = turn_responses[turn_idx] if turn_idx < len(turn_responses) else ""
            if not ai_response and cp.type != "chart_rendered":
                results.append({"turn": cp.turn, "type": cp.type, "passed": False, "score": 0.0, "reason": f"Turn {cp.turn}: no AI response found"})
                continue
            latency = turn_latencies[turn_idx] if turn_idx < len(turn_latencies) else None
            reaction = turn_reactions[turn_idx] if turn_idx < len(turn_reactions) else None
            if cp.type == "record_ack":
                result = self._check_record_ack(cp, ai_response, latency)
            elif cp.type == "retrieval_data":
                result = self._check_retrieval_data(cp, ai_response)
            elif cp.type == "chart_rendered":
                result = self._check_chart_rendered(cp, ai_response, reaction)
            else:
                result = {"turn": cp.turn, "type": cp.type, "passed": False, "score": 0.0, "reason": f"Unknown type: {cp.type}"}
            results.append(result)
        if not results:
            return EvalResult(result="pass", score=1.0, feedback="No checkpoints to evaluate (all skipped)")
        passed_count = sum(1 for r in results if r["passed"])
        total = len(results)
        score = passed_count / total
        passed = score >= cfg.threshold
        failed = [r for r in results if not r["passed"]]
        if failed:
            feedback = f"{passed_count}/{total} checkpoints passed. Failed: " + "; ".join(
                f"Turn {r['turn']}({r['type']}): {r['reason']}" for r in failed
            )
        else:
            feedback = f"All {total} checkpoints passed."
        return EvalResult(result="pass" if passed else "fail", score=score, feedback=feedback, trace=EvalTrace(eval_detail={"checkpoints": results}))

    @staticmethod
    def _extract_turn_data(memory_list: List[TestAgentMemory]) -> tuple[List[str], List[Optional[float]]]:
        """Extract per-turn AI response text and latency (seconds).

        Latency 优先使用 annotation timestamp（某些 target 的 annotation 早于完整 reply，
        更准确地反映用户感知到的确认时间）。无 annotation 时回退到完整响应时间。
        """
        responses = []
        latencies: List[Optional[float]] = []
        for mem in memory_list:
            if mem.test_reaction.is_finished and mem.target_response is None:
                continue
            text = mem.target_response.extract_text() if mem.target_response else ""
            responses.append(text)
            # Default: full response latency
            latency = None
            if mem.test_reaction_time and mem.target_response_time:
                latency = (mem.target_response_time - mem.test_reaction_time).total_seconds()
            # Override: use first annotation timestamp if present（比完整 reply 更早到达，
            # 更准确反映用户感知到的响应时间）
            if mem.target_response and mem.target_response.message_list and mem.test_reaction_time:
                for item in mem.target_response.message_list:
                    if isinstance(item, dict) and item.get("type") == "annotation" and "timestamp" in item:
                        try:
                            latency = (datetime.fromisoformat(item["timestamp"]) - mem.test_reaction_time).total_seconds()
                        except (ValueError, TypeError):
                            pass
                        break
            latencies.append(latency)
        return responses, latencies

    @staticmethod
    def _check_record_ack(cp: Checkpoint, response: str, latency: Optional[float] = None) -> Dict[str, Any]:
        issues = []
        if len(response) > cp.max_chars:
            issues.append(f"too long ({len(response)} > {cp.max_chars} chars)")
        response_lower = response.lower()
        found_advice = [kw for kw in _ADVICE_KEYWORDS if kw in response_lower]
        if found_advice:
            issues.append(f"unsolicited advice: {found_advice}")
        missing = [kw for kw in cp.echo_keywords if kw not in response]
        if missing:
            issues.append(f"missing echo: {missing}")
        if latency is not None and latency > cp.max_seconds:
            issues.append(f"too slow ({latency:.1f}s > {cp.max_seconds}s)")
        passed = len(issues) == 0
        result: Dict[str, Any] = {"turn": cp.turn, "type": "record_ack", "passed": passed, "score": 1.0 if passed else 0.0, "reason": "OK" if passed else "; ".join(issues), "response_length": len(response), "response_preview": response[:100]}
        if latency is not None:
            result["latency_seconds"] = round(latency, 2)
        return result

    @staticmethod
    def _check_retrieval_data(cp: Checkpoint, response: str) -> Dict[str, Any]:
        response_lower = response.lower()
        missing = [item for item in cp.must_contain if item.lower() not in response_lower]
        passed = len(missing) == 0
        return {"turn": cp.turn, "type": "retrieval_data", "passed": passed, "score": 1.0 if passed else 0.0, "reason": "OK" if passed else f"missing data: {missing}", "found": [item for item in cp.must_contain if item.lower() in response_lower], "missing": missing, "response_preview": response[:200]}

    @staticmethod
    def _extract_turn_reactions(memory_list: List[TestAgentMemory]):
        """Return the raw TargetAgentReaction per recorded turn (or None)."""
        out = []
        for mem in memory_list:
            if mem.test_reaction.is_finished and mem.target_response is None:
                continue
            out.append(mem.target_response)
        return out

    @staticmethod
    def _check_chart_rendered(cp: Checkpoint, response: str, reaction) -> Dict[str, Any]:
        """Pass when the turn produced a chart — detected via any of:
        * markdown image in reply text
        * annotation chunk whose content mentions chart-ish keys
        * tool_calls name contains a chart keyword
        Optionally verifies that ``must_contain`` terms appear in the reply
        (e.g. when the prompt was about sleep, check '睡眠' shows up).
        """
        signals: list[str] = []

        md = _MARKDOWN_IMAGE_RE.findall(response or "")
        if md:
            signals.append(f"markdown_image[{len(md)}]: {md[0][:60]}")

        if reaction is not None:
            for item in getattr(reaction, "message_list", None) or []:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                text = content if isinstance(content, str) else str(content or "")
                low = text.lower()
                if any(k in low for k in _CHART_HINT_KEYS):
                    signals.append(f"{item.get('type','chunk')}:chart-hint")
                    break
            for tc in getattr(reaction, "tool_calls", None) or []:
                name = (tc.get("name") or "") if isinstance(tc, dict) else ""
                if any(k in name.lower() for k in _CHART_HINT_KEYS):
                    signals.append(f"tool_call:{name}")
                    break

        response_lower = (response or "").lower()
        missing = [item for item in cp.must_contain if item.lower() not in response_lower]

        passed = bool(signals) and not missing
        if not signals:
            reason = "no chart signals found (no markdown image, annotation, or tool call)"
        elif missing:
            reason = f"chart present but reply missing terms: {missing}"
        else:
            reason = "OK: " + "; ".join(signals)
        return {
            "turn": cp.turn,
            "type": "chart_rendered",
            "passed": passed,
            "score": 1.0 if passed else 0.0,
            "reason": reason,
            "signals": signals,
            "missing": missing,
            "response_preview": (response or "")[:200],
        }
