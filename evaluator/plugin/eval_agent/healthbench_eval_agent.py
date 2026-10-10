"""
HealthBenchEvalAgent — Rubric evaluator based on the original HealthBench GRADER_TEMPLATE

Registered name: "healthbench"

How it works:
1. Extract the full conversation from TestAgent / TargetAgent memory_list + history
2. For each rubric criterion, call the grader LLM to determine if met (criteria_met: bool)
3. Calculate score: achieved_points / total_positive_points (consistent with original calculate_score)
4. Clip to [0, 1], result is scored (scoring only, no pass/fail judgment)

Grader prompt fully reuses the original HealthBench version (English) to ensure comparable results.
Reference: https://github.com/openai/simple-evals/blob/main/healthbench_eval.py
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import BaseMessage
from langchain_core.messages.ai import UsageMetadata
from pydantic import BaseModel, ConfigDict, Field

from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent
from evaluator.core.schema import (
    EvalResult,
    EvalTrace,
    SessionInfo,
    TestAgentMemory,
)
from evaluator.utils.llm import do_execute

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4.1"


# ============================================================
# Evaluation config model
# ============================================================


class HealthBenchRubric(BaseModel):
    """HealthBench rubric — a single scoring criterion

    Corresponds to each rubric criterion in the HealthBench dataset.
    Points can be negative (indicating undesirable behavior), range -10 to +10.
    """

    model_config = ConfigDict(extra="forbid")

    criterion: str = Field(description="Criterion text")
    points: float = Field(description="Point value (can be negative, -10 to +10)")
    tags: List[str] = Field(default_factory=list, description="Tags (e.g. axis:accuracy, level:example)")


class HealthBenchEvalInfo(BaseModel):
    """HealthBench rubric evaluation — per-criterion judgment, LLM-as-Judge

    Reuses the original HealthBench GRADER_TEMPLATE; grader model judges each criterion.
    Score calculation: achieved_points / total_positive_points, fully consistent with original calculate_score.
    Scoring only (result="scored"), no pass/fail judgment.

    Designed for running HealthBench datasets converted to mirobody-eval format.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "evaluator": "healthbench",
                    "rubrics": [
                        {"criterion": "Advises the patient to see a doctor", "points": 10, "tags": ["axis:accuracy"]},
                        {"criterion": "Is overly verbose", "points": -5, "tags": ["axis:communication_quality"]},
                    ],
                },
            ]
        },
    )

    evaluator: Literal["healthbench"] = Field(description="Evaluator type")
    model: Optional[str] = Field("gpt-4.1", description="Grader model (default gpt-4.1, consistent with original HealthBench)")
    rubrics: List[HealthBenchRubric] = Field(description="List of scoring rubrics")


# ============================================================
# Original HealthBench GRADER_TEMPLATE
# Source: https://github.com/openai/simple-evals/blob/main/healthbench_eval.py
# ============================================================

GRADER_TEMPLATE = """Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response."""


# ============================================================
# HealthBenchEvalAgent
# ============================================================


class HealthBenchEvalAgent(AbstractEvalAgent, name="healthbench", params_model=HealthBenchEvalInfo):
    """HealthBench rubric-based evaluator"""

    _display_meta = {
        "icon": (
            "M21 8.25c0-2.485-2.099-4.5-4.688-4.5-1.935 0-3.597 1.126-4.312 2.733-.715-1.607-2.377-2.733-4.313"
            "-2.733C5.1 3.75 3 5.765 3 8.25c0 7.22 9 12 9 12s9-4.78 9-12z"
        ),
        "color": "#ef4444",
        "features": ["HealthBench", "Rubric Scoring", "Concurrent Evaluation"],
    }
    _cost_meta = {
        "est_cost_per_case": 0.035,  # ~11 rubrics × gpt-4.1 grading, USD/case
    }

    def __init__(self, eval_config: HealthBenchEvalInfo, model: str | None = None, **kwargs):
        super().__init__(eval_config, **kwargs)
        self.eval_config: HealthBenchEvalInfo = eval_config
        self.model = model or eval_config.model or DEFAULT_MODEL
        self._cost = UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)

    @property
    def cost(self) -> UsageMetadata:
        return self._cost

    def _accumulate_cost(self, usage: dict[str, UsageMetadata] | None) -> None:
        from evaluator.utils.llm import accumulate_usage

        self._cost = accumulate_usage(self._cost, usage)

    # ----------------------------------------------------------
    # Framework interface
    # ----------------------------------------------------------

    async def run(
        self,
        memory_list: list[TestAgentMemory],
        session_info: SessionInfo | None = None,
    ) -> EvalResult:
        try:
            # Extract full conversation (including history)
            conversation = self._build_conversation(memory_list, self.history)
            if not conversation:
                return EvalResult(result="fail", score=0.0, feedback="No conversation records, cannot evaluate")

            rubrics = self.eval_config.rubrics
            if not rubrics:
                return EvalResult(result="fail", score=0.0, feedback="No rubrics configured, cannot evaluate")

            logger.info(
                "[HealthBenchEval] Starting evaluation — %d rubrics, model=%s",
                len(rubrics),
                self.model,
            )

            # Concurrently evaluate all rubrics
            conversation_str = "\n\n".join(f"{m['role']}: {m['content']}" for m in conversation)
            grading_results = await asyncio.gather(*[
                self._grade_rubric_item(conversation_str, rubric) for rubric in rubrics
            ])

            # Calculate score (consistent with original calculate_score)
            score = self._calculate_score(rubrics, grading_results)
            if score is None:
                return EvalResult(result="fail", score=0.0, feedback="No positive-point rubric items, cannot calculate score")

            clipped_score = max(0.0, min(1.0, score))

            # Build details
            rubric_details = []
            for rubric, result in zip(rubrics, grading_results):
                rubric_details.append({
                    "criterion": rubric.criterion,
                    "points": rubric.points,
                    "criteria_met": result.get("criteria_met", False),
                    "explanation": result.get("explanation", ""),
                    "tags": rubric.tags,
                })

            met_count = sum(1 for r in grading_results if r.get("criteria_met"))
            feedback = f"HealthBench rubric evaluation: {met_count}/{len(rubrics)} criteria met, score={clipped_score:.3f}"

            logger.info(
                "[HealthBenchEval] Evaluation complete — score=%.3f, scored (%d/%d met)",
                clipped_score,
                met_count,
                len(rubrics),
            )

            return EvalResult(
                result="scored",
                score=clipped_score,
                feedback=feedback,
                trace=EvalTrace(eval_detail={"rubric_results": rubric_details, "raw_score": score}),
            )

        except Exception as e:
            logger.error("[HealthBenchEval] Evaluation error: %s", e, exc_info=True)
            return EvalResult(result="fail", score=0.0, feedback=f"Evaluation error: {e}")

    # ----------------------------------------------------------
    # Core methods
    # ----------------------------------------------------------

    async def _grade_rubric_item(self, conversation_str: str, rubric: HealthBenchRubric) -> dict:
        """Call grader LLM to evaluate a single rubric criterion"""
        rubric_str = f"[{rubric.points}] {rubric.criterion}"
        grader_prompt = GRADER_TEMPLATE.replace("<<conversation>>", conversation_str).replace(
            "<<rubric_item>>", rubric_str
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await do_execute(
                    model=self.model,
                    system_prompt="You are a rubric grader for healthcare AI evaluation.",
                    input=grader_prompt,
                    max_tokens=500,
                )
                self._accumulate_cost(result.usage)

                parsed = self._parse_json_response(result.content)
                if parsed is not None and "criteria_met" in parsed:
                    if isinstance(parsed["criteria_met"], bool):
                        return parsed

                logger.warning(
                    "[HealthBenchEval] Grader output format error (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    result.content[:200],
                )
            except Exception as e:
                logger.warning(
                    "[HealthBenchEval] Grader call failed (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    e,
                )

        # Retries exhausted, default to criteria_met=False
        logger.error("[HealthBenchEval] Grader retries exhausted, defaulting to criteria_met=False: %s", rubric.criterion[:100])
        return {"explanation": "Grading failed after retries", "criteria_met": False}

    @staticmethod
    def _calculate_score(rubrics: List[HealthBenchRubric], grading_results: List[dict]) -> Optional[float]:
        """Calculate score — consistent with original HealthBench calculate_score"""
        total_possible = sum(r.points for r in rubrics if r.points > 0)
        if total_possible == 0:
            return None
        achieved = sum(r.points for r, g in zip(rubrics, grading_results) if g.get("criteria_met"))
        return achieved / total_possible

    @staticmethod
    def _parse_json_response(text: str) -> Optional[dict]:
        """Extract JSON from grader response (compatible with markdown wrapping)"""
        cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip())
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None

    # ----------------------------------------------------------
    # Conversation extraction
    # ----------------------------------------------------------

    @staticmethod
    def _build_conversation(
        memory_list: List[TestAgentMemory],
        history: List[BaseMessage] | None = None,
    ) -> List[Dict[str, str]]:
        """Extract full conversation from history + memory_list"""
        conversation: List[Dict[str, str]] = []

        # 1. Historical conversation (TestCase.history)
        if history:
            for msg in history:
                role = "user" if msg.type == "human" else "assistant"
                conversation.append({"role": role, "content": msg.content})

        # 2. Current conversation (memory_list)
        for mem in memory_list:
            if mem.test_reaction.is_finished and mem.target_response is None:
                continue

            user_text = mem.test_reaction.action.semantic_content
            if user_text:
                conversation.append({"role": "user", "content": user_text})

            if mem.target_response:
                target_text = mem.target_response.extract_text()
                if target_text:
                    conversation.append({"role": "assistant", "content": target_text})

        return conversation
