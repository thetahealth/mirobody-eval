"""
MedCalcEvalAgent — Medical calculation evaluator based on MedCalc-Bench

Registered name: "medcalc"

How it works:
1. Extract AI response text from TargetAgent's memory_list
2. Call LLM to extract structured answer from natural language response
3. Perform typed matching based on output_type:
   - decimal: answer falls within [lower_limit, upper_limit] range
   - integer: exact match after rounding
   - date: exact match in MM/DD/YYYY format
   - weeks_days: exact match after "X weeks, Y days" normalization
4. Binary scoring: pass(1.0) / fail(0.0)

Reference: https://arxiv.org/abs/2406.12036
"""

import json
import logging
import re
from datetime import datetime
from typing import List, Literal, Optional

from langchain_core.messages.ai import UsageMetadata
from pydantic import BaseModel, ConfigDict, Field

from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent
from evaluator.core.schema import EvalResult, EvalTrace, SessionInfo, TestAgentMemory
from evaluator.utils.llm import do_execute

logger = logging.getLogger(__name__)


# ============================================================
# Evaluation config model
# ============================================================


class MedCalcEvalInfo(BaseModel):
    """MedCalc-Bench medical calculation evaluation — LLM answer extraction + typed numeric matching

    Based on MedCalc-Bench dataset (https://arxiv.org/abs/2406.12036), evaluates LLM medical calculation ability.
    Flow: LLM extracts answer from target response -> typed matching based on output_type:
      - decimal: answer falls within [lower_limit, upper_limit] range
      - integer: exact match after rounding
      - date: exact match in MM/DD/YYYY format
      - weeks_days: exact match after "X weeks, Y days" normalization
    Binary scoring: pass(1.0) / fail(0.0), consistent with original MedCalc-Bench accuracy metric.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "evaluator": "medcalc",
                    "ground_truth": "25.2381",
                    "lower_limit": "23.97619",
                    "upper_limit": "26.50001",
                    "output_type": "decimal",
                },
                {
                    "evaluator": "medcalc",
                    "ground_truth": "18",
                    "output_type": "integer",
                },
            ]
        },
    )

    evaluator: Literal["medcalc"] = Field(description="Evaluator type")
    ground_truth: str = Field(description="Standard answer (string form, preserving precision)")
    lower_limit: str = Field("", description="Tolerance lower limit (decimal type only)")
    upper_limit: str = Field("", description="Tolerance upper limit (decimal type only)")
    output_type: Literal["decimal", "integer", "date", "weeks_days"] = Field(description="Output type")
    explanation: str = Field("", description="Standard answer solution steps (for debugging, not used in scoring)")
    extractor_model: Optional[str] = Field("gpt-4.1", description="Answer extraction model")

DEFAULT_MODEL = "gpt-4.1"

# ============================================================
# Answer extraction prompt
# ============================================================

EXTRACTION_PROMPT = """You are a medical answer extractor. Given an AI assistant's response to a medical calculation question, extract ONLY the final numerical answer.

# AI Response
{response_text}

# Expected Output Type: {output_type}

# Instructions
Extract the final answer from the response above. Return a JSON object:

For decimal: {{"answer": "25.2381"}} (just the number, no units)
For integer: {{"answer": "18"}} (just the integer, no units)
For date: {{"answer": "03/15/2025"}} (MM/DD/YYYY format)
For weeks_days: {{"answer": "17 weeks, 4 days"}} (this exact format)

If the response does not contain a clear answer, return: {{"answer": "NONE"}}

Return ONLY the JSON object, no other text."""


class MedCalcEvalAgent(AbstractEvalAgent, name="medcalc", params_model=MedCalcEvalInfo):
    """Medical calculation evaluator — LLM answer extraction + typed numeric matching"""

    _display_meta = {
        "icon": (
            "M15.75 15.75V18m-7.5-6.75h.008v.008H8.25v-.008zm0 2.25h.008v.008H8.25v-.008zm0 2.25h.008v.008H8.25v"
            "-.008zm0 2.25h.008v.008H8.25v-.008zm2.498-6.75h.007v.008h-.007v-.008zm0 2.25h.007v.008h-.007v-.008zm0"
            " 2.25h.007v.008h-.007v-.008zm0 2.25h.007v.008h-.007v-.008zm2.504-6.75h.008v.008h-.008v-.008zm0 2.25h"
            ".008v.008h-.008v-.008zm0 2.25h.008v.008h-.008v-.008zm0 2.25h.008v.008h-.008v-.008zm2.498-6.75h.008v"
            ".008h-.008v-.008zm0 2.25h.008v.008h-.008v-.008zM8.25 6h7.5v2.25h-7.5V6zM12 2.25c-1.892 0-3.758.11"
            "-5.593.322C5.307 2.7 4.5 3.65 4.5 4.757V19.5a2.25 2.25 0 002.25 2.25h10.5a2.25 2.25 0 002.25-2.25V4"
            ".757c0-1.108-.806-2.057-1.907-2.185A48.507 48.507 0 0012 2.25z"
        ),
        "color": "#8b5cf6",
        "features": ["MedCalc-Bench", "LLM Answer Extraction", "Typed Matching"],
    }
    _cost_meta = {
        "est_cost_per_case": 0.003,  # Single gpt-4.1 answer extraction call
    }

    def __init__(self, eval_config: MedCalcEvalInfo, **kwargs):
        super().__init__(eval_config, **kwargs)
        self.config: MedCalcEvalInfo = eval_config
        self.model = eval_config.extractor_model or DEFAULT_MODEL
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
            # 1. Extract AI response text
            response_text = self._extract_response_text(memory_list)
            if not response_text:
                return EvalResult(result="fail", score=0.0, feedback="Target AI has no response, cannot evaluate")

            logger.info(
                "[MedCalcEval] Starting evaluation — output_type=%s, ground_truth=%s, model=%s",
                self.config.output_type,
                self.config.ground_truth,
                self.model,
            )

            # 2. LLM answer extraction
            extracted = await self._extract_answer(response_text)
            if extracted is None or extracted.upper() == "NONE":
                return EvalResult(
                    result="fail",
                    score=0.0,
                    feedback=f"Cannot extract answer from AI response (output_type={self.config.output_type})",
                    trace=EvalTrace(eval_detail={"response_preview": response_text[:500], "extracted": None}),
                )

            # 3. Typed matching
            is_correct, detail = self._match_answer(extracted)

            score = 1.0 if is_correct else 0.0
            result = "pass" if is_correct else "fail"

            feedback = (
                f"MedCalc evaluation ({self.config.output_type}): {result} — "
                f"extracted={extracted}, ground_truth={self.config.ground_truth}, {detail}"
            )

            logger.info("[MedCalcEval] %s — extracted=%s, ground_truth=%s", result, extracted, self.config.ground_truth)

            return EvalResult(
                result=result,
                score=score,
                feedback=feedback,
                trace=EvalTrace(
                    eval_detail={
                        "extracted_answer": extracted,
                        "ground_truth": self.config.ground_truth,
                        "output_type": self.config.output_type,
                        "is_correct": is_correct,
                        "match_detail": detail,
                    }
                ),
            )

        except Exception as e:
            logger.error("[MedCalcEval] Evaluation error: %s", e, exc_info=True)
            return EvalResult(result="fail", score=0.0, feedback=f"Evaluation error: {e}")

    # ----------------------------------------------------------
    # Response extraction
    # ----------------------------------------------------------

    @staticmethod
    def _extract_response_text(memory_list: list[TestAgentMemory]) -> str:
        """Extract all AI response text from memory_list"""
        parts: List[str] = []
        for mem in memory_list:
            if mem.target_response is not None:
                text = mem.target_response.extract_text()
                if text:
                    parts.append(text)
        return "\n\n".join(parts)

    # ----------------------------------------------------------
    # LLM answer extraction
    # ----------------------------------------------------------

    async def _extract_answer(self, response_text: str) -> Optional[str]:
        """Call LLM to extract structured answer from natural language response"""
        prompt = EXTRACTION_PROMPT.format(
            response_text=response_text[:3000],  # Truncate to avoid excessive length
            output_type=self.config.output_type,
        )

        max_retries = 2
        for attempt in range(max_retries):
            try:
                result = await do_execute(
                    model=self.model,
                    system_prompt="You are a precise medical answer extractor. Return only JSON.",
                    input=prompt,
                    max_tokens=200,
                )
                self._accumulate_cost(result.usage)

                parsed = self._parse_json_response(result.content)
                if parsed and "answer" in parsed:
                    return str(parsed["answer"]).strip()

                logger.warning(
                    "[MedCalcEval] Extraction output format error (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    result.content[:200],
                )
            except Exception as e:
                logger.warning("[MedCalcEval] Extraction call failed (attempt %d/%d): %s", attempt + 1, max_retries, e)

        return None

    @staticmethod
    def _parse_json_response(text: str) -> Optional[dict]:
        """Extract JSON from LLM response"""
        cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip())
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None

    # ----------------------------------------------------------
    # Typed matching
    # ----------------------------------------------------------

    def _match_answer(self, extracted: str) -> tuple[bool, str]:
        """Route to matching method based on output_type"""
        output_type = self.config.output_type
        if output_type == "decimal":
            return self._match_decimal(extracted)
        elif output_type == "integer":
            return self._match_integer(extracted)
        elif output_type == "date":
            return self._match_date(extracted)
        elif output_type == "weeks_days":
            return self._match_weeks_days(extracted)
        else:
            return False, f"Unknown output_type: {output_type}"

    def _match_decimal(self, extracted: str) -> tuple[bool, str]:
        """decimal matching — answer falls within [lower_limit, upper_limit] range"""
        try:
            answer = float(extracted.replace(",", ""))
        except ValueError:
            return False, f"Cannot parse as number: {extracted}"

        try:
            lower = float(self.config.lower_limit)
            upper = float(self.config.upper_limit)
        except ValueError:
            return False, f"Tolerance range config error: lower={self.config.lower_limit}, upper={self.config.upper_limit}"

        is_correct = lower <= answer <= upper
        detail = f"Range match: {answer} {'in' if is_correct else 'not in'} [{lower}, {upper}]"
        return is_correct, detail

    def _match_integer(self, extracted: str) -> tuple[bool, str]:
        """integer matching — exact match after rounding"""
        try:
            answer = round(float(extracted.replace(",", "")))
        except ValueError:
            return False, f"Cannot parse as number: {extracted}"

        try:
            ground_truth = int(float(self.config.ground_truth))
        except ValueError:
            return False, f"Standard answer is not integer: {self.config.ground_truth}"

        is_correct = answer == ground_truth
        detail = f"Integer match: {answer} {'==' if is_correct else '!='} {ground_truth}"
        return is_correct, detail

    def _match_date(self, extracted: str) -> tuple[bool, str]:
        """date matching — exact comparison after MM/DD/YYYY normalization"""
        try:
            answer_date = datetime.strptime(extracted.strip(), "%m/%d/%Y")
            answer_normalized = answer_date.strftime("%-m/%-d/%Y")
        except ValueError:
            return False, f"Cannot parse date: {extracted}"

        try:
            gt_date = datetime.strptime(self.config.ground_truth.strip(), "%m/%d/%Y")
            gt_normalized = gt_date.strftime("%-m/%-d/%Y")
        except ValueError:
            return False, f"Standard answer date format error: {self.config.ground_truth}"

        is_correct = answer_normalized == gt_normalized
        detail = f"Date match: {answer_normalized} {'==' if is_correct else '!='} {gt_normalized}"
        return is_correct, detail

    def _match_weeks_days(self, extracted: str) -> tuple[bool, str]:
        """weeks_days matching — exact comparison after normalization"""
        answer_norm = self._normalize_weeks_days(extracted)
        gt_norm = self._normalize_weeks_days(self.config.ground_truth)

        if answer_norm is None:
            return False, f"Cannot parse weeks/days format: {extracted}"
        if gt_norm is None:
            return False, f"Standard answer weeks/days format error: {self.config.ground_truth}"

        is_correct = answer_norm == gt_norm
        detail = f"Weeks/days match: {answer_norm} {'==' if is_correct else '!='} {gt_norm}"
        return is_correct, detail

    @staticmethod
    def _normalize_weeks_days(text: str) -> Optional[tuple[int, int]]:
        """Normalize "X weeks, Y days" or MedCalc-Bench's "('X weeks', 'Y days')" to (weeks, days) tuple"""
        text = text.lower().strip()
        match = re.search(r"(\d+)\s*weeks?\W*(\d+)\s*days?", text)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None
