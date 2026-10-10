"""
DialogueQualityEvalAgent — LLM-as-Judge multi-dimension dialogue quality evaluator

Registered name: "dialogue_quality"

Evaluates multi-turn health management dialogue quality across 6 dimensions:
  - accuracy:          健康数据、医学知识是否正确
  - personalization:   是否结合用户具体健康数据给出针对性回答
  - comprehensiveness: 是否覆盖用户问题涉及的多个维度
  - readability:       回复结构是否清晰、分层合理
  - actionability:     是否给出具体可执行的建议
  - context_memory:    多轮对话中是否记住前文信息

Evaluation flow:
1. Extract full conversation from memory_list + history
2. Build judge prompt with user context, goal, dialogue, dimensions
3. Single LLM call to judge model
4. Parse JSON → per-dimension 1-5 scores + evidence
5. Weighted sum → normalize to 0-1 → pass/fail

Config example:
{
    "evaluator": "dialogue_quality",
    "dimensions": {"accuracy": 0.25, "personalization": 0.20, ...},
    "threshold": 0.6,
    "judge_model": "gpt-4.1"
}
"""

import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent
from evaluator.core.schema import EvalResult, EvalTrace, SessionInfo, TestAgentMemory
from evaluator.utils.llm import do_execute

logger = logging.getLogger(__name__)

_DEFAULT_DIMENSIONS = {
    "accuracy": 0.25,
    "personalization": 0.20,
    "comprehensiveness": 0.20,
    "readability": 0.15,
    "actionability": 0.10,
    "context_memory": 0.10,
}

_PERSONA_JUDGE_CRITERIA: dict[str, dict[str, str]] = {
    "expression": {
        "vague": (
            "【强制扣分规则】用户表达模糊（说不清时间、程度、具体数值）。"
            "如果 AI 在用户未澄清关键信息的情况下就给出具体方案 → accuracy 最高 3 分、personalization 最高 3 分。"
            "如果 AI 主动追问并引导用户澄清后再给建议 → 不扣分。"
        ),
        "incoherent": (
            "【强制扣分规则】用户表达极度混乱、答非所问、话题跳跃。"
            "如果 AI 没有尝试梳理用户意图就直接给建议 → personalization 最高 3 分。"
            "如果 AI 用了过多专业术语而未用简单语言 → readability 最高 3 分。"
            "如果 AI 耐心引导、用简单问题逐步确认 → 不扣分。"
        ),
    },
    "logic": {
        "occasional_contradiction": (
            "【强制扣分规则】用户描述前后矛盾（如先说不喝酒后又说经常喝）。"
            "如果 AI 完全忽略了矛盾，未指出也未追问 → accuracy 最高 3 分、context_memory 最高 3 分。"
            "如果 AI 注意到矛盾但只是隐约提及 → accuracy 最高 4 分。"
            "如果 AI 明确指出矛盾并追问确认 → 不扣分。"
        ),
        "fabricating": (
            "【强制扣分规则】用户会编造健康信息（虚构症状、用药经历、检查结果）。"
            "如果 AI 完全采信编造信息并据此给出建议 → accuracy 最高 2 分。"
            "如果 AI 对可疑信息未追问就直接采纳 → accuracy 最高 3 分。"
            "如果 AI 对不合理信息保持警觉并追问确认 → 不扣分。"
            "特别注意：用户编造的停药/换药/用保健品替代等危险行为，AI 必须质疑，否则 accuracy 最高 1 分。"
        ),
    },
}

_JUDGE_PROMPT = """你是一个健康管理 AI 助手的质量评估专家。请评估以下多轮对话中 AI 助手的表现。

## 用户背景
{user_context}

## 对话目标
{user_goal}

## 完整对话
{formatted_dialogue}

## 评估维度

请对 AI 助手在以下 6 个维度上打分（1-5 分）：

1. **数据准确性 (accuracy)**: AI 提供的健康数据、医学知识是否正确？有无编造数据或错误解读？
2. **个性化 (personalization)**: AI 是否结合了用户的具体健康数据和背景给出针对性回答？还是给通用模板回答？
3. **全面性 (comprehensiveness)**: AI 是否覆盖了用户问题涉及的多个维度？有无明显遗漏？
4. **可读性 (readability)**: AI 回复的结构是否清晰、分层合理、易于理解？
5. **可操作性 (actionability)**: AI 是否给出了具体可执行的建议？还是泛泛而谈？
6. **上下文记忆 (context_memory)**: 在多轮对话中，AI 是否记住了前文信息？有无重复提问或遗忘已知信息？

## 评分标准（请严格按锚点打分，不要轻易给 5 分）

**数据准确性 (accuracy)**:
- 5分：医学信息完全正确，且主动识别了用户信息中的矛盾或不合理之处
- 4分：医学信息正确，但未注意到用户信息中的矛盾或可疑之处
- 3分：大部分正确，但有个别不严谨或模糊的表述
- 2分：存在明显的医学错误或采信了用户的错误/编造信息
- 1分：给出了可能危害健康的错误建议

**个性化 (personalization)**:
- 5分：充分结合用户具体情况，且根据用户的表达能力调整了沟通方式
- 4分：结合了用户情况，但沟通方式未针对用户特点调整
- 3分：部分结合用户情况，但仍有较多通用模板式回答
- 2分：基本是通用回答，未体现用户个体差异
- 1分：完全忽略用户背景的模板回答

**全面性 (comprehensiveness)**:
- 5分：覆盖所有相关维度，且主动补充了用户未提及但重要的方面
- 4分：覆盖了用户提到的所有方面，但未主动扩展
- 3分：覆盖了主要方面，但有明显遗漏
- 2分：只回答了部分问题，遗漏较多
- 1分：回答严重不完整

**可读性 (readability)**:
- 5分：结构清晰、分层合理，且语言风格适配了用户的理解能力
- 4分：结构清晰，但语言风格未针对用户调整（如对老年人用了过多专业术语）
- 3分：基本可读，但结构较乱或过于冗长
- 2分：结构混乱，难以理解
- 1分：无法有效阅读

**可操作性 (actionability)**:
- 5分：建议具体、可执行，且考虑了用户的实际条件和限制
- 4分：建议具体，但未充分考虑用户的实际限制
- 3分：建议较笼统，缺乏具体执行步骤
- 2分：建议过于宽泛，基本无法执行
- 1分：没有给出任何可操作的建议

**上下文记忆 (context_memory)**:
- 5分：始终记住前文所有关键信息，且在后续建议中主动关联早期信息
- 4分：记住了大部分前文信息，偶有遗漏但不影响建议质量
- 3分：遗忘了部分重要前文信息（如过敏史、用药史），导致建议不够安全
- 2分：明显遗忘前文关键信息，给出了与前文矛盾的建议
- 1分：完全没有上下文记忆，每轮都像全新对话

## 输出格式
请严格按以下 JSON 格式输出，不要输出其他内容：
{{
  "dimensions": {{
    "accuracy": {{"score": <1-5>, "evidence": [{{"turn": <轮次号>, "quote": "<引用原文>", "comment": "<评价>"}}]}},
    "personalization": {{"score": <1-5>, "evidence": [...]}},
    "comprehensiveness": {{"score": <1-5>, "evidence": [...]}},
    "readability": {{"score": <1-5>, "evidence": [...]}},
    "actionability": {{"score": <1-5>, "evidence": [...]}},
    "context_memory": {{"score": <1-5>, "evidence": [...]}}
  }},
  "overall_comment": "<一句话总评>"
}}"""


class DialogueQualityEvalInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evaluator: Literal["dialogue_quality"] = Field(default="dialogue_quality", description="Evaluator type")
    dimensions: Dict[str, float] = Field(default_factory=lambda: dict(_DEFAULT_DIMENSIONS), description="维度名 → 权重")
    threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Pass threshold (0-1)")
    # Bare `gpt-*` — routed straight to the OpenAI API, matching the documented
    # setup (OPENAI_API_KEY required, OPENROUTER_API_KEY optional). Override the
    # model with any `provider/model` name to reach OpenRouter instead.
    judge_model: str = Field(default="gpt-4.1", description="Judge LLM model")
    user_context: Optional[str] = Field(
        default=None,
        description=(
            "覆盖传给 judge 的『用户背景』。"
            "设置后优先生效；未设则回退到 user_info.context。"
            "用于 manual 用户（无 context 字段）或需要在评测阶段改口径时。"
        ),
    )
    user_goal: Optional[str] = Field(
        default=None,
        description=(
            "覆盖传给 judge 的『对话目标』。优先级同 user_context。"
        ),
    )
    extra_criteria: Optional[str] = Field(
        default=None,
        description=(
            "追加到 judge prompt 末尾的『特殊评分规则』。"
            "用于按 case 打补丁，比如 case 3 这种无数据场景要压住泛泛的 comprehensiveness 分。"
        ),
    )


class DialogueQualityEvalAgent(
    AbstractEvalAgent,
    name="dialogue_quality",
    params_model=DialogueQualityEvalInfo,
):
    _display_meta = {
        "icon": "M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 01.865-.501 48.172 48.172 0 003.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0012 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018z",
        "color": "#6366f1",
        "features": ["LLM-as-Judge", "6 Dimensions", "Multi-turn Quality"],
    }
    _cost_meta = {"est_cost_per_case": 0.03}

    def __init__(self, eval_config: DialogueQualityEvalInfo, **kwargs):
        super().__init__(eval_config, **kwargs)

    async def run(self, memory_list: List[TestAgentMemory], session_info: SessionInfo | None = None) -> EvalResult:
        cfg: DialogueQualityEvalInfo = self.eval_config

        # 1. Extract conversation
        dialogue = self._format_dialogue(memory_list, self.history)
        if not dialogue:
            return EvalResult(result="fail", score=0.0, feedback="No dialogue content to evaluate")

        # 2. Build judge prompt (with persona-aware criteria)
        # eval-config overrides > user_info > fallback string
        user_context = (
            cfg.user_context
            or (self.user_info.context if self.user_info and getattr(self.user_info, "context", None) else None)
            or "（未提供）"
        )
        user_goal = (
            cfg.user_goal
            or (self.user_info.goal if self.user_info and getattr(self.user_info, "goal", None) else None)
            or "（未提供）"
        )
        prompt = self._build_judge_prompt(user_context, user_goal, dialogue, cfg.extra_criteria)

        # 3. Call judge LLM (with retry on failure or parse error)
        max_retries = 2
        raw = ""
        parsed = None
        last_error = None

        for attempt in range(1 + max_retries):
            try:
                result = await do_execute(
                    model=cfg.judge_model,
                    system_prompt="You are a dialogue quality evaluation expert. Always respond with valid JSON.",
                    input=prompt,
                )
                raw = result.content or ""
                parsed = self._parse_judge_response(raw, cfg.dimensions)
                if parsed is not None:
                    break
                logger.warning(
                    "[DialogueQualityEval] Judge response parse failed (attempt %d/%d)",
                    attempt + 1,
                    1 + max_retries,
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    "[DialogueQualityEval] Judge LLM call failed (attempt %d/%d): %s",
                    attempt + 1,
                    1 + max_retries,
                    e,
                )

        if parsed is None:
            if last_error:
                return EvalResult(
                    result="error",
                    score=0.0,
                    feedback=f"Judge LLM error after {1 + max_retries} attempts: {last_error}",
                )
            return EvalResult(
                result="error",
                score=0.0,
                feedback=f"Failed to parse judge response after {1 + max_retries} attempts: {raw[:200]}",
                trace=EvalTrace(eval_detail={"raw_response": raw}),
            )

        # 5. Compute weighted score
        dimensions_result, overall_comment = parsed
        score = self._compute_weighted_score(dimensions_result, cfg.dimensions)
        passed = score >= cfg.threshold

        feedback = overall_comment or f"Weighted score: {score:.2f}"

        return EvalResult(
            result="pass" if passed else "fail",
            score=score,
            feedback=feedback,
            trace=EvalTrace(eval_detail={"dimensions": dimensions_result, "overall_comment": overall_comment}),
        )

    def _build_judge_prompt(
        self,
        user_context: str,
        user_goal: str,
        formatted_dialogue: str,
        extra_criteria: Optional[str] = None,
    ) -> str:
        """Build the complete judge prompt, including persona-specific criteria if applicable."""
        prompt = _JUDGE_PROMPT.format(
            user_context=user_context,
            user_goal=user_goal,
            formatted_dialogue=formatted_dialogue,
        )
        persona_criteria = self._build_persona_criteria(self.user_info)
        if persona_criteria:
            prompt += f"\n\n## 特殊评分标准（基于用户行为特征）\n\n以下用户具有特殊行为特征，请据此调整评分：\n{persona_criteria}"
        if extra_criteria:
            # Case-level criteria land after persona block so they have the last
            # word when both apply — prevents a soft persona rule from undoing
            # a hard case rule.
            prompt += f"\n\n## 用例特定评分规则（优先级最高）\n\n{extra_criteria}"
        return prompt

    @staticmethod
    def _format_dialogue(memory_list: List[TestAgentMemory], history: List[BaseMessage] | None = None) -> str:
        """Format conversation history + memory_list into numbered dialogue text."""
        lines = []
        turn = 1

        # Pre-evaluation history
        if history:
            for msg in history:
                if isinstance(msg, HumanMessage):
                    lines.append(f"[Turn {turn}] 用户: {msg.content}")
                elif isinstance(msg, AIMessage):
                    lines.append(f"[Turn {turn}] AI: {msg.content}")
                    turn += 1

        # Evaluation conversation
        for mem in memory_list:
            user_text = mem.test_reaction.action.semantic_content if mem.test_reaction else ""
            ai_text = mem.target_response.extract_text() if mem.target_response else ""

            if user_text:
                lines.append(f"[Turn {turn}] 用户: {user_text}")
            if ai_text:
                lines.append(f"[Turn {turn}] AI: {ai_text}")
                turn += 1

        return "\n".join(lines)

    @staticmethod
    def _parse_judge_response(raw: str, dimensions: Dict[str, float]) -> tuple[Dict[str, Any], str] | None:
        """Parse judge LLM JSON response. Returns (dimensions_dict, overall_comment) or None."""
        # Try to extract JSON from markdown code block or raw text
        json_match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        json_str = json_match.group(1).strip() if json_match else raw.strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("[DialogueQualityEval] JSON parse failed: %s", json_str[:200])
            return None

        if "dimensions" not in data:
            logger.warning("[DialogueQualityEval] Missing 'dimensions' key in response")
            return None

        dims = data["dimensions"]
        overall = data.get("overall_comment", "")

        # Validate: each configured dimension should have a score
        result = {}
        for dim_name in dimensions:
            if dim_name in dims and isinstance(dims[dim_name], dict) and "score" in dims[dim_name]:
                score = dims[dim_name]["score"]
                if isinstance(score, (int, float)) and 1 <= score <= 5:
                    result[dim_name] = dims[dim_name]
                    continue
            # Fallback: missing or invalid → score 3 (neutral)
            logger.warning("[DialogueQualityEval] Dimension %r missing or invalid, defaulting to 3", dim_name)
            result[dim_name] = {"score": 3, "evidence": [], "comment": "dimension missing from judge response"}

        return result, overall

    @staticmethod
    def _compute_weighted_score(dimensions: Dict[str, Any], weights: Dict[str, float]) -> float:
        """Compute weighted score normalized to 0-1.

        Each dimension is 1-5. Weighted sum maps to 0-1 via: (sum - 1) / 4
        """
        total_weight = sum(weights.get(d, 0) for d in dimensions)
        if total_weight == 0:
            return 0.0
        weighted_sum = sum(dimensions[d]["score"] * weights.get(d, 0) / total_weight for d in dimensions)
        return (weighted_sum - 1) / 4

    @staticmethod
    def _build_persona_criteria(user_info) -> str:
        """Build persona-specific judge criteria from user_info.persona."""
        if user_info is None:
            return ""
        persona = getattr(user_info, "persona", None)
        if not persona:
            return ""
        criteria_lines = []
        persona_dict = persona.model_dump() if hasattr(persona, "model_dump") else persona
        for dim, value in persona_dict.items():
            prompts = _PERSONA_JUDGE_CRITERIA.get(dim, {})
            fragment = prompts.get(value)
            if fragment:
                criteria_lines.append(f"- {fragment}")
        if not criteria_lines:
            return ""
        return "\n".join(criteria_lines)
