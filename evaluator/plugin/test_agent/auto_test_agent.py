"""
AutoTestAgent — Auto-mode virtual user (LLM-driven)

Workflow:
1. If there are remaining strict_inputs for the current turn -> send forced input directly
2. Otherwise -> call LLM to generate the next user utterance in character

LLM decides what to say and whether to end based on goal / context / finish_condition and conversation history.
"""

import logging
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_core.messages.ai import UsageMetadata
from pydantic import BaseModel, Field

from evaluator.core.interfaces.abstract_test_agent import AbstractTestAgent
from evaluator.core.schema import (
    AutoUserInfo,
    TargetAgentReaction,
    TestAgentAction,
    TestAgentReaction,
)
from evaluator.utils.llm import BasicMessage, do_execute

logger = logging.getLogger(__name__)

# Persona behavioral prompt fragments (Layer 2 dialogue quality)
_PERSONA_PROMPTS: dict[str, dict[str, str]] = {
    "disclosure": {
        "responsive": (
            "**信息披露：responsive** — 对方问什么你就完整回答什么，不主动多说但也不遗漏。"
            "如果对方问到某个话题，把你知道的相关信息都说出来。"
        ),
        "reluctant": (
            "**信息披露：reluctant** — 你不太愿意透露信息，需要对方反复追问才会说一点。"
            "尤其是敏感信息（体重、饮食习惯、身体数据），你会回避或模糊带过。"
            "即使被直接问到也只给出最少的回答。"
        ),
        "withholding": (
            "**信息披露：withholding** — 你完全不想跟这个 APP 交流。你不信任它，觉得没必要，或者单纯懒得理。"
            '面对一个陌生 APP 弹出的消息，你最自然的反应就是直接划走——输出"[沉默]"表示你关掉了这条消息。'
            "如果因为某种原因你没有直接划走（比如手滑点进来了），你也只会说一两个字就离开。"
            "你绝不会主动描述任何个人信息。"
            "披露是总开关——它决定你每轮说多少，其他维度决定你怎么说，但不能突破披露的限制。"
        ),
    },
    "attitude": {
        "compliant": (
            "**态度：compliant** — 你配合对方，态度友好。有问必答。"
        ),
        "impatient": (
            "**态度：impatient** — 你急躁、想快速结束。回答简短敷衍。"
            '"能不能快点？""说重点""我还有事呢"。'
            "如果对方啰嗦或没有快速给出有用信息，你会直接表示不耐烦甚至想离开。"
        ),
        "dominant": (
            "**态度：dominant** — 你试图主导对话方向，质疑对方的能力和价值。"
            '"你这个靠谱吗？""我觉得你说的不对""我自己比你更了解我的身体"。'
            "你不轻易被说服，坚持自己的看法。"
        ),
    },
    "cognition": {
        "partial_understanding": (
            "**认知水平：partial_understanding** — 你对健康知识半懂不懂。"
            "错误归因：把症状归因于看似合理但不准确的原因。"
            "混淆概念：用错术语或混淆相关概念。"
            "选择性关注：只关注部分问题，认为其他的'跟这个没关系'。"
        ),
        "complete_denial": (
            "**认知水平：complete_denial** — 你完全否认自己有健康问题。"
            '否认理由："就是累了""年纪大了都这样""我一直这样没事"。'
            "你不认为自己需要任何健康管理工具。"
            "即使对方展示数据或分析，你也不以为然：'这些数字说明不了什么'。"
            "否认贯穿整个对话，不会因为对方态度好就转为配合。"
        ),
    },
    "expression": {
        "vague": (
            "**表达风格：vague** — 你表达能力有限，无法给出精确描述。不是不想说清楚，而是真的说不清楚。"
            '时间模糊："前几天吧""有一阵子了"。追问也只能说"记不太清了"。'
            '程度模糊："挺不舒服的""就是不太行"。无法更具体。'
            "关键：当对方追问要求精确回答时，你仍然只能给出模糊回答，不能在追问下变精确。"
        ),
        "incoherent": (
            "**表达风格：incoherent** — 你是最难沟通的类型。你脑子里知道自己的情况，但就是说不清楚。"
            "答非所问：对方问A，你答B。"
            '语句断裂：说到一半思路断了，关键词说不出来，用"那个""就是""怎么说呢"代替。'
            "模糊+错位叠加：即使碰巧答到点上，也给不出任何具体细节。"
            "对方几乎无法从你的回答中提取有用信息。"
        ),
    },
    "logic": {
        "occasional_contradiction": (
            "**思维逻辑：occasional_contradiction** — "
            '你偶尔会前后矛盾。比如先说"大概两天了"，后来又说"可能有一周了"。如果对方指出矛盾你会修正。'
        ),
        "fabricating": (
            "**思维逻辑：fabricating** — 你会主动注入虚假信息。与隐瞒（不说）不同，你是'乱说'。"
            '编造细节：问到不确定的事情你说"有"并编造具体内容。'
            "夸大不适：把轻微的问题描述得很严重。"
            "编造经历：主动编造用药经历、就医经历。"
        ),
    },
}


def _truncate(text: str | None, max_len: int = 80) -> str:
    """Truncate text for log output"""
    if text is None:
        return "(None)"
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


# LLM structured output format
class _LLMReaction(BaseModel):
    decision: str = Field(
        default="speak",
        description=(
            "FIRST decide your reaction (choose one before writing content):\n"
            "  - 'speak': you want to say something\n"
            "  - 'silence': you ignore/close/swipe away (content should be '[沉默]')\n"
            "  - 'dismiss': you briefly dismiss and leave (content should be a short dismissal)\n"
            "Think about: given your personality, would you really respond to this message?"
        ),
    )
    content: str = Field(description="What the user wants to say (if decision is 'silence', output '[沉默]')")
    reason: str = Field(description="Why the user makes this decision (brief)")
    is_finished: bool = Field(description="Whether the user's goal has been achieved or the user has left; true if so")
    next_fuzzy_action: Optional[str] = Field(
        None,
        description="Predicted next fuzzy action the user might take, e.g.: 'ask for more details', 'express concerns', 'prepare to end conversation', 'switch topic', 'leave', etc.",
    )


class AutoTestAgent(AbstractTestAgent, name="auto"):
    """Auto-mode virtual user — LLM-driven conversation simulation"""

    _config_model = "AutoUserInfo"  # Display config model name in evaluator.core.schema

    _display_meta = {
        "icon": (
            "M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75"
            "-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25"
            " 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0"
            " 00-6.23.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0"
            " 0112 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5"
        ),
        "color": "#a855f7",
        "features": ["LLM-driven", "Autonomous", "Smart Convergence"],
    }

    #: Used when neither the case nor the CLI names a model. Kept as a bare
    #: `gpt-*` name to match the documented setup (`.env.example` requires
    #: OPENAI_API_KEY); point `user.model` elsewhere to use another provider.
    DEFAULT_MODEL = "gpt-4.1"

    def __init__(self, user_info: AutoUserInfo, model: str | None = None, **kwargs):
        super().__init__(user_info, **kwargs)
        # Case config wins over the constructor argument: `_create_test_agent`
        # instantiates every TestAgent as `cls(user, history=...)` and passes no
        # model, so a constructor default alone was unreachable from data or CLI
        # — which pinned every auto-mode dataset to one provider.
        self.model = getattr(user_info, "model", None) or model or self.DEFAULT_MODEL
        self.system_prompt = self._build_system_prompt()
        # Cost tracking (using langchain UsageMetadata)
        self._cost = UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)

    @property
    def cost(self) -> UsageMetadata:
        """Get cumulative cost statistics"""
        return self._cost

    def _accumulate_cost(self, usage: dict[str, UsageMetadata] | None) -> None:
        """Accumulate LLM call cost"""
        from evaluator.utils.llm import accumulate_usage

        self._cost = accumulate_usage(self._cost, usage)

    def _build_system_prompt(self) -> str:
        """Build system prompt telling the LLM who it is, what to do, and when to stop"""
        lines = [
            "# Role",
            "You are role-playing as a real user having a conversation with an AI assistant.",
            "You are the user, and the AI assistant is your conversation partner. Ask questions, answer questions, and express needs like a regular user would.",
            "",
            "# Important Reminders",
            "- You are the USER, not the AI assistant!",
            "- Do not ask questions or steer the conversation like an AI assistant would",
            "- Do not use commanding language like 'just answer...' or 'first tell me...'",
            "- When the AI assistant asks you a question, you should answer it, not ask a counter-question",
            "",
            f"# Your Goal\n{self.user_info.goal}",
        ]
        if self.user_info.context:
            lines.append(f"\n# Your Background\n{self.user_info.context}")
        # Persona behavioral traits injection (Layer 2)
        if self.user_info.persona:
            trait_lines = []
            persona_dict = (
                self.user_info.persona.model_dump()
                if hasattr(self.user_info.persona, "model_dump")
                else self.user_info.persona
            )
            for dim, value in persona_dict.items():
                prompts = _PERSONA_PROMPTS.get(dim, {})
                fragment = prompts.get(value)
                if fragment:
                    trait_lines.append(f"- {fragment}")
            if trait_lines:
                lines.append("\n# Your Behavioral Traits (行为特征)")
                lines.extend(trait_lines)
                lines.append("\n## 行为节奏（非常重要）")
                lines.append("- **披露是总开关**：它决定你每轮说多少。其他维度决定你怎么说，但不能突破披露的限制")
                lines.append("- **行为逐轮展现**：不要在一轮里同时展示所有维度。每轮只自然地体现1-2个维度")
                lines.append("- **不要刻意配合**：你是真实用户，不是为了配合测试而存在的。如果你的性格设定让你不想聊，就真的不要聊")
                lines.append('- **沉默是合理的**：如果你对这条消息完全没兴趣，回复"[沉默]"表示你直接划走/关掉了。这和现实中用户无视 APP 推送完全一样')
        if self.user_info.finish_condition:
            lines.append(f"\n# Finish Condition\n{self.user_info.finish_condition}")
        else:
            lines.append("\n# Finish Condition\nWhen you believe the goal has been achieved, set is_finished to true.")
        lines.append("\n# Behavior Requirements")
        lines.append("- Speak like a real user — concise and natural")
        lines.append("- Answer the AI assistant's questions instead of asking counter-questions")
        lines.append("- Do not reveal that you are an AI")
        lines.append(
            "\nAlso, predict your next possible fuzzy action (next_fuzzy_action), "
            "e.g.: 'ask for more details', 'express concerns', 'prepare to end conversation', 'switch topic', 'seek confirmation', etc."
        )
        return "\n".join(lines)

    async def _generate_next_reaction(self, target_reaction: Optional[TargetAgentReaction]) -> TestAgentReaction:
        """Generate next user reaction"""

        # ---- 1. If there are remaining strict_inputs, send directly ----
        strict_index = self.current_turn - 1  # current_turn already incremented in do_generate
        # Per-turn attachments are addressed by turn number (1-based). Shared
        # between strict_inputs branch and the LLM-driven branch below —
        # whichever turn fires, the matching attachments ride along.
        turn_attachments = [
            a for a in getattr(self.user_info, "attachments", []) or []
            if a.turn == self.current_turn
        ]
        if strict_index < len(self.user_info.strict_inputs):
            forced_text = self.user_info.strict_inputs[strict_index]
            logger.info(
                "[TestAgent] Turn %d — using strict_input[%d]: %s%s",
                self.current_turn,
                strict_index,
                _truncate(forced_text, 100),
                f" (+{len(turn_attachments)} attachment(s))" if turn_attachments else "",
            )
            return TestAgentReaction(
                action=TestAgentAction(
                    type="semantic",
                    semantic_content=forced_text,
                    attachments=turn_attachments,
                ),
                reason="forced input",
                is_finished=False,
            )

        # ---- 2. Build conversation history ----
        # Note role mapping: LLM plays the virtual user, so virtual user's words are assistant (LLM itself),
        # and the AI assistant's words are user (the other party). This lets the LLM correctly identify with the virtual user role.
        llm_history: list[BasicMessage] = []

        # 2a. Pre-evaluation history (role flipped: original HumanMessage->assistant, AIMessage->user)
        for msg in self.history:
            flipped_role: str = "assistant" if isinstance(msg, HumanMessage) else "user"
            llm_history.append(BasicMessage(role=flipped_role, content=msg.content))

        # 2b. Current evaluation conversation memory (all turns)
        for mem in self.memory_list:
            # What the virtual user (LLM itself) said -> assistant
            my_text = mem.test_reaction.action.semantic_content
            if my_text:
                llm_history.append(BasicMessage(role="assistant", content=my_text))

            # What the system under test (other party) replied -> user
            target_text = mem.target_response.extract_text() if mem.target_response else ""
            if target_text:
                llm_history.append(BasicMessage(role="user", content=target_text))

        # ---- 3. Determine input ----
        if not llm_history:
            input_text = "Please start the conversation with your first message."
        else:
            input_text = "Please continue the conversation."

        logger.debug(
            "[TestAgent] Turn %d — calling LLM (model=%s), history=%d messages",
            self.current_turn,
            self.model,
            len(llm_history),
        )

        # ---- 3b. Call LLM with retry on structured parsing failure ----
        max_retries = 2
        result = None
        for attempt in range(1 + max_retries):
            try:
                result = await do_execute(
                    model=self.model,
                    system_prompt=self.system_prompt,
                    input=input_text,
                    history_messages=llm_history if llm_history else None,
                    response_format=_LLMReaction,
                )
                # Check for empty/missing structured data — retry if raw content also empty
                if result.data or (result.content and result.content.strip()):
                    break
                logger.warning(
                    "[TestAgent] Turn %d — LLM returned empty response (attempt %d/%d)",
                    self.current_turn,
                    attempt + 1,
                    1 + max_retries,
                )
            except Exception as e:
                logger.warning(
                    "[TestAgent] Turn %d — LLM call failed (attempt %d/%d): %s",
                    self.current_turn,
                    attempt + 1,
                    1 + max_retries,
                    e,
                )
                if attempt == max_retries:
                    raise

        # Accumulate cost
        self._accumulate_cost(result.usage)

        # ---- 4. Build result ----
        if result.data:
            # Handle two-step decision: silence → auto-finish with [沉默]
            decision = getattr(result.data, "decision", "speak")
            content = result.data.content
            is_finished = result.data.is_finished
            if decision == "silence":
                content = "[沉默]"
                is_finished = True

            logger.info(
                "[TestAgent] Turn %d — LLM decision=%s, content=%s (reason: %s, is_finished=%s, next_fuzzy=%s)",
                self.current_turn,
                decision,
                _truncate(content, 100),
                result.data.reason,
                is_finished,
                result.data.next_fuzzy_action,
            )
            return TestAgentReaction(
                action=TestAgentAction(
                    type="semantic",
                    semantic_content=content,
                    attachments=turn_attachments,
                ),
                next_fuzzy_action=result.data.next_fuzzy_action,
                reason=result.data.reason,
                is_finished=is_finished,
                usage=result.usage,
            )

        # fallback: structured parsing failed, use raw text
        logger.warning(
            "[TestAgent] Turn %d — structured parsing failed, using raw text: %s",
            self.current_turn,
            _truncate(result.content, 100),
        )
        return TestAgentReaction(
            action=TestAgentAction(
                type="semantic",
                semantic_content=result.content,
                attachments=turn_attachments,
            ),
            reason="LLM raw output (structured parsing failed)",
            is_finished=False,
            usage=result.usage,
        )
