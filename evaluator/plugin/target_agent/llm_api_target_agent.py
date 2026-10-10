"""
LlmApiTargetAgent — Unified LLM API caller

Registered name: "llm_api"

Based on evaluator/utils/llm.py do_execute interface, supports all integrated model providers
(OpenAI, Google Gemini, etc.), automatically maintains multi-turn conversation history,
compatible with TestCase.history field.

User-level params (LlmApiTargetInfo):
    model: Model name (required, e.g. gpt-5.2 / gemini-3-pro-preview)
    system_prompt: System prompt (optional)

Infrastructure params (read from env vars, managed by do_execute / init_chat_model):
    OPENAI_API_KEY / GOOGLE_API_KEY etc.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from evaluator.core.interfaces.abstract_target_agent import AbstractTargetAgent
from evaluator.core.schema import (
    SessionInfo,
    TargetAgentReaction,
    TestAgentAction,
)
from evaluator.utils.llm import BasicMessage, do_execute

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = "You are an AI assistant. Please provide helpful answers based on the user's questions."

# benchmark/data/ directory (3 levels up from evaluator/plugin/target_agent/ -> project root -> benchmark/data/)
_BENCHMARK_DATA_DIR = Path(__file__).resolve().parents[3] / "benchmark" / "data"

# ESLBench-only language awareness (V2-1).
# ESLBench 题面可能为中英文；基模默认用英文回答会被 kg_qa 判负。检测题面语言并在
# system_prompt 尾部追加语言指令，让基模用与提问一致的语言作答。仅对 eslbench/retrieve
# 作用域生效（见 LlmApiTargetAgent._generate_next_reaction 的 scope gate），其它 llm_api
# benchmark（healthbench / medcalc ...）行为字节级不变。
# CJK Ext-A (㐀-䶿) + Unified (一-鿿) + Compat Ideographs (豈-﫿)
_CJK_RE = re.compile(r"[㐀-鿿豈-﫿]")
_LANG_DIRECTIVE = {"zh": "\n\nReply in Chinese (请用中文回答).", "en": ""}


def _detect_lang(text: str) -> str:
    """检测题面语言：命中 CJK 字符即判 zh，否则 en。None/空串 视为 en。"""
    return "zh" if _CJK_RE.search(text or "") else "en"


def _load_tool_group(tool_group: str, tool_context: dict[str, Any]) -> tuple[list[BaseTool], type | None]:
    """Load tool group module from benchmark/data/

    tool_group format: "{benchmark}/{tool_name}" e.g. "eslbench/retrieve"
    Resolves to: benchmark/data/{benchmark}/tools/{tool_name}.py

    Module must export get_tools(**tool_context) -> list[BaseTool]
    Optionally exports ToolContext class (for langgraph context_schema, eliminates Pydantic serialization warnings)

    Returns:
        (tools, context_class) — context_class is None when falling back to dict
    """
    import importlib.util

    parts = tool_group.split("/")
    if len(parts) != 2:
        raise ValueError(f"tool_group format should be '{{benchmark}}/{{tool_name}}', got: {tool_group!r}")

    benchmark, tool_name = parts
    file_path = _BENCHMARK_DATA_DIR / benchmark / "tools" / f"{tool_name}.py"
    if not file_path.exists():
        raise FileNotFoundError(f"Tool group file not found: {file_path}")

    spec = importlib.util.spec_from_file_location(f"tool_group.{benchmark}.{tool_name}", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    get_tools_fn = getattr(module, "get_tools", None)
    if get_tools_fn is None:
        raise AttributeError(f"Tool group {tool_group!r} does not export get_tools() function")

    # Extract ToolContext class (optional, for context_schema)
    context_class = getattr(module, "ToolContext", None)

    return get_tools_fn(**tool_context), context_class


class LlmApiTargetInfo(BaseModel):
    """LLM API target config — unified LLM calling via do_execute"""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"_comment": "Minimal config, only model required", "type": "llm_api", "model": "gpt-5.4"},
                {
                    "_comment": "With custom system prompt",
                    "type": "llm_api",
                    "model": "gemini-3-pro-preview",
                    "system_prompt": "You are a professional health assistant.",
                },
                {
                    "_comment": "With tool group",
                    "type": "llm_api",
                    "model": "gpt-5.4",
                    "tool_group": "eslbench/retrieve",
                    "tool_context": {"user_email": "user110@demo"},
                },
            ],
        },
    )
    type: Literal["llm_api"] = Field(description="Target type")
    model: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Provider model ID, e.g. gpt-5.4-mini, openai/gpt-5-mini, or dashscope:qwen3.5-flash",
        examples=["gpt-5.4-mini", "openai/gpt-5-mini", "dashscope:qwen3.5-flash"],
    )
    system_prompt: Optional[str] = Field(None, description="System prompt (uses default prompt if not specified)")
    tool_group: Optional[str] = Field(
        None,
        description="Tool group path (format: '{benchmark}/{name}', e.g. 'eslbench/retrieve', corresponds to .py files under benchmark/data/{benchmark}/tools/)",
    )
    tool_context: Optional[Dict[str, Any]] = Field(
        None, description="Tool group runtime context (passed to get_tools(), e.g. user_email etc.)"
    )
    thinking_level: Optional[str] = Field(
        None, description='Thinking level — OpenAI: "low"/"medium"/"high"; Gemini: budget token count (e.g. "8192")'
    )


class LlmApiTargetAgent(AbstractTargetAgent, name="llm_api", params_model=LlmApiTargetInfo):
    """Unified LLM API caller"""

    _display_meta = {
        "icon": (
            "M8.25 3v1.5M4.5 8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M8.25 19.5V21M12"
            " 3v1.5m0 15V21m3.75-18v1.5m0 15V21m-9-1.5h10.5a2.25 2.25 0 002.25-2.25V6.75a2.25 2.25 0"
            " 00-2.25-2.25H6.75A2.25 2.25 0 004.5 6.75v10.5a2.25 2.25 0 002.25 2.25zm.75-12h9v9h-9v-9z"
        ),
        "color": "#6366f1",
        "features": ["OpenAI", "Gemini", "Multi-model"],
    }
    _cost_meta = {
        "est_input_tokens": 200,  # Estimated input tokens per call (system prompt + user message)
        "est_output_tokens": 600,  # Estimated output tokens per call
    }

    def __init__(self, target_config: LlmApiTargetInfo, history: list[BaseMessage] | None = None):
        super().__init__(target_config, history=history)
        self.config: LlmApiTargetInfo = target_config
        self.model: str = target_config.model

        # ESLBench-only latched per-session language (V2-1). None = not yet detected.
        self._lang: str | None = None

        # Convert history (List[BaseMessage]) to BasicMessage list for do_execute
        self._conversation: list[BasicMessage] = []
        for msg in self.history:
            if isinstance(msg, HumanMessage):
                self._conversation.append(BasicMessage(role="user", content=msg.content))
            elif isinstance(msg, AIMessage):
                self._conversation.append(BasicMessage(role="assistant", content=msg.content))

        # Load tool group (optional)
        self._tools: list[BaseTool] | None = None
        self._tool_context_typed: Any = None  # Typed tool_context (eliminates Pydantic serialization warnings)
        self._tool_context_schema: type | None = None
        # Skip tool loading when tool_context is missing — tools that read user data (eslbench/retrieve)
        # raise on ToolContext(**None) at first invocation. Cases without user_email (factual / citation /
        # numerical / relational) don't need tools and should answer from model knowledge.
        if target_config.tool_group and target_config.tool_context:
            self._tools, context_class = _load_tool_group(target_config.tool_group, target_config.tool_context)
            # Convert dict to ToolContext type declared by the tool module (if available)
            if context_class is not None:
                self._tool_context_typed = context_class(**target_config.tool_context)
                self._tool_context_schema = context_class
            else:
                self._tool_context_typed = target_config.tool_context
                self._tool_context_schema = None
            logger.info(
                "[LlmApiTargetAgent] Loaded tool_group=%r with %d tools",
                target_config.tool_group,
                len(self._tools),
            )
        elif target_config.tool_group:
            logger.info(
                "[LlmApiTargetAgent] tool_group=%r declared but no tool_context — skipping tool load",
                target_config.tool_group,
            )

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

    async def _generate_next_reaction(self, test_action: Optional[TestAgentAction]) -> TargetAgentReaction:
        """Send user input to LLM via do_execute and return model response"""
        if test_action is None:
            # First turn with no user input, return welcome message
            return TargetAgentReaction(
                type="message",
                message_list=[{"content": "Hello, how can I help you?"}],
            )

        # Extract user input
        user_text = self._extract_user_input(test_action)
        system_prompt = self.config.system_prompt or _DEFAULT_SYSTEM_PROMPT

        # V2-1 SCOPE GATE: language awareness only for the eslbench retrieve tool group.
        # Any other llm_api benchmark (healthbench / medcalc / ...) is byte-for-byte unchanged.
        if getattr(self.config, "tool_group", None) == "eslbench/retrieve":
            override = os.environ.get("HOLYEVAL_ESLBENCH_TARGET_LANG", "auto").lower()
            lang = override if override in ("en", "zh") else (self._lang or _detect_lang(user_text))
            if self._lang is None and override not in ("en", "zh"):
                self._lang = lang  # latch on first turn (auto-detect path only)
            # en appends "" => zero change for English tasks; augment the local var only,
            # leaving self.config.system_prompt untouched (get_session_info must see the base value).
            system_prompt = system_prompt + _LANG_DIRECTIVE.get(lang, "")

        logger.debug(
            "[LlmApiTargetAgent] Calling %s, history=%d messages",
            self.config.model,
            len(self._conversation),
        )

        # Call do_execute: history_messages is prior conversation, input is current user input
        # opt-in 确定性:HOLYEVAL_TARGET_TEMP 设值时钉被测模型 temperature(默认不设=不变)
        import os as _os
        _temp_env = _os.environ.get("HOLYEVAL_TARGET_TEMP")
        _temperature = float(_temp_env) if _temp_env not in (None, "") else None

        result = await do_execute(
            model=self.config.model,
            system_prompt=system_prompt,
            input=user_text,
            history_messages=self._conversation if self._conversation else None,
            tools=self._tools,
            thinking_level=self.config.thinking_level,
            tool_context=self._tool_context_typed or self.config.tool_context,
            tool_context_schema=self._tool_context_schema,
            temperature=_temperature,
        )

        assistant_content = result.content

        # Accumulate cost
        self._accumulate_cost(result.usage)

        # Append this turn's conversation to history
        self._conversation.append(BasicMessage(role="user", content=user_text))
        self._conversation.append(BasicMessage(role="assistant", content=assistant_content))

        logger.debug(
            "[LlmApiTargetAgent] Response: %d chars",
            len(assistant_content),
        )

        # Serialize tool_calls (ToolCallRecord -> dict)
        tc_dicts = [tc.model_dump() for tc in result.tool_calls] if result.tool_calls else None

        return TargetAgentReaction(
            type="message",
            message_list=[{"content": assistant_content}],
            tool_calls=tc_dicts,
            usage=result.usage,
        )

    def get_session_info(self) -> SessionInfo:
        """Notify EvalAgent that the LLM has access to user-specific data — either via a long
        system_prompt carrying the health profile, or via tool_context exposing user_email
        (tool-based retrieval path). Without this, contextual hallucination judges fall back
        to the 'base model — no access' rubric and penalise correct tool-fetched answers."""
        has_profile_in_prompt = bool(self.config.system_prompt and len(self.config.system_prompt) > 200)
        has_user_email_in_tools = bool(self.config.tool_context and self.config.tool_context.get("user_email"))
        return SessionInfo(has_user_data=has_profile_in_prompt or has_user_email_in_tools)

    def _extract_user_input(self, test_action: TestAgentAction) -> str:
        """Extract user input from TestAgentAction"""
        if test_action.type == "semantic":
            return test_action.semantic_content or ""
        elif test_action.type == "message":
            msg = test_action.message_content
            if isinstance(msg, dict):
                return msg.get("content", "")
            return ""
        else:
            return str(test_action.custom_content) if test_action.custom_content else ""
