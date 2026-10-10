"""
EvermemTargetAgent — EverMemOS 记忆增强问答（仅搜索，不灌入数据）

注册名称: "evermem"

实现流程:
1. 将 user_email 映射为 EverMemOS user_id: holyeval_eslbench_{dir_name}
2. 调用 EverMemOSClient.search(user_id, query, top_k) 检索记忆
3. 拼接 RAG prompt + few-shot 示例 → LLM 生成答案

数据灌入由独立的 preload 脚本完成，本 agent 仅负责搜索与问答。
"""

import logging
import os
from typing import Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.ai import UsageMetadata
from pydantic import BaseModel, ConfigDict, Field

from evaluator.core.interfaces.abstract_target_agent import AbstractTargetAgent
from evaluator.core.schema import (
    TargetAgentReaction,
    TestAgentAction,
)
from evaluator.plugin.target_agent.hippo_rag_api_target_agent import (
    _RAG_SYSTEM_PROMPT,
    _SHOT_LIST_INPUT,
    _SHOT_LIST_OUTPUT,
    _SHOT_NUMERIC_INPUT,
    _SHOT_NUMERIC_OUTPUT,
)
from evaluator.utils.evermemos_client import EverMemOSClient
from evaluator.utils.llm import BasicMessage, do_execute
from evaluator.utils.user_data_chunker import email_to_dir

logger = logging.getLogger(__name__)

# EverMemOS user_id 前缀
USER_ID_PREFIX = "holyeval_eslbench"

# 全局 EverMemOS 客户端单例
_CLIENT: EverMemOSClient | None = None


def _get_client() -> EverMemOSClient:
    """获取或创建 EverMemOS 客户端单例"""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    base_url = os.environ.get("EVERMEMOS_BASE_URL", "http://localhost:1995")
    _CLIENT = EverMemOSClient(base_url=base_url)
    return _CLIENT


def _build_rag_input(query: str, memories: list[dict]) -> str:
    """将 EverMemOS search 结果拼接为 RAG 输入"""
    if not memories:
        context = "(No relevant memories found)"
    else:
        parts = []
        for m in memories:
            # v1 优先 episode/summary，兼容 v0 的 memory/content/text
            text = m.get("episode") or m.get("summary") or m.get("memory") or m.get("content") or m.get("text") or str(m)
            parts.append(text)
        context = "\n\n".join(parts)
    return f"{context}\n\nQuestion: {query}\nThought: "


# ============================================================
# 配置模型
# ============================================================


class EvermemTargetInfo(BaseModel):
    """EverMemOS 记忆增强被测目标配置"""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "_comment": "默认配置",
                    "type": "evermem",
                    "model": "gemini-3-flash-preview",
                    "user_email": "user110@demo",
                },
            ],
        },
    )
    type: Literal["evermem"] = Field(description="目标类型")
    model: Literal[
        "gpt-5.4",
        "gpt-5.2",
        "gpt-4.1",
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        "anthropic/claude-opus-4.6",
        "anthropic/claude-sonnet-4.6",
        "minimax/minimax-m2.7",
        "z-ai/glm-5.1",
    ] = Field(description="生成模型名称")
    user_email: Optional[str] = Field(None, description="用户邮箱（映射到 EverMemOS user_id）")
    top_k: int = Field(10, description="检索记忆条数", ge=1, le=100)
    system_prompt: Optional[str] = Field(None, description="自定义系统提示词（默认使用 HippoRAG 同款）")


# ============================================================
# Agent 实现
# ============================================================


class EvermemTargetAgent(AbstractTargetAgent, name="evermem", params_model=EvermemTargetInfo):
    """EverMemOS 记忆增强问答（仅搜索）"""

    _display_meta = {
        "icon": (
            "M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375"
            " 3.375 0 00-3.375-3.375H8.25m5.231 13.481L15 17.25m-4.5-15H5.625c-.621 0-1.125.504-1.125"
            " 1.125v16.5c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0"
            " 00-9-9zm3.75 11.625a2.625 2.625 0 11-5.25 0 2.625 2.625 0 015.25 0z"
        ),
        "color": "#8b5cf6",
        "features": ["RAG", "EverMemOS", "预灌入"],
    }
    _cost_meta = {
        "est_input_tokens": 8000,
        "est_output_tokens": 800,
    }

    def __init__(self, target_config: EvermemTargetInfo, history: list[BaseMessage] | None = None):
        super().__init__(target_config, history=history)
        self.config: EvermemTargetInfo = target_config

        # 对话历史
        self._conversation: list[BasicMessage] = []
        for msg in self.history:
            if isinstance(msg, HumanMessage):
                self._conversation.append(BasicMessage(role="user", content=msg.content))
            elif isinstance(msg, AIMessage):
                self._conversation.append(BasicMessage(role="assistant", content=msg.content))

        # 映射 user_email → EverMemOS user_id
        self._user_id: str | None = None
        if target_config.user_email:
            dir_name = email_to_dir(target_config.user_email)
            self._user_id = f"{USER_ID_PREFIX}_{dir_name}"

        # 成本统计
        self._cost = UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)

    @property
    def cost(self) -> UsageMetadata:
        return self._cost

    def _accumulate_cost(self, usage: dict[str, UsageMetadata] | None) -> None:
        from evaluator.utils.llm import accumulate_usage

        self._cost = accumulate_usage(self._cost, usage)

    async def _generate_next_reaction(self, test_action: Optional[TestAgentAction]) -> TargetAgentReaction:
        if test_action is None:
            return TargetAgentReaction(
                type="message",
                message_list=[{"content": "你好，有什么可以帮你的？"}],
            )

        user_text = self._extract_user_input(test_action)

        if not self._user_id:
            return TargetAgentReaction(
                type="message",
                message_list=[{"content": "错误: 未配置 user_email，无法确定 EverMemOS user_id"}],
            )

        # 1. 获取 EverMemOS 客户端
        client = _get_client()

        # 2. 检索相关记忆
        memories, error = await client.search(
            user_id=self._user_id,
            query=user_text,
            top_k=self.config.top_k,
        )

        if error:
            logger.warning("[evermem] 搜索失败 (user_id=%s): %s", self._user_id, error)
            memories = []

        logger.debug(
            "[evermem] Query: %s, Retrieved %d memories, calling %s",
            user_text[:80],
            len(memories) if memories else 0,
            self.config.model,
        )

        # 3. 构建 RAG prompt
        rag_input = _build_rag_input(user_text, memories or [])
        system_prompt = self.config.system_prompt or _RAG_SYSTEM_PROMPT

        # 4. 调用 LLM（带 few-shot 示例，与 HippoRAG 一致）
        history_with_oneshot = [
            BasicMessage(role="user", content=_SHOT_NUMERIC_INPUT),
            BasicMessage(role="assistant", content=_SHOT_NUMERIC_OUTPUT),
            BasicMessage(role="user", content=_SHOT_LIST_INPUT),
            BasicMessage(role="assistant", content=_SHOT_LIST_OUTPUT),
        ]
        if self._conversation:
            history_with_oneshot.extend(self._conversation)

        result = await do_execute(
            model=self.config.model,
            system_prompt=system_prompt,
            input=rag_input,
            history_messages=history_with_oneshot,
        )

        self._accumulate_cost(result.usage)

        # 更新对话历史
        self._conversation.append(BasicMessage(role="user", content=user_text))
        self._conversation.append(BasicMessage(role="assistant", content=result.content))

        return TargetAgentReaction(
            type="message",
            message_list=[{"content": result.content}],
            usage=result.usage,
        )

    def _extract_user_input(self, test_action: TestAgentAction) -> str:
        if test_action.type == "semantic":
            return test_action.semantic_content or ""
        elif test_action.type == "message":
            msg = test_action.message_content
            if isinstance(msg, dict):
                return msg.get("content", "")
            return ""
        else:
            return str(test_action.custom_content) if test_action.custom_content else ""
