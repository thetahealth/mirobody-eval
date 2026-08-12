"""
llm — 底层大模型调用基础模块

提供统一的大模型调用接口 do_execute，基于 langchain create_agent 实现。

支持模型：
- gpt-5.2 及以上（OpenAI 原生）
- gemini-3 及以上（Google GenAI 原生）
- 通过 OpenRouter 访问 280+ 模型（自动路由，包括 Claude、Llama、Mistral 等）

依赖：langchain / langchain-openai / langchain-google-genai
"""

import asyncio
import logging
import time
from typing import Any, Generic, Literal, Optional, TypeVar

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import AIMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

T = TypeVar("T")
ResponseT = TypeVar("ResponseT")

# OpenAI 偶发 401 重试配置
_RETRYABLE_MAX_ATTEMPTS = 3
_RETRYABLE_BASE_DELAY = 1.0  # 秒，指数退避基数


class BasicMessage(BaseModel):
    """基础消息（支持多模态）"""

    role: Literal["user", "assistant"] = Field(description="消息角色")
    content: str | list[dict[str, Any]] = Field(
        description='消息内容 — 纯文本传 str；带图片传 list，如 [{"type":"text","text":"..."}, {"type":"image_url","image_url":{"url":"..."}}]'
    )


class ToolCallRecord(BaseModel):
    """单次工具调用记录（ReAct 循环中的一步）"""

    name: str = Field(description="工具名称")
    args: dict[str, Any] = Field(default_factory=dict, description="调用参数")
    result: str = Field(default="", description="工具返回结果")
    call_id: str = Field(default="", description="调用 ID")


class ExecuteResult(BaseModel, Generic[T]):
    """
    do_execute 函数的返回结构体

    Args:
        usage: langchain UsageMetadata — token 使用统计
        content: 原始输出文本
        data: response_format 解析后的结构体
        tool_calls: 工具调用记录列表（ReAct 循环中的所有 tool call，无工具调用时为空列表）
        time_cost: 耗时（毫秒）
    """

    usage: dict[str, UsageMetadata]
    content: str
    data: Optional[T] = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    time_cost: int  # ms


async def do_execute(
    model: str,
    system_prompt: str,
    input: str | BasicMessage,
    history_messages: list[BasicMessage] | None = None,
    tools: list[BaseTool] | None = None,
    response_format: type[ResponseT] | None = None,
    thinking_level: str | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
    tool_context: Any | None = None,
    tool_context_schema: type | None = None,
    temperature: float | None = None,
) -> ExecuteResult:
    """
    调用大模型

    内部通过 langchain 的 create_agent 实现，支持工具调用循环和结构化输出。
    无工具时 agent 退化为单次 LLM 调用。

    模型路由规则：
    - gpt-* → OpenAI 原生 API
    - gemini-* → Google GenAI 原生 API
    - 其他模型（如 anthropic/claude-*、meta-llama/*）→ 自动通过 OpenRouter 调用

    Args:
        model: 模型名称，如 "gpt-5.2" / "gemini-3-pro" / "anthropic/claude-3.7-sonnet"
        system_prompt: 系统提示词
        input: 用户输入（字符串或 BasicMessage）
        history_messages: 历史对话消息列表
        tools: LangChain 工具列表（为空时无工具调用循环）
        response_format: 结构化输出的 Pydantic Model 类型，解析结果放入 data 字段
        thinking_level: 思考级别 — OpenAI: "low"/"medium"/"high"; Gemini: budget token 数（如 "8192"）
        max_tokens: 最大输出 token 数
        timeout: 超时时间（秒）。为 None 时从 AGENT_LLM_TIMEOUT 环境变量读取，默认 420s
        tool_context: 工具运行时上下文，透传到 langgraph ToolRuntime.context（tools 使用 ToolRuntime 注入时需要）
        tool_context_schema: tool_context 的类型类（可选）。显式传入可消除 Pydantic 序列化警告

    Returns:
        ExecuteResult

    Examples:
        # OpenAI 原生调用
        await do_execute(
            model="gpt-5.2",
            system_prompt="...",
            input="...",
        )

        # OpenRouter 调用 Claude（自动识别并路由）
        await do_execute(
            model="anthropic/claude-3.7-sonnet",
            system_prompt="...",
            input="...",
        )

        # OpenRouter 调用其他模型（自动识别并路由）
        await do_execute(
            model="meta-llama/llama-3.1-405b",
            system_prompt="...",
            input="...",
        )
    """

    # ---- 1. 判断 provider，拼装模型参数 ----

    _NOVA_BARE = {"glm-5.2"}  # nova per-call 中转仅有裸名(无 [次] 渠道)的模型
    use_nova = False
    if model.startswith("[") or model in _NOVA_BARE:
        # nova per-call 中转(OpenAI 兼容);[次]/[限时] 前缀或 nova 裸名。
        # base_url/key 走 GEMINI_BASE_URL/GEMINI_API_KEY(见下 init_kwargs)。
        provider = "openai"
        use_openrouter = False
        use_nova = True
    elif model.startswith("gpt"):
        # OpenAI 原生模型
        provider = "openai"
        use_openrouter = False
    elif model.startswith("gemini"):
        # Google Gemini 模型
        provider = "google_genai"
        use_openrouter = False
        # Remote config 可能强制 GOOGLE_GENAI_USE_VERTEXAI=true（生产 Vertex 路径）。
        # 本地评测没有 gcloud ADC 凭证时会失败，回退到 GOOGLE_API_KEY 直连 Generative Language API。
        # 显式 export GOOGLE_GENAI_USE_VERTEXAI=true 时不覆盖（让用户保留主动选择）。
        import os as _os
        if _os.environ.get("GOOGLE_API_KEY") and not _os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            for _k in ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"):
                _os.environ.pop(_k, None)
    else:
        # 其他模型通过 OpenRouter 调用（如 anthropic/claude-3.7-sonnet）
        provider = "openai"
        use_openrouter = True

    model_kwargs: dict[str, Any] = {}
    if max_tokens is not None:
        model_kwargs["max_tokens"] = max_tokens

    if thinking_level is not None:
        if provider == "google_genai":
            # Gemini thinking: budget_tokens 控制思考深度
            model_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": int(thinking_level),
            }
        else:
            # OpenAI reasoning_effort: "low" / "medium" / "high"
            model_kwargs["reasoning_effort"] = thinking_level

    # ---- 2. 创建模型实例（需要传 model_kwargs 所以用 init_chat_model） ----

    # 设置 HTTP 级别超时（单次请求），触发 httpx.TimeoutException 后
    # OpenAI 客户端会自动重试（默认 max_retries=2），避免 API 挂起时只能等 asyncio 兜底
    if timeout is None:
        from evaluator.utils.config import get_agent_llm_timeout

        timeout = get_agent_llm_timeout()

    http_timeout = min(timeout, 180)  # 单次 HTTP 请求最长 180s，超时后触发客户端重试

    # 构建 init_chat_model 参数
    init_kwargs = {
        "model": model,
        "model_provider": provider,
        "timeout": http_timeout,
        **model_kwargs,
    }

    # 可选:确定性采样(opt-in)。temperature=0 消除同题重跑的生成随机性,用于可复现评测。
    if temperature is not None:
        init_kwargs["temperature"] = temperature

    # OpenAI + reasoning_effort + tools → 必须使用 Responses API
    if provider == "openai" and not use_openrouter and thinking_level is not None and tools:
        init_kwargs["use_responses_api"] = True

    # OpenRouter 配置（非 gpt/gemini 模型）
    if use_openrouter:
        import os

        init_kwargs["base_url"] = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        init_kwargs["api_key"] = os.getenv("OPENROUTER_API_KEY")

    # nova per-call 中转配置（[次]/[限时] 前缀或 nova 裸名）
    if use_nova:
        import os

        init_kwargs["base_url"] = os.getenv("GEMINI_BASE_URL", "https://once.novai.su/v1")
        init_kwargs["api_key"] = os.getenv("GEMINI_API_KEY")

    llm = init_chat_model(**init_kwargs)

    # ---- 3. 创建 agent ----
    #   有 tools → agent 会进入 ReAct 循环（LLM → tool → LLM → ...）
    #   无 tools → 退化为单次 LLM 调用

    create_kwargs: dict[str, Any] = {
        "model": llm,
        "tools": tools,
        "system_prompt": system_prompt,
        "response_format": response_format,
    }
    if tool_context is not None:
        create_kwargs["context_schema"] = tool_context_schema or (type(tool_context) if not isinstance(tool_context, dict) else dict)
    agent = create_agent(**create_kwargs)

    # ---- 4. 构建消息列表（system_prompt 已交给 create_agent，这里只放对话历史和当前输入） ----

    messages: list[dict[str, Any]] = []

    if history_messages:
        for msg in history_messages:
            messages.append({"role": msg.role, "content": msg.content})

    if isinstance(input, str):
        messages.append({"role": "user", "content": input})
    else:
        messages.append({"role": input.role, "content": input.content})

    # ---- 5. 调用 agent，用 get_usage_metadata_callback 收集所有 LLM 调用的 token 统计 ----
    #   OpenAI service account key 在高并发时偶发 401，SDK 默认不重试 401，这里手动兜底

    start = time.time()
    last_exc: Exception | None = None
    result = None
    cb_usage: dict[str, UsageMetadata] = {}

    for attempt in range(1, _RETRYABLE_MAX_ATTEMPTS + 1):
        try:
            with get_usage_metadata_callback() as cb:
                invoke_kwargs: dict[str, Any] = {}
                if tool_context is not None:
                    invoke_kwargs["context"] = tool_context
                # opt-in:调高 ReAct 递归上限(默认 25=langgraph 默认)。弱模型(flash)工具循环多,
                # 25 步会在产出最终答案前被截断→末条 AIMessage 只剩 tool_calls、content 空。
                import os as _os_rl
                _rl = _os_rl.environ.get("AGENT_RECURSION_LIMIT")
                if _rl:
                    invoke_kwargs["recursion_limit"] = int(_rl)
                result = await asyncio.wait_for(agent.ainvoke({"messages": messages}, **invoke_kwargs), timeout=timeout)
            cb_usage = cb.usage_metadata
            break
        except Exception as exc:
            # 仅对 OpenAI 401 AuthenticationError 重试（偶发性平台问题）
            from openai import AuthenticationError as OpenAIAuthError

            if isinstance(exc, OpenAIAuthError) and attempt < _RETRYABLE_MAX_ATTEMPTS:
                delay = _RETRYABLE_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "[do_execute] OpenAI 401 (attempt %d/%d), retrying in %.1fs: %s",
                    attempt,
                    _RETRYABLE_MAX_ATTEMPTS,
                    delay,
                    exc,
                )
                last_exc = exc
                await asyncio.sleep(delay)
                continue
            raise

    elapsed_ms = int((time.time() - start) * 1000)

    # ---- 6. 从 result 中提取内容 ----

    # 6a. 取最后一条 AIMessage 的文本内容
    content = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            raw = msg.content
            if isinstance(raw, str):
                content = raw
            elif isinstance(raw, list):
                # thinking 模型返回 list[dict]，只取 text 部分
                content = "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in raw)
            break

    # 6b. 结构化输出（create_agent 自动解析到 structured_response）
    data = result.get("structured_response")

    # 6c. 提取 tool call 记录（ReAct 循环: AIMessage.tool_calls → ToolMessage 配对）
    tool_call_records: list[ToolCallRecord] = []
    if tools:
        from langchain_core.messages import ToolMessage

        # 建立 tool_call_id → ToolMessage.content 映射
        tool_results: dict[str, str] = {}
        for msg in result["messages"]:
            if isinstance(msg, ToolMessage):
                tool_results[msg.tool_call_id] = msg.content if isinstance(msg.content, str) else str(msg.content)

        # 从 AIMessage.tool_calls 提取调用记录并配对结果
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_call_records.append(
                        ToolCallRecord(
                            name=tc.get("name", ""),
                            args=tc.get("args", {}),
                            result=tool_results.get(tc.get("id", ""), ""),
                            call_id=tc.get("id", ""),
                        )
                    )

    # ---- 7. usage 直接取 callback 收集的 usage_metadata ----

    return ExecuteResult(
        usage=cb_usage,
        content=content,
        data=data,
        tool_calls=tool_call_records,
        time_cost=elapsed_ms,
    )


def accumulate_usage(
    accumulated: UsageMetadata,
    usage: dict[str, UsageMetadata] | None,
) -> UsageMetadata:
    """将 do_execute 返回的 usage 累加到已有统计中（保留 input_token_details）

    Args:
        accumulated: 已有的累计统计
        usage: do_execute 返回的 {model_name: UsageMetadata}

    Returns:
        累加后的 UsageMetadata（含 input_token_details / output_token_details）
    """
    if not usage:
        return accumulated
    for model_usage in usage.values():
        # 基础 token 累加
        new = UsageMetadata(
            input_tokens=accumulated["input_tokens"] + model_usage.get("input_tokens", 0),
            output_tokens=accumulated["output_tokens"] + model_usage.get("output_tokens", 0),
            total_tokens=accumulated["total_tokens"] + model_usage.get("total_tokens", 0),
        )
        # 累加 input_token_details（cache_read / cache_creation）
        old_in = accumulated.get("input_token_details") or {}
        new_in = model_usage.get("input_token_details") or {}
        if old_in or new_in:
            new["input_token_details"] = {k: old_in.get(k, 0) + new_in.get(k, 0) for k in set(old_in) | set(new_in)}
        # 累加 output_token_details（reasoning）
        old_out = accumulated.get("output_token_details") or {}
        new_out = model_usage.get("output_token_details") or {}
        if old_out or new_out:
            new["output_token_details"] = {
                k: old_out.get(k, 0) + new_out.get(k, 0) for k in set(old_out) | set(new_out)
            }
        accumulated = new
    return accumulated
