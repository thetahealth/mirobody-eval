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


# provider → (api key 环境变量, base_url 覆盖变量, 默认 base_url)
#
# 与被测引擎 mirobody 同一张表（`mirobody/utils/config/llm.py` 的 `_OPENAI_COMPAT`），
# 刻意保持一致:同一个人拿同一个 key，在引擎侧和评测侧不该需要两套写法。
#
# 用法是模型名加 `provider:` 前缀 —— `dashscope:qwen3.5-flash`。不加前缀时行为
# 完全不变（见下面 do_execute 里的前缀规则），所以既有的报告和脚本不受影响。
#
# 为什么用冒号而不是 `provider/model`:OpenRouter 的模型 id 本身就是 `vendor/model`
# 形状（`openai/gpt-5-mini`、`anthropic/claude-sonnet-5`），拿斜杠当分隔符会把这些
# 名字抢走、路由到错误的端点。
_OPENAI_COMPAT: dict[str, tuple[str, str, str]] = {
    "openrouter": ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    "dashscope": (
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    "volcengine": (
        "VOLCENGINE_API_KEY",
        "VOLCENGINE_BASE_URL",
        "https://ark.cn-beijing.volces.com/api/v3",
    ),
    "zhipu": ("ZHIPU_API_KEY", "ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
    "moonshot": ("MOONSHOT_API_KEY", "MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1"),
}

#: 已经为哪些模型打过「改从 OpenRouter 走」的提示。进程级去重:这条提示是给人看
#: 「你的请求没去你以为的地方」，说一次就够,每次调用都说会把日志刷爆。
_REROUTE_NOTIFIED: set[str] = set()

#: 走各自原生 SDK、不套 OpenAI 兼容层的两个。写成 `openai:` / `google:` 是为了
#: 能显式压过前缀规则 —— 比如 `openai:gpt-oss-120b` 才是真的要打 OpenAI，而裸的
#: `gpt-oss-120b` 按前缀规则也会被判成 OpenAI，却往往并不是本意。
_NATIVE_PREFIXES = {"openai", "google"}


def _split_provider_prefix(model: str) -> str:
    """取出 `provider:` 前缀里的 provider 名，没有则返回空串。

    只认冒号出现在第一个斜杠**之前**的情况:OpenRouter 用后缀冒号表达变体
    （`anthropic/claude-sonnet-4.5:batch`），那种冒号不是 provider 分隔符。

    前缀不认识时直接报错，不静默退回 OpenRouter —— 打错一个 provider 名而被
    默默发去另一个端点，症状会是一个和拼写毫无关系的鉴权失败。
    """
    if model.startswith("["):  # 自建网关语法，整体保持原样
        return ""
    colon = model.find(":")
    if colon <= 0:
        return ""
    slash = model.find("/")
    if slash != -1 and slash < colon:
        return ""
    head = model[:colon]
    if head in _OPENAI_COMPAT or head in _NATIVE_PREFIXES:
        return head
    known = ", ".join(sorted(_OPENAI_COMPAT) + sorted(_NATIVE_PREFIXES))
    raise ValueError(f"模型名 {model!r} 的 provider 前缀 {head!r} 不认识。可用: {known}")


def _relax_vertex_env() -> None:
    """让 Gemini 走 AI Studio 直连，而不是需要 ADC 凭证的 Vertex。

    Remote config 在生产容器里会推 `GOOGLE_GENAI_USE_VERTEXAI=true`，而本地评测
    通常只有 `GOOGLE_API_KEY`、没有 gcloud ADC，那条路径会被 Vertex 端点拒掉。
    显式设了 `GOOGLE_APPLICATION_CREDENTIALS` 时不动 —— 那是用户主动选择 Vertex。
    """
    import os as _os

    if _os.environ.get("GOOGLE_API_KEY") and not _os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        for _k in ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"):
            _os.environ.pop(_k, None)


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
        model: 模型名称，如 "gpt-5.2" / "gemini-3-pro-preview" / "anthropic/claude-3.7-sonnet"
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

    # `provider:model` 显式指定（如 `dashscope:qwen3.5-flash`）。见 _OPENAI_COMPAT
    # 的说明；命中时直接拿到端点和 key 变量，不再靠模型名前缀猜。
    compat: tuple[str, str, str] | None = None
    forced_native = ""
    head = _split_provider_prefix(model)
    if head:
        forced_native_map = {"openai": "openai", "google": "google_genai"}
        model = model[len(head) + 1 :]
        if head in _OPENAI_COMPAT:
            compat = _OPENAI_COMPAT[head]
        else:
            forced_native = forced_native_map[head]

    # 方括号前缀的模型名走「自带的 OpenAI 兼容网关」(vLLM / LiteLLM / 自建中转均可):
    # base_url/key 取自 HOLYEVAL_GATEWAY_BASE_URL / HOLYEVAL_GATEWAY_API_KEY(见下 init_kwargs)。
    # 用独立变量而不是复用 GEMINI_API_KEY —— 后者是真正的 Google key,复用会把它发给任意第三方。
    use_gateway = False
    if compat is not None:
        # 显式 provider —— OpenAI 兼容协议，端点来自 _OPENAI_COMPAT
        provider = "openai"
        use_openrouter = False
    elif forced_native == "google_genai":
        provider = "google_genai"
        use_openrouter = False
        _relax_vertex_env()
    elif forced_native == "openai":
        provider = "openai"
        use_openrouter = False
    elif model.startswith("["):
        provider = "openai"
        use_openrouter = False
        use_gateway = True
    elif model.startswith("gpt"):
        # OpenAI 原生模型
        provider = "openai"
        use_openrouter = False
    elif model.startswith("gemini"):
        # Google Gemini 模型
        provider = "google_genai"
        use_openrouter = False
        _relax_vertex_env()
    else:
        # 其他模型通过 OpenRouter 调用（如 anthropic/claude-3.7-sonnet）
        provider = "openai"
        use_openrouter = True

    # 按可用的 key 兜住那些「默认模型假设你有某个 key」的情况。
    #
    # 七个判分器各自的默认模型全是 OpenAI 家族（gpt-4.1 / gpt-5.2 / gpt-5.4-mini），
    # 所以只配了 OPENROUTER_API_KEY 的用户跑任何默认配置都会拿到一个 `OpenAIError`
    # —— 一个完全指不到真实原因的报错（key 是对的，只是模型名的写法决定了它被发去
    # 哪个端点）。这里把同一个模型改从 OpenRouter 走:`openai/<name>` 在 OpenRouter
    # 上就是同一个模型，命名是稳定约定。
    #
    # 只对前缀规则判出来的 OpenAI 生效。显式写 `openai:` 的不动 —— 那是用户主动
    # 要求打官方端点，替他改道就成了违背指令。
    #
    # Gemini 刻意不做同样的事:OpenRouter 上的 Gemini 命名不是 1:1（本仓库文档里
    # 点名的 `gemini-3-pro-preview` 在那边就不存在），自动改道会变成 404。
    if provider == "openai" and not use_openrouter and not use_gateway and not forced_native and compat is None:
        import os

        if not os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENROUTER_API_KEY"):
            rerouted = f"openai/{model}"
            # 每个模型只提示一次。判分器对每条 rubric 都调一次模型 —— 实测 2 条用例
            # 就刷了 29 行一模一样的 WARNING，跑全量会是上千行,把真正该看的日志淹掉。
            if model not in _REROUTE_NOTIFIED:
                _REROUTE_NOTIFIED.add(model)
                logger.warning(
                    "模型 %r 需要 OPENAI_API_KEY(未配置)，改从 OpenRouter 取同一模型: %r。"
                    "想固定这个行为就直接写后者;想打 OpenAI 官方请配 OPENAI_API_KEY。",
                    model,
                    rerouted,
                )
            model = rerouted
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
    if provider == "openai" and not use_openrouter and compat is None and thinking_level is not None and tools:
        init_kwargs["use_responses_api"] = True

    # 显式 provider（`dashscope:` 等）—— 端点和 key 都来自 _OPENAI_COMPAT
    if compat is not None:
        import os

        key_env, base_env, default_base = compat
        api_key = os.getenv(key_env)
        if not api_key:
            raise RuntimeError(
                f"模型 {head + ':' + model!r} 要用 {key_env}，但该环境变量没有设置。"
                f"在 .env 里配上它，或换一个已配好 key 的 provider 前缀。"
            )
        init_kwargs["base_url"] = os.getenv(base_env) or default_base
        init_kwargs["api_key"] = api_key
        # `init_kwargs["model"]` 不用再设:前缀在上面第 1 步就从 `model` 剥掉了，
        # 构建 init_kwargs 时取到的已经是真实模型 id。

    # OpenRouter 配置（非 gpt/gemini 模型）
    if use_openrouter:
        import os

        init_kwargs["base_url"] = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        init_kwargs["api_key"] = os.getenv("OPENROUTER_API_KEY")

    # 自建 OpenAI 兼容网关(方括号前缀的模型名)
    if use_gateway:
        import os

        base = os.getenv("HOLYEVAL_GATEWAY_BASE_URL")
        if not base:
            raise RuntimeError(
                f"模型 {model!r} 以 '[' 开头，表示走自带的 OpenAI 兼容网关，"
                "但环境变量 HOLYEVAL_GATEWAY_BASE_URL 未设置。"
            )
        init_kwargs["base_url"] = base
        init_kwargs["api_key"] = os.getenv("HOLYEVAL_GATEWAY_API_KEY", "")

    # Let the Google SDK resolve API keys and Vertex ADC (including gcloud and
    # workload credentials); environment-variable checks cannot enumerate them.
    from google.auth.exceptions import DefaultCredentialsError

    try:
        llm = init_chat_model(**init_kwargs)
    except (ValueError, DefaultCredentialsError) as exc:
        if provider != "google_genai":
            raise
        raise RuntimeError(
            f"Google model {model!r} could not be initialized: {exc}. "
            "Use GOOGLE_API_KEY or GEMINI_API_KEY for Gemini, or configure Vertex AI "
            "with Application Default Credentials. To use OpenRouter instead, select "
            "its published model ID explicitly (for example google/gemini-2.5-pro)."
        ) from exc

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
