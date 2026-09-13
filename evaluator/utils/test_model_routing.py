"""模型名决定请求发去哪个端点 —— 这层的错会伪装成鉴权失败。

纯逻辑，不连网、不调模型。

    pytest evaluator/utils/test_model_routing.py -v

锁的是一个实测出来的事故形状：README 的四条快速开始命令都用 `gpt-5.4-mini`，
按前缀规则打 api.openai.com。而只配了 `OPENROUTER_API_KEY` 的用户（mirobody 自己
主推的单 key 方案）跑下来拿到的是 `OpenAIError` —— key 是对的，模型名的写法才是
原因，而报错完全指不到那里。

三组断言：

  1. 新增的 `provider:model` 显式写法能解析，且不吃掉三种既有语法
  2. 缺 key 时报错要指得到原因
  3. 只有 OpenRouter key 时，OpenAI 家族的默认模型仍然可用
"""

from __future__ import annotations

import pytest

from evaluator.utils.llm import _OPENAI_COMPAT, _split_provider_prefix

# ---------------------------------------------------------------------------
# 1. 显式 provider 前缀，且不与既有语法冲突
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,expected",
    [
        # 新语法
        ("dashscope:qwen3.5-flash", "dashscope"),
        ("deepseek:deepseek-chat", "deepseek"),
        ("moonshot:kimi-k2", "moonshot"),
        ("zhipu:glm-4", "zhipu"),
        ("volcengine:doubao-pro", "volcengine"),
        ("openrouter:openai/gpt-5-mini", "openrouter"),
        ("openai:gpt-oss-120b", "openai"),
        ("google:gemini-3-pro", "google"),
    ],
)
def test_an_explicit_provider_prefix_is_recognised(model, expected):
    assert _split_provider_prefix(model) == expected


@pytest.mark.parametrize(
    "model",
    [
        # 前缀规则原样保留
        "gpt-5.4-mini",
        "gemini-3-pro-preview",
        # OpenRouter 的模型 id 就是 `vendor/model` 形状 —— 拿斜杠当分隔符会把它们
        # 抢走。这是选冒号而不选斜杠的全部原因，所以必须锁住。
        "openai/gpt-5-mini",
        "anthropic/claude-sonnet-5",
        # OpenRouter 用后缀冒号表达变体，那个冒号在斜杠之后，不是 provider 分隔符
        "anthropic/claude-sonnet-4.5:batch",
        "google/gemini-2.5-flash:batch",
        # 自建网关语法
        "[myvllm]llama-3.1-70b",
    ],
)
def test_existing_model_name_shapes_are_left_alone(model):
    assert _split_provider_prefix(model) == "", (
        f"{model!r} 被当成带 provider 前缀了 —— 既有写法会被改道到错误的端点"
    )


def test_a_misspelled_provider_fails_loudly():
    """打错 provider 名不能静默退回 OpenRouter。

    静默退回的话，症状是一个和拼写毫无关系的鉴权失败 —— 而这正是本轮修复要消除
    的那类误导。
    """
    with pytest.raises(ValueError) as e:
        _split_provider_prefix("dashscop:qwen3.5-flash")  # 少一个 e

    assert "dashscop" in str(e.value)
    assert "dashscope" in str(e.value), "报错要列出可用的 provider，否则用户没法自己改对"



def test_the_provider_table_matches_the_engine():
    """与被测引擎 mirobody 的 `_OPENAI_COMPAT` 同一张表。

    同一个人拿同一个 key，在引擎侧和评测侧不该需要两套写法。表分叉了，两边就会
    对同一个模型给出不同的写法要求 —— 那是 bug 的温床。
    """
    expected = {"openrouter", "dashscope", "deepseek", "volcengine", "zhipu", "moonshot"}

    assert set(_OPENAI_COMPAT) == expected
    for name, (key_env, base_env, default_base) in _OPENAI_COMPAT.items():
        assert key_env.endswith("_API_KEY"), name
        assert base_env.endswith("_BASE_URL"), name
        assert default_base.startswith("https://"), name


# ---------------------------------------------------------------------------
# 2. 缺 key 的报错要指得到原因
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_provider_without_its_key_names_the_variable(monkeypatch):
    from evaluator.utils.llm import do_execute

    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)

    with pytest.raises(RuntimeError) as e:
        await do_execute(model="moonshot:kimi-k2", system_prompt="x", input="y")

    assert "MOONSHOT_API_KEY" in str(e.value), "报错必须点名缺哪个环境变量"


@pytest.mark.asyncio
async def test_gemini_without_a_google_key_says_so_instead_of_guessing(monkeypatch):
    """Gemini 刻意不自动改道去 OpenRouter。

    那边的 Gemini 命名不是 1:1（本仓库文档点名的 `gemini-3-pro-preview` 在
    OpenRouter 上不存在），自动改道会把「缺 key」变成更难查的 404。
    """
    from evaluator.utils import llm

    calls = []

    def reject_missing_key(**kwargs):
        calls.append(kwargs)
        raise ValueError("Missing key inputs argument")

    monkeypatch.setattr(llm, "init_chat_model", reject_missing_key)

    with pytest.raises(RuntimeError) as e:
        await llm.do_execute(model="gemini-3-pro-preview", system_prompt="x", input="y")

    msg = str(e.value)
    assert "GOOGLE_API_KEY" in msg
    assert "GEMINI_API_KEY" in msg
    assert "google/gemini" in msg, "配了 OpenRouter 的人应当被告知那边的命名不同"
    assert calls[0]["model_provider"] == "google_genai"
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# 3. 只有 OpenRouter key 时，OpenAI 家族的默认模型仍可用
# ---------------------------------------------------------------------------


def test_every_evaluator_default_model_is_covered_by_the_reroute():
    """七个判分器的默认模型全是 OpenAI 家族 —— 这就是「只有 OpenRouter 就全废」
    的根源。

    源码级断言:真调一遍要花钱且要联网。这里锁的是「默认值仍然落在改道能兜住的
    那一族里」—— 哪天有人把某个默认值改成别的家族，这条会失败并提醒他确认改道
    覆盖到了。
    """
    import re
    from pathlib import Path

    agents = Path(__file__).resolve().parents[1] / "plugin" / "eval_agent"
    defaults: dict[str, str] = {}
    for py in agents.glob("*_eval_agent.py"):
        src = py.read_text(encoding="utf-8")
        for m in re.finditer(r'(?:DEFAULT_(?:JUDGE_)?MODEL\s*=|default=)\s*"([^"]+)"', src):
            v = m.group(1)
            if re.match(r"^(gpt|gemini|o\d|claude|deepseek|qwen)", v):
                defaults[py.name] = v

    assert defaults, "一个判分器默认模型都没找到 —— 断言的前提没了，得改这条测试"
    for fname, model in defaults.items():
        assert model.startswith("gpt"), (
            f"{fname} 的默认模型是 {model!r}，不在自动改道覆盖的 OpenAI 家族里。"
            f"确认只配 OPENROUTER_API_KEY 的用户还能用这个默认值，再更新本断言。"
        )


class _ReplyAgent:
    async def ainvoke(self, *args, **kwargs):
        from langchain_core.messages import AIMessage

        return {"messages": [AIMessage(content="test reply")]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,provider,model_id,base_url",
    [
        ("dashscope:qwen3.5-flash", "openai", "qwen3.5-flash", _OPENAI_COMPAT["dashscope"][2]),
        ("deepseek:deepseek-chat", "openai", "deepseek-chat", _OPENAI_COMPAT["deepseek"][2]),
        ("moonshot:kimi-k2", "openai", "kimi-k2", _OPENAI_COMPAT["moonshot"][2]),
        ("zhipu:glm-4", "openai", "glm-4", _OPENAI_COMPAT["zhipu"][2]),
        ("volcengine:my-endpoint", "openai", "my-endpoint", _OPENAI_COMPAT["volcengine"][2]),
        ("openrouter:openai/gpt-5-mini", "openai", "openai/gpt-5-mini", _OPENAI_COMPAT["openrouter"][2]),
        ("openai/gpt-5-mini", "openai", "openai/gpt-5-mini", _OPENAI_COMPAT["openrouter"][2]),
        ("openai:gpt-5.4-mini", "openai", "gpt-5.4-mini", None),
        ("google:gemini-3-pro-preview", "google_genai", "gemini-3-pro-preview", None),
        ("gpt-5.4-mini", "openai", "gpt-5.4-mini", None),
    ],
)
async def test_target_model_reaches_the_selected_provider(monkeypatch, model, provider, model_id, base_url):
    """Exercise case validation and target execution, without an external API call."""
    import evaluator.plugin.eval_agent  # noqa: F401
    import evaluator.plugin.test_agent  # noqa: F401
    from evaluator.core.bench_schema import bench_item_to_test_case, find_target_spec
    from evaluator.core.schema import TestAgentAction
    from evaluator.plugin.target_agent.llm_api_target_agent import LlmApiTargetAgent
    from evaluator.utils import llm
    from evaluator.utils.benchmark_reader import load_benchmark

    for key, base, _ in _OPENAI_COMPAT.values():
        monkeypatch.setenv(key, "test-placeholder")
        monkeypatch.delenv(base, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-placeholder")
    calls = []

    def capture_client(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(llm, "init_chat_model", capture_client)
    monkeypatch.setattr(llm, "create_agent", lambda **kwargs: _ReplyAgent())
    bench = load_benchmark("healthbench", "sample")
    spec = find_target_spec(bench.target, "llm_api")
    case = bench_item_to_test_case(bench.items[0], spec, {"model": model})
    reply = await LlmApiTargetAgent(case.target).do_generate(
        TestAgentAction(type="message", message_content={"content": "test question"})
    )
    assert reply.extract_text() == "test reply"
    assert len(calls) == 1
    assert calls[0]["model"] == model_id
    assert calls[0]["model_provider"] == provider
    assert calls[0].get("base_url") == base_url


def test_model_schema_allows_free_text_for_the_web_form():
    from pydantic import ValidationError

    from evaluator.plugin.target_agent.llm_api_target_agent import LlmApiTargetInfo
    from evaluator.utils.agent_inspector import _extract_config_info

    schema, _ = _extract_config_info(LlmApiTargetInfo)
    model_field = schema["properties"]["model"]
    assert model_field["type"] == "string"
    assert "enum" not in model_field
    for value in ("", "  "):
        with pytest.raises(ValidationError):
            LlmApiTargetInfo(type="llm_api", model=value)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gemini-3-pro-preview", "google:gemini-3-pro-preview"])
@pytest.mark.parametrize("credential_mode", ["gemini_key", "vertex_adc"])
async def test_google_authentication_is_resolved_by_the_sdk(monkeypatch, model, credential_mode):
    from langchain_google_genai import ChatGoogleGenerativeAI

    from evaluator.utils import llm

    for key in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_GENAI_USE_VERTEXAI"):
        monkeypatch.delenv(key, raising=False)
    calls = []
    if credential_mode == "gemini_key":
        monkeypatch.setenv("GEMINI_API_KEY", "test-placeholder")
    else:
        # ADC need not be named by GOOGLE_APPLICATION_CREDENTIALS: it can come
        # from gcloud login or workload identity, both resolved inside the SDK.
        monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")

    def initialize(**kwargs):
        calls.append(kwargs)
        assert kwargs["model_provider"] == "google_genai"
        if credential_mode == "gemini_key":
            client = ChatGoogleGenerativeAI(model=kwargs["model"])
            assert client.google_api_key.get_secret_value() == "test-placeholder"
            return client
        return object()  # Isolate ADC discovery from the machine running CI.

    monkeypatch.setattr(llm, "init_chat_model", initialize)
    monkeypatch.setattr(llm, "create_agent", lambda **kwargs: _ReplyAgent())
    result = await llm.do_execute(model=model, system_prompt="test", input="test", timeout=1)
    assert result.content == "test reply"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_compatible_provider_keeps_chat_completions_with_tools(monkeypatch):
    from evaluator.utils import llm

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-placeholder")
    calls = []
    monkeypatch.setattr(llm, "init_chat_model", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(llm, "create_agent", lambda **kwargs: _ReplyAgent())
    await llm.do_execute(
        model="dashscope:qwen3.5-flash", system_prompt="test", input="test",
        tools=[object()], thinking_level="low", timeout=1,
    )
    assert not calls[0].get("use_responses_api", False)
