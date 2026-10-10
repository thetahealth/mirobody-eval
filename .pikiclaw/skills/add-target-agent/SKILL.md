---
name: add-target-agent
description: Scaffold a new TargetAgent plugin (config + implementation + registration).
argument-hint: [agent-name]
---

# Add TargetAgent Plugin

Create a new TargetAgent plugin with all required boilerplate: Pydantic config, implementation class, and plugin registration.

## Framework Protection（必读）

本 skill 仅通过**插件扩展点**添加功能，**严禁修改框架核心逻辑**。文件按修改权限分为三级：

### 🟢 Plugin 层 — 自由新建

| 文件 | 操作 |
|------|------|
| `evaluator/plugin/target_agent/<name>_target_agent.py` | 创建新文件（plugin 实现） |
| `evaluator/plugin/target_agent/__init__.py` | **不要改**：pkgutil 按 `_target_agent.py` 后缀自动导入，无 import 清单也无 `__all__` |

### 🟡 Schema 扩展点 — 仅限追加

`evaluator/core/schema.py` **不需要改**。`TargetInfo` 是
`Annotated[Any, BeforeValidator(_validate_target_info)]`，运行时按 `type` 字段从
`AbstractTargetAgent` 的注册表取出该插件的 `params_model`。配置类写在插件文件里，
用 `params_model=` 注册即可。

**禁止的操作**：
- 修改 `evaluator/core/schema.py` 的任何分发逻辑
- 修改其他插件的类定义

### 🔴 框架核心 — 严禁修改

以下文件为框架核心，任何修改都可能破坏全局功能：

- `evaluator/core/orchestrator.py` — 编排引擎
- `evaluator/core/bench_schema.py` — Benchmark 数据模型（resolve_effective_target / bench_item_to_test_case 等）
- `evaluator/core/interfaces/abstract_*.py` — 抽象基类
- `evaluator/utils/*.py` — 通用工具层（llm, benchmark_reader, report_reader, agent_inspector, config）
- `benchmark/basic_runner.py` — 跑分执行器
- `web/` — Web UI（通过 agent_inspector 自动适配新 plugin）

> **如果你发现需要修改 🔴 文件才能完成需求，请停下来通知用户** — 这通常意味着需求理解有误，或框架需要由维护者升级扩展点。

## Auto-Adaptation

完成以下步骤后，Web UI / CLI 会自动适配新 plugin：
- **Config Schema**: `agent_inspector` 从注册表里各插件的 `params_model` 自动派生 config map，无需手动维护
- **resolve_effective_target()**: 按 `type` 走注册表分发，新 target 类型的 per-case overrides 开箱即用
- **展示元数据**: 新 plugin 默认使用通用图标/颜色，可通过 `_display_meta` 类属性自定义（可选）

## Workflow

### Step 1: Gather Requirements

Ask the user (via AskUserQuestion) for the following if not provided in $ARGUMENTS:

1. **Plugin name** (snake_case, e.g. `openai_api`) — used as:
   - `__init_subclass__` registration name
   - `type: Literal["<name>"]` discriminator value
   - File name: `evaluator/plugin/target_agent/<name>_target_agent.py`
   - Class name: `<PascalCase>TargetAgent`

2. **Target system description** — what system this agent connects to and how (HTTP API, SDK, CLI, etc.)

3. **Config fields** — what parameters the user needs to provide in test case JSON (beyond the standard `type` field)

4. **Connection lifecycle** — whether the agent needs session management / cleanup

### Step 2: Validate Constraints

Before generating code, verify:

- [ ] Name is unique — check with `python -c "import evaluator.plugin.target_agent; from evaluator.core.interfaces.abstract_target_agent import AbstractTargetAgent; print(sorted(AbstractTargetAgent.get_all()))"`
- [ ] Name is valid snake_case identifier (lowercase, underscores only, no leading digits)
- [ ] Config field names don't shadow Pydantic reserved names

### Step 3: Define the config model **in the plugin file**

Add the Pydantic config model at the top of
`evaluator/plugin/target_agent/<name>_target_agent.py`:

- Use `model_config = ConfigDict(extra="forbid")`
- Include `type: Literal["<name>"]`
- Add `json_schema_extra.examples` with at least one minimal and one full example
- Add docstring in Chinese explaining the target system

Then register it with the class:

```python
class MyTargetAgent(AbstractTargetAgent, name="my_target", params_model=MyTargetInfo):
    ...
```

`evaluator/core/schema.py` needs no edit — it resolves the model from the registry.

> **Important**: 类上传了 `params_model=` 之后：
> - `agent_inspector` 会自动发现其 config schema 并在 Web UI 展示
> - `resolve_effective_target()` 会自动支持新类型的 per-case overrides 合并
> - 无需修改 `agent_inspector.py` 或 `bench_schema.py`

### Step 4: Create Implementation File

Create `evaluator/plugin/target_agent/<name>_target_agent.py` following this template:

```python
"""
<PascalCase>TargetAgent — <中文描述>

注册名称: "<name>"

<详细说明连接方式和交互协议>
"""

import logging
from typing import Optional

from evaluator.core.interfaces.abstract_target_agent import AbstractTargetAgent
from evaluator.core.schema import (
    TargetAgentReaction,
    TestAgentAction,
)

# 配置类就定义在本文件里（见 Step 3），不要从 schema.py import

logger = logging.getLogger(__name__)


class <PascalCase>TargetAgent(AbstractTargetAgent, name="<name>", params_model=<PascalCase>TargetInfo):
    """<中文一句话描述>"""

    # 可选：自定义 Web UI 展示元数据（不声明则使用默认图标/颜色）
    # _display_meta = {"color": "#6366f1", "features": ["HTTP API", "自定义协议"]}

    # 费用预估元数据 — 声明单次调用 token 量，实际费用 = tokens × 用户选择模型定价
    _cost_meta = {
        "est_input_tokens": 200,   # 预估单次调用输入 token
        "est_output_tokens": 600,  # 预估单次调用输出 token
    }

    def __init__(self, target_config: <PascalCase>TargetInfo):
        super().__init__(target_config)
        self.config: <PascalCase>TargetInfo = target_config
        # Initialize connection state here

    async def _generate_next_reaction(
        self, test_action: Optional[TestAgentAction]
    ) -> TargetAgentReaction:
        """将用户输入发送到被测系统，返回系统响应

        Args:
            test_action: 用户动作（首轮可能为 None）

        Returns:
            TargetAgentReaction: 被测系统响应
        """
        # 1. Extract user message from test_action
        user_text = ""
        if test_action and test_action.semantic_content:
            user_text = test_action.semantic_content

        # 2. Call target system
        # response = await self._call_system(user_text)

        # 3. Build and return reaction
        return TargetAgentReaction(
            type="message",
            message_list=[{"content": "TODO: implement"}],
        )

    async def cleanup(self):
        """释放连接资源（对话结束后由 orchestrator 调用）"""
        # Close HTTP sessions, SDK clients, etc.
        pass
```

Key patterns to follow (from `hermes_target_agent.py`):
- **Lazy initialization**: First call can trigger auth/setup
- **Response format**: Return `TargetAgentReaction(type="message", message_list=[{"content": "..."}])` for text responses
- **Cleanup**: Implement `async cleanup()` if holding network connections or sessions
- **Config from env**: Use `.env` for infrastructure params (base_url, timeout), keep only business params in config model
- **Error handling**: Log errors clearly, raise exceptions that orchestrator can catch

### Step 5: Register Plugin

**没有手工注册这一步。** `evaluator/plugin/target_agent/__init__.py` 用 `pkgutil` 遍历本包，
自动 import 所有文件名以 `_target_agent.py` 结尾的模块，从而触发 `__init_subclass__` 注册。
该文件里既没有 import 清单也没有 `__all__` —— **文件名即注册**，类上传 `params_model=` 即可。

> 该处的 `except ImportError: pass` 会吞掉依赖缺失，所以插件没出现时直接单独 import
> 该模块看真实报错。模块级的 SyntaxError / NameError 不会被吞，会炸掉整个包导入。

### Step 6: Verify

Run the following to verify:
```bash
# 插件注册检查
uv run python -c "import evaluator.plugin.target_agent; from evaluator.core.interfaces.abstract_target_agent import AbstractTargetAgent; print(sorted(AbstractTargetAgent.get_all()))"

# Web UI schema 自动发现检查
uv run python -c "from evaluator.utils.agent_inspector import list_target_agents; print([(a.name, list(a.config_schema.get('properties', {}).keys())) for a in list_target_agents()])"

# Lint
ruff check evaluator/plugin/target_agent/
ruff format evaluator/core/schema.py evaluator/plugin/target_agent/
```

### Step 7: Update Documentation

新增 TargetAgent 后，更新以下文档中的被测系统列表/表格，保持信息同步：

| 文件 | 更新内容 |
|------|---------|
| `web/guides/develop-target-agent.md` | 「现有被测系统」表格追加新行 + 「关键文件」表格追加参考实现 |
| `web/guides/overview.md` | 「对接服务系统」章节补充说明（如有新的接入模式） |
| `CLAUDE.md` | Plugin System 表格中 TargetAgent 行追加新名称 |
| `README.md` | 「已注册插件」表格 TargetAgent 区域追加新行 |

> 每个文件只需追加一行到已有表格，不要修改其他内容。

## Optional: Custom Display & Cost Metadata

新 plugin 默认使用灰色图标和空特性标签。若需自定义 Web UI 展示，在 plugin 类上声明 `_display_meta`:

```python
class <PascalCase>TargetAgent(AbstractTargetAgent, name="<name>", params_model=<PascalCase>TargetInfo):
    _display_meta = {
        "icon": "M5.25 14.25h13.5m...",  # heroicons SVG path (24x24 viewBox)
        "color": "#6366f1",               # CSS 颜色值
        "features": ["HTTP API", "自定义协议"],  # 特性标签
    }
```

费用预估通过 `_cost_meta` 声明，前端自动读取并结合用户选择的模型定价计算预估费用：

```python
    _cost_meta = {
        "est_input_tokens": 200,   # 预估单次调用输入 token（含 system prompt + 上下文）
        "est_output_tokens": 600,  # 预估单次调用输出 token
    }
```

- LLM 类 target（如 `llm_api`）：声明合理的 token 估算值
- 外部 API 类 target（如 `hermes`）：token 填 0（不按 token 计费）

`_display_meta` 和 `_cost_meta` 中的字段均为可选，未指定的使用默认值。

## Reference: Existing TargetAgent Plugins

| Name | Config | File | Description |
|------|--------|------|-------------|
| `llm_api` | `LlmApiTargetInfo` | `llm_api_target_agent.py` | Generic LLM API (OpenAI/Gemini via do_execute) |
| `hermes` | `HermesTargetInfo` | `hermes_target_agent.py` | External HTTP service (own session/token handling) |
| `evermem` | `EvermemTargetInfo` | `evermem_target_agent.py` | EverMem memory service |
| `mem0_rag_api` | `Mem0RagApiTargetInfo` | `mem0_rag_api_target_agent.py` | Mem0 RAG service |
| `naive_rag_api` | `NaiveRagApiTargetInfo` | `naive_rag_api_target_agent.py` | Baseline RAG |
| `hippo_rag_api` | `HippoRagApiTargetInfo` | `hippo_rag_api_target_agent.py` | HippoRAG service |
| `dyg_rag_api` | `DygRagApiTargetInfo` | `dyg_rag_api_target_agent.py` | DyG-GraphRAG service |

## TargetAgentReaction Types

The response must be one of:
- `type="message"` + `message_list=[{"content": "..."}]` — text response (most common)
- `type="gui"` + `gui_snapshots=["..."]` — GUI screenshot response
- `type="custom"` + `custom_content={...}` — custom structured response
