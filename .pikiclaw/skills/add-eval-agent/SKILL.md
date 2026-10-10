---
name: add-eval-agent
description: Scaffold a new EvalAgent plugin (config + implementation + registration).
argument-hint: [agent-name]
---

# Add EvalAgent Plugin

Create a new EvalAgent plugin with all required boilerplate: Pydantic config, implementation class, and plugin registration.

## Framework Protection（必读）

本 skill 仅通过**插件扩展点**添加功能，**严禁修改框架核心逻辑**。文件按修改权限分为三级：

### 🟢 Plugin 层 — 自由新建

| 文件 | 操作 |
|------|------|
| `evaluator/plugin/eval_agent/<name>_eval_agent.py` | 创建新文件（plugin 实现） |
| `evaluator/plugin/eval_agent/__init__.py` | **不要改**：pkgutil 按 `_eval_agent.py` 后缀自动导入，无 import 清单也无 `__all__` |

### 🟢 Schema — 无需修改

`evaluator/core/schema.py` 中的 `EvalInfo` 采用 `BeforeValidator` 动态分发机制，从 `AbstractEvalAgent._params_registry` 自动路由到对应插件的配置模型。**新增 EvalAgent 无需修改 schema.py**。

配置模型直接定义在插件文件中，通过 `params_model` 参数自动注册：
```python
class MyEvalInfo(BaseModel):
    evaluator: Literal["my_eval"] = ...
    ...

class MyEvalAgent(AbstractEvalAgent, name="my_eval", params_model=MyEvalInfo):
    ...
```

### 🔴 框架核心 — 严禁修改

以下文件为框架核心，任何修改都可能破坏全局功能：

- `evaluator/core/orchestrator.py` — 编排引擎
- `evaluator/core/bench_schema.py` — Benchmark 数据模型
- `evaluator/core/interfaces/abstract_*.py` — 抽象基类
- `evaluator/utils/*.py` — 通用工具层（llm, benchmark_reader, report_reader, agent_inspector, config）
- `benchmark/basic_runner.py` — 跑分执行器
- `web/` — Web UI（通过 agent_inspector 自动适配新 plugin）

> **如果你发现需要修改 🔴 文件才能完成需求，请停下来通知用户** — 这通常意味着需求理解有误，或框架需要由维护者升级扩展点。

## Auto-Adaptation

完成以下步骤后，Web UI / CLI 会自动适配新 plugin：
- **Config Schema**: `agent_inspector` 从 `AbstractEvalAgent._params_registry` 自动派生 config map，无需手动维护
- **展示元数据**: 新 plugin 默认使用通用图标/颜色，可通过 `_display_meta` 类属性自定义（可选）

## Workflow

### Step 1: Gather Requirements

Ask the user (via AskUserQuestion) for the following if not provided in $ARGUMENTS:

1. **Plugin name** (snake_case, e.g. `response_time`) — used as:
   - `__init_subclass__` registration name
   - `evaluator: Literal["<name>"]` discriminator value
   - File name: `evaluator/plugin/eval_agent/<name>_eval_agent.py`
   - Class name: `<PascalCase>EvalAgent`

2. **Evaluation logic description** — what this evaluator measures and how (LLM-based or deterministic)

3. **Config fields** — what parameters the user needs to provide in test case JSON (beyond the standard `evaluator`, `model`, `threshold`)

### Step 2: Validate Constraints

Before generating code, verify:

- [ ] Name is unique — check with `python -c "import evaluator.plugin.eval_agent; from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent; print(sorted(AbstractEvalAgent.get_all()))"`
- [ ] Name is valid snake_case identifier (lowercase, underscores only, no leading digits)
- [ ] Config field names don't shadow Pydantic reserved names

### Step 3: Create Implementation File

Create `evaluator/plugin/eval_agent/<name>_eval_agent.py` following this template:

```python
"""
<PascalCase>EvalAgent — <中文描述>

注册名称: "<name>"

<详细说明评估逻辑>
"""

import logging
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent
from evaluator.core.schema import EvalResult, SessionInfo, TestAgentMemory

logger = logging.getLogger(__name__)


# ============================================================
# 配置模型（与插件实现同文件，通过 params_model 自动注册）
# ============================================================


class <NewEvalInfo>(BaseModel):
    """<中文描述>"""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"evaluator": "<name>"}]},
    )

    evaluator: Literal["<name>"] = Field(default="<name>", description="评估器类型")
    model: Optional[str] = Field(None, description="LLM 模型")
    threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="通过阈值")
    # ... 自定义字段


# ============================================================
# 插件实现
# ============================================================


class <PascalCase>EvalAgent(AbstractEvalAgent, name="<name>", params_model=<NewEvalInfo>):
    """<中文一句话描述>"""

    # 可选：自定义 Web UI 展示元数据（不声明则使用默认图标/颜色）
    # _display_meta = {"color": "#8b5cf6", "features": ["LLM 驱动", "自定义评估"]}

    _cost_meta = {
        "est_cost_per_case": 0.010,  # LLM-based ~$0.01，纯规则填 0
    }

    def __init__(self, eval_config: <NewEvalInfo>, **kwargs):
        super().__init__(eval_config, **kwargs)
        self.config: <NewEvalInfo> = eval_config

    async def run(
        self,
        memory_list: list[TestAgentMemory],
        session_info: SessionInfo | None = None,
    ) -> EvalResult:
        """执行评估"""
        ...
```

Key patterns to follow (from existing implementations):
- **LLM-based evaluator**: See `semantic_eval_agent.py` (SemanticEvalAgent) — uses `do_execute()`, tracks cost via `self._cost`, exposes `self.model` and `self.cost` property
- **Deterministic evaluator**: See `record_retrieval_eval_agent.py` — no LLM, no cost tracking, pure logic
- Extract conversation from `memory_list` (each entry has `.test_reaction` for user action and `.target_response` for AI response)
- Static context available via `self.history`, `self.user_info`, `self.case_id`
- Return `EvalResult(result="pass"|"fail", score=0.0~1.0, feedback="...", trace=...)`

### Step 5: Register Plugin

**没有手工注册这一步。** `evaluator/plugin/eval_agent/__init__.py` 用 `pkgutil` 遍历本包，
自动 import 所有文件名以 `_eval_agent.py` 结尾的模块，从而触发 `__init_subclass__` 注册。
该文件里既没有 import 清单也没有 `__all__` —— **文件名即注册**，类上传 `params_model=` 即可。

> 该处的 `except ImportError: pass` 会吞掉依赖缺失，所以插件没出现时直接单独 import
> 该模块看真实报错。模块级的 SyntaxError / NameError 不会被吞，会炸掉整个包导入。

### Step 6: Verify

Run the following to verify:
```bash
# 插件注册检查
uv run python -c "import evaluator.plugin.eval_agent; from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent; print(sorted(AbstractEvalAgent.get_all()))"

# Web UI schema 自动发现检查
uv run python -c "from evaluator.utils.agent_inspector import list_eval_agents; print([(a.name, list(a.config_schema.get('properties', {}).keys())) for a in list_eval_agents()])"

# Lint
ruff check evaluator/plugin/eval_agent/
ruff format evaluator/core/schema.py evaluator/plugin/eval_agent/
```

### Step 7: Update Documentation

新增 EvalAgent 后，更新以下文档中的评估器列表/表格，保持信息同步：

| 文件 | 更新内容 |
|------|---------|
| `web/guides/develop-eval-agent.md` | 「现有评估器」表格追加新行 + 「关键文件」表格追加参考实现 |
| `web/guides/overview.md` | 「核心评估能力」表格追加新行 |
| `CLAUDE.md` | Plugin System 表格中 EvalAgent 行追加新名称 + Key Modules 追加说明 |
| `README.md` | 「已注册插件」表格 EvalAgent 区域追加新行 |

> 每个文件只需追加一行到已有表格，不要修改其他内容。

## Optional: Custom Display & Cost Metadata

新 plugin 默认使用灰色图标和空特性标签。若需自定义 Web UI 展示，在 plugin 类上声明 `_display_meta`:

```python
class <PascalCase>EvalAgent(AbstractEvalAgent, name="<name>", params_model=<PascalCase>EvalInfo):
    _display_meta = {
        "icon": "M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z",  # heroicons SVG path
        "color": "#8b5cf6",        # CSS 颜色值
        "features": ["LLM 驱动", "自定义评估"],  # 特性标签
    }
```

费用预估通过 `_cost_meta` 声明，前端自动读取并用于费用预估显示：

```python
    _cost_meta = {
        "est_cost_per_case": 0.035,  # 单 case 评估费用 (USD)，封装了模型 + 调用模式
    }
```

| 评估器类型 | 参考值 | 说明 |
|-----------|--------|------|
| LLM 单次调用 (semantic) | 0.010 | 1-2 次 LLM 调用 |
| LLM 多次调用 (healthbench) | 0.035 | N 条 rubric × 独立 LLM grading |
| 纯规则 (record_retrieval) | 0 | 无 LLM 调用 |

`_display_meta` 和 `_cost_meta` 中的字段均为可选，未指定的使用默认值。

## Reference: EvalAgent Plugins Shipped with the Framework

| Name | Config | File | Type |
|------|--------|------|------|
| `semantic` | `SemanticEvalInfo` | `semantic_eval_agent.py` | LLM-based |
| `rubric` | `RubricEvalInfo` | `rubric_eval_agent.py` | LLM + deterministic (criteria / signals / latency) |
| `healthbench` | `HealthBenchEvalInfo` | `healthbench_eval_agent.py` | LLM-based (rubric) |
| `medcalc` | `MedCalcEvalInfo` | `medcalc_eval_agent.py` | LLM + deterministic |
| `kg_qa` | `KgQaEvalInfo` | `kg_qa_eval_agent.py` | LLM + deterministic (numeric tolerance / stepwise) |
| `record_retrieval` | `RecordRetrievalEvalInfo` | `record_retrieval_eval_agent.py` | Deterministic (per-turn checkpoints) |
| `dialogue_quality` | `DialogueQualityEvalInfo` | `dialogue_quality_eval_agent.py` | LLM-based (multi-dimension) |
| `engagement` | `EngagementEvalInfo` | `engagement_eval_agent.py` | LLM-based |

当前实际注册了什么以这条为准，别以表为准：

```bash
python -c "import evaluator.plugin.eval_agent; from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent; print(sorted(AbstractEvalAgent.get_all()))"
```
