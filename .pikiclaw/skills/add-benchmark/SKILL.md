---
name: add-benchmark
description: 集成外部 benchmark 数据集到 HolyEval 框架 — 从论文/仓库到可执行评测的端到端流程。
argument-hint: "[benchmark paper URL or GitHub repo URL]"
---

# Add External Benchmark

将外部 benchmark 数据集（如论文、开源仓库）集成到 HolyEval 框架中，使其可以通过 `python -m benchmark.basic_runner <benchmark> <dataset>` 执行评测。

**交互原则**: 本 skill 采用「分析 → 确认 → 执行」模式。在每个关键决策点，必须通过 `AskQuestion` 工具向用户展示分析结论并获得确认，确认通过后才继续执行。**绝不在未经用户确认的情况下开始写代码。**

---

## 〇、Framework Protection（必读）

本 skill 仅通过**插件扩展点和数据目录**集成新 benchmark，**严禁修改框架核心逻辑**。文件按修改权限分为三级：

### 🟢 数据 & Plugin 层 — 自由新建

| 文件 | 操作 |
|------|------|
| `generator/<name>/__init__.py` | 新建（空文件） |
| `generator/<name>/converter.py` | 新建（数据转换器） |
| `benchmark/data/<name>/metadata.json` | 新建（数据集元信息） |
| `benchmark/data/<name>/<dataset>.jsonl` | 新建（由 converter 生成） |
| `evaluator/plugin/eval_agent/<name>_eval_agent.py` | **仅 Case B**: 新建（EvalAgent plugin） |
| `evaluator/plugin/eval_agent/__init__.py` | **不要改**：pkgutil 按 `_eval_agent.py` 后缀自动导入 |

### 🟡 Schema 扩展点 — 仅限追加（Case B）

`evaluator/core/schema.py` **不需要改**。`EvalInfo` 是
`Annotated[Any, BeforeValidator(_validate_eval_info)]`，运行时按 `evaluator` 字段从
`AbstractEvalAgent` 的注册表取出该插件的 `params_model`。配置类写在插件文件里，
用 `params_model=` 注册即可。

**禁止的操作**：
- 修改 `evaluator/core/schema.py` 的任何分发逻辑
- 修改其他插件的类定义

> Case A（复用现有评估器）**完全不需要修改 schema.py**。

### 🔴 框架核心 — 严禁修改

以下文件为框架核心，任何修改都可能破坏全局功能：

- `evaluator/core/orchestrator.py` — 编排引擎（do_single_test / BatchSession）
- `evaluator/core/bench_schema.py` — Benchmark 数据模型（BenchItem / resolve_effective_target / bench_item_to_test_case）
- `evaluator/core/interfaces/abstract_*.py` — 三类 Agent 抽象基类
- `evaluator/utils/*.py` — 通用工具层（llm, benchmark_reader, report_reader, agent_inspector, config）
- `evaluator/plugin/test_agent/` — 已有 TestAgent 插件（manual / auto）
- `evaluator/plugin/target_agent/` — 已有 TargetAgent 插件（llm_api / hermes / evermem / *_rag_api）
- `evaluator/plugin/eval_agent/` 中的**已有**文件 — 不得修改随框架发布的任何评测器（semantic / rubric / healthbench / medcalc / kg_qa / record_retrieval / dialogue_quality / engagement）
- `benchmark/basic_runner.py` — 跑分执行器
- `web/` — Web UI

> **如果你发现需要修改 🔴 文件才能完成需求，请停下来通知用户** — 这通常意味着需求理解有误，或框架需要由维护者升级扩展点。绝不"顺手"改一下核心代码来适配新 benchmark。

---

## 一、Architecture Overview

### 执行流水线

```
外部数据集 → Converter → BenchItem JSONL → basic_runner → Orchestrator → TestAgent ↔ TargetAgent → EvalAgent → BenchReport
                ↑                                                              ↑             ↑              ↑
           [需实现]                                                       [复用即可]     [复用即可]     [可能需实现]
```

### 三类 Agent 角色

| 角色 | 职责 | Benchmark 集成时是否需要新增？ |
|------|------|-------------------------------|
| **TestAgent**（虚拟用户） | 模拟真实用户发送消息 | **几乎不需要** — 现有 `manual`/`auto` 覆盖所有场景 |
| **TargetAgent**（被测系统） | 封装被测系统的调用 | **原则上不需要** — 现有 `llm_api` / `hermes` / `*_rag_api` 已足够 |
| **EvalAgent**（评估器） | 评判对话质量 | **可能需要** — 取决于评估方法论是否已有对应实现 |

### 已有 TestAgent 插件（虚拟用户）

| 名称 | 类 | 工作方式 | 典型场景 |
|------|-----|---------|---------|
| `manual` | `ManualTestAgent` | 按序发送 `strict_inputs` 预设输入，用完自动结束。零 LLM 调用、完全确定性、零成本。轮次 = `len(strict_inputs) + 1`，忽略 `max_turns` | **绝大多数 benchmark 都用此类型**：标准问答、rubric 评测、答案匹配等 |
| `auto` | `AutoTestAgent` | 前 N 轮消费 `strict_inputs`，之后 LLM 自主生成对话。根据 `goal`/`context`/`finish_condition` 判断何时结束 | 需要 AI 模拟多轮追问的开放式对话场景（极少数 benchmark 需要） |

> **选择原则**: 如果 benchmark 数据中包含完整的用户输入文本，一律使用 `manual`。仅当 benchmark 要求虚拟用户自主生成多轮对话时才考虑 `auto`。

### 已有 TargetAgent 插件（被测系统）

| 名称 | 类 | 工作方式 | 配置字段 |
|------|-----|---------|---------|
| `llm_api` | `LlmApiTargetAgent` | 通过 `do_execute()` 统一调用大模型（OpenAI/Gemini/Anthropic/GLM），自动维护多轮对话历史 | `model`（必填）, `system_prompt`（可选） |
| `hermes` | `HermesTargetAgent` | 通过 HTTP API 调用外部被测服务，自行管理 session/token | `base_url`, `user_email` 等（见 `HermesTargetInfo`） |
| `*_rag_api` | `EvermemTargetAgent` / `Mem0RagApiTargetAgent` / `NaiveRagApiTargetAgent` / `HippoRagApiTargetAgent` / `DygRagApiTargetAgent` | 各记忆/RAG 服务的接入 | `user_email` 等（见各自的 `*TargetInfo`） |

> **⚠️ 关于自定义 TargetAgent**: 对于外部 benchmark 集成，**原则上不需要自定义 TargetAgent**。
> - 如果 benchmark 是评测大模型能力 → 使用 `llm_api`（运行时通过 `--target-model` 指定模型）
> - 如果 benchmark 是评测某个记忆/RAG 服务 → 使用对应的 `*_rag_api` / `hermes`
> - **如果你判断需要自定义 TargetAgent，这几乎一定意味着理解有误**。请务必在 Checkpoint 1 中向用户确认，说明为什么现有 target 不够用，并获得明确同意后才继续。

### 已有 EvalAgent 插件（评估器）

| 名称 | 适用场景 | LLM | 评估方式 |
|------|---------|-----|---------|
| `semantic` | 通用多维度语义评估 | 是 | LLM 按 criteria 独立打分 → 加权总分 → 对比 threshold → pass/fail |
| `healthbench` | HealthBench rubric 评测 | 是 | LLM 逐条判定 criterion → 按 points 加权计算 → scored（不做 pass/fail） |
| `rubric` | 逐轮 rubric（自然语言判据 + 信号检查 + 时延预算） | 是 | 按 turn 逐条判定 criteria → 加权 → pass/fail |
| `medcalc` | 医学计算题标准答案比对 | 是 | LLM 抽取答案 → 按类型做数值容差匹配 → pass/fail |
| `kg_qa` | 知识图谱问答 | 是 | 数值容差 + 分步给分 |
| `record_retrieval` | 记录确认与检索准确性 | 否 | 逐轮 checkpoint 规则判定 → pass/fail |
| `dialogue_quality` | 多维对话质量 | 是 | LLM 多维打分 |
| `engagement` | 虚拟用户是否真的参与 | 是 | LLM 判定 engaged/not |

### 决策树：判断需要实现哪些组件

```
外部 benchmark 的评估方法论是否已有对应的 EvalAgent？
│
├─ YES → 仅需 Converter + 数据目录 （Case A: 轻量集成）
│        例：数值答案匹配类 → 复用 medcalc
│        例：逐轮判据类 → 复用 rubric
│        例：多维度语义评估 → 复用 semantic
│
└─ NO  → Converter + EvalInfo Config + EvalAgent Plugin + 数据目录 （Case B: 完整集成）
         例：HealthBench rubric 评估 → 自定义 healthbench eval
         例：MMLU 多选题评估 → 自定义 mcq eval
```

### 不需要修改的模块

以下模块完全通用，新增 benchmark 时**绝不修改**：
- `benchmark/basic_runner.py` — 执行器（基于 BenchItem 架构，自动适配）
- `evaluator/core/orchestrator.py` — 编排器（do_single_test / BatchSession）
- `evaluator/core/bench_schema.py` — 通用数据模型（BenchItem / BenchMark / BenchReport）
- `evaluator/utils/benchmark_reader.py` — 自动发现 `benchmark/data/` 目录
- `evaluator/utils/report_reader.py` — 自动发现 `benchmark/report/` 目录
- Web UI — 通过 `agent_inspector` 自动适配新 plugin

---

## 二、Workflow

### Phase 0: 研究外部 Benchmark

**这是最关键的一步**。在写任何代码前，必须彻底理解外部 benchmark。

#### 0.1 获取信息源

用户会提供论文 URL 或 GitHub 仓库 URL：
- **论文**: 使用 `WebFetch` 阅读论文内容，重点关注评估方法论章节（Evaluation / Metrics / Scoring）
- **仓库**: 使用 `WebFetch` 阅读 README、数据格式说明、评估脚本源码（重点：scoring / grading 函数）
- **数据样例**: 尽量获取 1-2 条原始数据样例，理解每个字段的含义

#### 0.2 提取关键信息

系统性分析以下三个维度：

**数据格式**：
- 原始数据文件格式（JSONL? CSV? JSON?）
- 每条数据包含哪些字段？
- 哪些字段映射到 `user.strict_inputs`（用户输入）？
- 是否有多轮对话上下文？如有，应拆分为 `history`（前置轮次）+ `strict_inputs`（最后一条用户输入）
- 哪些字段映射到评估标准？
- 哪些字段可用作 `tags`？
- 是否有多个数据子集（full, hard, easy 等）？

**评估方法论**：
- 评估方式是什么？（rubric 评分? 答案匹配? LLM-as-Judge? 多选题? 自动化指标?）
- 是否可以复用现有 EvalAgent？
- 评分公式是什么？什么算 pass/fail/scored？
- 原版评估脚本的核心逻辑（prompt、scoring 函数）

**对话模式**：
- 单轮还是多轮？如果多轮，前面的轮次应作为 `history`（评测前对话上下文），最后一条 user message 作为 `strict_inputs`
- 用户输入是否已经确定？（确定 → `manual`，需生成 → `auto`）
- 是否需要自定义 TargetAgent？（答案几乎一定是「不需要」）

---

#### 0.3 Checkpoint 1: 方案确认（必须执行）

**在进入 Phase 1 之前，必须通过 `AskQuestion` 向用户确认以下所有决策。**

先向用户展示一段分析总结文本（Markdown），包含：
1. benchmark 概述（一段话说明这个 benchmark 评测什么）
2. 原始数据格式说明（字段列举 + 样例）
3. 你的字段映射方案
4. 评估方法论分析

然后使用 `AskQuestion` 工具发起确认：

```
Question 1: Benchmark 名称
- prompt: "确认 benchmark 目录名（snake_case，用于 benchmark/data/<name>/ 和 generator/<name>/）"
- options: [推荐名称, 备选名称, "自定义（请在下方说明）"]

Question 2: 虚拟用户类型 (TestAgent)
- prompt: "虚拟用户类型 — 基于数据分析的推荐如下"
- options:
  - "manual（脚本驱动）— 使用原始数据中的确定性输入，零 LLM 成本【推荐】"
  - "auto（LLM 驱动）— 需要 AI 自主生成多轮对话"

Question 3: 被测系统类型 (TargetAgent)
- prompt: "被测系统类型 — 以下选项使用已有 TargetAgent，运行时通过 CLI 参数指定模型"
- options:
  - "llm_api — 通用大模型 API（OpenAI/Gemini/Anthropic 等）【推荐】"
  - "hermes / *_rag_api — 外部被测服务（记忆、RAG 等）"
  - "⚠️ 需要自定义 TargetAgent（请说明原因）"

Question 4: 评估器类型 (EvalAgent)
- prompt: "评估器类型 — 基于评估方法论分析的推荐如下"
- options:
  - 列出可能匹配的已有 eval + "[推荐原因]"
  - "需要自定义评估器（Case B）"

Question 5: 数据子集方案
- prompt: "数据子集划分"
- options: [列出原始数据中的子集方案]
- allow_multiple: true
```

> **关键规则**:
> - 如果用户在 Question 3 选择了「需要自定义 TargetAgent」，必须追问具体原因，并尝试用现有方案替代。只有用户二次确认确实无法复用时才执行。
> - 如果用户选择了意料之外的选项，主动解释可能的影响。

---

### Phase 1: 数据转换器（Converter）

**创建**: `generator/<benchmark_name>/converter.py` + `generator/<benchmark_name>/__init__.py`

参考实现: `generator/healthbench/converter.py`

#### 1.1 文件结构

```
generator/
├── <benchmark_name>/
│   ├── __init__.py          # 空文件
│   └── converter.py         # 转换器
└── ...
```

#### 1.2 Converter 核心模式

```python
"""
<BenchmarkName> → HolyEval 数据转换器

将 <原始格式> 转换为 HolyEval BenchItem JSONL。

转换映射:
  <原始字段A>  →  strict_inputs（用户输入）
  <原始字段B>  →  history（可选，多轮对话上下文）
  <原始字段C>  →  eval.<评估配置>
  <原始字段D>  →  tags

用法:
  python -m generator.<benchmark_name>.converter input_file output.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _convert_single(entry: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    """将单条原始数据转换为 HolyEval BenchItem dict

    Returns:
        BenchItem dict，转换失败返回 None
    """
    bench_item: Dict[str, Any] = {
        "id": "<prefix>_<unique_id>",
        "title": "<生成标题>",
        "description": "<描述>",
        "user": {
            "type": "manual",         # Checkpoint 1 确认的类型
            "goal": "<评测目标>",
            "strict_inputs": [...],   # 用户输入列表
        },
        "eval": {
            "evaluator": "<eval_type>",  # Checkpoint 1 确认的评估器
            # ... 评估器特定配置
        },
        "tags": [...],
    }

    # 如果原始数据含多轮对话上下文，添加 history
    # history: [{role: "user", content: "..."}, {role: "assistant", content: "..."}]
    if history:
        bench_item["history"] = history

    return bench_item


def convert(
    input_path: str | Path,
    output_path: str | Path,
    limit: int | None = None,
) -> int:
    """批量转换，返回成功条数"""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            if limit is not None and converted >= limit:
                break
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("第 %d 行 JSON 解析失败: %s", i + 1, e)
                skipped += 1
                continue
            bench_item = _convert_single(entry, i)
            if bench_item is None:
                skipped += 1
                continue
            fout.write(json.dumps(bench_item, ensure_ascii=False) + "\n")
            converted += 1

    logger.info("转换完成: %d 条成功, %d 条跳过, 输出: %s", converted, skipped, output_path)
    return converted


def main() -> None:
    """CLI 入口"""
    parser = argparse.ArgumentParser(
        description="将 <BenchmarkName> 转换为 HolyEval BenchItem JSONL",
    )
    parser.add_argument("input", help="源文件路径")
    parser.add_argument("output", help="输出 BenchItem JSONL 路径")
    parser.add_argument("--limit", type=int, default=None, help="最大转换条数")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    count = convert(args.input, args.output, limit=args.limit)
    print(f"转换完成: {count} 条 BenchItem → {args.output}")


if __name__ == "__main__":
    main()
```

#### 1.3 BenchItem 字段映射规则

**user 配置**：

| 场景 | user.type | strict_inputs | max_turns |
|------|-----------|---------------|-----------|
| 单轮问答（最常见） | `manual` | `["用户提问"]` | 不填（自动计算） |
| 多轮预注入 + 提问 | `manual` | `["背景数据...", "补充信息...", "正式提问"]` | 不填 |
| 需要 LLM 自主生成 | `auto` | `[]` 或前几轮 | 必填 |

> **`strict_inputs` 是一个列表（`List[str]`）**，`manual` 模式下逐条按序发送，每条都会触发被测系统回复，对话轮次 = `len(strict_inputs) + 1`。
>
> 当列表包含多条输入时，前面的条目用于**向被测系统预注入上下文信息**（如病历、检查指标、用药记录等），中间的回复不影响评估，评估器只关注完整对话的最终质量。这种方式适用于：
> - 需要先提供背景数据、再提问的场景（如先发患者病历，再问诊断建议）
> - 需要模拟多步交互的流程（如先报告症状，再补充检查结果，最后问治疗方案）
> - 不关心中间回复内容、只评估最终对话效果的评测设计
>
> **`history` vs `strict_inputs`**：两者都支持多轮对话，但机制不同：
>
> | | `history` | `strict_inputs`（多条） |
> |---|---|---|
> | **注入方式** | 作为预加载上下文，双方 Agent 直接"看到"，不经过对话循环 | 逐条发送，被测系统逐条回复，走完整对话循环 |
> | **被测系统行为** | 被测系统感知历史对话存在，像"接续"之前的对话 | 被测系统逐条处理每条输入并生成回复 |
> | **适用场景** | 原始数据本身包含多轮已有对话（如 HealthBench 的多轮 prompt） | 需要主动向被测系统"灌入"信息再提问 |
> | **格式** | `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]` | `["第一条输入", "第二条输入", ...]` |
>
> 两者可以组合使用：`history` 提供已有对话背景 + `strict_inputs` 在此基础上继续多步交互。
>
> Web UI 中 `history` 消息会以半透明样式展示并标注"以上为历史对话"，与评测对话视觉区分。

**eval 配置**（根据 Checkpoint 1 的选择）：

| 评估器 | eval 配置示例 |
|--------|-------------|
| `semantic` | `{"evaluator": "semantic", "criteria": [...], "threshold": 0.7}` |
| `medcalc` | `{"evaluator": "medcalc", "ground_truth": "42", "output_type": "integer"}` |
| `rubric` | `{"evaluator": "rubric", "turns": [{"turn": 1, "criteria": [{"name": "...", "llm_rubric": "..."}]}]}` |
| `healthbench` | `{"evaluator": "healthbench", "rubrics": [{"criterion": "...", "points": 10, "tags": [...]}]}` |
| 自定义 | `{"evaluator": "<new_name>", ...}`（需完成 Phase 2） |

**id 格式**: `<2-3字母前缀>_<原始ID或序号>`（如 `hb_<prompt_id>`，`mmlu_<subject>_<index>`），必须全局唯一。

---

### Phase 2: 自定义 EvalAgent（仅 Case B）

如果 Checkpoint 1 确认需要自定义评估器，按以下步骤执行。

> **推荐**: 直接调用 `add-eval-agent` skill 完成此步骤，它会自动处理 schema 修改、plugin 实现和注册。
> 以下为手动步骤说明，供理解整体流程。

#### 2.1 添加 EvalInfo 配置类

**新建**: `evaluator/plugin/eval_agent/<name>_eval_agent.py`（配置类与实现同文件）

在文件顶部添加 Pydantic 配置类：

```python
class <Name>EvalInfo(BaseModel):
    """<中文描述> — <评估方法简述>"""
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{...}]})

    evaluator: Literal["<name>"] = Field(description="评估器类型")
    model: Optional[str] = Field(None, description="LLM 模型")
    # ... benchmark 特有的评估配置字段
```

配置类与实现放在同一个插件文件里，用 `params_model=` 注册；`schema.py` 不需要改。

#### 2.2 实现 EvalAgent Plugin

**创建**: `evaluator/plugin/eval_agent/<name>_eval_agent.py`

核心要点：
- 继承 `AbstractEvalAgent`，使用 `name="<name>"` 注册
- 实现 `async def run(self, memory_list, session_info=None) -> EvalResult`
- **尽可能复用原版评估逻辑**（prompt、scoring 公式）
- 提取对话：`memory_list`（含 `.test_reaction` + `.target_response`），历史上下文通过 `self.history` 访问
- 如需 LLM：使用 `evaluator/utils/llm.py` 的 `do_execute()`
- 声明 `_cost_meta` 和 `_display_meta`

参考实现:
- **LLM rubric 评估**: `evaluator/plugin/eval_agent/healthbench_eval_agent.py`（`_build_conversation` + 并发 grade + `_calculate_score`）
- **LLM 语义评估**: `evaluator/plugin/eval_agent/semantic_eval_agent.py`
- **规则评估（零 LLM）**: `evaluator/plugin/eval_agent/record_retrieval_eval_agent.py`

#### 2.3 注册 Plugin

**无需修改任何文件**：文件名以 `_eval_agent.py` 结尾即被 pkgutil 自动导入，
类上的 `params_model=` 把配置模型写进注册表。

---

### Phase 3: 数据目录 + 执行转换

#### 3.1 创建 metadata.json

**创建**: `benchmark/data/<benchmark_name>/metadata.json`

```json
{
  "description": "# <Benchmark 名称>\n\n<Markdown 描述>\n\n## 子集\n\n| 子集 | 数量 | 说明 |\n|------|------|------|\n| ... | ... | ... |\n\n**评估器**: `<evaluator_name>`",
  "target": [
    {
      "type": "llm_api",
      "fields": {
        "model": {"default": "gpt-5.4-mini", "editable": true, "required": true}
      }
    }
  ]
}
```

字段说明：
- `description`: Markdown 格式，Web UI 渲染展示
- `target`: **TargetSpec 数组**，每个元素定义一种被测系统类型（参考 `benchmark/data/healthbench/metadata.json`）
  - `type`: agent 类型（`llm_api` / `hermes` / `*_rag_api`）
  - `fields`: 各字段的默认值、是否可编辑（`editable`）、是否必填（`required`）
  - 单 target → CLI/Web 自动使用；多 target → CLI `--target-type` 指定
- `params`（可选）: 共享参数字典，供 JSONL 条目通过 `$ref` 引用（见下方说明）

##### params & $ref 引用机制

当**多条用例共享相同的大块数据**（如 history 对话上下文）时，在 metadata.json 中定义 `params`，JSONL 条目通过 `{"$ref": "key"}` 引用，避免重复：

```json
// metadata.json
{
  "target": [...],
  "params": {
    "diabetic_user_history": [
      {"role": "user", "content": "I have type 2 diabetes..."},
      {"role": "assistant", "content": "Thank you for sharing..."}
    ]
  }
}

// sample.jsonl — 多条用例引用同一 history
{"id": "case_1", "history": {"$ref": "diabetic_user_history"}, "user": {...}, "eval": {...}}
{"id": "case_2", "history": {"$ref": "diabetic_user_history"}, "user": {...}, "eval": {...}}
```

规则：
- 仅扫描 JSONL 条目的**顶层字段**（嵌套 `$ref` 不处理）
- 字段值必须是 `{"$ref": "key"}` 且仅含此一个键才触发替换
- 引用 key 未找到时输出警告，保持原值
- 典型场景：同一用户画像的多条用例共享 `history`

写法：共享片段定义在 `metadata.json` 的 `params` 里，用例侧用 `{"$ref": "<key>"}` 整体替换该字段。

```jsonc
// metadata.json
{ "params": { "diabetic_user_history": [
      {"role": "user", "content": "I'm a 45-year-old male with type 2 diabetes ..."},
      {"role": "assistant", "content": "Thank you for sharing that ..."}
] } }

// <dataset>.jsonl —— 多条用例共享同一段 history
{"id": "hd_001", "user": {...}, "eval": {...}, "history": {"$ref": "diabetic_user_history"}}
```

#### 3.2 执行转换

```bash
uv run python -m generator.<benchmark_name>.converter <input_file> benchmark/data/<benchmark_name>/<dataset>.jsonl
```

---

#### Checkpoint 2: 转换结果确认（必须执行）

转换完成后，向用户展示：
1. 成功转换的条数 / 跳过的条数
2. 首条 BenchItem 的完整 JSON（格式化）
3. 末条 BenchItem 的完整 JSON（格式化）

然后使用 `AskQuestion` 确认：

```
Question 1: 转换结果
- prompt: "已转换 N 条数据，上方展示了首条和末条样例。请确认数据映射是否正确"
- options:
  - "确认正确，继续验证"
  - "有问题，需要调整（请说明）"
```

---

### Phase 4: 验证

#### 4.1 数据加载验证

```bash
uv run python -c "
from evaluator.utils.benchmark_reader import load_bench_items
items = load_bench_items('benchmark/data/<benchmark_name>/<dataset>.jsonl')
print(f'成功加载 {len(items)} 条 BenchItem')
print(f'首条 ID: {items[0].id}')
print(f'评估器: {items[0].eval.evaluator}')
"
```

#### 4.2 Plugin 注册验证（仅 Case B）

```bash
uv run python -c "
import evaluator.plugin.eval_agent
from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent
print('已注册 EvalAgent:', list(AbstractEvalAgent.get_all().keys()))
"
```

#### 4.3 Benchmark 发现验证

```bash
uv run python -c "
from evaluator.utils.benchmark_reader import list_benchmarks
for b in list_benchmarks():
    print(f'{b.name}: {[d.name for d in b.datasets]}')
"
```

#### 4.4 Lint

```bash
ruff check generator/<benchmark_name>/ evaluator/core/schema.py evaluator/plugin/eval_agent/
ruff format generator/<benchmark_name>/ evaluator/core/schema.py evaluator/plugin/eval_agent/
```

---

#### Checkpoint 3: 端到端测试确认（必须执行）

所有验证通过后，使用 `AskQuestion` 确认是否跑端到端测试：

```
Question 1: 端到端测试
- prompt: "数据加载和 plugin 注册均通过。是否执行小规模端到端测试？（会调用 LLM API，产生少量费用）"
- options:
  - "是，跑 3 条端到端测试"
  - "是，跑 1 条端到端测试"
  - "跳过，我稍后手动测试"
```

用户确认后，**启动 Web 服务并通过浏览器执行**（见下方「Web 驱动执行」说明），或用 CLI 快速跑：

```bash
uv run python -m benchmark.basic_runner <benchmark_name> <dataset> \
  --target-type llm_api --target-model gpt-5.4-mini \
  --limit <N> -v
```

---

### Phase 5: Sample 跑分与论文基准对比（必须执行）

**目的**: 用 sample 子集实际跑分，与原始论文/实验公开的基准数据对比，验证迁移后的评测管线是否产出合理且可比的结果。如果分数偏差过大，说明 converter 映射或 EvalAgent 实现有问题。

#### 5.1 收集论文基准数据

在 Phase 0 阅读论文/仓库时，就应记录以下信息（如有）：

- **原版基准分数**: 论文中同一模型（如 gpt-5.4-mini）在同一数据子集上的分数
- **分数指标定义**: avg_score? pass_rate? accuracy? 与 HolyEval 的 `avg_score` / `pass_rate` 如何对应
- **测试条件**: 原版使用的 grader 模型、temperature、采样次数等
- **已知差异**: 例如原版可能跑 3 次取平均，HolyEval 默认跑 1 次

> 如果论文没有公开基准数据，在 Checkpoint 1 中向用户确认是否有内部参考数据，或标注"无可比基准，仅做冒烟验证"。

#### 5.2 Web 驱动执行（确保进度可视）

> **重要**: CLI 和 Web 是独立的执行通道 — CLI 跑的任务在 Web 上看不到进度。
> 要获得实时进度可视化，必须通过 Web 执行跑分。

**Step 1: 启动 Web 服务**

检查 Web 服务是否已在运行。如果未运行，后台启动：

```bash
# 后台启动 Web 服务（block_until_ms: 0）
uv run python -m web
```

等待服务就绪（检查 `http://localhost:8000` 可访问）。

**Step 2: 浏览器打开任务页面**

使用 `open` 命令打开浏览器：

```bash
open http://localhost:8000/tasks
```

**Step 3: 通过 Web API 创建跑分任务**

通过 `POST /api/tasks` 创建任务（等效于在 Web UI 上点击「开始评测」）：

```bash
curl -X POST http://localhost:8000/api/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "benchmark": "<benchmark_name>",
    "dataset": "sample",
    "target_type": "llm_api",
    "target_model": "<与论文对齐的模型>",
    "max_concurrency": 5
  }'
```

API 返回 `task_id`，用户可在浏览器 `http://localhost:8000/tasks/{task_id}` 实时查看进度。

**Step 4: 等待任务完成**

轮询任务状态直到完成：

```bash
curl http://localhost:8000/api/tasks/<task_id>
```

响应中的 `snapshot.completed == snapshot.total` 时表示完成。

> `--target-model` 应尽量与论文中的被测模型一致，以便直接对比分数。

#### 5.3 生成对比报告

跑分完成后，整理以下对比表格向用户展示：

```markdown
## 迁移验证报告: <Benchmark 名称>

### 测试条件
| 项目 | 原版 (论文) | HolyEval 迁移版 |
|------|-----------|----------------|
| 数据子集 | <子集名> (<N>条) | sample (<M>条) |
| 被测模型 | <model> | <model> |
| Grader 模型 | <model> | <model> |
| 评估器 | <原版实现> | <evaluator name> |

### 分数对比
| 指标 | 原版基准 | HolyEval 结果 | 偏差 | 判定 |
|------|---------|-------------|------|------|
| avg_score | <X> | <Y> | <±Z%> | ✅/⚠️/❌ |
| pass_rate (如有) | <X> | <Y> | <±Z%> | ✅/⚠️/❌ |

偏差判定标准:
- ✅ 偏差 ≤5%: 正常范围（LLM 非确定性 + 采样差异）
- ⚠️ 偏差 5-15%: 需关注，可能因采样数不足或 prompt 微调
- ❌ 偏差 >15%: 需排查，converter 映射或 eval 逻辑可能有误

### 按标签维度对比 (如有)
| 标签 | 原版 | HolyEval | 偏差 |
|------|------|---------|------|
| <tag1> | ... | ... | ... |

### 分析 & 结论
<对偏差的解释，已知差异因素，是否通过验证>
```

#### 5.4 Checkpoint 4: 对比结果确认（必须执行）

展示上述对比报告后，使用 `AskQuestion` 确认：

```
Question 1: 对比结果
- prompt: "上方为 sample 跑分与论文基准的对比报告。请确认迁移结果是否可接受"
- options:
  - "结果可接受，迁移完成"
  - "偏差较大，需要排查（请说明关注点）"
  - "无原版基准数据，冒烟通过即可"
```

如果用户选择"偏差较大，需要排查"，按以下方向排查：

1. **Converter 映射问题**: 抽查几条原始数据和转换后的 BenchItem，确认字段映射正确
2. **EvalAgent 逻辑问题**: 对比原版 grading prompt 和 HolyEval 实现，检查是否有遗漏
3. **测试条件差异**: 模型版本、temperature、sample 大小等差异导致的正常波动
4. **分数计算口径**: 原版和 HolyEval 的 score 计算方式是否完全一致

排查修复后重新执行 5.2-5.4，直到用户确认通过。

---

### Phase 6: 更新文档

集成完成后，更新以下文档，确保新 benchmark 在所有入口可见：

| 文件 | 更新内容 |
|------|---------|
| `README.md` | 「Benchmark 数据集」表格追加新行 + CLI 示例追加新命令 |
| `CLAUDE.md` | Commands 区 CLI 示例追加 + Benchmark Data 目录树追加 + Data Conversion 追加转换器说明 |
| `web/guides/run-benchmark.md` | 「可用数据集」表格追加新行 + CLI 示例追加 |
| `web/guides/generate-benchmark.md` | 「已内置的转换器」表格追加新行 |
| `web/guides/overview.md` | generator 目录树追加 + 评估能力表追加（仅 Case B 新增 EvalAgent 时） |

> **Case B 额外更新**（新增了 EvalAgent）：
>
> | 文件 | 更新内容 |
> |------|---------|
> | `web/guides/develop-eval-agent.md` | 「现有评估器」表格追加 + 「关键文件」表格追加参考实现 |
> | `CLAUDE.md` | EvalAgent 实现表 + Key Modules 追加说明 |
> | `README.md` | 「已注册插件」表格 EvalAgent 区域追加新行 |

---

## 三、Reference

### HealthBench 集成实例（Case B 完整案例）

| 组件 | 文件 | 作用 |
|------|------|------|
| Converter | `generator/healthbench/converter.py` | HealthBench JSONL → BenchItem JSONL |
| EvalInfo | `evaluator/plugin/eval_agent/healthbench_eval_agent.py` → `HealthBenchEvalInfo` | rubrics 配置结构 |
| EvalAgent | `evaluator/plugin/eval_agent/healthbench_eval_agent.py` | 原版 GRADER_TEMPLATE + scoring |
| 注册 | 文件名以 `_eval_agent.py` 结尾即被 pkgutil 自动导入 | `params_model=` 写入注册表 |
| 数据 | `benchmark/data/healthbench/metadata.json` | 元信息（target_configurable: true） |
| 数据 | `benchmark/data/healthbench/sample.jsonl` 等 | 转换后的 BenchItem 数据 |

**数据映射**:
```
HealthBench 原始               →  HolyEval BenchItem
─────────────────────────────────────────────────
prompt (多轮对话)               →  拆分为 history + strict_inputs
  prompt[:-1] (历史轮次)        →  history [{role, content}]（41.7% 用例有多轮）
  prompt[-1].content (user msg) →  user.strict_inputs[0]
rubrics[].criterion            →  eval.rubrics[].criterion
rubrics[].points               →  eval.rubrics[].points
example_tags                   →  tags
prompt_id                      →  id (加 "hb_" 前缀)
user.type                      =  "manual"（预设输入，零 LLM）
target                         =  不在 BenchItem 中（运行时决定）
```

**评估逻辑**:
```
achieved = Σ(points for rubric where criteria_met=True)
total_possible = Σ(points for rubric where points > 0)
score = clip(achieved / total_possible, 0, 1)
result = "scored"（不做 pass/fail 判定）
```

### Key Data Models（速查）

**BenchItem**（数据集用例 — 没有 target）:
```python
class BenchItem(BaseModel):
    id: str                         # 唯一标识
    title: str                      # 一句话标题
    description: Optional[str]      # 补充说明
    user: BenchUserInfo             # 虚拟用户配置（含 target_overrides）
    eval: EvalInfo                  # 评估配置（按 evaluator 字段走注册表分发）
    history: List[Dict[str, str]]   # 可选，评测前历史对话 [{role, content}]
    tags: List[str]                 # 分类标签
```

**运行时转换链**:
```
BenchItem + CLI runtime_target
    ↓ bench_item_to_test_case()
TestCase（包含 user, target, eval, history, tags）
    ↓ do_single_test()
TestResult（包含 score, result, feedback, trace, cost）
    ↓ build_bench_report()
BenchReport
```

---

## 四、Checklist

完成集成后，确认以下所有项目：

- [ ] `generator/<name>/__init__.py` 存在
- [ ] `generator/<name>/converter.py` 可正确转换数据
- [ ] `benchmark/data/<name>/metadata.json` 格式正确
- [ ] `benchmark/data/<name>/<dataset>.jsonl` 可被 `load_bench_items()` 加载
- [ ] （Case B）`evaluator/plugin/eval_agent/<name>_eval_agent.py` 内含配置类与实现，
      并在类上传了 `params_model=`（不传则 `未注册的 evaluator`）
- [ ] `benchmark_reader.list_benchmarks()` 可发现新数据集
- [ ] `ruff check` 和 `ruff format` 通过
- [ ] 端到端小规模测试通过
- [ ] Sample 跑分与论文基准对比报告已生成，偏差在可接受范围内（或用户确认无基准数据）
- [ ] 文档已更新（README.md, CLAUDE.md, web/guides/ 中的相关表格和示例）
