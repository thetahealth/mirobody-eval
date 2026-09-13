<h1 align="center">
  <br>
  mirobody-eval
  <br>
</h1>

<p align="center">
  <strong>五个基准随仓库分发，一条命令复现；虚拟用户、被测系统、判分器都是可换的插件。<br>七条 Claude Code 斜杠命令，从接入一篇论文引导到分析一份报告。</strong>
</p>

<p align="center">
  <a href="https://github.com/thetahealth/mirobody-eval/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-3776AB.svg?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="https://docs.anthropic.com/en/docs/claude-code"><img src="https://img.shields.io/badge/Claude_Code-native-cc785c.svg?logo=anthropic&logoColor=white" alt="Claude Code Native"></a>
  <a href="https://arxiv.org/abs/2604.02834"><img src="https://img.shields.io/badge/arXiv-2604.02834-b31b1b.svg" alt="arXiv Paper"></a>
  <a href="https://huggingface.co/datasets/healthmemoryarena/ESL-Bench"><img src="https://img.shields.io/badge/%F0%9F%A4%97_HuggingFace-ESL--Bench-FFD21E.svg" alt="HuggingFace Dataset"></a>
  <a href="http://healthmemoryarena.ai"><img src="https://img.shields.io/badge/%F0%9F%8C%90_Live-Health_Memory_Arena-black.svg" alt="Live Demo"></a>
  <a href="https://github.com/thetahealth/mirobody-eval/stargazers"><img src="https://img.shields.io/github/stars/thetahealth/mirobody-eval?style=social" alt="GitHub Stars"></a>
</p>

<p align="center">
  <strong><a href="README.md">English</a></strong> &middot; <strong>简体中文</strong>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> &middot;
  <a href="#评测你自己的-mirobody-部署">评测 mirobody</a> &middot;
  <a href="#用-claude-code-做-ai-原生开发">Claude Code</a> &middot;
  <a href="#web-ui">Web UI</a> &middot;
  <a href="http://healthmemoryarena.ai">在线演示</a> &middot;
  <a href="https://arxiv.org/abs/2604.02834">论文</a> &middot;
  <a href="https://huggingface.co/datasets/healthmemoryarena/ESL-Bench">数据集</a> &middot;
  <a href="#基准">基准</a> &middot;
  <a href="#参与贡献">参与贡献</a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.02834">
    <img src="docs/screenshots/eslbench_overview.png" alt="ESL-Bench: Event-Driven Longitudinal Health Agent Benchmark" width="80%">
  </a>
</p>

<p align="center">
  <em>ESL-Bench —— 面向健康 agent 的事件驱动合成纵向基准。
  <br>100 个合成用户，10,000 道题，5 个维度，标准答案由程序算出。
  <br>论文：<a href="https://arxiv.org/abs/2604.02834">arXiv:2604.02834</a></em>
</p>

---

mirobody-eval 是 [mirobody](https://github.com/thetahealth/mirobody)（开源健康数据引擎）的评测半边。它存在的意义是让 mirobody 的说法带上数字：把一个合成用户灌进你自己的部署，对它跑一遍基准，拿到一份带分数的报告。底层框架是通用的，所以同一条命令也能对任何你接进来的被测系统复现任何已发表的基准。放进一个基准数据集，跑一条命令，拿到评分报告。评测器、被测系统、虚拟用户都可以通过可插拔的 agent 架构自行扩展。

从一开始就按 [Claude Code](https://docs.anthropic.com/en/docs/claude-code) 原生项目来设计——从初始化到把一篇论文里的新基准接进来，每个流程都是一条交互式斜杠命令。你用自然语言描述意图，剩下的交给 Claude Code。**用它或扩展它，你一行代码都不必写。**

### 从论文到基准 —— `/add-benchmark`

<p align="center">
  <img src="docs/screenshots/holyeval_add_benchmark.gif" alt="Add Benchmark Demo" width="80%">
</p>

> 粘一个论文链接，`/add-benchmark` 引导 Claude Code 读论文、写转换器、生成数据集，最后跑一份样本、**拿论文公布的数字校验转换是否忠实**。产物只落在数据和插件目录，框架核心不被触碰。

### 同一条命令跑全部五个基准

<p align="center">
  <img src="docs/screenshots/holyeval_run_benchmark.gif" alt="Run Benchmark Demo" width="80%">
</p>

```bash
# 先跑少量用例；费用和耗时取决于模型及基准
uv run python -m benchmark.basic_runner healthbench sample --target-model gpt-5.4-mini --limit 3
uv run python -m benchmark.basic_runner medcalc sample --target-model gpt-5.4-mini --limit 3
uv run python -m benchmark.basic_runner virtual_user round1 --target-type llm_api --target-model gpt-5.4-mini --limit 3

# ESLBench 需要先准备数据（见下面的快速开始）
uv run python -m benchmark.basic_runner eslbench sample50-20260331 --target-type llm_api --target-model gpt-5.4-mini --limit 3

# 想跑全量？去掉 --limit
```

## 为什么用 mirobody-eval

| | |
|---|---|
| **论文 → 基准** | 粘一个论文链接，`/add-benchmark` 读论文、写转换器、生成数据集，最后拿论文公布的分数校验转换是否忠实 |
| **一条命令复现** | 随附基准都走同一条 `basic_runner` 命令；报告带批次号和 sha256 校验和，分数能指回它来自哪一期发布 |
| **判不成分就说判不成** | 超时、断连、判分器故障记 `error`，不算进平均分；未判成数和「已评分/总数」印在每份汇总里——假零分和虚高分都进不来 |
| **可插拔架构** | 虚拟用户、被测系统、判分器各是一个插件类；文件落进插件目录即自动注册，框架核心不用改 |
| **评测单位是会话** | TestAgent ↔ TargetAgent 多轮对话直到结束条件，判分器拿到完整轨迹，不只是单轮问答 |
| **批量执行** | 并发、可取消；中断留下检查点，`--resume` 加载后跳过已完成的用例接着跑 |
| **Web UI** | 发起评测、SSE 进度、逐用例报告——与 CLI 汇聚到同一个执行入口 `do_single_test()` |
| **AI 原生** | 七条引导式斜杠命令，从 `/quick-start` 到 `/eslbench-report-analysis`，装环境到读报告全程有引导 |

## 快速开始

用 [Claude Code](https://docs.anthropic.com/en/docs/claude-code)：直接 `/quick-start`，它会自动处理一切。

或者手动：

```bash
git clone https://github.com/thetahealth/mirobody-eval.git && cd mirobody-eval
uv sync
cp .env.example .env                    # 编辑 .env，填入所用服务商的真实 key

# 配好 OPENAI_API_KEY 或 OPENROUTER_API_KEY 后运行
uv run python -m benchmark.basic_runner healthbench sample --target-model gpt-5.4-mini --limit 2

# 启动 Web UI
uv run python -m web                    # http://localhost:8000
```

> **前置条件：** Python 3.11+、[uv](https://docs.astral.sh/uv/)、至少一个 LLM API key。
> 上面的示例中，被测模型和判分模型都属于 OpenAI。可以配置 `OPENAI_API_KEY`，
> 或只配置 `OPENROUTER_API_KEY`、不设置 `OPENAI_API_KEY`，让这些模型 ID 通过
> OpenRouter 调用。账号需有对应模型的访问权限。这配置的是评测进程；mirobody
> 部署自身的对话模型和 embedding 仍需单独配置。
>
> **ESLBench 数据准备：** 评测前运行 `uv run python -m generator.eslbench.prepare_data`
>（Web UI 也会在后台启动准备，请等完成后再跑）。它会检查 manifest 中的全部批次，
> 下载缺失或有变更的批次，包括用户数据和题库，再生成 DuckDB 文件。它不是只下载
> 一个用户，首次准备可能占用较多磁盘和时间。其他基准的题目文件随仓库提供。

如果只有 Google 或 DashScope 的 key，还需要为调用 LLM 的角色分别指定模型。
先在 `.env` 配好对应的 key，再把下面的 `YOUR_MODEL_ID` 换成账号可用的模型 ID：

```bash
MODEL=google:YOUR_MODEL_ID              # 或 dashscope:YOUR_MODEL_ID
uv run python -m benchmark.basic_runner healthbench sample \
    --target-type llm_api --target-model "$MODEL" --eval-model "$MODEL" --limit 2
```

运行 `virtual_user` 时还需加 `--user-model "$MODEL"`。仅设置服务商的 key 不会替换
其他角色原有的 OpenAI 默认模型。所有可用前缀见[配置](#配置)。

## 评测你自己的 mirobody 部署

先准备一个运行中的 [mirobody](https://github.com/thetahealth/mirobody) 部署，并确保
评测进程能访问它的 Postgres 和 Redis。部署已有的演示数据可能属于其他用户，
评测前仍需灌入所选题目对应的用户。

在 mirobody-eval 仓库目录中，按快速开始配置评测用的 LLM key。mirobody 部署自身
需要可用的对话和 embedding provider，评测进程需使用同一部署的 JWT 配置。
如果配置文件经过加密，seeder 和 runner 也需要该部署的 `CONFIG_ENCRYPTION_KEY`。
先安装可选依赖，再灌数据（此依赖要求 Python 3.12+）：

```bash
# 1. 安装引擎集成依赖
uv sync --extra mirobody --python 3.12

# 2. 准备全部缺失或有变更的 ESLBench 批次，不只是一个用户
uv run python -m generator.eslbench.prepare_data

# 3. 指向自己的部署；把路径和地址替换为实际配置
export MIROBODY_CONFIG=/绝对路径/到/你的/mirobody/config.localdb.yaml
export MIROBODY_BASE_URL=http://localhost:18080

# 4. 灌入接下来要评测的用户
uv run python -m generator.eslbench.seed_mirobody --users user5086@demo

# 5. 首次只跑该用户的指定题目；--limit 本身不会按用户筛选
uv run python -m benchmark.basic_runner eslbench sample200-20260430 \
    --target-type mirobody \
    --ids user5086_AT_demo_Q001,user5086_AT_demo_Q087,user5086_AT_demo_Q067 -p 1
```

HTTP 地址以部署实际暴露的地址为准；如果端口是 18060，就填
`http://localhost:18060`。配置中的 Docker 容器内数据库地址未必能从宿主机访问。
灌入完成后 seeder 会检查指标的可检索性，有缺失向量会先报错。

这三题用于确认连接和判分流程，不覆盖 ESL-Bench 的全部五个维度，也不能作为完整
基准成绩。扩大评测前，请先灌入所选题目涉及的全部用户。要与使用检索工具的纯模型
对比，保留相同数据集和 ID，改用 `--target-type llm_api` 并填写 `--target-model`。
`--target-type mirobody` 背后的模型由部署自身配置。要从 Web UI 发起评测，
在上述已配置的终端中启动 `uv run python -m web`，再选相同的题目 ID。

### 文件上传演示

`labreport` 把这个合成用户的某次化验面板渲染成 PDF，让 mirobody 的入库链路有真东西可嚼。灌数据时把那次面板留出来，上传就带来了库里确实还没有的数据——「我的血脂什么趋势」这才是个真问题，而不是一个孤立的点：

```bash
uv run python -m generator.eslbench.seed_mirobody --users user5086@demo --hold-out-exams 1
uv run python -m generator.eslbench.labreport     --users user5086@demo -o samples/lab_report.pdf
```

这个演示会从灌入的历史中留出最新体检。之后若要评测依赖完整历史的题目，
请去掉 `--hold-out-exams` 再灌入一次。

`user5086@demo` 是一个生成出来的 58 岁 2 型糖尿病人，血脂在四次面板里先改善、后回落。每个数值都是合成的；PDF 首页就写着这一点。

打印出来的十二行里，八行能拿到 LOINC 编码，四行拿不到——报告上印的是 `High-Density Lipoprotein`，而 LOINC 编的是 `Cholesterol in HDL`；`LDL/HDL Ratio` 是个派生比值，压根没有对应的观测编码。这个混合是刻意的：它同时检验术语解析的命中与失手，而一个每行都能解析的面板做不到这件事。

## 用 Claude Code 做 AI 原生开发

mirobody-eval 的设计目标是完全通过 [Claude Code](https://docs.anthropic.com/en/docs/claude-code) 操作。每个常见任务都有专门的斜杠命令。你用自然语言说意图；Claude Code 去读代码、生成文件、跑测试、校验结果。

**你不用背 CLI 参数、不用读源码、不用写样板。** 敲斜杠命令，然后顺着对话走。

### 斜杠命令一览

| 你想做什么 | 命令 | Claude Code 替你做的事 |
|---|---|---|
| **初始化项目** | `/quick-start` | 检查 Python/uv、装依赖、把 API key 配进 `.env`、启动 Web UI |
| **跑一个基准** | `/run-benchmark` | 问你要哪个基准和数据集，然后按你选的模型和并发度执行 |
| **接一个新基准** | `/add-benchmark` | 端到端：读论文/仓库 → 分析数据格式 → 写转换器 → 生成数据集 → 校验 |
| **加一个自定义评测器** | `/add-eval-agent` | 生成配置模型 + 插件实现 + 注册。CLI 和 Web UI 里立即可用 |
| **加一个被测系统** | `/add-target-agent` | 为新的被测系统生成连接处理、消息处理和清理逻辑 |
| **审查架构** | `/review-architecture` | 检查 GitOps 合规、插件隔离、共享层复用，报告违规并给修法 |
| **分析跑分报告** | `/eslbench-report-analysis` | 按难度维度拆分数、方法间对比、看失败率与每题耗时/成本 |

### 工作流示例

**「我想在 GPT-4.1 上复现 HealthBench」**
```
> /run-benchmark
# Claude 问：哪个基准？→ healthbench
# 哪个数据集？→ sample
# 哪个模型？→ gpt-5.4-mini
# 跑多少条？→ 5（先从小开始！）
# 运行中…… 5 条 → 报告已保存
```

**「我需要一个检查引用准确性的评测器」**
```
> /add-eval-agent
# Claude 问：插件名？→ citation_accuracy
# 它评什么？→ 检查 AI 回答是否引用了有效来源
# 生成：evaluator/plugin/eval_agent/citation_accuracy_eval_agent.py
# 通过 __init_subclass__ 自动注册 —— 可以直接用
```

> **提示：** 你不必局限于斜杠命令。Claude Code 理解整个代码库——用自然语言问它任何事，比如*「插件系统是怎么工作的」*或*「这个用例为什么失败了」*。

## Web UI

`uv run python -m web` 启动，然后访问 http://localhost:8000。

<table>
<tr>
<td width="50%">

**发起评测** —— 选基准、配参数、启动任务，SSE 实时跟进度。

<img src="docs/screenshots/holyeval_tasks.jpg" alt="Run Evaluations" width="100%">
</td>
<td width="50%">

**评测报告** —— 带分数的结果，用例可展开，含对话历史和逐条反馈。

<img src="docs/screenshots/holyeval_report.jpg" alt="Evaluation Report" width="100%">
</td>
</tr>
<tr>
<td width="50%">

**浏览基准** —— 所有基准数据集的概览，含用例数和统计。
</td>
<td width="50%">

**Agent 注册表** —— 查看所有已注册插件的配置 schema、特性和成本估算。
</td>
</tr>
</table>

## Health Memory Arena —— 在线评测平台

[Health Memory Arena](http://healthmemoryarena.ai)（HMA）是由 mirobody-eval 驱动的公开评测平台。它托管着 ESL-Bench 排行榜，健康 AI agent 在结构化纵向推理任务上同台竞争。

> **现状说明**：凡是答案已发布的批次，本仓库能让你复现榜单上的分数——数据集拉取、跑分、报告里带批次号和校验和，这条链是完整的。按滚动发布政策，**最新一期在下一期上线前不公开答案**，那一期的分数在此之前无法在本地判分。此外，**把你自己的题库发布到 HuggingFace、以及把成绩提交上榜，这两步的工具还没有开源化**。想接入的话先开一个 issue，我们按需推进。

<table>
<tr>
<td width="33%">
<a href="http://healthmemoryarena.ai"><img src="docs/screenshots/hma_home.jpg" alt="HMA Home" width="100%"></a>
<p align="center"><em>平台首页</em></p>
</td>
<td width="33%">
<a href="http://healthmemoryarena.ai/leaderboard"><img src="docs/screenshots/hma_leaderboard.jpg" alt="HMA Leaderboard" width="100%"></a>
<p align="center"><em>Agent 排行榜</em></p>
</td>
<td width="33%">
<a href="http://healthmemoryarena.ai/dataset"><img src="docs/screenshots/hma_dataset.jpg" alt="HMA Dataset" width="100%"></a>
<p align="center"><em>数据集浏览</em></p>
</td>
</tr>
</table>

## 架构

```
TestCase (JSON) → Orchestrator
  1. 通过插件注册表按配置初始化各 agent
  2. 对话循环：TestAgent ↔ TargetAgent（直到结束或达到最大轮数）
  3. EvalAgent.run(conversation, session) → EvalResult
  4. 返回 TestResult（分数、通过/失败、反馈、成本）
```

所有执行路径（CLI、Web UI、程序调用）都汇聚到同一个入口：`do_single_test()`。

### 插件系统

三类 agent，每类都通过 `__init_subclass__` 自动注册来扩展：

```python
# 定义一个自定义评测器 —— 就这样，已经注册好了
class MyEvalAgent(AbstractEvalAgent, name="my_eval", params_model=MyEvalInfo):
    async def run(self, memory_list, session_info):
        # 你的评测逻辑
        return EvalResult(result="pass", score=0.95, feedback="...")
```

| Agent 类型 | 角色 | 内置插件 |
|---|---|---|
| **TestAgent** | 虚拟用户 | `auto`（LLM 驱动）、`manual`（照剧本） |
| **TargetAgent** | 被测系统 | `mirobody`（自部署实例）、`llm_api`（OpenAI / Gemini / OpenRouter）、`hermes`、`evermem`、`mem0_rag_api`、`naive_rag_api`、`hippo_rag_api`、`dyg_rag_api` |
| **EvalAgent** | 评测器 | `semantic`、`rubric`、`healthbench`、`medcalc`、`kg_qa`、`record_retrieval`、`dialogue_quality`、`engagement` |

### 项目结构

```
mirobody-eval/
├── evaluator/          # 核心引擎：schema、orchestrator、插件接口
├── benchmark/          # runner + 数据集（JSONL）+ 报告
│   └── data/eslbench/  # ESLBench：数据 + 工具（retrieve.py，JSON/DuckDB）
├── generator/          # 数据集转换器 + 数据准备脚本
│   └── eslbench/       # ESLBench 数据下载 + DuckDB 构建
└── web/                # Web UI（FastAPI + htmx）
```

## 基准

| 基准 | 论文 / 来源 | 数据集 | 评什么 |
|---|---|---|---|
| **HealthBench** | [OpenAI HealthBench](https://arxiv.org/abs/2505.07469) | `sample`（100）、`full`、`hard`、`consensus` | 医疗 AI 质量 |
| **MedCalc-Bench** | [MedCalc-Bench](https://arxiv.org/abs/2406.12036) | `sample`、`full` | 医学计算 |
| **ESLBench** | [arXiv:2604.02834](https://arxiv.org/abs/2604.02834) | `sample50-20260331`（50）、`sample500-20260331`（500）、`full-20260331`（4500） | 纵向健康推理 |
| **ESLBench-Distractor** | —— | `sample`（60）、`distractor-behavioral-20260723`（120）、`distractor-computable-20260723`（400） | 对干扰上下文的稳健性（复用 ESLBench 已准备好的数据） |
| **Virtual User** | —— | `round1`（15）、`round2`（60） | 开场白的互动质量 |

### ESLBench —— 事件驱动的合成纵向基准

ESLBench（[arXiv:2604.02834](https://arxiv.org/abs/2604.02834)）评测纵向健康推理能力——在多来源患者轨迹（设备数据流、临床检查、生活事件）之间对齐、聚合与归因的能力。它建立在一个事件驱动的合成框架上：每个用户的轨迹被建模为一个基线健康状态，加上一系列带明确时间核（sigmoid 起效、指数衰减）的离散事件，因此标准答案可由程序算出。

<p align="center">
  <img src="docs/screenshots/eslbench_trajectory.png" alt="ESL-Bench Trajectory Visualization" width="70%">
</p>

<p align="center"><em>四个月轨迹片段 —— 事件驱动的指标动态，sigmoid 起效与指数衰减。</em></p>

**100 个合成用户**，轨迹跨度 1–5 年，**10,000 道评测题**，覆盖五个维度、三档难度：

| 维度 | 考什么 | 例子 |
|---|---|---|
| **Lookup** | 直接取数 | 「2024-03-15 的静息心率是多少？」 |
| **Trend** | 时序规律分析 | 「哪个月步数最高？」 |
| **Comparison** | 跨事件/跨来源对比 | 「开始跑步之后平均步数变化多少？」 |
| **Anomaly** | 异常检测 | 「血糖有异常过吗？」 |
| **Explanation** | 因果归因 | 「按对血糖下降的影响给事件排序」 |

<details>
<summary><strong>基准结果 —— 3 种范式下的 13 种方法</strong></summary>
<br>
主要发现：数据库型 agent（48–58%）显著优于记忆型 RAG（30–38%），差距集中在 Comparison 和 Explanation 这两类需要多跳推理和证据归因的题上。
</details>

**需要先准备数据** —— ESLBench 会从 HuggingFace 下载用户数据，并为每个用户建 DuckDB 索引：

```bash
# 首次：准备数据（走 Web UI 会自动做，CLI 需手动）
uv run python -m generator.eslbench.prepare_data

# 快速验证：先跑 3 条用例检查环境
uv run python -m benchmark.basic_runner eslbench sample50-20260331 --target-type llm_api --target-model gpt-5.4-mini --limit 3

# 抽样数据集
uv run python -m benchmark.basic_runner eslbench sample50-20260331 --target-type llm_api --target-model gpt-5.4-mini      # 50 条
uv run python -m benchmark.basic_runner eslbench sample500-20260331 --target-type llm_api --target-model gpt-5.4-mini -p 5 # 500 条

# 全量（4500 条 —— API 成本可观，跑之前先想清楚）
uv run python -m benchmark.basic_runner eslbench full-20260331 --target-type llm_api --target-model gpt-5.4-mini -p 5
```

被测的 LLM 会拿到一组工具（`eslbench/retrieve`）：读 JSON 文件、查 DuckDB、查指标——它必须用这些工具在用户的健康数据里找答案。

### 新增一个基准

两种方式：

**A）用 Claude Code skill（推荐）：**
```
/add-benchmark    # 引导式：读论文 → 转数据 → 校验
```

**B）手动：**
1. 建 `benchmark/data/<name>/metadata.json`，写好被测目标配置
2. 建 `benchmark/data/<name>/<dataset>.jsonl`，用 BenchItem 格式
3. 跑：`uv run python -m benchmark.basic_runner <name> <dataset> --target-model gpt-5.4-mini`

最小的 `metadata.json` + `sample.jsonl` 组合可参考 [benchmark/data/medcalc/](benchmark/data/medcalc/)。

## 扩展 mirobody-eval

### 新增评测器

```python
# evaluator/plugin/eval_agent/my_eval_agent.py
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent
from evaluator.core.schema import EvalResult


class MyEvalInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evaluator: Literal["my_eval"] = "my_eval"
    threshold: float = Field(0.8, ge=0.0, le=1.0)


class MyEvalAgent(AbstractEvalAgent, name="my_eval", params_model=MyEvalInfo):
    async def run(self, memory_list, session_info):
        conversation = memory_list[-1].target_response
        score = your_scoring_logic(conversation)
        return EvalResult(result="pass" if score > 0.8 else "fail", score=score, feedback="...")
```

文件名必须以 `_eval_agent.py` 结尾——这个后缀是包内 `pkgutil` 自动导入的依据，而自动导入正是触发注册的东西。不需要往 `__init__.py` 里加任何东西。

### 新增被测系统

```python
# evaluator/plugin/target_agent/my_target_agent.py
from typing import Literal
from pydantic import BaseModel, ConfigDict

from evaluator.core.interfaces.abstract_target_agent import AbstractTargetAgent
from evaluator.core.schema import TargetAgentReaction


class MyTargetInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["my_target"] = "my_target"
    base_url: str


class MyTargetAgent(AbstractTargetAgent, name="my_target", params_model=MyTargetInfo):
    async def _generate_next_reaction(self, test_action):
        response = await call_your_api(test_action)
        return TargetAgentReaction(type="message", message_list=[{"content": response}])
```

想要引导式脚手架，用 Claude Code 的 `/add-eval-agent` 或 `/add-target-agent`。

## CLI 参考

```bash
# 准备基准数据（ESLBench 必需；其余基准数据自带）
uv run python -m generator.eslbench.prepare_data          # 下载 HF 数据 + 建 DuckDB
uv run python -m generator.eslbench.prepare_data --force   # 强制重建

# 跑基准
uv run python -m benchmark.basic_runner <benchmark> <dataset> [options]
  --target-type TYPE      # 被测 agent 类型（多 target 的基准需要指定）
  --target-model MODEL    # 被测模型（如 gpt-5.4-mini、anthropic/claude-sonnet-4.6）
  --user-model MODEL      # 扮演虚拟用户的模型（仅 auto 模式数据集有效）
  --eval-model MODEL      # 判分用的模型（纯规则判分的题型不受影响）
  --system-prompt TEXT    # 覆盖被测系统的 system prompt
  --target-override K=V   # 覆盖某个可编辑的 target 字段，如 agent=Deep
  --limit N               # 最多跑多少条
  --ids id1,id2           # 只跑指定 ID 的用例
  -p, --parallel N        # 并发数（默认 0 = 不限）
  -v, --verbose           # 详细日志
  --resume                # 从上次检查点续跑

# 转换外部数据集
uv run python -m generator.healthbench.converter input.jsonl output.jsonl
uv run python -m generator.medcalc.converter input.csv output.jsonl
uv run python -m generator.virtual_user case_gen --seed 42 \
    --output benchmark/data/virtual_user/my_round.jsonl   # ⚠ 不传 --output 会覆盖随附的 round1.jsonl

# Web UI
uv run python -m web             # http://localhost:8000
```

一次评测最多驱动三个模型——虚拟用户、被测系统、判分器——每个都单独指定，所以一个 provider 可以同时充当三者：

```bash
uv run python -m benchmark.basic_runner virtual_user round1 \
    --target-type llm_api --target-model anthropic/claude-sonnet-4.6 \
    --user-model anthropic/claude-sonnet-4.6 \
    --eval-model anthropic/claude-sonnet-4.6
```

对于 ESL-Bench 的 `kg_qa` 判分器，`text` 和 `behavioral` 使用 LLM 判分；
`numeric_value`、`boolean` 和 `list` 使用规则。规则判分不需要判分用的 key，
但被测系统回答问题时仍可能需要调用模型。

超时、取消和执行异常记录为 `error`。平均分只统计 `pass`、`fail`、`scored` 用例。
CLI 汇总会显示错误数量和“已评分/总数”；没有任何用例判成时，平均分显示为 `—`。
比较成绩时必须同时看完成覆盖率：只完成少数题的高分不能代表完整题库成绩。
保存的 JSON 在没有成绩时仍保留 `avg_score: 0.0`，读取报告时需结合 `error_count`
和用例总数判断。其他判分器可能有各自的失败处理方式。

## 配置

在 `.env` 中配置被测模型、判分器和虚拟用户所用服务商的 key。随附默认配置使用
OpenAI 模型，可通过 OpenAI 或 OpenRouter 调用；使用其他服务商时，请分别指定
每个 LLM 角色的模型。

| 变量 | 何时需要 | 说明 |
|---|---|---|
| `OPENAI_API_KEY` | 使用 OpenAI 模型 | 直连 OpenAI API |
| `OPENROUTER_API_KEY` | 使用 OpenRouter 模型 | 未设置 `OPENAI_API_KEY` 时，也用于不带前缀的 `gpt*` 模型名 |
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | 使用 Google 模型 | 选择 `google:` 或 `gemini*` 模型；Vertex AI 也可使用应用默认凭据 |
| `DASHSCOPE_API_KEY` | 使用 `dashscope:` | 阿里云百炼 |
| `DEEPSEEK_API_KEY` | 使用 `deepseek:` | DeepSeek 官方直连 |
| `MOONSHOT_API_KEY` | 使用 `moonshot:` | 月之暗面（Kimi）官方直连 |
| `ZHIPU_API_KEY` | 使用 `zhipu:` | 智谱（GLM）官方直连 |
| `VOLCENGINE_API_KEY` | 使用 `volcengine:` | 火山方舟（豆包）官方直连 |
| `HF_TOKEN` | 公开数据集可选 | HuggingFace token；若选择私有或受限数据集，需要相应访问权限 |

### 模型名怎么写

| 写法 | 路由 |
|---|---|
| `gpt-5.4-mini` | OpenAI；若只配置了 OpenRouter key，则通过它请求 `openai/gpt-5.4-mini`，并记录改道日志 |
| `openai/gpt-5-mini` | OpenRouter，斜杠属于它的模型 ID |
| `openai:gpt-4.1` | 显式指定 OpenAI 官方端点，不会自动改道 |
| `google:YOUR_MODEL_ID` | Google SDK；请把占位符替换为可用模型 ID |
| `dashscope:YOUR_MODEL_ID` | 百炼端点；请把占位符替换为可用模型 ID |

支持的前缀：`openrouter`、`dashscope`、`deepseek`、`volcengine`、`zhipu`、
`moonshot`、`openai`、`google`。前缀只负责选择端点，不保证模型存在或账号有访问
权限；不认识的前缀会报错。不带前缀时，`gpt*` 走 OpenAI，`gemini*` 走 Google，
其他模型名走 OpenRouter；`[label]model` 走配置的自建网关。

这套写法同时适用于 `--target-type llm_api` 的 `--target-model`、判分器的
`--eval-model`，以及自动虚拟用户的 `--user-model`。改其中一个不会改变另外两个。
Web UI 的 Model 输入框支持自由填写模型 ID。这些评测设置不会修改正在运行的
mirobody 服务端的模型配置。

分隔符是斜杠之前的冒号。OpenRouter 的 `anthropic/claude-sonnet-4.5:batch` 等
模型 ID 会保留原后缀。六个 OpenAI 兼容服务商（`openrouter`、`dashscope`、
`deepseek`、`volcengine`、`zhipu`、`moonshot`）还支持对应的
`<PROVIDER>_BASE_URL` 环境变量。`google:` 使用 Google SDK 的端点配置，
不读取 `GOOGLE_BASE_URL`。

### 运行和网关配置

| 变量 | 何时需要 | 说明 |
|---|---|---|
| `HOLYEVAL_GATEWAY_BASE_URL` | 使用 `[label]model` | 自建 OpenAI 兼容网关地址 |
| `HOLYEVAL_GATEWAY_API_KEY` | 网关鉴权 | 网关的 API key |
| `HOLYEVAL_WEB_PORT` | 可选 | Web UI 端口，默认 8000 |
| `HOLYEVAL_HEALTH_PORT` | 可选 | 健康检查端口，默认 8001 |
| `HOLYEVAL_RELOAD` | 可选 | `true` 开启 uvicorn 自动重载，默认 false |
| `AGENT_LLM_TIMEOUT` | 可选 | 框架 LLM 超时秒数，默认 840 |

### mirobody 部署配置

这些参数用于指定要灌入数据并评测的部署，请在启动 runner 或 Web UI 前设置。
如果部署使用加密配置，还需通过评测进程的环境变量提供该部署的
`CONFIG_ENCRYPTION_KEY`。

| 变量 | 何时需要 | 说明 |
|---|---|---|
| `MIROBODY_CONFIG` | 使用 `--target-type mirobody` 时建议配置 | 部署配置文件的绝对路径，其中的数据库和 Redis 地址需能从当前进程访问 |
| `MIROBODY_BASE_URL` | 填部署的实际地址 | 默认 `http://localhost:18080`，请使用实际暴露的端口 |
| `MIROBODY_TIMEOUT` | 可选 | 单次对话请求的超时秒数；不配则用 `AGENT_LLM_TIMEOUT`，默认 840。一次请求可能包含多次模型和工具调用 |
| `MIROBODY_PROVIDER` | 可选 | 选择 mirobody 服务端已配置的 provider；留空使用部署默认值 |

## 路线图

### 进行中
- [ ] **GUI TargetAgent** —— 通过真实产品的 Web 界面评测，而不只是 API 端点。浏览器里的 agent 像真人一样操作你的应用，从而对任何带前端的产品做端到端评测

### 计划中
- [ ] **评测驱动的优化闭环** —— 跑基准 → 自动分析失败模式 → 生成针对性的 prompt/系统改进 → 重跑验证。把评测和迭代接成一个环
- [ ] **CI/CD 集成** —— `pip install mirobody-eval` 加上 `mirobody_eval.run("healthbench", model="gpt-5.4-mini")`，在你的 CI 流水线里一行搞定。跨轮次回归检测，分数掉了在上线前就告警
- [ ] **行业 agent 与 App 深度评测** —— 对主流 AI agent 和健康类 App（如 ChatGPT、Gemini、各类健康助手）出完整评测报告。在安全性、准确性、用户体验上做标准化打分，作为可复现的社区基准发布

## 开发

```bash
# 单元测试 —— 纯逻辑，不连库、不联网、不调模型
uv run --group dev python -m pytest generator/ evaluator/ -q

# 自检 —— 插件注册表能否加载
uv run python -c "import evaluator.plugin.eval_agent, evaluator.plugin.target_agent; \
from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent; \
print(sorted(AbstractEvalAgent.get_all()))"

# Lint 与格式检查（安装 lint 依赖组）
uv run --group lint ruff check .
uv run --group lint ruff format --check .
```

CI 会运行单元测试、插件注册和数据集插件引用检查。Lint 和格式检查目前只报告问题，
不会阻断 CI，因此即使 CI 通过，本地仍可能看到既有检查项。请检查自己改动文件的
检查结果；CI 通过不代表整个仓库已无 lint 问题。

## 参与贡献

欢迎贡献！最省事的路径是通过 Claude Code——下面每种都有对应的引导式斜杠命令：

| 贡献类型 | 怎么开始 | 难度 |
|---|---|---|
| **加一个基准** | `/add-benchmark` —— 最快的贡献方式 | 容易 |
| **加一个评测器** | `/add-eval-agent` —— 搭一套新的打分方法 | 中等 |
| **加一个被测系统** | `/add-target-agent` —— 接一个新的 API/服务来评测 | 中等 |
| **改进已有基准** | 补更多用例、边界情况，或更好的 prompt | 容易 |

较大的改动请先开 issue 讨论。

## 引用

如果你在研究中用到 ESL-Bench 或 mirobody-eval，请引用：

```bibtex
@article{li2026eslbench,
  title={ESL-Bench: An Event-Driven Synthetic Longitudinal Benchmark for Health Agents},
  author={Li, Chao and Liu, Cailiang and Gao, Ang and Deng, Kexin and Zhang, Shu and Xu, Langping and Shi, Xiaotong and Ding, Xionghao and Pei, Jian and Jiang, Xun},
  journal={arXiv preprint arXiv:2604.02834},
  year={2026}
}
```

## 许可

[MIT](LICENSE)。第三方组件及其许可证列在 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
