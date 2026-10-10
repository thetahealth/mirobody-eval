---
name: quick-start
description: 引导项目初始化 — 环境安装、配置、验证、跑通第一个 benchmark、启动 Web UI。
---

# Quick Start

引导用户完成 HolyEval 的初始化设置。按顺序执行以下步骤，每步完成后继续下一步，不要等待用户确认。

## Workflow

### Step 1: 检查环境

```bash
python3 --version   # 需要 Python >= 3.11
uv --version        # 需要 uv 包管理器
```

如果缺少，给出安装指引并停止：
- Python: https://www.python.org/downloads/ 或 `brew install python@3.11`
- uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Step 2: 安装依赖

```bash
uv sync
```

### Step 3: 配置环境变量

检查根目录 `.env` 是否存在。不存在则从 `.env.example` 复制：

```bash
cp .env.example .env
```

读取 `.env`，检查 API Key 是否已填。如果全是占位符（`sk-xxx` / `xxx` / `hf_xxx`），
**暂停并提示用户**至少配一个 LLM 提供商：

| 变量 | 说明 | 获取方式 |
|------|------|----------|
| `OPENAI_API_KEY` | OpenAI API 密钥（gpt-5.4 等） | https://platform.openai.com/api-keys |
| `GOOGLE_API_KEY` | Google AI API 密钥（gemini-3 等） | https://aistudio.google.com/apikey |
| `OPENROUTER_API_KEY` | OpenRouter 统一多提供商访问（可选） | https://openrouter.ai/keys |
| `HF_TOKEN` | HuggingFace Token（仅 ESLBench 数据下载需要；不配不影响其他 benchmark） | https://huggingface.co/settings/tokens |

如果已有有效 key，直接继续。

### Step 4: 验证安装

```bash
uv run python -c "import evaluator, benchmark, generator; print('OK')"
```

如果报错，诊断并修复后再继续。

### Step 5: 跑通第一个 benchmark

这一步是为了让用户**立刻看到一次真实的评测**，而不是只装好环境。
下面两个 benchmark 的数据随仓库发布，不需要额外下载：

```bash
# MedCalc — 医学计算题，确定性判分，最便宜
uv run python -m benchmark.basic_runner medcalc sample --target-model gpt-5.4-mini --limit 3

# HealthBench — rubric 评分
uv run python -m benchmark.basic_runner healthbench sample --target-model gpt-5.4-mini --limit 3
```

跑完后把报告路径告诉用户（默认写在 `benchmark/report/` 下）。
`--limit 3` 是为了先花几分钱确认链路通，确认后去掉即可跑全量。

### Step 5.5: 准备 ESLBench 数据（可选）

ESLBench 需要从 HuggingFace 下载合成健康 KG 数据并建 per-user DuckDB 索引。
Web UI 启动时会自动执行；CLI 场景手动跑：

```bash
uv run python -m generator.eslbench.prepare_data
```

如果 `.env` 里 `HF_TOKEN` 还是占位符 `hf_xxx`，**提示用户**：
- ESLBench 需要 HuggingFace Token 才能下载数据，获取：https://huggingface.co/settings/tokens
- 不配置不影响其他 benchmark，可以后续再配

注意：
- 首次下载可能几分钟（取决于网络）
- 数据落在 `benchmark/data/eslbench/.data/`
- 支持增量更新，重复运行会跳过已下载的部分
- 出错且非关键（如网络抖动）时可重试：`uv run python -m generator.eslbench.prepare_data --force`

### Step 6: 启动 Web UI

用 Bash 工具的 `run_in_background: true` 启动服务，再单独打开浏览器：

```bash
uv run python -m web            # 后台启动
```

```bash
# 等服务就绪后打开浏览器（macOS 用 open，Linux 用 xdg-open）
sleep 2 && (open http://localhost:8000 2>/dev/null || xdg-open http://localhost:8000 2>/dev/null || echo "请手动访问 http://localhost:8000")
```

端口可用 `HOLYEVAL_WEB_PORT` 环境变量自定义（默认 8000）。

### Step 7: 展示项目概览

读取 `web/guides/overview.md`，在对话窗口输出项目概览。

**输出内容包括**：
- **项目简介**：一句话描述 HolyEval 的定位
- **核心架构**：
  - 三大目录：evaluator（评测框架 + 对接服务）、generator（数据转换）、benchmark（批量调度）
  - 三层 Agent 架构：TestAgent / TargetAgent / EvalAgent
  - 执行流程：对话循环 → 评估 → 生成报告
- **核心功能**：插件化 Agent 系统、批量评测执行、多维度评估、数据管理
- **技术栈**：Python 3.11+, FastAPI, uv workspace, Pydantic v2, LangChain

**引导用户**：
- Web UI 已在浏览器打开：**http://localhost:8000**
- 项目概览页：**http://localhost:8000/guides/overview**
- 推荐后续操作（下表）

**输出格式**：markdown 表格和列表；简洁易读。

**推荐后续操作**：

| Skill | 说明 |
|-------|------|
| `/run-benchmark` | 运行 benchmark 跑分 |
| `/add-benchmark` | 集成外部 benchmark（研究 → 转换 → 验证全流程） |
| `/add-eval-agent` | 新增评测逻辑插件 |
| `/add-target-agent` | 新增被测系统插件 |
| `/review-architecture` | 审查项目架构健康度 |
