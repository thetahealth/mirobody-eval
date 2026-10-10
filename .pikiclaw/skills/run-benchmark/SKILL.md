---
name: run-benchmark
description: Run all benchmark test cases or a filtered subset.
argument-hint: "<benchmark> <dataset> [options]"
---

# Run Benchmark

运行 benchmark 跑分，对数据集中的用例批量执行评测并生成报告。

## Usage

当用户要求运行 benchmark 跑分时使用此 skill。

## Commands

```bash
# 语法: python -m benchmark.basic_runner <benchmark> <dataset> --target-type <type> --target-model <model> [options]

# HealthBench
python -m benchmark.basic_runner healthbench sample --target-type llm_api --target-model gpt-5.4-mini
python -m benchmark.basic_runner healthbench hard --target-type llm_api --target-model gpt-5.4-mini -p 5

# MedCalc-Bench
python -m benchmark.basic_runner medcalc sample --target-type llm_api --target-model gpt-5.4-mini
python -m benchmark.basic_runner medcalc full --target-type llm_api --target-model gpt-5.4-mini --limit 50

# ESL-Bench（需先跑 generator.eslbench.prepare_data 下载数据）
python -m benchmark.basic_runner eslbench sample50-20260331 --target-type llm_api --target-model gpt-5.4-mini

# 组合选项
python -m benchmark.basic_runner healthbench sample --target-type llm_api --target-model gpt-5.4-mini --limit 10 -p 3 -v
```

## Options

| 选项 | 说明 |
|------|------|
| `--target-type` | 被测系统类型（`llm_api` / `hermes` / `evermem` / `mem0_rag_api` / `naive_rag_api` / `hippo_rag_api` / `dyg_rag_api`）。可省略——省略时取 metadata.json 里的**第一个** target（顺序变了会静默换目标）|
| `--target-model` | 模型名称（如 `gpt-5.4-mini`、`gemini-3-pro-preview`） |
| `--limit N` | 只跑前 N 条用例 |
| `--ids x,y,z` | 指定用例 ID |
| `-p N` | 并发数（默认 0 = 不限制） |
| `-v` | 详细日志 |

## Directories

- Benchmark 数据集: `benchmark/data/<benchmark>/<dataset>.jsonl`
- 评测报告: `benchmark/report/<benchmark>/<dataset>_<target>_<timestamp>.json`

## Available Datasets

| 评测套件 | 数据集 | 数量 | 评估器 |
|----------|--------|------|--------|
| healthbench | sample / hard / consensus / full | 100 ~ 5,000 | healthbench |
| medcalc | sample / full | 50 ~ 1,100 | medcalc |
| virtual_user | round1 / round2 | 15 ~ 60 | engagement |
| eslbench | sample50-* / sample500-* / full-* | 50 ~ 4,500 | kg_qa |
| eslbench_distractor | sample / distractor-* | 60 ~ 400 | kg_qa |

## Web UI Alternative

也可通过 Web UI 执行跑分，支持实时进度查看：

```bash
python -m web
# 访问 http://localhost:8000/tasks 创建任务
```

## Output

命令输出：
1. 每条用例的执行进度
2. 汇总统计（通过率、平均分、成本）
3. 报告文件保存到 `benchmark/report/` 目录
