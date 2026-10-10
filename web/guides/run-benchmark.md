# Benchmark Scoring

> **Using in Claude Code**: Type `/run-benchmark`, followed by your benchmarking requirements (e.g., `healthbench sample --target-model gpt-5.4-mini`, specify concurrency). Claude will automatically configure and execute the benchmark run.

## Overview

Benchmark scoring batch-executes evaluations for all cases in a dataset, generating structured reports containing pass rates, score distributions, and cost statistics. Both CLI and Web UI methods are supported.

## CLI Benchmarking

### Basic Usage

```bash
# Syntax: python -m benchmark.basic_runner <benchmark> <dataset> --target-type <type> --target-model <model> [options]

# HealthBench — physician-written rubric scoring
python -m benchmark.basic_runner healthbench sample --target-type llm_api --target-model gpt-5.4-mini
python -m benchmark.basic_runner healthbench hard --target-type llm_api --target-model gpt-5.4-mini -p 5

# MedCalc-Bench — medical calculation, deterministic grading
python -m benchmark.basic_runner medcalc sample --target-type llm_api --target-model gpt-5.4-mini
python -m benchmark.basic_runner medcalc full --target-type llm_api --target-model gpt-5.4-mini --limit 50

# Virtual User — LLM-generated patients, engagement scoring
python -m benchmark.basic_runner virtual_user round1 --target-type llm_api --target-model gpt-5.4-mini

# ESL-Bench — event-driven longitudinal health reasoning (run prepare_data first, see below)
python -m benchmark.basic_runner eslbench sample50-20260331 --target-type llm_api --target-model gpt-5.4-mini
python -m benchmark.basic_runner eslbench sample500-20260331 --target-type llm_api --target-model gpt-5.4-mini -p 5

# ESL-Bench Distractor — same tasks with distractor context injected
python -m benchmark.basic_runner eslbench_distractor sample --target-type llm_api --target-model gpt-5.4-mini
```

> **ESL-Bench data prep**: `eslbench` and `eslbench_distractor` download synthetic health-KG data from
> HuggingFace and build a per-user DuckDB index. Run `python -m generator.eslbench.prepare_data` once
> (requires `HF_TOKEN` in `.env`). The Web UI does this automatically. The other benchmarks ship with
> their data included.

### Options

| Option | Description |
|------|------|
| `--target-type` | System-under-test type (`llm_api`, `hermes`, `evermem`, `mem0_rag_api`, `naive_rag_api`, `hippo_rag_api`, `dyg_rag_api`), optional — when omitted, the **first** target in `metadata.json` is used |
| `--target-model` | Model name (e.g., `gpt-5.4-mini`, `gemini-3-pro-preview`) |
| `--limit N` | Run only the first N cases |
| `--ids x,y,z` | Specify case IDs |
| `-p N` | Concurrency level (default 0 = unlimited) |
| `-v` | Verbose logging |

### Combined Examples

```bash
# Specify IDs + verbose logging
python -m benchmark.basic_runner healthbench sample --target-type llm_api --target-model gpt-5.4-mini --ids hb_abc,hb_def -v

# Limit count + concurrency
python -m benchmark.basic_runner medcalc full --target-type llm_api --target-model gpt-5.4-mini --limit 10 -p 3 -v
```

## Web UI Benchmarking

1. Start the web server: `python -m web`
2. Navigate to http://localhost:8000
3. On the **Run Evaluations** page, select a benchmark and dataset
4. Configure target parameters (type, model) and concurrency
5. Click to create the task; progress is pushed in real time via SSE

The Web UI provides:
- Real-time progress tracking (SSE push)
- Per-case conversation process viewing
- Task cancellation
- Automatic report generation upon completion

## Reports

### Report Location

```
benchmark/report/<benchmark>/<dataset>_<target>_<timestamp>.json
```

Example: `benchmark/report/healthbench/sample_gpt-5.4-mini_20260213_183356.json`

Override the root with the `HOLYEVAL_REPORT_DIR` environment variable.

### Report Contents

- Overall pass rate and average score
- Per-case pass/fail, score, and feedback
- Conversation turn statistics
- Cost statistics (test / eval / target)

### Viewing Reports

- **CLI**: Reports are printed to the terminal after benchmarking completes
- **Web UI**: View in the "Historical Reports" list at the bottom of the Run Evaluations page; supports filtering by benchmark and filename

## Available Datasets

| Benchmark Suite | Dataset | Case Count | Evaluator | Description |
|----------|--------|--------|--------|------|
| healthbench | sample | 100 | healthbench | Proportionally sampled by topic, quick validation |
| healthbench | hard | 1,000 | healthbench | High-difficulty subset |
| healthbench | consensus | 3,671 | healthbench | Physician consensus subset |
| healthbench | full | 5,000 | healthbench | Full dataset |
| medcalc | sample | 50 | medcalc | Covers multiple calculator types |
| medcalc | full | 1,100 | medcalc | Full test set |
| virtual_user | round1 | 15 | engagement | Opening-line engagement, LLM-generated patient personas |
| virtual_user | round2 | 60 | engagement | Wider opening-line × persona grid |
| virtual_user | layer2_explore | 10 | semantic | Exploratory follow-up quality |
| virtual_user | layer2_insight | 5 | semantic | Insight-quality probes |
| eslbench | sample50-20260331 | 50 | kg_qa | Quick validation slice |
| eslbench | sample500-20260331 | 500 | kg_qa | Mid-size slice |
| eslbench | full-20260331 | 4,500 | kg_qa | Full snapshot |
| eslbench_distractor | sample | 60 | kg_qa | Distractor-injected smoke slice |
| eslbench_distractor | distractor-computable-20260723 | 400 | kg_qa | Computable-answer distractor set |
| eslbench_distractor | distractor-behavioral-20260723 | 120 | kg_qa | Behavioral-answer distractor set |

`eslbench` ships several dated snapshots beyond the three listed above (`sample50-*`, `sample200-*`,
`sample500-*`, `full-*`); run `ls benchmark/data/eslbench/*.jsonl` to see everything available.
