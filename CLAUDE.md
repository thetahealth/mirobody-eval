# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HolyEval is an open-source LLM evaluation framework. Reproduce any published benchmark with one command; extend with custom evaluators via a pluggable agent architecture.

## Commands

```bash
# Install dependencies (uv workspace)
uv sync

# Run benchmarks
python -m benchmark.basic_runner healthbench sample --target-model gpt-4.1
python -m benchmark.basic_runner healthbench full --target-model gpt-4.1 --limit 50
python -m benchmark.basic_runner healthbench hard --target-model gemini-3-pro -p 5
python -m benchmark.basic_runner medcalc sample --target-model gpt-4.1
python -m benchmark.basic_runner eslbench sample50-20260324 --target-model gpt-4.1        # ESLBench quick (50 cases)
python -m benchmark.basic_runner eslbench full-20260324 --target-model gpt-4.1 -p 5       # ESLBench full (1800 cases)
python -m benchmark.basic_runner healthbench sample --target-model gpt-4.1 --ids hb_abc
python -m benchmark.basic_runner healthbench sample --target-model gpt-4.1 --limit 10 -p 3 -v
python -m benchmark.basic_runner healthbench sample --resume

# Data preparation (required before running ESLBench via CLI; automatic via Web UI)
python -m generator.eslbench.prepare_data            # download HF data + build per-user DuckDB
python -m generator.eslbench.prepare_data --force    # force re-download + rebuild

# Data conversion (external datasets → HolyEval format)
python -m generator.healthbench.converter input.jsonl output.jsonl --target-model gpt-4.1
python -m generator.medcalc.converter

# Web UI
python -m web    # uvicorn :8000, auto-reload

# Tests
pytest evaluator/tests/
pytest evaluator/tests/test_e2e.py

# Lint
ruff check .
ruff format .
```

## Architecture

### Execution Flow

```
TestCase (JSON) → Orchestrator (do_single_test)
  1. Initialize agents from TestCase config via plugin registry
  2. Dialogue loop: TestAgent ↔ TargetAgent (until is_finished or max_turns)
  3. EvalAgent.run(memory_list, session_info) → EvalResult
  4. Return TestResult (score, pass/fail, feedback, cost)
```

All call paths (CLI, batch, API) funnel through `evaluator/core/orchestrator.py:do_single_test()`.

### Batch Execution

```python
session = BatchSession(cases, max_concurrency=5, on_progress=callback)
report = await session.run()       # Returns TestReport
session.snapshot()                  # JSON-serializable progress snapshot
session.cancel()                    # Cooperative cancellation
```

### Plugin System

Three agent types use `__init_subclass__` auto-registration:

```python
class CustomTestAgent(AbstractTestAgent, name="custom"):
    ...
# Lookup: AbstractTestAgent.get("custom")
```

Plugins activate on import (in `evaluator/plugin/`). The `core/` layer depends only on abstract interfaces.

| Agent Type | Interface | Built-in Plugins |
|---|---|---|
| **TestAgent** (virtual user) | `core/interfaces/abstract_test_agent.py` | `auto` (LLM-driven), `manual` (scripted) |
| **TargetAgent** (system under test) | `core/interfaces/abstract_target_agent.py` | `llm_api` (OpenAI/Gemini) |
| **EvalAgent** (evaluator) | `core/interfaces/abstract_eval_agent.py` | `semantic`, `healthbench`, `medcalc`, `kg_qa` |

Add custom plugins by inheriting from the abstract base classes. Use `/add-eval-agent` or `/add-target-agent` skills for guided scaffolding.

#### Plugin Metadata

Plugins can declare class attributes for inspector discovery:

| Attribute | Applies to | Description |
|---|---|---|
| `_display_meta` | All | Display metadata: `icon`, `color`, `features` |
| `_cost_meta` | EvalAgent | `{"est_cost_per_case": float}` (USD/case) |
| `_cost_meta` | TargetAgent | `{"est_input_tokens": int, "est_output_tokens": int}` |
| `_config_model` | TestAgent | Config model class name in schema.py |

### Key Modules

- **`evaluator/core/schema.py`** — Pydantic v2 data models: TestCase, UserInfo, TargetInfo, EvalInfo, TestResult, SessionInfo
- **`evaluator/core/orchestrator.py`** — `do_single_test()`, `do_batch_test()`, `BatchSession`, `CaseContext`, `CaseStatus`
- **`evaluator/utils/llm.py`** — Unified LLM interface `do_execute()` via langchain. Supports OpenAI and Google Gemini
- **`evaluator/core/bench_schema.py`** — Benchmark models: BenchItem, BenchMark, BenchReport, conversion functions
- **`evaluator/utils/benchmark_reader.py`** — Read/load `benchmark/data/` (shared by CLI + Web)
- **`evaluator/utils/report_reader.py`** — Read/write `benchmark/report/` (shared by CLI + Web)
- **`evaluator/utils/agent_inspector.py`** — Reflect plugin registry for metadata (shared by CLI + Web)
- **`evaluator/utils/checkpoint.py`** — Checkpoint manager for resume-on-interrupt

### Workspace Structure

uv workspace monorepo with four members:
- **`evaluator/`** — Core evaluation engine
- **`benchmark/`** — Benchmark runner + data + reports
- **`generator/`** — Dataset converters
- **`web/`** — Web UI (FastAPI + htmx + Alpine.js + Tailwind CSS)

### Benchmark Data

```
benchmark/
├── data/
│   ├── eslbench/         # ESLBench health KG Q&A (requires data preparation)
│   │   ├── tools/        # retrieve.py — JSON lookup + DuckDB query tools
│   │   └── .data/        # Downloaded user data + DuckDB (auto-created by prepare_data)
│   ├── healthbench/      # HealthBench medical AI
│   └── medcalc/          # MedCalc-Bench calculations
├── report/               # Output reports
└── basic_runner.py       # CLI runner
```

Each benchmark directory contains `<dataset>.jsonl` (data) + `metadata.json` (suite config).

Report filename format: `{dataset}_{target_label}_{YYYYMMDD_HHmmss}.json`

**metadata.json** supports multiple target types via TargetSpec array:

```json
{
  "description": "...",
  "target": [
    { "type": "llm_api", "fields": { "model": {"default": "gpt-4.1", "editable": true, "required": true} } }
  ],
  "params": {
    "shared_history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
  }
}
```

- Single target → auto-selected by CLI/Web
- Multiple targets → CLI uses `--target-type`, Web shows selector
- `params` (optional): shared data dict. JSONL fields with `{"$ref": "key"}` resolve to `params[key]` at load time

### Test Cases

JSONL files in `benchmark/data/` (benchmarks) and `evaluator/tests/fixtures/cases/` (dev). Each case specifies user config, target config, eval config, and optional `history`.

Key fields:
- **`strict_inputs`** (`List[str]`): Manual mode sends these sequentially
- **`history`** (`List[Dict]`): Pre-loaded conversation context (skips dialogue loop). Combinable with `strict_inputs`

### Data Converters

`generator/` transforms external datasets into HolyEval BenchItem format:

- **`generator/eslbench/prepare_data.py`** — ESLBench data preparation: HuggingFace download + per-user DuckDB creation
- **`generator/healthbench/converter.py`** — HealthBench JSONL → BenchItem
- **`generator/medcalc/converter.py`** — MedCalc-Bench CSV → BenchItem

### Web UI

```bash
python -m web    # uvicorn :8000, auto-reload
```

| Page | Route | Description |
|------|-------|-------------|
| Run evaluations | `/tasks` | Select benchmark, configure, launch with SSE progress |
| Task details | `/tasks/{id}` | Progress cards, expandable case list |
| Reports | `/reports/{benchmark}/{file}` | Report viewer |
| Datasets | `/benchmarks` | Browse benchmark datasets |
| Agent registry | `/agents/*` | Inspect registered plugins |

## Environment Variables

Configure in `.env` (copy from `.env.example`):

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | At least one LLM | OpenAI API key |
| `GOOGLE_API_KEY` | At least one LLM | Google Gemini API key |
| `HF_TOKEN` | ESLBench | HuggingFace token for downloading ESLBench data |
| `OPENROUTER_API_KEY` | Optional | OpenRouter multi-provider access |
| `HOLYEVAL_PORT` | Optional | Web UI port (default: 8000) |

## Code Style

- Python 3.11+, async/await throughout
- Ruff for linting/formatting, line-length 120
- Pydantic v2 for all data models
