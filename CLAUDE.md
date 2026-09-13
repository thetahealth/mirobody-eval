# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mirobody-eval is the evaluation half of [mirobody](https://github.com/thetahealth/mirobody): seed a synthetic user into your own deployment, score it, and reproduce any published benchmark with one command. Extend it with custom evaluators via a pluggable agent architecture.

## Commands

```bash
# Install dependencies (uv workspace)
uv sync

# Add the engine, only needed for `--target-type mirobody`. The engine is 3.12+,
# while this repo runs on 3.11 — on 3.11 the extra installs nothing.
uv sync --extra mirobody --python 3.12

# Score a self-hosted mirobody deployment (seed the selected question's user first).
# MIROBODY_CONFIG names the deployment; both the seeder and the target agent read it.
export MIROBODY_CONFIG=/abs/path/to/deployment/config.localdb.yaml
export MIROBODY_BASE_URL=http://localhost:18080
uv run python -m generator.eslbench.prepare_data  # all missing or changed batches, not one user
python -m generator.eslbench.seed_mirobody --users user5086@demo
python -m benchmark.basic_runner eslbench sample200-20260430 --target-type mirobody \
    --ids user5086_AT_demo_Q001,user5086_AT_demo_Q087,user5086_AT_demo_Q067 -p 1

# Run benchmarks
python -m benchmark.basic_runner healthbench sample --target-model gpt-5.4-mini
python -m benchmark.basic_runner healthbench full --target-model gpt-5.4-mini --limit 50
python -m benchmark.basic_runner healthbench hard --target-model gemini-3-pro-preview -p 5
python -m benchmark.basic_runner medcalc sample --target-model gpt-5.4-mini
# eslbench declares six targets, so --target-type is required — without it the
# runner used to silently take the first (mirobody) and fail on a DB connection.
python -m benchmark.basic_runner eslbench sample50-20260324 --target-type llm_api --target-model gpt-5.4-mini   # quick (50 cases)
python -m benchmark.basic_runner eslbench full-20260324 --target-type llm_api --target-model gpt-5.4-mini -p 5  # full (1800 cases)
python -m benchmark.basic_runner healthbench sample --target-model gpt-5.4-mini --ids hb_abc
python -m benchmark.basic_runner healthbench sample --target-model gpt-5.4-mini --limit 10 -p 3 -v
python -m benchmark.basic_runner healthbench sample --resume

# A run drives up to three models independently: the virtual user, the target, the judge
python -m benchmark.basic_runner eslbench sample50-20260324 --target-type llm_api \
    --user-model gpt-4.1 --target-model gpt-5.4-mini --eval-model gpt-5.4-mini
# Set a target field the dataset leaves editable. Which fields those are is per
# dataset: eslbench pins the identity in each case and leaves `agent` open, so
# `--target-override user_email=…` there is ignored with a warning, not applied.
python -m benchmark.basic_runner eslbench sample50-20260324 \
    --target-type mirobody --target-override agent=Mix

# Data preparation (required before running ESLBench via CLI; automatic via Web UI)
python -m generator.eslbench.prepare_data            # download HF data + build per-user DuckDB
python -m generator.eslbench.prepare_data --force    # force re-download + rebuild

# Data conversion (external datasets → mirobody-eval format)
python -m generator.healthbench.converter input.jsonl output.jsonl
python -m generator.medcalc.converter input.csv output.jsonl

# Web UI
python -m web    # uvicorn :8000 (+ health :8001)

# Lint (currently advisory in CI; existing findings may remain)
uv run --group lint ruff check .
uv run --group lint ruff format --check .
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
| **TargetAgent** (system under test) | `core/interfaces/abstract_target_agent.py` | `mirobody` (a self-hosted deployment), `llm_api`, `hermes`, `evermem`, `mem0_rag_api`, `naive_rag_api`, `hippo_rag_api`, `dyg_rag_api` |
| **EvalAgent** (evaluator) | `core/interfaces/abstract_eval_agent.py` | `semantic`, `rubric`, `healthbench`, `medcalc`, `kg_qa`, `record_retrieval`, `dialogue_quality`, `engagement` |

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
│   ├── eslbench_distractor/  # Distractor-injected variant (shares eslbench's prepared data)
│   ├── healthbench/      # HealthBench medical AI
│   ├── medcalc/          # MedCalc-Bench calculations
│   └── virtual_user/     # Virtual-user opening-line engagement
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
    { "type": "llm_api", "fields": { "model": {"default": "gpt-5.4-mini", "editable": true, "required": true} } }
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

JSONL files in `benchmark/data/`. Each case specifies user config, target config, eval config, and optional `history`.

Key fields:
- **`strict_inputs`** (`List[str]`): Manual mode sends these sequentially
- **`history`** (`List[Dict]`): Pre-loaded conversation context (skips dialogue loop). Combinable with `strict_inputs`

### Data Converters

`generator/` transforms external datasets into mirobody-eval BenchItem format:

- **`generator/eslbench/prepare_data.py`** — ESLBench data preparation: HuggingFace download + per-user DuckDB creation
- **`generator/healthbench/converter.py`** — HealthBench JSONL → BenchItem
- **`generator/medcalc/converter.py`** — MedCalc-Bench CSV → BenchItem

### Web UI

```bash
python -m web    # uvicorn :8000 (+ health :8001)
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
| `OPENROUTER_API_KEY` | OpenRouter models | Used for unprefixed OpenAI defaults when `OPENAI_API_KEY` is absent. The account needs access to the requested models |
| `OPENAI_API_KEY` | OpenAI models | Direct OpenAI API key |
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | Google models | Also choose Google models for the target, judge and automatic user as needed. The SDK resolves keys or Vertex AI Application Default Credentials |
| `HF_TOKEN` | Optional for public data | HuggingFace token; private or gated repositories require access |
| `DASHSCOPE_API_KEY` `DEEPSEEK_API_KEY` `MOONSHOT_API_KEY` `ZHIPU_API_KEY` `VOLCENGINE_API_KEY` | Optional | Reached by prefixing the model name (`dashscope:qwen3.5-flash`). Each honours a matching `<PROVIDER>_BASE_URL` |
| `HOLYEVAL_WEB_PORT` | Optional | Web UI port (default: 8000) |
| `HOLYEVAL_HEALTH_PORT` | Optional | Health-check port (default: 8001) |
| `HOLYEVAL_RELOAD` | Optional | `true` enables uvicorn auto-reload (default: false) |
| `HOLYEVAL_GATEWAY_BASE_URL` | Optional | Your own OpenAI-compatible gateway (vLLM / LiteLLM / a proxy). Required only when a model name is written as `[label]model` |
| `HOLYEVAL_GATEWAY_API_KEY` | Optional | API key for that gateway |
| `MIROBODY_CONFIG` | `--target-type mirobody` | Absolute path to the deployment's own `config.*.yaml`. Without it the engine searches the CWD — this repo, not the deployment — and falls back to built-in defaults, which surfaces as a connection error naming a database rather than a missing setting |
| `MIROBODY_BASE_URL` | Optional | Deployment address (default `http://localhost:18080`) |
| `MIROBODY_TIMEOUT` | Optional | Per-turn seconds. Unset falls back to the framework-wide `AGENT_LLM_TIMEOUT` (default 840) — one turn is one `/api/chat` in which the deployment runs a whole agent loop, measured at 60+ model calls on the heavier questions |
| `MIROBODY_PROVIDER` | Optional | Override the agent's LLM provider. Empty uses the deployment's own default |

## Model Names Decide the Endpoint

Provider keys apply to the evaluation process, not the remote mirobody server.
Changing `--target-model` does not change the judge or virtual user defaults:
when only a Google, DashScope or other direct-provider key is available, also
select `--eval-model` and, for auto users, `--user-model` for that provider.

`evaluator/utils/llm.py` resolves a model name to a provider. Unprefixed names
follow the inference rules (`gpt*` → OpenAI, `gemini*` → Google, anything with a
slash → OpenRouter). A `provider:` prefix names it outright, using the same
provider table as the engine under test (`mirobody/utils/config/llm.py`):
`openrouter` `dashscope` `deepseek` `volcengine` `zhipu` `moonshot`, plus
`openai:` / `google:` to force a native SDK.

Colon, not slash — OpenRouter's own ids are already `vendor/model`
(`openai/gpt-5-mini`), and a slash separator would capture them. A colon after a
slash is a variant suffix (`anthropic/claude-sonnet-4.5:batch`), not a provider.

**An OpenAI-family name with no `OPENAI_API_KEY` but an OpenRouter key set is
taken from OpenRouter as `openai/<name>`, with a WARNING naming both.** Every
bundled evaluator defaults to a `gpt*` judge, so without this a user holding
only the single key this project recommends gets `OpenAIError` on the first
command in the README — a failure that points nowhere near the cause. An
explicit `openai:` prefix is never rerouted. `gemini*` is never rerouted either
and raises instead, because OpenRouter's Gemini ids are not 1:1 and guessing
turns a missing key into a 404.

## Ungraded Cases Are Not Zeros

`EvalResult.result` splits on whether grading happened at all. `pass` / `fail` /
`scored` mean a grade was produced. `error` means one was not — and it covers
every way that happens: a target timeout, a connection failure, a judge outage
(`kg_qa` raising `JudgeUnavailable`), a cancelled batch, or a bug in this
framework. The exception type lands in `trace.eval_detail`.

**`error` cases are excluded from `avg_score`, and `error_count` is published
beside it.** Both halves are load-bearing. A timed-out case carries `score=0.0`,
but that zero is not a result: the same question scored 0.00 under a 300s budget
and 0.89 under 900s, so the number moved with the harness's own configuration
rather than with the system under test. Averaging it in produced a fake low
score; excluding it without publishing the count would produce a fake high one.

The CLI displays `error_count`, the grading denominator, and a separate list of
ungraded case IDs. An all-error or empty group has no displayed average (`—`).
`pass` / `fail` counts do not include the continuous `scored` result type, so
zero pass and fail counts alone do not indicate a broken run. For example:

```
通过: 0   失败: 0   未判成: 0    平均得分 0.06 （3/3 条已评分）   ← three continuous scores
通过: 0   失败: 0   未判成: 3    平均得分 —    （0/3 条已评分）   ← no grade was produced
通过: 2   失败: 1   未判成: 4    平均得分 0.71 （3/7 条已评分）   ← read the denominator
```

The third line is the one to watch: a healthy-looking average over less than
half the set. `未判成` is usually a timeout budget (`MIROBODY_TIMEOUT`) or a
missing judge key, not a property of what you are evaluating.

## Code Style

- Python 3.11+ (`--extra mirobody` needs 3.12+), async/await throughout
- Ruff for linting/formatting, line-length 120
- Pydantic v2 for all data models
