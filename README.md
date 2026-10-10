<h1 align="center">
  <br>
  mirobody-eval
  <br>
</h1>

<p align="center">
  <strong>Five benchmarks ship in the repo; one command reproduces any of them. The virtual user, the target and the judge are all pluggable.<br>Seven Claude Code slash commands guide you from ingesting a paper to reading a report.</strong>
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
  <strong>English</strong> &middot; <strong><a href="README.zh-CN.md">简体中文</a></strong>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#evaluate-your-own-mirobody-deployment">Evaluate mirobody</a> &middot;
  <a href="#ai-native-development-with-claude-code">Claude Code</a> &middot;
  <a href="#web-ui">Web UI</a> &middot;
  <a href="http://healthmemoryarena.ai">Live Demo</a> &middot;
  <a href="https://arxiv.org/abs/2604.02834">Paper</a> &middot;
  <a href="https://huggingface.co/datasets/healthmemoryarena/ESL-Bench">Dataset</a> &middot;
  <a href="#benchmarks">Benchmarks</a> &middot;
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.02834">
    <img src="docs/screenshots/eslbench_overview.png" alt="ESL-Bench: Event-Driven Longitudinal Health Agent Benchmark" width="80%">
  </a>
</p>

<p align="center">
  <em>ESL-Bench — an event-driven synthetic longitudinal benchmark for health agents.
  <br>100 synthetic users, 10,000 queries, 5 dimensions, programmatic ground truth.
  <br>Read the paper: <a href="https://arxiv.org/abs/2604.02834">arXiv:2604.02834</a></em>
</p>

---

mirobody-eval is the evaluation half of [mirobody](https://github.com/thetahealth/mirobody) — the open-source health data engine. It exists so mirobody's claims come with numbers attached: seed a synthetic user into your own deployment, run a benchmark against it, get a scored report. The framework underneath is general, so the same one command reproduces any published benchmark against any target you plug in. Drop in a benchmark dataset, run one command, get a scored report. Extend it with custom evaluators, target systems, and virtual users via a pluggable agent architecture.

Built from the ground up as a [Claude Code](https://docs.anthropic.com/en/docs/claude-code) native project — every workflow, from initial setup to integrating a new benchmark from a research paper, is an interactive slash command. You describe what you want in natural language, and Claude Code handles the rest. **You don't need to write a single line of code to use or extend this framework.**

### From paper to benchmark — `/add-benchmark`

<p align="center">
  <img src="docs/screenshots/holyeval_add_benchmark.gif" alt="Add Benchmark Demo" width="80%">
</p>

> Paste a paper link and `/add-benchmark` walks Claude Code through reading the paper, writing the converter and generating the datasets — then runs a sample and **checks the conversion against the numbers the paper published**. Everything lands in data and plugin directories; the framework core is never touched.

### One command runs all five benchmarks

<p align="center">
  <img src="docs/screenshots/holyeval_run_benchmark.gif" alt="Run Benchmark Demo" width="80%">
</p>

```bash
# Start with a few cases; cost and runtime depend on the models and benchmark
uv run python -m benchmark.basic_runner healthbench sample --target-model gpt-5.4-mini --limit 3
uv run python -m benchmark.basic_runner medcalc sample --target-model gpt-5.4-mini --limit 3
uv run python -m benchmark.basic_runner virtual_user round1 --target-type llm_api --target-model gpt-5.4-mini --limit 3

# ESLBench requires data preparation first (see Quick Start below)
uv run python -m benchmark.basic_runner eslbench sample50-20260331 --target-type llm_api --target-model gpt-5.4-mini --limit 3

# Ready for a full run? Remove --limit to run the entire dataset
```

## Why mirobody-eval?

| | |
|---|---|
| **Paper → benchmark** | Paste a paper link and `/add-benchmark` reads it, writes the converter, generates the datasets — then validates the conversion against the scores the paper published |
| **One-command reproduction** | Every bundled benchmark runs through the same `basic_runner` command; reports carry the batch id and sha256 checksum a score came from |
| **Ungraded is not zero** | Timeouts, connection failures and judge outages are recorded as `error` and excluded from the average; every summary prints the ungraded count next to the graded denominator — no fake zeros, no inflated averages |
| **Pluggable architecture** | The virtual user, the target and the judge are each one plugin class; a file dropped into the plugin directory registers itself, and the framework core stays untouched |
| **Sessions, not single turns** | TestAgent ↔ TargetAgent converse until a stop condition; the judge sees the full trajectory, not one Q&A pair |
| **Batch execution** | Concurrent and cancellable; an interrupted run leaves a checkpoint and `--resume` skips the completed cases |
| **Web UI** | Launch runs, SSE progress, per-case reports — converging on the same execution entry point as the CLI, `do_single_test()` |
| **AI-native** | Seven guided slash commands, `/quick-start` through `/eslbench-report-analysis`, from environment setup to report analysis |

## Quick Start

With [Claude Code](https://docs.anthropic.com/en/docs/claude-code): just run `/quick-start` — it handles everything automatically.

Or manually:

```bash
git clone https://github.com/thetahealth/mirobody-eval.git && cd mirobody-eval
uv sync
cp .env.example .env                    # edit .env and enter your provider's real key

# With OPENAI_API_KEY or OPENROUTER_API_KEY configured
uv run python -m benchmark.basic_runner healthbench sample --target-model gpt-5.4-mini --limit 2

# Launch Web UI
uv run python -m web                    # http://localhost:8000
```

> **Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/), at least one LLM API key.
> The example above uses OpenAI models for both the target and the judge. Configure
> `OPENAI_API_KEY`, or configure `OPENROUTER_API_KEY` and leave `OPENAI_API_KEY`
> unset to route those model IDs through OpenRouter. Your account must have access
> to the requested models. This configures the evaluation process; a mirobody
> deployment needs its own model and embedding configuration.
>
> **ESLBench data prep:** Run `uv run python -m generator.eslbench.prepare_data`
> before evaluating ESLBench (also started in the background by the Web UI; wait
> for it to finish). It checks all manifest batches and downloads missing or changed
> batches, including user data and question banks, then builds DuckDB files. It is
> not a single-user download and can take substantial disk space and time. Other
> benchmarks ship their question files with the repository.

If you only have a Google or DashScope key, also select models for the roles that
call an LLM. Set the matching key in `.env`, replace `YOUR_MODEL_ID` below with a
model available to your account, and run:

```bash
MODEL=google:YOUR_MODEL_ID              # or dashscope:YOUR_MODEL_ID
uv run python -m benchmark.basic_runner healthbench sample \
    --target-type llm_api --target-model "$MODEL" --eval-model "$MODEL" --limit 2
```

For `virtual_user`, also pass `--user-model "$MODEL"`. Setting a provider key alone
does not replace the other roles' OpenAI defaults. See [Configuration](#configuration)
for all supported prefixes.

## Evaluate your own mirobody deployment

Use a running [mirobody](https://github.com/thetahealth/mirobody) deployment whose
Postgres and Redis are reachable from the evaluation process. Demo data already in
the deployment may belong to different users; seed the user required by the chosen
questions before evaluating it.

From the mirobody-eval repository, configure an evaluation LLM key as in Quick Start.
The deployment needs a working chat provider and embedding provider, plus the same
JWT configuration used by the evaluation process. If its config is encrypted, make
the deployment's `CONFIG_ENCRYPTION_KEY` available to the seeder and runner as well.
Install the optional dependency before seeding (it requires Python 3.12+):

```bash
# 1. Install the engine integration
uv sync --extra mirobody --python 3.12

# 2. Prepare all missing or changed ESLBench batches (not just one user)
uv run python -m generator.eslbench.prepare_data

# 3. Point to your deployment; replace both values with your actual settings
export MIROBODY_CONFIG=/abs/path/to/your/mirobody/config.localdb.yaml
export MIROBODY_BASE_URL=http://localhost:18080

# 4. Seed the user whose questions will be run
uv run python -m generator.eslbench.seed_mirobody --users user5086@demo

# 5. Smoke-test only that user's questions; --limit alone does not filter users
uv run python -m benchmark.basic_runner eslbench sample200-20260430 \
    --target-type mirobody \
    --ids user5086_AT_demo_Q001,user5086_AT_demo_Q087,user5086_AT_demo_Q067 -p 1
```

Use the HTTP address actually exposed by your deployment; if it publishes port
18060, use `http://localhost:18060`. A Docker-only database address in its config
may not be reachable from a runner on the host. Seeding checks indicator searchability
and reports missing embeddings before you start the evaluation.

These three cases check the connection and scoring; they do not cover all five
ESL-Bench dimensions or establish a benchmark score. For a larger run, seed every
user referenced by the selected cases. To compare with a model using retrieval
tools, keep the same dataset and IDs and use `--target-type llm_api` with
`--target-model`. The model behind `--target-type mirobody` is configured in that
deployment. To launch runs in the Web UI, start `uv run python -m web` from this
configured shell, then select the same case IDs.

### The file-upload demo

`labreport` renders one of the synthetic user's lab panels as a PDF, so mirobody's ingest path has
something real to chew on. Hold that panel back when seeding and the upload contributes data the
database genuinely does not have yet — which is what turns "what is my lipid trend?" into a real
question instead of a single point:

```bash
uv run python -m generator.eslbench.seed_mirobody --users user5086@demo --hold-out-exams 1
uv run python -m generator.eslbench.labreport     --users user5086@demo -o samples/lab_report.pdf
```

This demo omits the latest exam from the seeded history. Before running benchmark
questions that use the complete history, seed again without `--hold-out-exams`.

`user5086@demo` is a generated 58-year-old with type 2 diabetes whose lipids improve and then drift
back across four panels. Every value is synthetic; the PDF says so on its front page.

Eight of the twelve printed rows come back with a LOINC code and four do not — a report prints
`High-Density Lipoprotein` where LOINC codes `Cholesterol in HDL`, and `LDL/HDL Ratio` is a derived
ratio with no observation code at all. That mix is deliberate: it exercises the resolver's misses
alongside its hits, which a panel where every row resolved would not.

## AI-Native Development with Claude Code

mirobody-eval is designed to be operated entirely through [Claude Code](https://docs.anthropic.com/en/docs/claude-code). Every common task has a dedicated slash command. You describe your intent in natural language; Claude Code reads the code, generates files, runs tests, and validates the result.

**You don't need to memorize CLI flags, read source code, or write boilerplate.** Just type the slash command and follow the conversation.

### Slash Command Reference

| What you want to do | Command | What Claude Code does for you |
|---|---|---|
| **Set up the project** | `/quick-start` | Checks Python/uv, installs dependencies, configures `.env` with your API keys, launches Web UI |
| **Run a benchmark** | `/run-benchmark` | Asks which benchmark & dataset, then executes with your chosen model and concurrency |
| **Integrate a new benchmark** | `/add-benchmark` | End-to-end: reads the paper/repo → analyzes data format → writes the converter → creates dataset → validates |
| **Add a custom evaluator** | `/add-eval-agent` | Scaffolds config model + plugin implementation + registration. Immediately available in CLI & Web UI |
| **Add a new target system** | `/add-target-agent` | Scaffolds connection handling, message processing, and cleanup for a new system under test |
| **Audit architecture** | `/review-architecture` | Checks GitOps compliance, plugin isolation, shared-layer reuse. Reports violations with fix suggestions |
| **Read a run's report** | `/eslbench-report-analysis` | Breaks scores down by difficulty dimension, compares methods, surfaces failure rates and per-question cost/duration |

### Workflow Examples

**"I want to reproduce HealthBench on GPT-4.1"**
```
> /run-benchmark
# Claude asks: which benchmark? → healthbench
# Which dataset? → sample
# Which model? → gpt-5.4-mini
# How many cases? → 5 (start small!)
# Running... 5 cases → report saved
```

**"I need a custom evaluator that checks citation accuracy"**
```
> /add-eval-agent
# Claude asks: plugin name? → citation_accuracy
# What does it evaluate? → checks if AI responses cite valid sources
# Generates: evaluator/plugin/eval_agent/citation_accuracy_eval_agent.py
# Registered automatically via __init_subclass__ — ready to use
```

> **Tip:** You're not limited to slash commands. Claude Code understands the full codebase — ask it anything in natural language, like *"explain how the plugin system works"* or *"why did this test case fail?"*.

## Web UI

Launch with `uv run python -m web`, then visit http://localhost:8000.

<table>
<tr>
<td width="50%">

**Run Evaluations** — Select benchmark, configure parameters, launch tasks with real-time SSE progress tracking.

<img src="docs/screenshots/holyeval_tasks.jpg" alt="Run Evaluations" width="100%">
</td>
<td width="50%">

**Evaluation Report** — Scored results with expandable cases, dialogue history, and per-case feedback.

<img src="docs/screenshots/holyeval_report.jpg" alt="Evaluation Report" width="100%">
</td>
</tr>
<tr>
<td width="50%">

**Browse Benchmarks** — Overview of all benchmark datasets with case counts and statistics.
</td>
<td width="50%">

**Agent Registry** — Inspect all registered plugins with config schemas, features, and cost estimates.
</td>
</tr>
</table>

## Health Memory Arena — Live Evaluation Platform

[Health Memory Arena](http://healthmemoryarena.ai) (HMA) is the public evaluation platform powered by mirobody-eval. It hosts the ESL-Bench leaderboard where health AI agents compete on structured longitudinal reasoning tasks.

> **Where this repository stops.** For any batch whose answers are released,
> reproducing its leaderboard scores works end to end here — fetch the dataset,
> run it, and the report carries the batch id and checksum it came from. Under
> the rolling release policy **the newest batch's answers stay unreleased until
> the next one ships**, so its scores cannot be graded locally until then.
> Separately, **publishing a benchmark of your own to HuggingFace, and
> submitting a result to the leaderboard, are the two steps whose tooling is
> not open-sourced yet.** Open an issue if you want to get on the board and we
> will prioritise accordingly.

<table>
<tr>
<td width="33%">
<a href="http://healthmemoryarena.ai"><img src="docs/screenshots/hma_home.jpg" alt="HMA Home" width="100%"></a>
<p align="center"><em>Platform Home</em></p>
</td>
<td width="33%">
<a href="http://healthmemoryarena.ai/leaderboard"><img src="docs/screenshots/hma_leaderboard.jpg" alt="HMA Leaderboard" width="100%"></a>
<p align="center"><em>Agent Leaderboard</em></p>
</td>
<td width="33%">
<a href="http://healthmemoryarena.ai/dataset"><img src="docs/screenshots/hma_dataset.jpg" alt="HMA Dataset" width="100%"></a>
<p align="center"><em>Dataset Browser</em></p>
</td>
</tr>
</table>

## Architecture

```
TestCase (JSON) → Orchestrator
  1. Initialize agents from config via plugin registry
  2. Dialogue loop: TestAgent ↔ TargetAgent (until finished or max turns)
  3. EvalAgent.run(conversation, session) → EvalResult
  4. Return TestResult (score, pass/fail, feedback, cost)
```

All execution paths (CLI, Web UI, programmatic) funnel through a single entry point: `do_single_test()`.

### Plugin System

Three agent types, each extensible via `__init_subclass__` auto-registration:

```python
# Define a custom evaluator — that's it, it's registered
class MyEvalAgent(AbstractEvalAgent, name="my_eval", params_model=MyEvalInfo):
    async def run(self, memory_list, session_info):
        # your evaluation logic
        return EvalResult(result="pass", score=0.95, feedback="...")
```

| Agent Type | Role | Built-in Plugins |
|---|---|---|
| **TestAgent** | Virtual user | `auto` (LLM-driven), `manual` (scripted) |
| **TargetAgent** | System under test | `mirobody` (a self-hosted deployment), `llm_api` (OpenAI / Gemini / OpenRouter), `hermes`, `evermem`, `mem0_rag_api`, `naive_rag_api`, `hippo_rag_api`, `dyg_rag_api` |
| **EvalAgent** | Evaluator | `semantic`, `rubric`, `healthbench`, `medcalc`, `kg_qa`, `record_retrieval`, `dialogue_quality`, `engagement` |

### Project Structure

```
mirobody-eval/
├── evaluator/          # Core engine: schema, orchestrator, plugin interfaces
├── benchmark/          # Runner + datasets (JSONL) + reports
│   └── data/eslbench/  # ESLBench: data + tools (retrieve.py for JSON/DuckDB)
├── generator/          # Dataset converters + data preparation scripts
│   └── eslbench/       # ESLBench data downloader + DuckDB builder
└── web/                # Web UI (FastAPI + htmx)
```

## Benchmarks

| Benchmark | Paper / Source | Datasets | What it evaluates |
|---|---|---|---|
| **HealthBench** | [OpenAI HealthBench](https://arxiv.org/abs/2505.07469) | `sample` (100), `full`, `hard`, `consensus` | Medical AI quality |
| **MedCalc-Bench** | [MedCalc-Bench](https://arxiv.org/abs/2406.12036) | `sample`, `full` | Medical calculations |
| **ESLBench** | [arXiv:2604.02834](https://arxiv.org/abs/2604.02834) | `sample50-20260331` (50), `sample500-20260331` (500), `full-20260331` (4500) | Longitudinal health reasoning |
| **ESLBench-Distractor** | — | `sample` (60), `distractor-behavioral-20260723` (120), `distractor-computable-20260723` (400) | Robustness to distractor context (shares ESLBench's prepared data) |
| **Virtual User** | — | `round1` (15), `round2` (60) | Opening-line engagement |

### ESLBench — Event-Driven Synthetic Longitudinal Benchmark

ESLBench ([arXiv:2604.02834](https://arxiv.org/abs/2604.02834)) evaluates longitudinal health reasoning — the ability to align, aggregate, and attribute across multi-source patient trajectories combining device streams, clinical exams, and life events. Built on an event-driven synthesis framework where each user trajectory is modeled as a baseline health state plus discrete events with explicit temporal kernels (sigmoid onset, exponential decay), making ground truth programmatically computable.

<p align="center">
  <img src="docs/screenshots/eslbench_trajectory.png" alt="ESL-Bench Trajectory Visualization" width="70%">
</p>

<p align="center"><em>Four-month trajectory excerpt — event-driven indicator dynamics with sigmoid onset and exponential decay.</em></p>

**100 synthetic users** with 1–5 year trajectories, **10,000 evaluation queries** across five dimensions and three difficulty tiers:

| Dimension | What it tests | Example |
|---|---|---|
| **Lookup** | Direct data retrieval | "What was resting heart rate on 2024-03-15?" |
| **Trend** | Temporal pattern analysis | "In which month was step count highest?" |
| **Comparison** | Cross-event/source comparisons | "How did mean steps change after jogging started?" |
| **Anomaly** | Abnormality detection | "Has glucose ever been abnormal?" |
| **Explanation** | Causal attribution | "Rank events by impact on glucose drop" |

<details>
<summary><strong>Benchmark results — 13 methods across 3 paradigms</strong></summary>
<br>
Key findings: DB agents (48–58%) substantially outperform memory RAG (30–38%), with the gap concentrated on Comparison and Explanation queries where multi-hop reasoning and evidence attribution are required.
</details>

**Data preparation required** — ESLBench downloads user data from HuggingFace and builds per-user DuckDB indexes:

```bash
# First time: prepare data (automatic via Web UI, manual for CLI)
uv run python -m generator.eslbench.prepare_data

# Quick test: start with 3 cases to verify setup
uv run python -m benchmark.basic_runner eslbench sample50-20260331 --target-type llm_api --target-model gpt-5.4-mini --limit 3

# Sample datasets
uv run python -m benchmark.basic_runner eslbench sample50-20260331 --target-type llm_api --target-model gpt-5.4-mini      # 50 cases
uv run python -m benchmark.basic_runner eslbench sample500-20260331 --target-type llm_api --target-model gpt-5.4-mini -p 5 # 500 cases

# Full benchmark (4500 cases — significant API cost, review before running)
uv run python -m benchmark.basic_runner eslbench full-20260331 --target-type llm_api --target-model gpt-5.4-mini -p 5
```

The LLM target is equipped with a tool group (`eslbench/retrieve`) that provides JSON file reading, DuckDB queries, and indicator lookup — the LLM must use these tools to find answers in the user's health data.

### Add a new benchmark

Two ways:

**A) Use the Claude Code skill (recommended):**
```
/add-benchmark    # guided: research paper → convert data → validate
```

**B) Manual:**
1. Create `benchmark/data/<name>/metadata.json` with target config
2. Create `benchmark/data/<name>/<dataset>.jsonl` in BenchItem format
3. Run: `uv run python -m benchmark.basic_runner <name> <dataset> --target-model gpt-5.4-mini`

See [benchmark/data/medcalc/](benchmark/data/medcalc/) for a minimal `metadata.json` + `sample.jsonl` pair.

## Extending mirobody-eval

### Add an evaluator

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

The filename must end with `_eval_agent.py` — that suffix is what the package's `pkgutil`
auto-import picks up, which is what triggers registration. Nothing to add to `__init__.py`.

### Add a target system

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

Use `/add-eval-agent` or `/add-target-agent` Claude Code skills for guided scaffolding.

## CLI Reference

```bash
# Prepare benchmark data (required for ESLBench; other benchmarks ship with data)
uv run python -m generator.eslbench.prepare_data          # download HF data + build DuckDB
uv run python -m generator.eslbench.prepare_data --force   # force rebuild

# Run benchmark
uv run python -m benchmark.basic_runner <benchmark> <dataset> [options]
  --target-type TYPE      # Target agent type (for multi-target benchmarks)
  --target-model MODEL    # the model under test (e.g. gpt-5.4-mini, anthropic/claude-sonnet-4.6)
  --user-model MODEL      # the model playing the virtual user (auto-mode datasets only)
  --eval-model MODEL      # the model doing the judging (rule-scored answers ignore it)
  --system-prompt TEXT    # override the target's system prompt
  --target-override K=V   # override an editable target field, e.g. agent=Deep
  --limit N               # Max cases to run
  --ids id1,id2           # Run specific case IDs
  -p, --parallel N        # Concurrency (default: 0 = unlimited)
  -v, --verbose           # Verbose output
  --resume                # Resume from last checkpoint

# Convert external datasets
uv run python -m generator.healthbench.converter input.jsonl output.jsonl
uv run python -m generator.medcalc.converter input.csv output.jsonl
uv run python -m generator.virtual_user case_gen --seed 42 \
    --output benchmark/data/virtual_user/my_round.jsonl   # ⚠ 不传 --output 会覆盖随附的 round1.jsonl

# Web UI
uv run python -m web             # http://localhost:8000
```

A run drives up to three models — the virtual user, the target, and the judge — and each is named
separately, so one provider can serve all three:

```bash
uv run python -m benchmark.basic_runner virtual_user round1 \
    --target-type llm_api --target-model anthropic/claude-sonnet-4.6 \
    --user-model anthropic/claude-sonnet-4.6 \
    --eval-model anthropic/claude-sonnet-4.6
```

For ESL-Bench's `kg_qa` evaluator, `text` and `behavioral` answers use an LLM judge;
`numeric_value`, `boolean` and `list` use rules. Rule-based grading does not require
a judge key, but the target may still require an LLM key to answer the question.

Timeouts, cancellations and execution exceptions are reported as `error`. The
average includes only `pass`, `fail` and `scored` cases. CLI summaries show the
error count and the number graded out of the total; if none were graded, the
average displays as `—`. Always compare coverage alongside the average: a high
score over a few completed cases is not a full-dataset result. Saved JSON keeps
`avg_score: 0.0` when no grades exist, so consumers must also read `error_count`
and the case count. Other evaluators may have their own failure handling.

## Configuration

Set the key for each provider used by the target, judge and virtual user in `.env`.
The bundled defaults use OpenAI models. OpenAI or OpenRouter can serve those defaults;
other providers require explicit model choices for each LLM role.

| Variable | When needed | Description |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI models | Direct OpenAI API access |
| `OPENROUTER_API_KEY` | OpenRouter models | Also used for unprefixed `gpt*` names when `OPENAI_API_KEY` is unset |
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | Google models | Select a `google:` or `gemini*` model; Vertex AI can use Application Default Credentials instead |
| `DASHSCOPE_API_KEY` | `dashscope:` | Alibaba DashScope |
| `DEEPSEEK_API_KEY` | `deepseek:` | DeepSeek, direct |
| `MOONSHOT_API_KEY` | `moonshot:` | Moonshot (Kimi), direct |
| `ZHIPU_API_KEY` | `zhipu:` | Zhipu (GLM), direct |
| `VOLCENGINE_API_KEY` | `volcengine:` | Volcengine Ark (Doubao), direct |
| `HF_TOKEN` | Optional for public datasets | HuggingFace access token; required for private or gated datasets if selected |

### Writing a model name

| Syntax | Routing |
|---|---|
| `gpt-5.4-mini` | OpenAI; if only an OpenRouter key is configured, requests `openai/gpt-5.4-mini` there and logs the route change |
| `openai/gpt-5-mini` | OpenRouter; the slash is part of its model ID |
| `openai:gpt-4.1` | Explicit OpenAI endpoint; never automatically rerouted |
| `google:YOUR_MODEL_ID` | Google SDK; replace the placeholder with an available model ID |
| `dashscope:YOUR_MODEL_ID` | DashScope endpoint; replace the placeholder with an available model ID |

Supported prefixes: `openrouter`, `dashscope`, `deepseek`, `volcengine`, `zhipu`,
`moonshot`, `openai`, `google`. The prefix selects an endpoint; it does not guarantee
that a model exists or that your account can use it. Unknown prefixes raise an error.
For an unprefixed name, `gpt*` selects OpenAI, `gemini*` selects Google, and other
names select OpenRouter; `[label]model` selects your configured gateway.

The same syntax works for `--target-model` with `--target-type llm_api`,
`--eval-model` for the judge, and `--user-model` for an automatic virtual user.
Changing one role does not change the other two. The Web UI's Model field accepts
free-form model IDs. These evaluation settings do not change a running mirobody
server's model configuration.

The separator is a colon before any slash. An OpenRouter ID such as
`anthropic/claude-sonnet-4.5:batch` keeps its suffix intact. The six OpenAI-compatible
providers (`openrouter`, `dashscope`, `deepseek`, `volcengine`, `zhipu`, `moonshot`)
also accept their corresponding `<PROVIDER>_BASE_URL` environment variable.
`google:` uses the Google SDK's endpoint configuration, not `GOOGLE_BASE_URL`.

### Runtime and gateway settings

| Variable | When needed | Description |
|---|---|---|
| `HOLYEVAL_GATEWAY_BASE_URL` | `[label]model` | Your OpenAI-compatible gateway URL |
| `HOLYEVAL_GATEWAY_API_KEY` | Gateway authentication | The gateway's API key |
| `HOLYEVAL_WEB_PORT` | Optional | Web UI port (default 8000) |
| `HOLYEVAL_HEALTH_PORT` | Optional | Health-check port (default 8001) |
| `HOLYEVAL_RELOAD` | Optional | `true` enables uvicorn auto-reload (default false) |
| `AGENT_LLM_TIMEOUT` | Optional | Framework LLM timeout in seconds (default 840) |

### mirobody deployment settings

These identify the deployment to seed and evaluate. Set them before starting the
runner or Web UI. If the deployment uses encrypted configuration, provide its
`CONFIG_ENCRYPTION_KEY` through the evaluation process's environment too.

| Variable | When needed | Description |
|---|---|---|
| `MIROBODY_CONFIG` | Recommended for `--target-type mirobody` | Absolute path to the deployment's config file, with database and Redis addresses reachable from this process |
| `MIROBODY_BASE_URL` | Set to the deployment's URL | Defaults to `http://localhost:18080`; use the actual exposed port |
| `MIROBODY_TIMEOUT` | Optional | Per-chat-request timeout in seconds; unset uses `AGENT_LLM_TIMEOUT` (default 840). A request may contain many model and tool calls |
| `MIROBODY_PROVIDER` | Optional | Select a provider configured on the mirobody server; empty uses the deployment's default |

## Roadmap

### In Progress
- [ ] **GUI TargetAgent** — evaluate real products through their web UI, not just API endpoints. Browser-based agent interacts with your app like a real user, enabling end-to-end evaluation of any product with a frontend

### Planned
- [ ] **Eval-driven optimization loop** — run benchmark → auto-analyze failure patterns → generate targeted prompt/system improvements → re-run to verify. Close the loop between evaluation and iteration
- [ ] **CI/CD integration** — `pip install mirobody-eval` + `mirobody_eval.run("healthbench", model="gpt-5.4-mini")` as a one-liner in your CI pipeline. Regression detection across runs, alerting on score drops before deployment
- [ ] **Industry agent & app deep evaluation** — comprehensive evaluation reports for mainstream AI agents and health apps (e.g. ChatGPT, Gemini, health assistants). Standardized scoring across safety, accuracy, and user experience, published as reproducible community benchmarks

## Development

```bash
# Unit tests — pure logic, no database, no network, no model calls
uv run --group dev python -m pytest generator/ evaluator/ -q

# Sanity check — plugin registries load
uv run python -c "import evaluator.plugin.eval_agent, evaluator.plugin.target_agent; \
from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent; \
print(sorted(AbstractEvalAgent.get_all()))"

# Lint and formatting checks (install the lint dependency group)
uv run --group lint ruff check .
uv run --group lint ruff format --check .
```

CI runs unit tests, plugin registration and dataset plugin-reference checks.
Lint and formatting checks currently report findings without blocking CI, so
existing findings may appear locally even when CI is green. Review findings in
the files you change; a successful CI run does not mean the repository is lint-clean.

## Contributing

Contributions are welcome! The easiest way to contribute is through Claude Code — every workflow below has a guided slash command:

| Contribution type | How to start | Difficulty |
|---|---|---|
| **Add a benchmark** | `/add-benchmark` — the fastest way to contribute | Easy |
| **Add an evaluator** | `/add-eval-agent` — scaffold a new scoring methodology | Medium |
| **Add a target system** | `/add-target-agent` — connect a new API/service to evaluate | Medium |
| **Improve existing benchmarks** | Add more test cases, edge cases, or better prompts | Easy |

Please open an issue first to discuss significant changes.

## Citation

If you use ESL-Bench or mirobody-eval in your research, please cite:

```bibtex
@article{li2026eslbench,
  title={ESL-Bench: An Event-Driven Synthetic Longitudinal Benchmark for Health Agents},
  author={Li, Chao and Liu, Cailiang and Gao, Ang and Deng, Kexin and Zhang, Shu and Xu, Langping and Shi, Xiaotong and Ding, Xionghao and Pei, Jian and Jiang, Xun},
  journal={arXiv preprint arXiv:2604.02834},
  year={2026}
}
```

## License

[MIT](LICENSE). Third-party components and their licenses are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
