# Generating Benchmark Data

> **Using in Claude Code**: Type `/add-benchmark`, followed by your benchmark source (paper link, GitHub repository). Claude will automatically complete the full research -> conversion -> validation workflow.

## Overview

Benchmark datasets are the input for evaluations, stored in `benchmark/data/<benchmark>/<dataset>.jsonl`. You can convert data from external evaluation benchmarks (such as HealthBench, MedCalc-Bench) or manually write test cases.

## Data Format

Each line is a JSON object corresponding to a `BenchItem` (automatically converted to `TestCase` at runtime):

```json
{
  "id": "case_001",
  "title": "Example case",
  "tags": ["topic:cardiology"],
  "user": {
    "type": "manual",
    "strict_inputs": ["I've been having a rapid heartbeat lately, what's going on?"]
  },
  "eval": {
    "evaluator": "semantic",
    "threshold": 0.7
  }
}
```

> **Note**: `BenchItem` typically does not include a `target` configuration — the system under test is specified at runtime via CLI arguments or the Web UI, allowing the same dataset to be reused across different models. If a benchmark requires a fixed target configuration, specify it in `metadata.json`.

### Key Fields

| Field | Description |
|------|------|
| `user.type` | `manual` (script-driven) or `auto` (LLM-driven) |
| `user.strict_inputs` | Preset input list (`List[str]`), sent sequentially to the system under test in manual mode. Single item = single-turn Q&A; multiple items = sequential context injection before the final question (e.g., send medical records first, then ask for diagnosis), intermediate responses do not affect evaluation |
| `eval.evaluator` | Evaluator type (`semantic` / `rubric` / `healthbench` / `medcalc` / `kg_qa` / `record_retrieval` / `dialogue_quality` / `engagement`) |
| `history` | Optional, pre-evaluation conversation history `[{role, content}]`. Unlike `strict_inputs`: history is injected directly as preloaded context without going through the conversation loop |

## Converting from External Datasets

### Built-in Converters

| Converter | Command | Description |
|--------|------|------|
| HealthBench | `python -m generator.healthbench.converter input.jsonl output.jsonl` | prompt -> history + strict_inputs, rubrics -> eval.rubrics |
| MedCalc-Bench | `python -m generator.medcalc.converter input.csv output.jsonl` | Patient Note + Question -> strict_inputs, Answer -> eval.ground_truth |
| ESL-Bench (data prep) | `python -m generator.eslbench.prepare_data` | Downloads the synthetic health-KG data from HuggingFace and builds a per-user DuckDB index (`--force` to rebuild) |
| Virtual User (personas) | `python -m generator.virtual_user profile_gen --count 20` | LLM-generated patient personas -> `profiles.jsonl` |
| Virtual User (cases) | `python -m generator.virtual_user case_gen --seed 42 --output <new>.jsonl` | personas x opening lines -> BenchItem JSONL. **Without `--output` it overwrites the shipped `round1.jsonl`.** |
| Virtual User (analysis) | `python -m generator.virtual_user analyzer report.json` | Aggregates a finished report by opening line / adversarial dimension |

### Custom Converters

Create a conversion script in the `generator/<benchmark>/` directory. Core steps:

1. Read the source data (JSONL / CSV / other formats)
2. Map to the `BenchItem` structure (user + eval, without target)
3. Write the JSONL file to `benchmark/data/<benchmark>/`
4. Create a `metadata.json` to describe the dataset

> **Recommended**: Use the `/add-benchmark` skill to automate the entire workflow above.

## Directory Structure

```
benchmark/
├── data/
│   ├── healthbench/          # Benchmark suite name
│   │   ├── full.jsonl        # Dataset
│   │   ├── sample.jsonl
│   │   └── metadata.json     # Suite metadata (displayed in Web UI)
│   ├── medcalc/
│   │   ├── full.jsonl
│   │   ├── sample.jsonl
│   │   └── metadata.json
│   ├── virtual_user/
│   │   ├── round1.jsonl
│   │   └── metadata.json
│   └── eslbench/
│       ├── sample50-20260331.jsonl
│       ├── full-20260331.jsonl
│       └── metadata.json
└── report/                   # Report output (auto-generated)
```

### metadata.json Format

Each benchmark suite directory contains a `metadata.json` that provides metadata. The `description` field (Markdown format) is displayed on the dataset detail page in the Web UI:

```json
{
  "description": "# Benchmark Suite Name\n\nBrief description...\n\n## Subsets\n\n| Subset | Count | Description |\n|------|------|------|\n| sample | 100 | Quick validation |\n\n**Evaluator**: `evaluator_name`",
  "target": {
    "type": "llm_api",
    "model": "gpt-5.4-mini"
  },
  "target_configurable": true
}
```

| Field | Description |
|------|------|
| `description` | Markdown-formatted suite description, displayed in Web UI |
| `target` | Default target configuration, used as initial values when creating tasks in Web UI / CLI |
| `target_configurable` | `true` allows users to modify target parameters, `false` locks them (e.g., a suite pinned to one target type) |

## Validating Data

```bash
# Run a few cases for quick validation
python -m benchmark.basic_runner <benchmark> <dataset> --target-type llm_api --target-model gpt-5.4-mini --limit 5 -v
```

Or execute via the Web UI: start `python -m web`, create a task at http://localhost:8000/tasks, and view progress in real time.
