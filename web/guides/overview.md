<!-- Project Overview - mirobody-eval -->

## Introduction

mirobody-eval is a **virtual user evaluation framework** designed for AI medical assistants. It systematically evaluates AI assistant performance through automated multi-turn conversation testing and multi-dimensional assessment.

## Architecture Overview

mirobody-eval adopts a modular design. Its components map onto the four workspace members:

- **evaluator** — Virtual user evaluation framework + system-under-test integration
- **generator** — Benchmark Generator + Benchmark Wrapper
- **benchmark** — Benchmark Scheduler (batch scheduling center)
- **web** — This management interface

### Virtual User Evaluation Framework (evaluator/)

Provides core evaluation capabilities based on a three-layer Agent architecture:

| Agent Type | Responsibility | Capabilities |
|-----------|---------|---------|
| **TestAgent** | Simulates virtual user behavior | Supports LLM-driven natural conversation and script-driven exact replay |
| **TargetAgent** | Integrates with the system under test | Unified wrapper for real system APIs and simulated LLM calls |
| **EvalAgent** | Evaluates conversation quality | Offers semantic, rubric, knowledge-graph QA, rule-based, and other multi-dimensional evaluation methods |

**Core Evaluation Flow**:
1. TestAgent initiates conversation (based on user goal and context)
2. TargetAgent calls the system under test to obtain responses
3. Conversation repeats until the goal is met or the turn limit is exceeded
4. EvalAgent performs multi-dimensional evaluation based on the complete conversation history
5. Generates TestResult (score, pass rate, detailed feedback, cost statistics)

### System-Under-Test Integration (evaluator/plugin/target_agent/)

Supports two execution modes for integrating different types of systems under test:

- **Real system integration**: Talks to a live system under test over its HTTP API to verify real-world performance
- **Simulated system integration**: Directly calls LLM APIs to simulate system behavior for quick validation of evaluation logic

Eight targets ship: `mirobody` — a self-hosted deployment of the engine this
project evaluates, seeded with synthetic data and driven as a signed-in user —
plus `llm_api` as a retrieval-only baseline and six external memory/RAG systems
to compare against.

### Benchmark Generator (generator/)

**Capabilities**: Converts external evaluation datasets into mirobody-eval standard format

- **Data conversion**: Supports mainstream AI evaluation datasets including HealthBench, MedCalc-Bench, ESL-Bench, and more
- **Business abstraction**: Maps raw evaluation scenarios (prompt + rubrics) to the BenchItem standard format, supporting multi-turn conversation context (`history` field)
- **Flexible configuration**: Supports custom evaluation criteria, conversation history, and virtual user configuration

### Benchmark Scheduler (benchmark/)

**Capabilities**: Scheduling, execution, and progress management for batch evaluations

- **Batch scheduling**: Supports concurrent execution of thousands of test cases (configurable concurrency)
- **Real-time monitoring**: Tracks execution status of each case (pending/in-dialogue/evaluating/completed/cancelled)
- **Flexible control**: Supports task cancellation at any time and progress snapshot export
- **Data organization**: Manages test data in directories organized by evaluation type (healthbench/medcalc/eslbench)

### Benchmark Wrapper (shared with generator/)

**Capabilities**: Standardized packaging and persistence of evaluation results

- **Result aggregation**: Aggregates individual case results into complete evaluation reports (total score, pass rate, statistics)
- **Metadata tracking**: Records the Agent type used for each case, facilitating subsequent analysis
- **Report management**: Automatically saves reports to the `benchmark/report/` directory, supporting historical report queries

## Directory Structure and Responsibilities

```
mirobody-eval/
├── evaluator/              # Virtual user evaluation framework + system integration
│   ├── core/              # Core evaluation engine (orchestrator, data models)
│   ├── plugin/            # Three-layer Agent plugin implementations
│   └── utils/             # Shared utilities (LLM calls, data reading)
│
├── generator/             # Data generation + result wrapping
│   ├── healthbench/       # HealthBench data converter
│   ├── medcalc/           # MedCalc-Bench data converter
│   ├── eslbench/          # ESL-Bench data prep (HuggingFace download + DuckDB index)
│   └── virtual_user/      # Virtual-user persona / case generation + report analysis
│
├── benchmark/             # Batch scheduling center
│   ├── data/              # Evaluation datasets (organized by type)
│   ├── report/            # Evaluation reports (auto-generated)
│   └── basic_runner.py    # CLI runner
│
└── web/                   # Web management interface
    ├── app/               # API services (task scheduling, report queries)
    ├── templates/         # Frontend pages
    └── guides/            # Developer guides
```

## Core Evaluation Capabilities

mirobody-eval ships eight evaluators covering different evaluation scenarios:

| Evaluator | Use Case | Evaluation Method |
|--------|---------|---------|
| **semantic** | General semantic quality | LLM-based subjective quality assessment (safety, empathy, professionalism) |
| **rubric** | Per-turn rubric scoring | Natural-language criteria + registered signal checks + latency budgets |
| **healthbench** | HealthBench standard | Rubric-based multi-dimensional scoring (concurrent evaluation of all rubric items) |
| **medcalc** | Medical calculation | LLM answer extraction + typed numerical matching (decimal/integer/date/weeks_days) |
| **kg_qa** | Knowledge-graph QA | Numeric tolerance + stepwise credit, routed by answer type (used by ESL-Bench) |
| **record_retrieval** | Record & retrieval accuracy | Zero-LLM per-turn checkpoints |
| **dialogue_quality** | Multi-turn dialogue quality | LLM-as-Judge multi-dimension scoring |
| **engagement** | Virtual-user engagement | LLM-as-Judge on whether the simulated user actually engaged |

## Execution Modes

Three execution methods are supported to fit different use cases:

- **CLI execution**: Command-line batch benchmarking, suitable for CI/CD integration and automated testing
- **Web UI execution**: Visual interface with real-time progress and results, suitable for manual validation and analysis
- **API execution**: HTTP interface integration, suitable for embedding in other systems or services

## Next Steps

| Action | Description |
|------|------|
| [Run Evaluations](/tasks) | Select a dataset, configure parameters, view progress in real time |
| [View Reports](/tasks) | Browse historical evaluation results |
| [Benchmark Data](/benchmarks) | Browse datasets by directory |
| [Agent Registry](/agents/test) | View registered plugins |

**Skills Quick Commands**:

- `/add-benchmark` — Integrate an external benchmark (research -> conversion -> validation end-to-end)
- `/add-eval-agent` — Scaffold: add a new evaluation logic plugin
- `/add-target-agent` — Scaffold: add a new system-under-test plugin
- `/run-benchmark` — Run benchmark scoring
