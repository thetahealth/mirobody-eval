# Developing Evaluation Logic (EvalAgent)

> **Using in Claude Code**: Type `/add-eval-agent <agent-name>`, followed by your evaluation logic description (e.g., scoring rules, input/output requirements). Claude will automatically generate the configuration, implementation, and registration code.

## Overview

EvalAgent is a core component of the evaluation framework, responsible for assessing the system under test after a conversation ends. The framework supports multiple evaluation strategies through a plugin mechanism, allowing you to develop custom evaluation logic based on your needs.

## Existing Evaluators

| Name | Type | Description |
|------|------|------|
| `semantic` | LLM | Semantic understanding evaluation, uses LLM to judge response quality |
| `rubric` | LLM + Rules | Generic per-turn rubric: natural-language criteria, registered signal checks, and latency budgets |
| `healthbench` | LLM | HealthBench rubric scoring, based on multi-dimensional grading criteria |
| `medcalc` | LLM + Rules | MedCalc-Bench medical calculation evaluation (LLM answer extraction + typed numerical matching) |
| `kg_qa` | LLM + Rules | Knowledge-graph QA grading with numeric tolerance and stepwise credit (used by ESL-Bench) |
| `record_retrieval` | Rules | Zero-LLM per-turn checkpoints for record acknowledgment and retrieval accuracy |
| `dialogue_quality` | LLM | Multi-dimension dialogue quality scoring (LLM-as-Judge) |
| `engagement` | LLM | Whether the virtual user actually engaged, used for opening-line comparisons |

## Development Steps

### 1. Design the Evaluation Logic

Determine what your evaluator needs:
- **LLM-based** or **deterministic rules**? LLM is suitable for semantic judgment; rules are suitable for exact matching
- What **configuration parameters** are needed? (e.g., threshold, keyword list, model name, etc.)
- Is the scoring method **binary** (pass/fail) or **continuous** (0~1)?

### 2. Define the Configuration Model

Put the Pydantic model **next to the plugin**, in `evaluator/plugin/eval_agent/my_eval_agent.py`:

```python
class MyEvalInfo(BaseModel):
    """My custom evaluation configuration"""
    model_config = ConfigDict(extra="forbid")

    evaluator: Literal["my_eval"] = "my_eval"
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    # Custom fields...
```

You do **not** edit `evaluator/core/schema.py` — it dispatches dynamically through
the plugin's registered `params_model`, which is populated by the `params_model=` argument
in the next step.

### 3. Implement the Evaluator

Create the implementation file in `evaluator/plugin/eval_agent/`:

```python
class MyEvalAgent(AbstractEvalAgent, name="my_eval", params_model=MyEvalInfo):
    async def run(self, memory_list, session_info=None) -> EvalResult:
        # memory_list: List[TestAgentMemory] — full conversation (user actions + target system responses)
        # session_info: SessionInfo | None — target system session info (authentication data, etc.)
        # self.history / self.user_info / self.case_id — static context (injected at init)
        # Return EvalResult(result="pass"|"fail", score=0.8, feedback="...")
```

### 4. Register the Plugin

Nothing to import by hand: `evaluator/plugin/eval_agent/__init__.py` walks the package with
`pkgutil` and auto-imports every module whose filename **ends with `_eval_agent.py`**, which
triggers `__init_subclass__` registration. Just name the file accordingly.

> Broken plugins are skipped silently (`except ImportError: pass`), so if your evaluator does
> not show up, import the module directly to see the real error.

### 5. Verify

```bash
# Check registration
python -c "import evaluator.plugin.eval_agent; \
from evaluator.core.interfaces.abstract_eval_agent import AbstractEvalAgent; \
print(sorted(AbstractEvalAgent.get_all()))"

# Lint check
ruff check evaluator/plugin/eval_agent/
```

## Key Files

| File | Description |
|------|------|
| `evaluator/core/schema.py` | Dynamic dispatch to each plugin's config model (no edit needed) |
| `evaluator/core/interfaces/abstract_eval_agent.py` | Abstract interface |
| `evaluator/plugin/eval_agent/` | Plugin implementation directory |
| `evaluator/plugin/eval_agent/semantic_eval_agent.py` | LLM evaluation reference implementation |
| `evaluator/plugin/eval_agent/medcalc_eval_agent.py` | LLM + rules hybrid reference implementation |
| `evaluator/plugin/eval_agent/record_retrieval_eval_agent.py` | Rule-based (zero-LLM) reference implementation |
| `evaluator/plugin/eval_agent/rubric_eval_agent.py` | Generic rubric reference implementation (criteria + signals + latency) |
| `evaluator/plugin/eval_agent/kg_qa_eval_agent.py` | Numeric-tolerance / stepwise-credit reference implementation |
