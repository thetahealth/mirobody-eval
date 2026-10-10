# Developing a System Under Test (TargetAgent)

> **Using in Claude Code**: Type `/add-target-agent <agent-name>`, followed by your system-under-test description (e.g., API endpoint, authentication method, protocol details). Claude will automatically generate the configuration, implementation, and registration code.

## Overview

TargetAgent encapsulates the integration layer for the system under test, responsible for sending virtual user inputs to the target system and retrieving responses. Each type of system under test (HTTP API, SDK, GUI, etc.) corresponds to a TargetAgent plugin.

## Existing Target Systems

| Name | Description |
|------|------|
| `llm_api` | Generic LLM API (OpenAI / Gemini / Anthropic / GLM), based on the unified `do_execute` LLM layer |
| `hermes` | External HTTP service that manages its own session/token |
| `evermem` | EverMem memory service |
| `mem0_rag_api` | Mem0 RAG service |
| `naive_rag_api` | Baseline RAG over the case's own corpus |
| `hippo_rag_api` | HippoRAG service |
| `dyg_rag_api` | DyG-GraphRAG service (vendored under `evaluator/vendor/dyg_graphrag/`) |

## Development Steps

### 1. Determine the Integration Method

- **HTTP API** — Most common, interacts with the target system via HTTP requests
- **SDK/Client** — Uses the target system's Python SDK
- **CLI** — Invokes the target system via command line
- **LLM API** — Directly calls an LLM provider's API

### 2. Define the Configuration Model

Put the configuration model **next to the plugin**, in `evaluator/plugin/target_agent/my_target_agent.py`:

```python
class MyTargetInfo(BaseModel):
    """My target system configuration"""
    model_config = ConfigDict(extra="forbid")

    type: Literal["my_target"] = "my_target"
    # Connection parameters (infrastructure params should go in .env, business params in the config model)
```

Register it on the class with `params_model=` (next step). You do **not** edit
`evaluator/core/schema.py` — it resolves the model from `AbstractTargetAgent`'s registry.

### 3. Implement the TargetAgent

Create the implementation file in `evaluator/plugin/target_agent/`:

```python
class MyTargetAgent(AbstractTargetAgent, name="my_target", params_model=MyTargetInfo):
    async def _generate_next_reaction(self, test_action):
        # 1. Extract user input from test_action.semantic_content
        # 2. Call the target system
        # 3. Return TargetAgentReaction(type="message", message_list=[...])

    async def cleanup(self):
        # Release connection resources
```

### 4. Register the Plugin

Nothing to import by hand: `evaluator/plugin/target_agent/__init__.py` walks the package with
`pkgutil` and auto-imports every module whose filename **ends with `_target_agent.py`**, which
triggers `__init_subclass__` registration. Just name the file accordingly, and pass
`params_model=` on the class so the config model lands in the registry.

> `ImportError` is swallowed there (`except ImportError: pass`), so if your target does not show
> up, import the module directly to see the real error. A `SyntaxError` or `NameError` at module
> level is **not** swallowed and will break the whole package import.

### 5. Verify

```bash
python -c "import evaluator.plugin.target_agent; \
from evaluator.core.interfaces.abstract_target_agent import AbstractTargetAgent; \
print(sorted(AbstractTargetAgent.get_all()))"
ruff check evaluator/plugin/target_agent/
```

## Key Design Patterns

- **Lazy initialization**: Authentication/connection setup is triggered on first call
- **Response format**: `TargetAgentReaction(type="message", message_list=[{"content": "..."}])`
- **Resource cleanup**: Implement the `cleanup()` method to release network connections
- **Environment variables**: Infrastructure parameters (base_url, timeout) go in `.env`; business parameters go in the config model

## Key Files

| File | Description |
|------|------|
| `evaluator/core/schema.py` | Dynamic dispatch to each plugin's config model (no edit needed) |
| `evaluator/core/interfaces/abstract_target_agent.py` | Abstract interface |
| `evaluator/plugin/target_agent/` | Plugin implementation directory |
| `evaluator/plugin/target_agent/hermes_target_agent.py` | HTTP API reference implementation |
| `evaluator/plugin/target_agent/llm_api_target_agent.py` | LLM API reference implementation |
