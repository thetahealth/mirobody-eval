# Dialogue Quality (RECORD + RETRIEVAL)

Two things a health assistant does constantly, and two different ways of getting
them wrong:

- **RECORD** — the user states a reading ("blood pressure 138/92 today"). A good
  reply confirms it in one line. A bad one launches into advice nobody asked
  for, or paraphrases the numbers away.
- **RETRIEVAL** — the user asks for it back. A good reply returns the value that
  was recorded. A bad one returns a plausible number that is not the one.

Both failures are invisible to a single-turn benchmark, because each turn on its
own looks fine.

## Evaluators

Layer 1 is **rule-based, zero LLM** — `record_retrieval`, checking per-turn
checkpoints ([`record_retrieval_eval_agent.py`](../../../evaluator/plugin/eval_agent/record_retrieval_eval_agent.py)):

| checkpoint | what it checks |
|---|---|
| `record_ack` | reply within a character budget, no advice keywords, echoes the user's numbers |
| `retrieval_data` | reply contains the expected values (substring match) |
| `skip` | not checked |

Layer 2 is **LLM-as-judge** — `dialogue_quality`, scoring six dimensions
(accuracy, personalization, comprehensiveness, readability, actionability,
context memory) with per-persona penalty rules.

`smoke2` runs on the generic `rubric` evaluator instead: per-turn latency
budgets plus a `criteria` list, either a natural-language `llm_rubric` or a
`signal_check` against registered signals such as `has_chart`, `has_table` or
`char_count`. It always returns `scored` — no pass/fail.

## Datasets

| Dataset | Cases | Purpose |
|---|---|---|
| `smoke` | 5 | fast gate |
| `smoke2` | 4 | image attachment + proactive chart + fabrication guard + latency |
| `core` | 15 | every record type |
| `l2_core` | 12 | interpretation + planning + Q&A, across persona combinations |
| `l2_hard` | 9 | adversarial: contradictory input, unsafe requests, memory pressure |

## Running

```bash
# Against a plain model — tests terseness and recall within the conversation
uv run python -m benchmark.basic_runner dialogue_quality smoke \
    --target-type llm_api --target-model gpt-5.4-mini -p 1 -v

# Layer 2 needs a judge as well as a target
uv run python -m benchmark.basic_runner dialogue_quality l2_core \
    --target-type llm_api --target-model gpt-5.4-mini \
    --eval-model gpt-5.4-mini -p 1
```

To score a self-hosted mirobody deployment instead, name the seeded user the
cases should run as — they carry no user identity of their own:

```bash
uv run python -m benchmark.basic_runner dialogue_quality smoke \
    --target-type mirobody --target-override user_email=user5086@demo -p 1
```

## One thing to know before comparing scores

The `record_ack` character budget is a **product convention**, not a clinical or
universal one: these cases were written against an assistant whose spec said a
recording confirmation should stay short. A system that answers correctly but
verbosely scores low here, and that is the intended reading — the dataset
measures conformance to that convention, alongside whether the data survived the
round trip.

So a Layer 1 score is comparable between two systems you run yourself. It is not
a statement about which assistant is better in general, and the budget is worth
adjusting in the case files if your product's convention differs.
