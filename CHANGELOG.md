# Changelog

Versions are git tags on `main` (`0.1.0`, `0.2.0`, …). Each entry links the pull request that made the change.

## Unreleased

### MedCalc-Bench

- Gestational-age ground truths stored as tuple strings (`"('35 weeks', '4 days')"`) are parsed, so the 20 such cases in `full.jsonl` are graded on the answer instead of always failing ([#40](https://github.com/thetahealth/mirobody-eval/pull/40)).

### ESL-Bench

- New dataset `demo-events-20260930`: 11 questions about life events, phrased the way a person asks about their own data (why a reading changed, a wrong suspect, a medicine's expected effect, which readings show an illness, a change that did not happen), on six synthetic users from batch `202609`. Each answer was checked against the readings and diary events the model can see ([#38](https://github.com/thetahealth/mirobody-eval/pull/38)).

### Web UI

- In a reply's tool-call list, consecutive calls to the same tool are one row (`28–43  query_duckdb  ×16`) that opens to the calls. Every call is still listed with its own step number; the report data is unchanged ([#37](https://github.com/thetahealth/mirobody-eval/pull/37)).
- The report list is a table: benchmark / dataset, model, cases, average score, cost, finish time. The internal target label, the always-`completed` badge and the raw filename are gone from the row; the filename still searches and shows on hover ([#33](https://github.com/thetahealth/mirobody-eval/pull/33)).
- A model missing from the price table is no longer priced as gpt-4.1. A total that leaves one out reads `—` or `≥ $x`, and the report page says how many models are unpriced. `provider:` and `openai/` prefixes are looked up by the bare model ([#33](https://github.com/thetahealth/mirobody-eval/pull/33)).

## 0.2.0 — 2026-11-19

Nothing to migrate. Reports written by 0.1.0 open as before, and the new report-list columns are computed from them when the list loads. ESL-Bench data already downloaded stays as it is; from now on, `prepare_data --dataset <name>` fetches only what that dataset needs, and the Web UI no longer downloads every batch at startup.

### Grading

- `kg_qa`: the behavioral judge gets `max_tokens=2000` (was 300). A reasoning-model judge spent the budget on reasoning, the verdict came back truncated, and the case was recorded as ungraded: 3–26% of behavioral cases per configuration in one 320-case run ([#21](https://github.com/thetahealth/mirobody-eval/pull/21)).
- `kg_qa`: when an answer contains more than one `ANSWER:` line, the last one is graded. An answer that drafted a value and then corrected it was graded on the draft ([#22](https://github.com/thetahealth/mirobody-eval/pull/22)).

### ESL-Bench

- `prepare_data --dataset <name>` syncs the question banks (~34 MB), then fetches only the users that dataset's questions point at: 27 users for `sample50-20260331`, against 335 users and about 5 GB for every batch. The Web UI syncs the banks at startup and, when a run starts, fetches what its dataset still lacks. A dataset that points at users outside every published batch is refused up front ([#31](https://github.com/thetahealth/mirobody-eval/pull/31)).
- Data preparation loads each indicator table in one statement instead of one `INSERT` per row ([#23](https://github.com/thetahealth/mirobody-eval/pull/23)).

### Web UI

- The report page shows the per-tag stats, which were never displayed, and clicking a tag filters the cases ([#24](https://github.com/thetahealth/mirobody-eval/pull/24)).
- Dialogue turns in the case detail render as Markdown. Raw HTML is escaped and images are not fetched ([#25](https://github.com/thetahealth/mirobody-eval/pull/25)).
- Each case in a report has its own link (`…#case-<id>`), which opens the report with that case expanded ([#26](https://github.com/thetahealth/mirobody-eval/pull/26)).
- The case detail shows the expected answer next to the model's final answer ([#27](https://github.com/thetahealth/mirobody-eval/pull/27)).
- The report list shows each report's case count, average score, cost and number of ungraded cases ([#28](https://github.com/thetahealth/mirobody-eval/pull/28)).
- A reply's tool calls are shown as a collapsed step list under the answer, with each call's arguments and result one click away ([#30](https://github.com/thetahealth/mirobody-eval/pull/30)).

## 0.1.0 — 2026-10-10

The first tagged version. This is `main` as of [#20](https://github.com/thetahealth/mirobody-eval/pull/20), the state the README described at that point:

- Six benchmarks in the repo: ESLBench, ESLBench-Distractor, HealthBench, MedCalc-Bench, Virtual User and Dialogue Quality. Any of them runs with one `benchmark.basic_runner` command against a pluggable target: `llm_api`, a mirobody deployment, or your own.
- `mirobody` target and seeder: seed a synthetic ESL-Bench user into your own mirobody deployment and score it ([#11](https://github.com/thetahealth/mirobody-eval/pull/11), [#13](https://github.com/thetahealth/mirobody-eval/pull/13), [#15](https://github.com/thetahealth/mirobody-eval/pull/15)).
- Question banks are fetched from the release they are quoted from, and every report records which bank its score came from ([#17](https://github.com/thetahealth/mirobody-eval/pull/17)).
- A case that raised an exception is recorded as `error` and kept out of the average, rather than scored 0 ([#18](https://github.com/thetahealth/mirobody-eval/pull/18)).
- A `provider:` prefix on a model name picks the endpoint (`dashscope:qwen3.5-flash`), for each of the three models in a run: target, judge and virtual user ([#18](https://github.com/thetahealth/mirobody-eval/pull/18)).
- `kg_qa` has a stricter behavioral judge, with deterministic adjudication for contract items. Behavioral scores from before this change are not comparable with later ones ([#20](https://github.com/thetahealth/mirobody-eval/pull/20)).
- Web UI for launching runs and reading reports, Claude Code slash commands from adding a benchmark to reading a report, English and Chinese READMEs whose commands are checked against the code, and CI that runs lint and the unit tests.
