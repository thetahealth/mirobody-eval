---
name: eslbench-report-analysis
description: Use when analyzing ESL-Bench evaluation reports - extracting scores by difficulty dimension, comparing methods, checking fail rates, token costs, duration per query. Triggers on keywords like eslbench report, benchmark results, group by difficulty.
---

# ESL-Bench Report Analysis

Extract and compare evaluation results from HolyEval benchmark report JSON files.

## Report Locations

```
benchmark/report/eslbench/
  {dataset}_{target_type}_{model}_{date}_{time}.json

（报告根目录可用 HOLYEVAL_REPORT_DIR 环境变量覆盖，见 evaluator/utils/paths.py）
```

Example report paths for recent 3 days: YYYYMMDD-3 to YYYYMMDD: `benchmark/report/eslbench/sample50-20260331_*_{YYYYMMDD-2,YYYYMMDD-1,YYYYMMDD}*.json`

## Report JSON Structure

```python
report = {
    "benchmark_name": "eslbench",
    "dataset_name": "sample50-20260331",
    "runtime_target": {...},
    "cases": [...],           # Per-case results
    "pass_count": 0,          # Cases scoring >= threshold
    "fail_count": 3,          # Cases with eval errors (not low scores)
    "pass_rate": 0.0,         # pass_count / total (NOTE: often 0 due to strict threshold)
    "avg_score": 0.512,       # Mean score across all cases (0-1 scale)
    "total_duration_seconds": 8492
}
```

### Per-Case Structure

```python
case = {
    "id": "user5022_AT_demo_Q001",
    "title": "[Lookup] ...",
    "tags": ["difficulty:Trend", "answer_type:numeric_value"],
    "eval": {
        "result": "scored",    # or "error"
        "score": 0.85,         # 0.0 - 1.0
        "feedback": "..."
    },
    "cost": {
        "target": {
            "gpt-5.4": {
                "input_tokens": 8621,
                "output_tokens": 314,
                "total_tokens": 8935,
                "input_token_details": {"cache_read": 7168},
                "output_token_details": {"reasoning": 0}
            }
        }
    },
    "start": "...", "end": "..."
}
```

### Key Fields

| Field | Where | Notes |
|---|---|---|
| Difficulty dimension | `case.tags` → `"difficulty:Lookup"` | 5 values: Lookup, Trend, Comparison, Anomaly, Explanation |
| Score | `case.eval.score` | 0.0-1.0, multiply by 100 for percentage |
| Answer type | `case.tags` → `"answer_type:text"` | text, numeric_value, list, boolean |
| Fail | `report.fail_count` | Eval errors, not low scores |
| Token usage | `case.cost.target.{model}.input_tokens` | May be None/empty if not tracked |
| Duration | `report.total_duration_seconds` | Wall-clock for entire run |

## Resolving Input Files

Input can be any of:
- **Glob pattern**: `benchmark/report/eslbench/sample50-20260331_*.json`
- **Directory**: scan all `*.json` inside
- **Explicit file list**: one or more JSON file paths
- **Zip URL**: download, extract, scan `*.json` inside
- **Zip file path**: extract, scan `*.json` inside

The analysis output MUST show the **full resolved file path** for each report processed.

**CRITICAL: Script stdout may be truncated when output is large (>2KB).** To ensure file paths are always visible to the user:
1. Write the resolved file path list into the **markdown report file** (under a `## Processed Report Files` section)
2. Also print a **brief summary** (file count + sources) directly to the user in your text response, e.g. "31 files analyzed: 21 LOCAL + 10 ZIP"

Do NOT rely on script stdout alone for showing file paths — it will be truncated for large analyses.

Default report location (if no input specified): `benchmark/report/eslbench/`

## Analysis Script Template

```python
import json, glob, os, re, zipfile, tempfile
from collections import defaultdict

DIMS = ["Lookup", "Trend", "Comparison", "Anomaly", "Explanation"]

def resolve_report_files(inputs):
    """Resolve various input forms to a sorted list of JSON file paths.
    inputs: list of paths, glob patterns, directories, or zip files."""
    files = []
    for inp in inputs:
        inp = os.path.expanduser(inp)
        if inp.endswith(".zip") and os.path.isfile(inp):
            tmpdir = tempfile.mkdtemp()
            with zipfile.ZipFile(inp) as z:
                z.extractall(tmpdir)
            files.extend(sorted(glob.glob(f"{tmpdir}/**/*.json", recursive=True)))
        elif os.path.isdir(inp):
            files.extend(sorted(glob.glob(f"{inp}/**/*.json", recursive=True)))
        elif "*" in inp or "?" in inp:
            files.extend(sorted(glob.glob(inp)))
        elif os.path.isfile(inp):
            files.append(inp)
    # Filter out __MACOSX etc
    return [f for f in files if "/__MACOSX/" not in f and f.endswith(".json")]

def analyze_reports(inputs=None):
    if inputs is None:
        inputs = [os.environ.get("HOLYEVAL_REPORT_DIR", "benchmark/report") + "/eslbench"]
    files = resolve_report_files(inputs)
    rows = []
    for fpath in files:
        fname = os.path.basename(fpath)
        with open(fpath) as f:
            report = json.load(f)
        cases = report.get("cases", [])
        n = len(cases)
        if n == 0: continue
        fail_n = report.get("fail_count", 0)
        dur = report.get("total_duration_seconds", 0)

        by_diff = defaultdict(list)
        in_tok = out_tok = 0
        for c in cases:
            diff = next((t.split(":",1)[1] for t in c.get("tags",[]) if t.startswith("difficulty:")), None)
            score = c.get("eval",{}).get("score")
            if diff and score is not None:
                by_diff[diff].append(score)
            for model_usage in (c.get("cost",{}).get("target") or {}).values():
                if isinstance(model_usage, dict):
                    in_tok += model_usage.get("input_tokens",0) or 0
                    out_tok += model_usage.get("output_tokens",0) or 0

        all_scores = [s for scores in by_diff.values() for s in scores]
        rows.append({
            "fpath": fpath,  # full resolved path
            "label": fname,  # filename for display
            "n": n,
            "dims": {d: (sum(by_diff[d])/len(by_diff[d])*100 if by_diff[d] else None) for d in DIMS},
            "total": sum(all_scores)/len(all_scores)*100 if all_scores else 0,
            "fail": f"{fail_n}/{n}", "fail_n": fail_n,
            "dur": dur,
            "per_q": dur/n if n else 0,
            "in_tok": in_tok, "out_tok": out_tok,
        })
    return rows

def fmt_tok(n):
    """Format token count: 1234567 -> '1.2M', 51245 -> '51K', 0 -> '-'."""
    if n == 0: return "-"
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000: return f"{n//1_000}K"
    return str(n)

def print_table(rows):
    """Print score overview table. Always list resolved file paths first."""
    # Show resolved file paths
    print("**Processed report files:**\n")
    for i, r in enumerate(rows, 1):
        print(f"{i}. `{r['fpath']}`")
    print()
    # Score table
    def f(v): return f"{v:5.1f}" if v is not None else "  N/A"
    print(f"| {'Report (file)':<52} | {'Look':>5} | {'Trend':>5} | {'Comp':>5} | {'Anom':>5} | {'Expl':>5} | {'Avg%':>5} | {'Fail':>5} | {'Dur':>6} | {'s/q':>4} | {'InTok':>6} | {'OutTok':>6} |")
    print(f"|{'-'*54}|{'-'*7}|{'-'*7}|{'-'*7}|{'-'*7}|{'-'*7}|{'-'*7}|{'-'*7}|{'-'*8}|{'-'*6}|{'-'*8}|{'-'*8}|")
    for r in rows:
        d = r["dims"]
        print(f"| {r['label']:<52} | {f(d['Lookup'])} | {f(d['Trend'])} | {f(d['Comparison'])} | {f(d['Anomaly'])} | {f(d['Explanation'])} | {r['total']:5.1f} | {r['fail']:>5} | {r['dur']:5.0f}s | {r['per_q']:3.0f}s | {fmt_tok(r['in_tok']):>6} | {fmt_tok(r['out_tok']):>6} |")
```

### After Score Overview: Write a Summary

After printing the score table, **always write a free-form summary** covering:

1. **Best overall performer** — which report/model has highest Avg%, noting fail rate trade-off
2. **Per-dimension winners** — which model leads each dimension (Look/Trend/Comp/Anom/Expl)
3. **Weakest dimension** — which dimension is universally low across all methods and why
4. **Efficiency** — token usage vs. score (cost-effectiveness), duration per query
5. **Reliability** — fail rates, runs that should be discarded (high fail = unreliable)
6. **Notable surprises** — unexpected results (e.g. structured API outperforming LLM, score jumps between runs)

Format as a `## Summary` section with bullet points. Be specific with numbers, don't just say "good" or "bad".

## Fail Reason Analysis

Cases with `eval.result == "fail"` are execution errors, not low scores. Classify by extracting error from `eval.feedback` traceback.

```python
from collections import Counter

def analyze_fails(report):
    """Classify fail reasons from a report. Returns Counter of reason -> count."""
    reasons = Counter()
    for c in report.get("cases", []):
        ev = c.get("eval", {})
        if ev.get("result") != "fail":
            continue
        fb = ev.get("feedback", "")
        if "TimeoutError" in fb or "CancelledError" in fb:
            reasons["Timeout/CancelledError"] += 1
        elif "orchestrator.py" in fb:
            reasons["Orchestrator error (model API failure)"] += 1
        elif "asyncio/tasks.py" in fb:
            reasons["Async task error"] += 1
        else:
            reasons[fb[:80] if fb else "unknown"] += 1
    return reasons
```

### Common Fail Categories

| Category | Cause | Fix |
|---|---|---|
| Timeout/CancelledError | Query took too long, asyncio cancelled | Increase timeout or reduce concurrency |
| Orchestrator model error | Target model API returned error (rate limit, bad response) | Check API key/quota, retry |
| Async task error | General asyncio failure | Check logs for root cause |

### When to Run Fail Reason Analysis

**For every report with fail_count > 0**, run `analyze_fails()` and include the breakdown in:
1. The **markdown report** — add a `## Fail Reason Breakdown` section listing each report's fail sub-categories (Timeout vs Orchestrator vs Async vs other)
2. The **per-dimension error analysis** — when showing `FAIL_execution_error:N`, also note the sub-category if available (e.g. `FAIL_execution_error:3 (2 Timeout, 1 Orchestrator)`)

Do NOT just report the total `FAIL_execution_error` count without sub-classifying.

## Per-Agent Per-Dimension Error Analysis

Classify WHY each agent scores low on each dimension. Combines fail cases and low-score cases into one view.

### Feedback Classification

`eval.feedback` format: `[answer_type] score=X.XXX — detail`

| Category | Pattern in feedback | Meaning |
|---|---|---|
| `no_data_found` | "no_data", "no relevant" | Agent couldn't retrieve data |
| `numeric_mismatch` | "Number match 0/" | Computed number wrong |
| `partial_list_match` | "F1=" | Incomplete list returned |
| `rubric_low_score` | "LLM judge", "Rubric" | LLM judge scored low on rubric |
| `exact_match_fail` | "exact" + "0/" | Text/boolean exact match failed |
| `FAIL_execution_error` | `eval.result == "fail"` | Execution error (timeout/API) |

### Analysis Script

```python
import json, glob, os, re
from collections import Counter

REPORT_DIR = os.environ.get("HOLYEVAL_REPORT_DIR", "benchmark/report") + "/eslbench"
DIMS = ["Lookup", "Trend", "Comparison", "Anomaly", "Explanation"]

def classify_feedback(fb):
    """Classify a single eval.feedback string into an error category."""
    m = re.match(r'\[(\w+)\]\s*score=([\d.]+)\s*[—-]\s*(.*)', fb)
    if not m:
        return fb[:50] or "unknown"
    _, _, detail = m.groups()
    dl = detail.lower()
    if "no_data" in dl or "no relevant" in dl:
        return "no_data_found"
    if "number match 0/" in dl:
        return "numeric_mismatch"
    if "f1=" in dl:
        return "partial_list_match"
    if "llm judge" in dl or "rubric" in dl:
        return "rubric_low_score"
    if "exact" in dl and "0/" in dl:
        return "exact_match_fail"
    return detail[:50]

def analyze_report_by_dim(report, threshold=0.8):
    """Per-dimension: avg score, low-score count, top error reasons, sample case IDs."""
    results = {}
    for dim in DIMS:
        reasons = Counter()
        reason_cases = defaultdict(list)  # reason -> [case_id, ...]
        scores = []
        for c in report.get("cases", []):
            diff = next((t.split(":",1)[1] for t in c.get("tags",[])
                         if t.startswith("difficulty:")), None)
            if diff != dim:
                continue
            ev = c.get("eval", {})
            cid = c.get("id", "?")
            if ev.get("result") == "fail":
                reasons["FAIL_execution_error"] += 1
                reason_cases["FAIL_execution_error"].append(cid)
                continue
            score = ev.get("score")
            if score is None:
                continue
            scores.append(score)
            if score < threshold:
                cat = classify_feedback(ev.get("feedback", ""))
                reasons[cat] += 1
                reason_cases[cat].append(cid)
        results[dim] = {
            "avg": sum(scores)/len(scores)*100 if scores else 0,
            "scored": len(scores),
            "low": sum(reasons.values()),
            "reasons": reasons,
            "reason_cases": reason_cases,
        }
    return results

def print_full_analysis(inputs=None, show_cases=3):
    """Print per-report-file per-dimension error breakdown with sample case IDs.
    inputs: same as analyze_reports(). show_cases: max case IDs per error category (0 to hide)."""
    if inputs is None:
        inputs = [os.environ.get("HOLYEVAL_REPORT_DIR", "benchmark/report") + "/eslbench"]
    files = resolve_report_files(inputs)
    for fpath in files:
        fname = os.path.basename(fpath)
        with open(fpath) as f:
            report = json.load(f)
        fail_n = report.get("fail_count", 0)
        n = len(report.get("cases", []))
        print(f"\n{'='*70}")
        print(f"  {fpath}")
        print(f"  fails={fail_n}/{n}, avg={report.get('avg_score',0):.3f}")
        print(f"{'='*70}")
        results = analyze_report_by_dim(report)
        for dim in DIMS:
            r = results[dim]
            top3 = r["reasons"].most_common(3)
            top_str = ", ".join(f"{k}:{v}" for k,v in top3) if top3 else "all good"
            print(f"  {dim:<12} avg={r['avg']:5.1f}%  low={r['low']}/{r['scored']}  | {top_str}")
            if show_cases > 0:
                for reason, count in top3:
                    ids = r["reason_cases"][reason][:show_cases]
                    suffix = f" +{count-show_cases} more" if count > show_cases else ""
                    print(f"    {reason}: {', '.join(ids)}{suffix}")
```

### Cross-Agent Dimension Patterns

When multiple agent types are present (LLM-direct, RAG, Structured API), **always generate a cross-agent comparison table** in the markdown report under `## Cross-Agent Error Patterns`. Group agents into categories and show the top 2 error types per dimension per category.

Template (populate from actual data, not hardcoded):

| Dimension | LLM-direct (top errors) | RAG (top errors) | Structured API (top errors) |
|---|---|---|---|
| **Lookup** | ... | ... | ... |
| **Trend** | ... | ... | ... |
| **Comparison** | ... | ... | ... |
| **Anomaly** | ... | ... | ... |
| **Explanation** | ... | ... | ... |

Agent category mapping:
- **LLM-direct**: `llm_api` target type (gpt-5.4, gemini-flash, claude-sonnet, minimax, glm, etc.)
- **RAG**: `*_rag_api` target types (hippo_rag, dyg_rag, mem0_rag, evermem)
- **Structured API**: `hermes` and any privately registered structured-backend target types

After the table, add a **Key insights** paragraph noting which categories dominate which dimensions and why.

Reference patterns (from prior analyses — verify against current data):
- **LLM-direct** tends to fail on Comparison (`partial_list_match`) and Explanation (`numeric_mismatch`)
- **RAG methods** suffer `numeric_mismatch` across all dimensions — retrieval noise degrades computation
- **Structured API at 0%** may be connectivity issue — check if all scores are exactly 0.0
- **Comparison** is typically the weakest dimension across all agent types
- **Anomaly** discriminates most: LLM-direct scores 72-98% while RAG drops to 1-48%

## Error Sample Cases Export

When running analysis, **always export error/low-score cases to a JSON file** alongside the markdown report. This enables case-level debugging without re-parsing report files.

### Export Script

```python
def export_error_cases(inputs=None, threshold=0.8, output_path=None):
    """Export all low-score and failed cases to a JSON file for debugging.
    
    Args:
        inputs: same as analyze_reports()
        threshold: score below this is considered error (default 0.8)
        output_path: output JSON path. Default: docs/eslbench_error_cases-YYYYMMDD.json
    
    Returns: output file path
    """
    import datetime
    if inputs is None:
        inputs = [os.environ.get("HOLYEVAL_REPORT_DIR", "benchmark/report") + "/eslbench"]
    if output_path is None:
        today = datetime.date.today().strftime("%Y%m%d")
        output_path = f"docs/eslbench_error_cases-{today}.json"

    files = resolve_report_files(inputs)
    error_cases = []

    for fpath in files:
        fname = os.path.basename(fpath)
        with open(fpath) as f:
            report = json.load(f)
        for c in report.get("cases", []):
            ev = c.get("eval", {})
            score = ev.get("score")
            result = ev.get("result", "")
            is_fail = result in ("fail", "error")
            is_low = score is not None and score < threshold

            if not is_fail and not is_low:
                continue

            diff = next((t.split(":",1)[1] for t in c.get("tags",[])
                         if t.startswith("difficulty:")), None)
            atype = next((t.split(":",1)[1] for t in c.get("tags",[])
                          if t.startswith("answer_type:")), None)
            category = "FAIL_execution_error" if is_fail else classify_feedback(ev.get("feedback", ""))

            error_cases.append({
                "report_file": fname,
                "case_id": c.get("id", "?"),
                "difficulty": diff,
                "answer_type": atype,
                "score": score,
                "eval_result": result,
                "error_category": category,
                "feedback": ev.get("feedback", ""),
                "title": c.get("title", ""),
                "description": c.get("description", ""),
            })

    # Sort by report_file, then difficulty, then score
    diff_order = {d: i for i, d in enumerate(DIMS)}
    error_cases.sort(key=lambda x: (
        x["report_file"],
        diff_order.get(x["difficulty"], 99),
        x["score"] if x["score"] is not None else -1,
    ))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(error_cases, f, indent=2, ensure_ascii=False)

    # Print summary
    from collections import Counter
    by_cat = Counter(c["error_category"] for c in error_cases)
    by_diff = Counter(c["difficulty"] for c in error_cases)
    print(f"\nExported {len(error_cases)} error cases to: {output_path}")
    print(f"  By category: {dict(by_cat.most_common())}")
    print(f"  By difficulty: {dict(by_diff.most_common())}")
    return output_path
```

### Output Format

Each exported case contains:

| Field | Description |
|---|---|
| `report_file` | Source report filename |
| `case_id` | e.g. `user5022_AT_demo_Q001` |
| `difficulty` | Lookup/Trend/Comparison/Anomaly/Explanation |
| `answer_type` | text/numeric_value/list/boolean |
| `score` | 0.0-1.0, null for execution errors |
| `eval_result` | "scored", "fail", "error" |
| `error_category` | Classified: numeric_mismatch, partial_list_match, etc. |
| `feedback` | Full eval feedback string |
| `title` | Question title |
| `description` | Full question description |

### When to Export

**Always call `export_error_cases()` after `print_table()` and `print_full_analysis()`.** The exported file is the entry point for case-level debugging — use it to trace specific failures back to dataset issues or model weaknesses.

## Interpretation Notes

- **`pass_rate`**: Often 0.0 because it uses a strict binary threshold; use `avg_score` instead for meaningful comparison
- **`fail_count`**: Eval-level errors (API timeout, parse failure), NOT low-scoring answers. High fail = unreliable run
- **A structured-API target at 0%**: Usually means API connectivity issue, not model failure
- **Token cost = 0**: Means token tracking not enabled for that target type (e.g. mem0_rag)
- **Multiple runs of same model**: Take the latest or lowest-fail run; early runs may have config bugs
- **`s/q` (seconds per query)**: Dominated by retrieval + generation time; <30s usually means cached/fast path

## Per-Agent Performance Analysis (Verified Runs Only)

**Filter criteria**: Only include runs where `fail_rate < 40%` AND `scored_count > 30` (i.e. at least 30 successfully scored answers). This excludes partial/broken runs and ensures statistically meaningful comparisons.

When multiple runs of the same agent exist, **keep only the latest run** that passes the filter (by timestamp in filename).

### Agent Performance Table

Generate a `## Agent Performance (Verified)` section with a filtered, deduplicated table:

```python
def get_verified_agents(rows):
    """Filter to verified runs: fail_rate < 40% and scored > 30. Deduplicate by agent, keep latest."""
    verified = []
    for r in rows:
        scored = r["n"] - r["fail_n"]
        fail_rate = r["fail_n"] / r["n"] if r["n"] else 1.0
        if fail_rate < 0.40 and scored > 30:
            verified.append(r)

    # Deduplicate: extract agent key (target_type + model), keep latest by filename timestamp
    from collections import OrderedDict
    by_agent = OrderedDict()
    for r in sorted(verified, key=lambda x: x["label"]):
        # Extract agent key: everything before the date_time suffix
        # e.g. "sample50-20260407_llm_api_gpt-5.4" from "sample50-20260407_llm_api_gpt-5.4_20260407_190032.json"
        parts = r["label"].rsplit("_", 2)  # split off YYYYMMDD_HHMMSS.json
        agent_key = parts[0] if len(parts) >= 3 else r["label"]
        if agent_key not in by_agent or r["label"] > by_agent[agent_key]["label"]:
            by_agent[agent_key] = r
    return list(by_agent.values())
```

### Table Format

```
| Agent | Look | Trend | Comp | Anom | Expl | Avg% | Scored | Fail | s/q | InTok |
```

Sort by `Avg%` descending. This is the **primary comparison table** — the Score Overview shows all raw runs, this shows only verified, deduplicated results.

### Per-Agent Dimension Radar Summary

For each verified agent, add a one-line text radar showing relative strengths:

```
gpt-5.4: Trend(70.0) > Anom(57.1) > Comp(44.4) > Look(30.1) > Expl(14.2)
```

This helps quickly identify each agent's strength/weakness profile.

## Output to File

When saving analysis, produce **two files**:

1. **Markdown report**: `docs/eslbench_report_analysis-YYYYMMDD.md` — must contain ALL of:
   - `## Processed Report Files` — numbered list of full resolved file paths (with source tag if mixed)
   - `## Score Overview` — the score table
   - `## Summary` — the 6-point free-form analysis
   - `## Fail Reason Breakdown` — per-report sub-classification of fails (Timeout/Orchestrator/Async/other), only for reports with fail_count > 0
   - `## Cross-Agent Error Patterns` — the cross-agent dimension comparison table (if multiple agent types present)
   - `## Agent Performance (Verified)` — deduplicated table of agents with fail_rate < 40% and scored > 30, plus per-agent dimension radar
   - `## Error Distribution` — aggregate error category and difficulty counts
2. **Error cases JSON**: `<report-dir>/eslbench_error_cases-YYYYMMDD.json` — all low-score/failed cases for debugging

**Always** print where the two artifacts were written, e.g.:

```
Report:      <report-dir>/eslbench_report_analysis-20260401.md
Error cases: <report-dir>/eslbench_error_cases-20260401.json (N cases)
```

## Final: Verify Output Sections

**After writing the markdown report, MUST run this verification before completing:**

```bash
REQUIRED_SECTIONS=("## Processed Report Files" "## Score Overview" "## Summary" "## Fail Reason Breakdown" "## Cross-Agent Error Patterns" "## Agent Performance (Verified)" "## Error Distribution")
MISSING=()
for section in "${REQUIRED_SECTIONS[@]}"; do
  grep -qF "$section" "$OUTPUT_FILE" || MISSING+=("$section")
done
if [ ${#MISSING[@]} -eq 0 ]; then
  echo "OK: All 6 required sections present"
else
  echo "MISSING sections:"
  printf "  - %s\n" "${MISSING[@]}"
fi
```

If any section is MISSING, add it to the report file before completing. Exceptions:
- `## Fail Reason Breakdown` may be omitted ONLY if ALL reports have `fail_count == 0`
- `## Cross-Agent Error Patterns` may be omitted ONLY if all reports are from the same agent type
- `## Agent Performance (Verified)` may be omitted ONLY if no runs pass the filter (fail_rate < 40% AND scored > 30)
