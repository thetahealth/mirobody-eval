"""Multi-model 2D behavioral eval for the eslbench distractor (interference) layer.

Cohort user5200–5209. Default dataset distractor-behavioral-20260723.jsonl (behavioral);
pass --data distractor-computable-20260723.jsonl for the computable-distractor set.

Reuses production pieces (eslbench/retrieve tool group + kg_qa behavioral judge) so results
are comparable to the runner. For each target model, runs every item through the agent,
judges (behavioral: 3 booleans → PASS/FAIL; else by answer_type), and writes to the canonical
runner report dir (benchmark/report/eslbench_distractor/):
  cases_<dataset_stem>_<safe_model>.jsonl   — one scored case per line, appended+flushed
                                              (crash-safe + resumable; skips ids already done)
  <dataset_stem>_<safe_model>.json          — final {"cases":[...]} report (for report_2d)

Targets routed via do_execute (gpt*→OpenAI, gemini*→Google, [次]→nova, qwen→dashscope, nb:→nebula).
Judge default gpt-5.4-mini (independent of target). Concurrency gentle.

Usage: python run_distractor_eval.py --data distractor-behavioral-20260723.jsonl \
           --models "gpt-5.2,[次]claude-sonnet-4-6" [--judge gpt-5.4-mini] [--concurrency 3]
"""
import argparse, asyncio, json, os, re, sys
from pathlib import Path

# nova relay (once.novai.su) must bypass the http proxy 14.103.45.158 — the proxy 301-loops
# that host (curl via proxy → 301; --noproxy → 401 reachable). Ensure it's in no_proxy so the
# [次]/[限时] relay models (claude/kimi/glm/gemini) connect. Harmless if already present.
for _v in ("no_proxy", "NO_PROXY"):
    _cur = os.environ.get(_v, "")
    if "once.novai.su" not in _cur:
        os.environ[_v] = (_cur + "," if _cur else "") + "once.novai.su,novai.su"

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))

from evaluator.utils.llm import do_execute
from evaluator.plugin.target_agent.llm_api_target_agent import _load_tool_group
from evaluator.plugin.eval_agent.kg_qa_eval_agent import KgQaEvalAgent, KgQaEvalInfo
from report_2d_metrics import compute_2d_metrics, print_report

DATA = HERE / "distractor-behavioral-20260723.jsonl"
META = json.load(open(HERE / "metadata.json"))
SYS_PROMPT = META["target"][0]["fields"]["system_prompt"]["default"]
# results land in the canonical runner report dir (benchmark/report/<benchmark>/),
# NOT nested under data/, so all eslbench_distractor results share one home
RPTDIR = HERE.parents[1] / "report" / "eslbench_distractor"; RPTDIR.mkdir(parents=True, exist_ok=True)
TGT_TIMEOUT = 240  # per-item target-agent timeout (s); raised via --timeout for slow relay models
DATA_TAG = "distractor-behavioral-20260723"  # case-file namespace; set from --data stem so datasets don't collide


def safe(m):
    return re.sub(r'[^A-Za-z0-9._-]', '_', m)


_ORIG_OR_BASE = os.environ.get("OPENROUTER_BASE_URL")
_ORIG_OR_KEY = os.environ.get("OPENROUTER_API_KEY")


def route_env(model):
    """Route the process-global OpenRouter endpoint per model:
      nb:<name> → nebula (llm.ai-nebula.com, key from NEBULA_KEY env)
      *qwen*    → dashscope OpenAI-compatible endpoint
      else      → restore the original OpenRouter/default routing."""
    if model.startswith("nb:"):
        os.environ["OPENROUTER_BASE_URL"] = os.environ.get("NEBULA_BASE_URL", "https://llm.ai-nebula.com/v1")
        os.environ["OPENROUTER_API_KEY"] = (os.environ.get("NEBULA_TARGET_KEY")
                                            or os.environ.get("NEBULA_KEY_6")
                                            or os.environ.get("NEBULA_API_KEY", ""))
    elif "qwen" in model:
        os.environ["OPENROUTER_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        os.environ["OPENROUTER_API_KEY"] = os.environ.get("DASHSCOPE_API_KEY", "")
    else:
        if _ORIG_OR_BASE is not None:
            os.environ["OPENROUTER_BASE_URL"] = _ORIG_OR_BASE
        else:
            os.environ.pop("OPENROUTER_BASE_URL", None)
        if _ORIG_OR_KEY is not None:
            os.environ["OPENROUTER_API_KEY"] = _ORIG_OR_KEY


def load_items(limit=None, data=None, shard=None):
    items = [json.loads(l) for l in open(data or DATA) if l.strip()]
    if shard:  # (i, n) → every n-th item starting at i, for parallel sharding
        i, n = shard
        items = items[i::n]
    return items[:limit] if limit else items


async def run_target(item, model):
    real_model = model[3:] if model.startswith("nb:") else model  # strip nebula routing prefix
    em = item["user"]["target_overrides"]["theta_api"]["email"]
    tools, ctx_class = _load_tool_group("eslbench/retrieve", {"user_email": em})
    ctx = ctx_class(user_email=em) if ctx_class else {"user_email": em}
    hist = None
    if item.get("history"):
        hist = [type("M", (), {"role": m["role"], "content": m["content"]})() for m in item["history"]]
    res = await do_execute(model=real_model, system_prompt=SYS_PROMPT, input=item["description"],
                           tools=tools, history_messages=hist, tool_context=ctx,
                           tool_context_schema=ctx_class, max_tokens=1500)
    c = res.content
    return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)


async def judge(item, response_text, judge_model):
    ev = item["eval"]; sd = ev["source_data"]
    # empty/blank response = model failed to answer → error, NOT a resist-PASS (would inflate 抗假率)
    if not (response_text or "").strip():
        return 0.0, "empty_response — no answer produced (not counted as resist)"
    at = ev.get("answer_type", "text")          # route by the item's REAL answer_type
    kw = {}
    if ev.get("number_tolerance") is not None:
        kw["number_tolerance"] = ev["number_tolerance"]
    cfg = KgQaEvalInfo(answer_type=at, expected_value=ev["expected_value"],
                       key_points=ev.get("key_points", []), source_data=sd,
                       behavior_kind=sd.get("behavior_kind") or "existence",
                       model=judge_model, **kw)
    ag = KgQaEvalAgent(cfg)
    if at == "numeric_value":
        return ag._eval_numeric(response_text)
    if at == "boolean":
        return ag._eval_boolean(response_text)
    if at == "list":
        return ag._eval_list(response_text)
    if at == "behavioral":
        return await ag._eval_behavioral(response_text)
    return await ag._eval_text(response_text)


async def one(sem, item, model, judge_model, fh, done):
    # skip only if already scored (score not None); a saved-but-unjudged case is retried
    prev = done.get(item["id"])
    if prev and prev.get("eval", {}).get("score") is not None:
        return None
    async with sem:
        sd = item["eval"]["source_data"]
        tag = f"{sd.get('behavior_kind') or sd.get('fl_primitive','computable')}/{sd.get('premise_polarity')}"
        base = {"id": item["id"], "eval_config": {"source_data": sd},
                "tags": [f"behavior_kind:{sd.get('behavior_kind') or 'computable'}",
                         f"polarity:{sd.get('premise_polarity')}",
                         f"tier:{sd.get('difficulty_tier','?')}"]}
        # target — expensive. A timeout / target error is an AGENT failure (the agent could not
        # produce an answer): score it 0 (FAIL) and PERSIST, rather than dropping the item.
        # Dropping silently shrinks n and hides agentic give-up; scoring 0 counts it honestly.
        try:
            resp = prev.get("response") if prev else None
            if resp is None:
                resp = await asyncio.wait_for(run_target(item, model), timeout=TGT_TIMEOUT)
        except Exception as e:
            detail = f"target_failed (agent problem, scored 0): {str(e)[:80]}"
            case = {**base, "eval": {"score": 0.0, "feedback": detail}, "response": ""}
            fh.write(json.dumps(case, ensure_ascii=False) + "\n"); fh.flush(); os.fsync(fh.fileno())
            print(f"  {tag:20s} TGT-ERR→score=0  {item['id'][-28:]}", flush=True)
            return case
        # judge — flaky; persist the target response regardless so it can be re-judged
        try:
            score, detail = await asyncio.wait_for(judge(item, resp, judge_model), timeout=90)
        except Exception as e:
            score, detail = None, f"JUDGE_FAILED: {str(e)[:80]}"
            print(f"  {tag:20s} JDG-ERR (resp kept)  {item['id'][-28:]}", flush=True)
        case = {**base, "eval": {"score": score, "feedback": detail}, "response": resp}
        fh.write(json.dumps(case, ensure_ascii=False) + "\n"); fh.flush(); os.fsync(fh.fileno())
        if score is not None:
            print(f"  {tag:20s} score={score}  {item['id'][-28:]}", flush=True)
        return case if score is not None else None


async def eval_model(model, items, judge_model, concurrency, shard=None):
    route_env(model)
    base = f"cases_{DATA_TAG}_{safe(model)}"
    # each shard writes its own file; resume done-set reads ALL shards + main for this model
    casef = RPTDIR / (f"{base}.s{shard[0]}.jsonl" if shard else f"{base}.jsonl")
    done = {}
    import glob as _glob
    for df in _glob.glob(str(RPTDIR / f"{base}*.jsonl")):
        for l in open(df):
            if l.strip():
                c = json.loads(l)
                # dedup by id, last line wins BUT a scored case never regresses to unjudged
                if c["id"] in done and done[c["id"]].get("eval", {}).get("score") is not None \
                        and c.get("eval", {}).get("score") is None:
                    continue
                done[c["id"]] = c
    scored = sum(1 for c in done.values() if c.get("eval", {}).get("score") is not None)
    print(f"\n=== target={model}  judge={judge_model}  items={len(items)}  "
          f"resume={scored} scored / {len(done)} seen  concurrency={concurrency} ===", flush=True)
    sem = asyncio.Semaphore(concurrency)
    with open(casef, "a") as fh:
        fresh = [c for c in await asyncio.gather(
            *[one(sem, it, model, judge_model, fh, done) for it in items]) if c]
    # merge: fresh (re-judged) overrides done; keep only scored for the report
    merged = dict(done)
    for c in fresh:
        merged[c["id"]] = c
    cases = [c for c in merged.values() if c.get("eval", {}).get("score") is not None]
    report = {"benchmark_name": "eslbench_distractor", "dataset_name": DATA_TAG,
              "runtime_target": {"type": "llm_api", "model": model}, "cases": cases}
    (RPTDIR / f"{DATA_TAG}_{safe(model)}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n--- {model}: {len(cases)}/{len(items)} scored ---", flush=True)
    print_report(compute_2d_metrics(cases))
    return model, cases


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--judge", default="gpt-5.4-mini")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=240, help="per-item target timeout (s)")
    ap.add_argument("--shard", default=None, help="i/n → process items[i::n] (parallel sharding)")
    args = ap.parse_args()
    global TGT_TIMEOUT, DATA_TAG
    TGT_TIMEOUT = args.timeout
    if args.data:
        DATA_TAG = Path(args.data).stem
    shard = None
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        shard = (i, n)
    items = load_items(args.limit, args.data, shard)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        await eval_model(m, items, args.judge, args.concurrency, shard)


if __name__ == "__main__":
    asyncio.run(main())
