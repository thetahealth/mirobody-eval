"""Build a tier-aware sample200-0730 for the leaderboard.

干扰加权(对齐 0530 的 200 规模)+ 按混淆度 tier 抽样:
  --pro       : 偏 hard tier(集中难题,像 0530-pro)——hard:mid:easy ≈ 6:3:1
  --balanced  : easy/mid/hard 均衡(默认)——画"混淆度递增下的抵抗曲线"用

组成目标(可调): existence/noop 各 40、attribution 20、computable-distractor 20(干扰 120)
+ 正常 5 维各 16(80)= 200。tier 从 source_data.difficulty_tier 读,否则从 observed 量回算。
不足则 log 缺口(no silent cap)。

用法: python build_sample200.py <pool.jsonl> [--pro|--balanced] [--out FILE]
"""
import json, sys, random, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
_ESL = HERE.parent / "eslbench"   # benchmark/data/eslbench
POOL = sys.argv[1] if len(sys.argv) > 1 else str(_ESL / "full-20260730.clean.jsonl")
PRO = "--pro" in sys.argv
OUT = _ESL / (f"sample200-0730-{'pro' if PRO else 'balanced'}.jsonl")
for i, a in enumerate(sys.argv):
    if a == "--out" and i + 1 < len(sys.argv):
        OUT = Path(sys.argv[i + 1])

DISTRACTOR_TARGET = {"existence": 40, "noop": 40, "attribution": 20}  # behavioral
COMPUTABLE_TARGET = 20
NORMAL_PER_DIM = 16
TIER_MIX = {"pro": {"hard": 0.6, "mid": 0.3, "easy": 0.1},
            "balanced": {"hard": 0.34, "mid": 0.33, "easy": 0.33}}["pro" if PRO else "balanced"]
rng = random.Random(730)


def tier(sd):
    t = sd.get("difficulty_tier")
    if t:
        return t
    mx = sd.get("max_observed_pct")
    if mx is not None:
        return "hard" if mx >= 2.0 else ("mid" if mx >= 1.0 else "easy")
    tp, dp = sd.get("true_deviation_pct"), sd.get("decoy_deviation_pct") or sd.get("poisoned_prior_deviation_pct")
    if tp and dp is not None:
        r = dp / tp if tp else 0
        return "hard" if r >= 0.75 else ("mid" if r >= 0.5 else "easy")
    return "untiered"


def dim(o):
    for t in o.get("tags", []) or []:
        if isinstance(t, str) and t.startswith("difficulty:"):
            return t.split(":", 1)[1]
    return "?"


def pick_tiered(pool, n):
    """按 TIER_MIX 从 pool 抽 n 条;某 tier 不够就从其它 tier 补。"""
    buckets = collections.defaultdict(list)
    for o in pool:
        buckets[tier(o["eval"]["source_data"])].append(o)
    for b in buckets.values():
        rng.shuffle(b)
    out = []
    for t, frac in TIER_MIX.items():
        want = round(n * frac)
        out += buckets[t][:want]; buckets[t] = buckets[t][want:]
    leftover = [o for b in buckets.values() for o in b]
    rng.shuffle(leftover)
    out += leftover[:max(0, n - len(out))]
    return out[:n]


def main():
    items = [json.loads(l) for l in open(POOL)]
    by_kind = collections.defaultdict(list); normal = collections.defaultdict(list); comp = []
    for o in items:
        sd = o.get("eval", {}).get("source_data", {}) or {}
        bk = sd.get("behavior_kind")
        if bk:
            by_kind[bk].append(o)
        elif dim(o) == "Adversarial":  # computable-distractor
            comp.append(o)
        else:
            normal[dim(o)].append(o)

    sample = []
    for k, n in DISTRACTOR_TARGET.items():
        got = pick_tiered(by_kind.get(k, []), n)
        if len(got) < n:
            print(f"  [shortfall] {k}: {len(got)}/{n}(池里只有 {len(by_kind.get(k,[]))})")
        sample += got
    cg = comp[:COMPUTABLE_TARGET]; rng.shuffle(comp)
    if len(cg) < COMPUTABLE_TARGET:
        print(f"  [shortfall] computable: {len(cg)}/{COMPUTABLE_TARGET}")
    sample += cg
    for dname, lst in normal.items():
        rng.shuffle(lst); sample += lst[:NORMAL_PER_DIM]

    Path(OUT).write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in sample) + "\n")

    # report
    kt = collections.Counter((sd.get("behavior_kind") or "computable" if dim(o) == "Adversarial" else "normal",
                              tier(sd) if (sd := o["eval"]["source_data"]).get("behavior_kind") else "-")
                             for o in sample)
    print(f"\nwrote {OUT}  ({len(sample)} 题, mode={'pro' if PRO else 'balanced'})")
    print("组成 (类型, tier):")
    for k, v in sorted(kt.items()):
        print(f"  {str(k):28s} {v}")


if __name__ == "__main__":
    main()
