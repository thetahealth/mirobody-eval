"""
Strip "answer-key" fields from user JSON files so LLM tools (which read JSON)
see the same information as the backend under test (which reads user.duckdb).

DuckDB exam_indicators schema is: user_id, time, indicator, exam_type,
exam_location, value, unit. Anything richer in JSON is a spoiler.

Stripped fields (per-indicator):
  - reference_range
  - status
  - clinical_significance

Stripped fields (per-exam):
  - abnormal_findings
  - overall_assessment
  - recommendations
  - generation_summary

Operates on:
  - benchmark/data/eslbench/.data/userXXX_AT_demo/exam_data.json (the source
    LLM agents actually read in production runs)
  - Any additional staging copies of the same user directories, if present

Idempotent — running twice is safe.

Usage:
  uv run python -m benchmark.strip_user_data_spoilers
  uv run python -m benchmark.strip_user_data_spoilers --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ESL_DATA = ROOT / "benchmark" / "data" / "eslbench" / ".data"

PER_INDICATOR_SPOILERS = ("reference_range", "status", "clinical_significance")
PER_EXAM_SPOILERS = (
    "abnormal_findings",
    "overall_assessment",
    "recommendations",
    "generation_summary",
)


def strip_exam_data(path: Path, dry_run: bool = False) -> tuple[int, int]:
    """Returns (per_indicator_stripped, per_exam_stripped)."""
    if not path.exists():
        return 0, 0
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return 0, 0

    ind_stripped = 0
    exam_stripped = 0
    for exam in data:
        if not isinstance(exam, dict):
            continue
        # Per-exam strip
        for f in PER_EXAM_SPOILERS:
            if f in exam:
                del exam[f]
                exam_stripped += 1

        indicators = exam.get("indicators", {})
        if isinstance(indicators, dict):
            for ind_key, ind in indicators.items():
                if not isinstance(ind, dict):
                    continue
                for f in PER_INDICATOR_SPOILERS:
                    if f in ind:
                        del ind[f]
                        ind_stripped += 1

    if not dry_run:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return ind_stripped, exam_stripped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-dir", default=str(ESL_DATA), help="Root containing userXXX_AT_demo/ subdirs")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"!! data dir not found: {data_dir}")
        return

    user_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir() and d.name.endswith("_AT_demo"))
    print(f"Scanning {len(user_dirs)} user dirs in {data_dir}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'LIVE'}\n")

    total_ind = 0
    total_exam = 0
    for ud in user_dirs:
        exam_path = ud / "exam_data.json"
        ind_n, exam_n = strip_exam_data(exam_path, dry_run=args.dry_run)
        total_ind += ind_n
        total_exam += exam_n
        print(f"  {ud.name}: stripped {ind_n} per-indicator + {exam_n} per-exam fields")

    print(f"\nTotal: {total_ind} per-indicator spoilers, {total_exam} per-exam spoilers stripped")


if __name__ == "__main__":
    main()
