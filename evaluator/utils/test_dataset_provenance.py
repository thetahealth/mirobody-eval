"""A score has to be able to say which release it came from.

Two copies of a question bank can exist: the one `prepare_data` fetched from the
dataset host, and the one vendored in this repo. Before this, only the vendored
one was ever read — so the dataset the leaderboard actually quotes could not be
run at all, and a number produced here had no version relationship to any
published release. That is what an outside reader hit when asking which
immutable files a leaderboard row corresponds to.

Three things are pinned:

  1. a fetched bank wins over a vendored one, and reports the batch and checksum
  2. EVERY path-resolution site goes through the one resolver — the bug while
     writing this was exactly a missed site, where the runner kept its own
     `bench_dir / f"{dataset}.jsonl"` and so never saw a fetched bank
  3. the writer and the reader agree on the directory name

    pytest evaluator/utils/test_dataset_provenance.py -v
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from evaluator.utils import paths
from evaluator.utils.benchmark_reader import resolve_dataset

_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def fetched(tmp_path, monkeypatch):
    """A fetched bank for `probe/ds`, plus the local manifest that anchors it."""
    monkeypatch.setenv("HOLYEVAL_USER_DATA_DIR", str(tmp_path))
    bank = tmp_path / "probe" / paths.BANKS_DIRNAME / "202607" / "ds.jsonl"
    bank.parent.mkdir(parents=True)
    bank.write_text('{"id": "x", "eval": {"evaluator": "kg_qa"}}\n', encoding="utf-8")
    (tmp_path / "probe" / ".manifest.json").write_text(
        json.dumps({"version": 15, "batches": {"202607": {"checksum": "sha256:deadbeef"}}}),
        encoding="utf-8",
    )
    return bank


# ---------------------------------------------------------------------------
# 1. The fetched copy wins, and carries the anchor
# ---------------------------------------------------------------------------


def test_a_fetched_bank_is_preferred_and_reports_its_release(fetched):
    path, prov = resolve_dataset("probe", "ds")

    assert path == fetched
    assert prov["source"] == "fetched"
    assert prov["batch"] == "202607"
    assert prov["manifest_version"] == 15
    assert prov["batch_checksum"] == "sha256:deadbeef", (
        "without the checksum the provenance cannot pin a release, which is the "
        "whole point of recording it"
    )


def test_without_a_fetched_bank_the_answer_says_so(tmp_path, monkeypatch):
    """`vendored` is not a failure — it is the offline path CI depends on. It
    just must not be mistaken for a release."""
    monkeypatch.setenv("HOLYEVAL_USER_DATA_DIR", str(tmp_path))

    _, prov = resolve_dataset("eslbench", "sample200-20260430")

    assert prov == {"source": "vendored"}


def test_a_partial_local_manifest_does_not_lose_the_batch(tmp_path, monkeypatch):
    """A manifest that does not mention this batch still leaves the batch known
    from the path. Dropping the whole provenance because one field is missing
    would throw away the part that was never in doubt."""
    monkeypatch.setenv("HOLYEVAL_USER_DATA_DIR", str(tmp_path))
    bank = tmp_path / "probe" / paths.BANKS_DIRNAME / "202608" / "ds.jsonl"
    bank.parent.mkdir(parents=True)
    bank.write_text("{}\n", encoding="utf-8")
    (tmp_path / "probe" / ".manifest.json").write_text('{"version": 15, "batches": {}}', encoding="utf-8")

    _, prov = resolve_dataset("probe", "ds")

    assert prov["batch"] == "202608"
    assert prov["batch_checksum"] is None


# ---------------------------------------------------------------------------
# 2. No resolution site may bypass the resolver
# ---------------------------------------------------------------------------

_SOURCE_DIRS = ("evaluator", "benchmark", "web")

# `<something> / f"{dataset}.jsonl"` — the shape of a hand-rolled resolution.
_HANDROLLED = re.compile(r'/\s*f?"\{dataset\}\.jsonl"')


def test_no_module_resolves_a_dataset_path_on_its_own():
    """The bug this catches was made while writing this feature: three sites were
    converted and a fourth — the one the CLI runner actually calls — was not, so
    a fetched dataset was reported as nonexistent, with a list of alternatives
    that omitted it."""
    # The two modules that implement resolution are where the path is SUPPOSED to
    # be built: `paths.fetched_bank` finds the fetched copy, `resolve_dataset`
    # falls back to the vendored one. Everything else must call through them.
    implementors = {"paths.py", "benchmark_reader.py"}

    offenders = []
    for d in _SOURCE_DIRS:
        for py in (_ROOT / d).rglob("*.py"):
            if py.name.startswith("test_") or ".venv" in py.parts or py.name in implementors:
                continue
            for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                if _HANDROLLED.search(line):
                    offenders.append(f"{py.relative_to(_ROOT)}:{i}")

    assert not offenders, (
        "these build a dataset path directly and so cannot see a fetched bank; "
        f"call resolve_dataset() instead: {offenders}"
    )


# ---------------------------------------------------------------------------
# 3. Writer and reader agree
# ---------------------------------------------------------------------------


def test_the_fetch_writes_where_the_read_looks():
    """`prepare_data` puts banks somewhere and the reader looks somewhere. Spelled
    twice, they drift; the constant is shared so that cannot happen silently."""
    prepare = (_ROOT / "generator/eslbench/prepare_data.py").read_text(encoding="utf-8")

    assert "paths.BANKS_DIRNAME" in prepare, (
        "prepare_data must take the banks directory from paths, not spell it again"
    )
    assert not re.search(r'BANKS_DIRNAME\s*=\s*"', prepare), (
        "a literal here is the drift this shared constant exists to prevent"
    )


def test_a_fetched_dataset_appears_in_the_listing(tmp_path, monkeypatch):
    """Runnable and findable are two different things, and the first does not
    imply the second.

    The listing globbed the vendored directory only, so a fetched release could
    be run from the CLI by name while being absent from the Web UI's dataset
    picker — which is where someone would go looking for it. Reported from the
    picker, not from a test.
    """
    from evaluator.utils.benchmark_reader import list_benchmarks

    monkeypatch.setenv("HOLYEVAL_USER_DATA_DIR", str(tmp_path))
    bank = tmp_path / "eslbench" / paths.BANKS_DIRNAME / "202607" / "probe-fetched-only.jsonl"
    bank.parent.mkdir(parents=True)
    bank.write_text('{"id": "a", "eval": {"evaluator": "kg_qa"}}\n', encoding="utf-8")

    listed = {d.name for b in list_benchmarks() if b.name == "eslbench" for d in b.datasets}

    assert "probe-fetched-only" in listed, (
        "a fetched dataset must show up in the picker, or nobody can select it"
    )


def test_a_closed_book_batch_is_listed_with_the_reason(tmp_path, monkeypatch):
    """The newest batch ships questions without answers, on purpose.

    Offering it in the picker as if it were runnable makes every selection of it
    fail at load; hiding it makes the dataset the leaderboard quotes look like it
    does not exist. It is listed, and the reason distinguishes "wait for the next
    release" from "this file is broken" — taken from the manifest rather than
    inferred from the missing field.
    """
    from evaluator.utils.benchmark_reader import list_benchmarks

    monkeypatch.setenv("HOLYEVAL_USER_DATA_DIR", str(tmp_path))
    banks = tmp_path / "eslbench" / paths.BANKS_DIRNAME
    (banks / "202607").mkdir(parents=True)
    (banks / "202608").mkdir(parents=True)
    # released: carries an evaluator
    (banks / "202607" / "probe-open.jsonl").write_text(
        '{"id": "a", "eval": {"evaluator": "kg_qa"}}\n', encoding="utf-8"
    )
    # newest: questions only
    (banks / "202608" / "probe-closed.jsonl").write_text(
        '{"id": "b", "eval": {"answer_type": "text"}}\n', encoding="utf-8"
    )
    (tmp_path / "eslbench" / ".manifest.json").write_text(
        json.dumps(
            {
                "version": 15,
                "batches": {
                    "202607": {"released_answers": ["x.jsonl"]},
                    "202608": {},
                },
            }
        ),
        encoding="utf-8",
    )

    listed = {d.name: d for b in list_benchmarks() if b.name == "eslbench" for d in b.datasets}

    assert "probe-closed" in listed, "the closed batch must still be visible"
    assert "202608" in listed["probe-closed"].unavailable_reason
    assert "not released" in listed["probe-closed"].unavailable_reason
    assert listed["probe-open"].unavailable_reason == "", (
        "a released dataset must not be marked unavailable"
    )


def test_the_report_schema_can_carry_provenance():
    """A resolver nobody records is a resolver that changes nothing: the anchor
    has to reach the artifact a reader is handed."""
    from evaluator.core.bench_schema import BenchReport

    assert "dataset_provenance" in BenchReport.model_fields
