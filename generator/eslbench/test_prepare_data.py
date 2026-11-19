"""Preparing ESLBench data fetches what a dataset needs, not every published batch.

A first run of `sample50-20260331` needs 27 users (~0.5 GB); fetching every batch
was 335 users (~5 GB). These tests pin which users are fetched, when a fetched
user counts as current, and that a user no batch publishes is refused up front.

Pure logic: `snapshot_download` is replaced by a fake hub on disk, no network.

    pytest generator/eslbench/test_prepare_data.py -v
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from evaluator.utils import paths
from generator.eslbench import prepare_data as pd


def _user_files(root, batch, user):
    d = root / "data" / batch / user
    d.mkdir(parents=True)
    for name in ("profile.json", "timeline.json", "exam_data.json", "kg_evaluation_queries_x.json"):
        (d / name).write_text("{}", encoding="utf-8")


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A fake dataset host: two batches on disk, and a record of every download asked for."""
    root = tmp_path / "hub"
    for user in ("u1_AT_demo", "u2_AT_demo", "u3_AT_demo"):
        _user_files(root, "A", user)
    _user_files(root, "B", "v1_AT_demo")
    (root / "data" / "A" / "bankA.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "data" / "B" / "bankB.jsonl").write_text("{}\n", encoding="utf-8")

    calls: list[dict] = []

    def snapshot_download(repo_id, repo_type, allow_patterns, ignore_patterns=None):
        calls.append({"allow": allow_patterns, "ignore": ignore_patterns or []})
        return str(root)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    return calls


MANIFEST = {
    "version": 2,
    "batches": {
        "A": {"checksum": "a1", "users": ["u1_AT_demo", "u2_AT_demo", "u3_AT_demo"]},
        "B": {"checksum": "b1", "users": ["v1_AT_demo"]},
    },
}


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """`.data/` for eslbench, where both the writer and `missing_users` look."""
    monkeypatch.setenv("HOLYEVAL_USER_DATA_DIR", str(tmp_path / "userdata"))
    d = tmp_path / "userdata" / "eslbench"
    d.mkdir(parents=True)
    monkeypatch.setattr(pd, "DATA_DIR", d)
    return d


def _dataset(data_dir, name: str, emails: list[str]) -> None:
    """A fetched bank whose questions point at *emails*."""
    bank = data_dir / paths.BANKS_DIRNAME / "A" / f"{name}.jsonl"
    bank.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"id": f"q{i}", "user": {"target_overrides": {"llm_api": {"tool_context": {"user_email": e}}}}}
        for i, e in enumerate(emails)
    ]
    bank.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _ready(data_dir, user: str) -> None:
    (data_dir / user).mkdir(parents=True, exist_ok=True)
    (data_dir / user / "profile.json").write_text("{}", encoding="utf-8")
    (data_dir / user / "user.duckdb").write_text("", encoding="utf-8")


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def test_only_the_named_users_are_fetched(hub, data_dir):
    fetched = pd.fetch_users(data_dir, MANIFEST, {"u1_AT_demo", "u3_AT_demo"})

    assert fetched == {"u1_AT_demo", "u3_AT_demo"}
    assert hub == [
        {"allow": ["data/A/u1_AT_demo/*.json", "data/A/u3_AT_demo/*.json"], "ignore": ["*/kg_evaluation_queries*"]}
    ]
    assert not (data_dir / "u2_AT_demo").exists() and not (data_dir / "v1_AT_demo").exists()
    # the answers stay out of .data/, even when the hub's cache still holds them
    assert sorted(p.name for p in (data_dir / "u1_AT_demo").iterdir()) == ["exam_data.json", "profile.json", "timeline.json"]


def test_a_user_already_here_is_not_fetched_again_unless_forced(hub, data_dir):
    _ready(data_dir, "u1_AT_demo")

    assert pd.fetch_users(data_dir, MANIFEST, {"u1_AT_demo", "u3_AT_demo"}) == {"u3_AT_demo"}
    assert pd.fetch_users(data_dir, MANIFEST, {"u1_AT_demo", "u3_AT_demo"}, force=True) == {"u1_AT_demo", "u3_AT_demo"}


def test_a_changed_batch_drops_its_users_and_refetches_its_banks(hub, data_dir):
    pd._save_local_manifest(MANIFEST, data_dir)
    _ready(data_dir, "u1_AT_demo")
    _ready(data_dir, "v1_AT_demo")
    remote = json.loads(json.dumps(MANIFEST))
    remote["version"] = 3
    remote["batches"]["A"]["checksum"] = "a2"

    pd.sync_manifest(data_dir, remote)

    assert not (data_dir / "u1_AT_demo").exists(), "u1_AT_demo was fetched under A's old checksum"
    assert (data_dir / "v1_AT_demo" / "user.duckdb").exists(), "B did not change"
    assert hub == [{"allow": ["data/A/*.jsonl"], "ignore": []}]
    assert (data_dir / paths.BANKS_DIRNAME / "A" / "bankA.jsonl").is_file()
    assert pd._load_local_manifest(data_dir) == remote


def test_a_first_sync_fetches_every_bank_and_no_user(hub, data_dir):
    pd.sync_manifest(data_dir, MANIFEST)

    assert [c["allow"] for c in hub] == [["data/A/*.jsonl"], ["data/B/*.jsonl"]]
    assert pd.discover_user_dirs(data_dir) == []


# ---------------------------------------------------------------------------
# What a dataset still needs
# ---------------------------------------------------------------------------


def test_missing_users_lists_only_what_is_not_ready(data_dir):
    pd._save_local_manifest(MANIFEST, data_dir)
    _dataset(data_dir, "ds", ["u1@demo", "u3@demo", "u1@demo"])
    _ready(data_dir, "u1_AT_demo")
    (data_dir / "u3_AT_demo").mkdir()
    (data_dir / "u3_AT_demo" / "profile.json").write_text("{}", encoding="utf-8")  # fetched, no DuckDB yet

    assert pd.missing_users("eslbench", "ds") == ["u3_AT_demo"]

    _ready(data_dir, "u3_AT_demo")
    assert pd.missing_users("eslbench", "ds") == []


def test_a_user_no_batch_publishes_is_refused(data_dir):
    pd._save_local_manifest(MANIFEST, data_dir)
    _dataset(data_dir, "ds", ["u1@demo", "ghost@demo"])

    with pytest.raises(ValueError, match="ghost_AT_demo"):
        pd.missing_users("eslbench", "ds")


def test_before_any_sync_every_user_is_missing(data_dir):
    _dataset(data_dir, "ds", ["u1@demo"])

    assert pd.missing_users("eslbench", "ds") == ["u1_AT_demo"]


# ---------------------------------------------------------------------------
# DuckDB and the user directory listing
# ---------------------------------------------------------------------------


def test_the_banks_directory_is_not_a_user(data_dir):
    _ready(data_dir, "u1_AT_demo")
    (data_dir / paths.BANKS_DIRNAME / "A").mkdir(parents=True)

    assert [d.name for d in pd.discover_user_dirs(data_dir)] == ["u1_AT_demo"]


def test_duckdb_is_built_for_new_users_and_rebuilt_for_refetched_ones(data_dir, monkeypatch):
    built: list[str] = []
    monkeypatch.setattr(pd, "create_user_duckdb", lambda d, force: built.append(d.name))
    _ready(data_dir, "u1_AT_demo")  # has a DuckDB
    _ready(data_dir, "u2_AT_demo")  # has a DuckDB, but its data was just fetched again
    (data_dir / "u3_AT_demo").mkdir()
    (data_dir / "u3_AT_demo" / "profile.json").write_text("{}", encoding="utf-8")

    pd.build_all_duckdb(data_dir, ["u1_AT_demo", "u2_AT_demo", "u3_AT_demo", "absent"], rebuild={"u2_AT_demo"})

    assert built == ["u2_AT_demo", "u3_AT_demo"]
