"""The Web UI fetches a dataset's data when a run needs it, not every batch at startup.

Pure logic: the prepare module and the subprocess are faked, no network.

    pytest web/app/services/test_prepare_manager.py -v
"""

from __future__ import annotations

import sys
import types

import pytest

from web.app.services.prepare_manager import PrepareManager


@pytest.fixture
def manager(monkeypatch):
    """Two benchmarks sharing one prepare module; runs are recorded, not executed."""
    missing: dict[str, list[str]] = {}

    def missing_users(benchmark, dataset):
        if dataset == "broken":
            raise ValueError("users outside every batch")
        return missing.get(dataset, [])

    monkeypatch.setitem(sys.modules, "fake_prepare", types.SimpleNamespace(missing_users=missing_users))

    m = PrepareManager()
    m._modules = {"bench": "fake_prepare", "bench_variant": "fake_prepare"}
    started: list[list[str]] = []

    async def run_script(entry):
        started.append(entry.args)

    monkeypatch.setattr(m, "_run_script", run_script)
    m.missing = missing
    m.started = started
    return m


async def test_a_dataset_that_is_ready_starts_nothing(manager):
    assert manager.prepare_dataset("bench", "ds") is None
    assert manager.get_status("bench") is None


async def test_a_dataset_with_missing_users_fetches_just_that_dataset(manager):
    manager.missing["ds"] = ["u1", "u2"]

    waiting = manager.prepare_dataset("bench_variant", "ds")

    assert "2 users" in waiting
    entry = manager.get_status("bench_variant")
    assert entry.args == ["--benchmark", "bench_variant", "--dataset", "ds"]
    # the two benchmarks share a data directory, so they share the run
    assert manager.is_preparing("bench") and manager.is_preparing("bench_variant")


async def test_a_dataset_that_cannot_be_prepared_raises(manager):
    with pytest.raises(ValueError):
        manager.prepare_dataset("bench", "broken")


async def test_a_benchmark_without_a_prepare_module_needs_nothing(manager):
    assert manager.prepare_dataset("medcalc", "sample") is None
