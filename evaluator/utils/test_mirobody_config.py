"""`MIROBODY_CONFIG` must reach BOTH halves of the mirobody path.

The bug this pins: the target agent honoured the variable and the seeder did not.
The README's flow seeds first and scores second, so `MIROBODY_CONFIG=… python -m
generator.eslbench.seed_mirobody` failed against a database nobody configured:
the engine had searched this repo for a config, found none, and fallen back to
defaults that name the PG database after the OS user. The eval step it precedes
worked, so nothing about the failure pointed at the asymmetry.

Two things are checked, and the second is the one that actually matters:

  1. the shared helper passes an explicit path through, and rejects a bad one
  2. NEITHER caller reaches `Config.init` by itself — both go through the helper

Check 2 is a source-level assertion rather than a behavioural one on purpose.
Running the seeder needs a live deployment, so a behavioural test would be
skipped in CI, which is exactly where this must not regress. A bare
`Config.init()` in either file is the defect, and it is visible in the source.

    pytest evaluator/utils/test_mirobody_config.py -v
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from evaluator.utils.mirobody_config import ensure_mirobody_config

_ROOT = Path(__file__).resolve().parents[2]

_CALLERS = (
    "generator/eslbench/seed_mirobody.py",
    "evaluator/plugin/target_agent/mirobody_target_agent.py",
)


# ---------------------------------------------------------------------------
# 1. The helper itself
# ---------------------------------------------------------------------------


async def test_a_missing_path_is_named_rather_than_ignored(monkeypatch):
    """A typo in the path must not degrade into the CWD search.

    Falling through to `Config.init()` here is what produced the misleading
    database error: the reader gets told about a database they never configured
    instead of about the path they got wrong.

    Skipped where the engine is absent — the extra installs nothing below
    Python 3.12, which is what CI runs.
    """
    pytest.importorskip("mirobody", reason="`--extra mirobody` not installed")
    import mirobody.utils.config as engine_config

    # A process that already holds a configuration must not be reset, so the
    # helper returns early on one. Force the not-yet-configured branch.
    monkeypatch.setattr(engine_config, "global_config", lambda: None)
    monkeypatch.setenv("MIROBODY_CONFIG", "/nonexistent/config.localdb.yaml")

    with pytest.raises(RuntimeError) as exc:
        await ensure_mirobody_config()

    assert "MIROBODY_CONFIG" in str(exc.value)
    assert "/nonexistent/config.localdb.yaml" in str(exc.value)


# ---------------------------------------------------------------------------
# 2. Both callers actually use it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rel", _CALLERS)
def test_caller_does_not_initialize_the_engine_config_itself(rel):
    """`await Config.init(...)` outside the helper is the defect, by definition.

    Matched on the CALL form rather than the name: both files discuss
    `Config.init()` in prose while explaining this bug, so a check for the bare
    name would fail on the explanation instead of on the defect. `#` comments are
    dropped first; prose inside a docstring would still match if someone wrote
    the `await` form there, which is a trade this check accepts — the false
    positive is loud and immediate, the false negative is the bug shipping again.
    """
    source = (_ROOT / rel).read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    calls = re.findall(r"await\s+Config\.init\s*\(", code)

    assert not calls, f"{rel} initializes mirobody's config directly; use ensure_mirobody_config()"


@pytest.mark.parametrize("rel", _CALLERS)
def test_caller_routes_through_the_shared_helper(rel):
    """The other half of the check above: not calling `Config.init` is only
    correct if the helper is called instead. A file that does neither would pass
    the previous test while leaving the engine unconfigured."""
    source = (_ROOT / rel).read_text(encoding="utf-8")

    assert "ensure_mirobody_config" in source, f"{rel} never loads the deployment's config"


def test_the_env_var_is_read_as_a_literal():
    """`test_readme.py` finds documented-vs-read variables by matching
    `environ.get("MIROBODY_…")` in the source. Behind a module constant that gate
    silently stops covering this variable — it keeps passing, and would keep
    passing after the README row is deleted. That regression was introduced once
    already while extracting this helper."""
    source = (_ROOT / "evaluator/utils/mirobody_config.py").read_text(encoding="utf-8")

    assert re.search(r'environ\.get\(\s*"MIROBODY_CONFIG"', source), (
        "read MIROBODY_CONFIG as a literal so evaluator/test_readme.py can see it"
    )
