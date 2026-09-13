"""The two READMEs must stay usable and stay in step.

A translation drifts silently. The English README gains a flag, a table row, an
environment variable — and the Chinese one keeps describing the version before
it, with nothing failing. Same for links: rename a module and every `.md`
pointing at it becomes a 404 that only a reader discovers.

Three cheap checks, none of which need network or a model:

  1. every relative link in either README resolves to something on disk
  2. both READMEs document the same CLI flags, the same environment variables,
     and the same registered plugins as the code actually has
  3. each points at the other, so a reader can switch language

    pytest evaluator/test_readme.py -v
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_READMES = {"README.md": "English", "README.zh-CN.md": "简体中文"}

pytestmark = pytest.mark.skipif(
    not (_ROOT / "README.md").is_file(),
    reason="repo root not present",
)


def _text(name: str) -> str:
    path = _ROOT / name
    if not path.is_file():
        pytest.fail(f"{name} is missing — the language switcher promises it")
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Relative links resolve
# ---------------------------------------------------------------------------

# `[text](target)` and `src="target"`, skipping anything absolute or anchored.
_MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_HTML_SRC = re.compile(r'(?:src|href)="([^"]+)"')


def _relative_targets(text: str) -> set[str]:
    out = set()
    for pattern in (_MD_LINK, _HTML_SRC):
        for target in pattern.findall(text):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            out.add(target.split("#", 1)[0])
    return {t for t in out if t}


@pytest.mark.parametrize("name", sorted(_READMES))
def test_every_relative_link_resolves(name):
    """CONTRIBUTING says to grep the `.md` files after a rename. This is that
    rule, enforced rather than remembered."""
    missing = sorted(t for t in _relative_targets(_text(name)) if not (_ROOT / t).exists())

    assert not missing, f"{name} points at paths that do not exist: {missing}"


# ---------------------------------------------------------------------------
# 2. Both languages describe the code, not an earlier version of it
# ---------------------------------------------------------------------------


def _runner_flags() -> set[str]:
    source = (_ROOT / "benchmark" / "basic_runner.py").read_text(encoding="utf-8")
    return set(re.findall(r'"(--[a-z][a-z-]+)"', source))


def _documented_flags(text: str) -> set[str]:
    """Flags that appear on an EXPLAINED line, not merely somewhere in the file.

    Substring presence is not enough: every flag used in an example command is
    also "present", so deleting a flag from the reference list while an example
    still mentions it would pass. The reference form is a flag followed by its
    explanation, so that is what gets counted.
    """
    out = set()
    for line in text.splitlines():
        if "#" not in line:
            continue
        before = line.split("#", 1)[0]
        out |= set(re.findall(r"(--[a-z][a-z-]+)", before))
    return out


@pytest.mark.parametrize("name", sorted(_READMES))
def test_documents_every_runner_flag(name):
    """A flag nobody documents is a flag nobody uses."""
    undocumented = sorted(_runner_flags() - _documented_flags(_text(name)))

    assert not undocumented, f"{name} does not document: {undocumented}"


@pytest.mark.parametrize("name", sorted(_READMES))
def test_documents_every_registered_target(name):
    """The plugin table is the only place a reader learns what can be evaluated.

    `mirobody` — the target this repo exists for — was absent from it until
    this test was written, while appearing all over the command examples. So
    the check is scoped to the TABLE ROW: a target named in an example is not
    a target a reader can discover.
    """
    import evaluator.plugin.target_agent  # noqa: F401
    from evaluator.core.interfaces.abstract_target_agent import AbstractTargetAgent

    # The row itself, not the prose row in "Why …" that also names the three
    # agent types — matching on the cell start is what separates them.
    rows = [ln for ln in _text(name).splitlines() if ln.startswith("| **TargetAgent**")]
    assert rows, f"{name} has no TargetAgent row in its plugin table"
    row = rows[0]

    missing = sorted(t for t in AbstractTargetAgent.get_all() if f"`{t}`" not in row)

    assert not missing, f"{name}'s plugin table omits registered targets: {missing}"


@pytest.mark.parametrize("name", sorted(_READMES))
def test_documents_every_environment_variable_the_code_reads(name):
    """`MIROBODY_CONFIG` decides which deployment gets seeded and scored. An
    undocumented one of those is worse than a missing flag: the failure it
    produces names a database, not a setting."""
    read_by_code = set()
    for path in (_ROOT / "evaluator").rglob("*.py"):
        if path.name.startswith("test_"):
            continue
        source = path.read_text(encoding="utf-8")
        read_by_code |= set(re.findall(r'environ(?:\.get)?[\(\[]"(MIROBODY_[A-Z_]+)"', source))

    # Scoped to table rows for the same reason as the plugin check: a variable
    # mentioned in prose is not a variable a reader can look up.
    rows = "\n".join(ln for ln in _text(name).splitlines() if ln.startswith("|"))
    missing = sorted(v for v in read_by_code if f"`{v}`" not in rows)

    assert not missing, f"{name}'s configuration table does not document: {missing}"


def test_env_example_covers_what_the_readme_documents():
    """`.env.example` is the first file a reader copies.

    A variable documented in the README but absent from the template is one the
    reader has to invent from prose — and the ones most worth having in front of
    you are exactly the ones you would not guess, like the path to the
    deployment you are pointing at.
    """
    documented = set()
    rows = [ln for ln in _text("README.md").splitlines() if ln.startswith("|")]
    for row in rows:
        documented |= set(re.findall(r"`([A-Z][A-Z0-9_]{3,})`", row))

    # An ASSIGNMENT line, commented or not — not a mention in the surrounding
    # prose. Every variable this file explains is also "present" in it, so a
    # substring check passes after the line itself is deleted. (That mistake was
    # made three times while writing these checks, in three different ways.)
    template = (_ROOT / ".env.example").read_text(encoding="utf-8")
    assigned = set(re.findall(r"(?m)^\s*#?\s*([A-Z][A-Z0-9_]{3,})=", template))
    missing = sorted(documented - assigned)

    assert not missing, f".env.example has no line for: {missing}"


# ---------------------------------------------------------------------------
# 3. A reader can get from one language to the other
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_READMES))
def test_links_to_the_other_language(name):
    others = [other for other in _READMES if other != name]
    text = _text(name)

    for other in others:
        assert other in text, f"{name} has no link to {other}"


@pytest.mark.parametrize("name", sorted(_READMES))
def test_both_cover_the_same_sections(name):
    """Not the same wording — the same count of top-level sections. A
    translation that quietly stops one section short is the failure mode."""
    counts = {n: len(re.findall(r"(?m)^## ", _text(n))) for n in _READMES}

    assert len(set(counts.values())) == 1, f"section counts differ: {counts}"


@pytest.mark.parametrize("name", ["README.md", "README.zh-CN.md", "CLAUDE.md"])
def test_first_mirobody_run_only_uses_seeded_users(name, monkeypatch):
    """Follow the documented seed/run selection using real case conversion.

    Limit selects rows, not users. A single-user setup must not immediately run
    cases whose identities have not been seeded. Use the shipped question bank
    so this check does not depend on downloads in the developer's local cache.
    """
    import shlex

    import evaluator.plugin.eval_agent  # noqa: F401
    import evaluator.plugin.target_agent  # noqa: F401
    import evaluator.plugin.test_agent  # noqa: F401
    from evaluator.core.bench_schema import bench_item_to_test_case, find_target_spec
    from evaluator.utils import benchmark_reader

    commands = []
    for block in re.findall(r"```bash\n(.*?)```", _text(name), re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            tokens = shlex.split(line, comments=True)
            if tokens:
                commands.append(tokens)

    seed_at = next(i for i, args in enumerate(commands) if "generator.eslbench.seed_mirobody" in args)
    assert any(
        "sync" in args and "--extra" in args and "mirobody" in args
        for args in commands[:seed_at]
    ), f"{name}: install the mirobody extra before seeding"
    seed_args = commands[seed_at]
    users_at = seed_args.index("--users") + 1
    seeded = set()
    for token in seed_args[users_at:]:
        if token.startswith("-"):
            break
        seeded.add(token)

    run = next(
        args for args in commands[seed_at + 1:]
        if "benchmark.basic_runner" in args and "--target-type" in args
        and args[args.index("--target-type") + 1] == "mirobody"
    )
    module_at = run.index("benchmark.basic_runner")
    benchmark, dataset = run[module_at + 1:module_at + 3]
    monkeypatch.setattr(
        benchmark_reader, "resolve_dataset",
        lambda bench, bank: (_ROOT / "benchmark" / "data" / bench / f"{bank}.jsonl", {"source": "vendored"}),
    )
    bench = benchmark_reader.load_benchmark(benchmark, dataset)
    ids = run[run.index("--ids") + 1] if "--ids" in run else None
    limit = int(run[run.index("--limit") + 1]) if "--limit" in run else None
    items = benchmark_reader.filter_bench_items(bench.items, ids=ids, limit=limit)
    assert items, f"{name}: example selects no cases"
    if ids:
        assert {item.id for item in items} == set(ids.split(",")), f"{name}: an example ID does not exist"
    spec = find_target_spec(bench.target, "mirobody")
    missing = {
        item.id: case.target.user_email
        for item in items
        if (case := bench_item_to_test_case(item, spec)).target.user_email not in seeded
    }
    assert not missing, f"{name}: examples select users not seeded by the preceding command: {missing}"


@pytest.mark.parametrize("doc", ["README.md", "README.zh-CN.md", ".env.example"])
def test_the_docs_list_every_provider_key(doc):
    """表里新增一个 provider，文档必须跟着写。

    本文件的环境变量门禁对这些是盲的:它的正则只找 `MIROBODY_*` 字面量，而这些
    key 是通过 `_OPENAI_COMPAT` 间接读的（`os.getenv(key_env)`），扫源码的正则
    找不到。事实来源是那张表，所以拿表比。
    """
    from pathlib import Path

    from evaluator.utils.llm import _OPENAI_COMPAT

    text = (Path(__file__).resolve().parents[1] / doc).read_text(encoding="utf-8")

    missing = sorted(key_env for key_env, _, _ in _OPENAI_COMPAT.values() if key_env not in text)
    assert not missing, f"{doc} 没有提到这些 provider 的 key: {missing}"

    prefixes = sorted(p for p in _OPENAI_COMPAT if p not in text)
    assert not prefixes, f"{doc} 没有提到这些 provider 前缀: {prefixes}"


@pytest.mark.parametrize("doc", ["README.md", "README.zh-CN.md", "CLAUDE.md"])
def test_documented_commands_name_a_target_when_there_are_several(doc):
    """多个可测系统时，示例命令必须写明 `--target-type`。

    `find_target_spec` 以前在没指定时无条件返回第一个,而 eslbench 的 metadata
    把 `mirobody` 排在首位 —— 于是漏写这个参数的示例命令会静默选中 mirobody、
    跑到连数据库时才失败,报错是「数据库 <你的用户名> 不存在」,跟真实原因毫无
    关系。现在 `find_target_spec` 会拒绝猜,所以这些命令是真的跑不了,不是只是
    有歧义。三份文档一起查:上一轮就是只改了两个 README、漏了 CLAUDE.md。
    """
    import json
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    counts = {
        d.name: len(json.loads((d / "metadata.json").read_text(encoding="utf-8")).get("target") or [])
        for d in (root / "benchmark" / "data").iterdir()
        if (d / "metadata.json").is_file()
    }

    # 续行也算同一条命令 —— 反斜杠续行的命令把参数分散在多行上。
    text = re.sub(r"\\\n\s*", " ", (root / doc).read_text(encoding="utf-8"))

    offenders = [
        line.strip()[:90]
        for line in text.splitlines()
        if (m := re.search(r"basic_runner (\w+) ", line))
        and counts.get(m.group(1), 0) > 1
        and "--target-type" not in line
    ]
    assert not offenders, f"{doc} 里这些命令没指定 --target-type，会直接跑不起来: {offenders}"
