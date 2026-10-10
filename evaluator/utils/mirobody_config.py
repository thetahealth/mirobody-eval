"""Point the mirobody engine at the deployment being evaluated.

Both halves of the mirobody path need this and need it identically: the seeder
writes through the deployment's own `execute_query` and `IndicatorSyncTask`, and
the target agent reads the deployment's `JWT_KEY` to sign a token. Neither works
until mirobody's own configuration is loaded, and this repo's runner loads only
its own.

It lived in the target agent alone, and the seeder called bare `Config.init()`.
That is the failure this module exists to prevent: `MIROBODY_CONFIG` was
documented as the way to name a deployment, the README's flow seeds first and
scores second, and step one ignored the variable — so the documented command
died against a database named after the OS user, while the eval step it
precedes worked. One implementation, imported twice, cannot drift that way
again.

    from evaluator.utils.mirobody_config import ensure_mirobody_config
    await ensure_mirobody_config()
"""

from __future__ import annotations

import os


async def ensure_mirobody_config() -> None:
    """Load the target deployment's mirobody configuration, once.

    `MIROBODY_CONFIG` gives an explicit path. Without it, `Config.init()`
    searches the CURRENT WORKING DIRECTORY for `config.{ENV}.yaml` — and the
    runner's working directory is this repo, not the deployment. It then falls
    back to built-in defaults, which take the PG database name from the OS user,
    so the error names a database you never configured — one called after your
    OS user — and points nowhere near the missing setting.

    Only initializes when nothing is loaded yet. A process that already has a
    configuration — the engine's own, if this code ever runs inside it — must not
    be reset.
    """
    from mirobody.utils.config import global_config

    if global_config() is not None:
        return

    from mirobody.utils import Config

    # The literal, not a constant: `test_readme.py` finds the variables the code
    # reads by matching `environ.get("MIROBODY_…")` in the source. Behind a
    # module constant the gate stops seeing this one, and then it passes whether
    # or not the README still documents it — which is the drift the gate exists
    # to catch.
    explicit = os.environ.get("MIROBODY_CONFIG", "").strip()
    if explicit:
        if not os.path.isfile(explicit):
            raise RuntimeError(f"MIROBODY_CONFIG 指向的文件不存在: {explicit!r}")
        await Config.init(explicit)
    else:
        await Config.init()

    if global_config() is None:  # pragma: no cover - 配置缺失时的兜底提示
        raise RuntimeError(
            "mirobody 配置加载失败。用 MIROBODY_CONFIG 指定那个部署的 "
            "config.{ENV}.yaml 绝对路径，或把工作目录切到它所在的目录。"
        )
