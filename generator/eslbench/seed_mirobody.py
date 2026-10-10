"""把 ESL-Bench 合成用户灌进一个自部署的 mirobody 实例。

补的是 mirobody ←→ mirobody-eval 之间缺的那一环。mirobody 的 README 说
「its generator also produces the synthetic (PHI-free) health data used in demos
and tests」，但两个仓库里都没有代码把这批数据送进 mirobody 的库——没有它，
`mirobody` 这个 target 连可问的用户都没有。

只写 mirobody 已有的表：

* ``health_app_user``  —— 用户本体。``mirobody.indicator.search`` 是拿邮箱去这张表
  换数字 id 的，没有这行，后面每个查询都返回空。
* ``th_series_data``   —— 指标读数。
* ``th_files``         —— events / profile / 体检报告文档，DeepAgent 会自动只读
  镜像到 ``/library/``，`read_file` / `grep` 就能读到。
* ``th_series_dim``    —— **不直接写**。交给 ``IndicatorSyncTask``（那张表的
  canonical writer），它能解析的 ``fhir_id`` 就解析，剩下的补 embedding。

最后一步不是可选的装饰。agent 发现指标走的是向量检索，而
``mirobody/indicator/fhir/adapter.py`` 里两条检索路径都硬要求 embedding 非空：
已映射的走 ``fhir_indicators.embedding_*``，未映射的走
``th_series_dim.embedding_*``。只 insert 读数不补向量，数据在库里但 agent 看不见，
只会回答"我没有你的健康数据"，而且日志里看不出原因。所以 :func:`seed_user`
灌完会回头校验覆盖率，有漏的直接报错，而不是留下一个哑掉的部署。

需要目标部署的 mirobody 可 import（``uv sync --extra mirobody``）：复用它的
``execute_query`` / ``Config`` / ``IndicatorSyncTask``，而不是在这里重写一遍
连接管理和 embedding 逻辑。

用法::

    python -m generator.eslbench.seed_mirobody --users user5086@demo
    python -m generator.eslbench.seed_mirobody --users user5086@demo --hold-out-exams 1
    python -m generator.eslbench.seed_mirobody --all
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .mapping import (
    SOURCE_TABLE,
    MappingStats,
    events_document,
    exam_documents,
    profile_document,
    series_rows,
)
from .prepare_data import DATA_DIR, dir_to_email, discover_user_dirs

logger = logging.getLogger(__name__)

# 每批 executemany 的行数。2000 让单次往返的参数量还在 psycopg 的舒适区内，
# 同时把 ~67k 行的延迟摊薄。
_BATCH = 2000

# ``IndicatorSyncTask.embed`` 每轮最多处理 10k 个 dim 行。单用户约 260 个，
# 但 --all 会超过上限，所以只要还在推进就继续扫。设上界是为了让一个永久失败的
# provider（key 不对）暴露成错误，而不是死循环。
_MAX_EMBED_SWEEPS = 20


def _require_mirobody() -> None:
    """mirobody 不可 import 时给出可操作的报错，而不是裸 ImportError。

    分两种情况说，因为它们的处置完全不同：Python 太旧的话 ``--extra mirobody``
    会**装成功但什么都没装**（mirobody 是 3.12+，extra 上带着版本标记），此时
    再叫人重跑一次安装只会让人原地转圈。
    """
    if sys.version_info < (3, 12):
        raise RuntimeError(
            f"seed 需要 mirobody，而它要求 Python >= 3.12（当前 "
            f"{sys.version_info.major}.{sys.version_info.minor}）。本项目自身支持 3.11，"
            "所以 `--extra mirobody` 在 3.11 上不装任何东西 —— 换用 3.12+ 的解释器：\n"
            "    uv sync --extra mirobody --python 3.12"
        )
    try:
        import mirobody  # noqa: F401
    except ImportError as e:  # pragma: no cover - 环境问题，不便单测
        raise RuntimeError(
            "seed 需要能 import mirobody（它复用 mirobody 的 execute_query / "
            "IndicatorSyncTask，不自己重写连接与 embedding 逻辑）。\n"
            "    uv sync --extra mirobody\n"
            "并确保配置指向你要灌的那个部署的 Postgres。"
        ) from e


@dataclass
class SeedResult:
    email: str
    user_id: str
    mapping: MappingStats
    documents: int = 0
    indicators: int = 0
    embedded: int = 0
    unembedded: list[str] = field(default_factory=list)
    held_out_exams: list[str] = field(default_factory=list)

    def summary(self) -> str:
        text = (
            f"{self.email} (user_id={self.user_id}): {self.mapping.summary()}; "
            f"{self.documents} 个文档; "
            f"{self.indicators} 个指标, {self.embedded} 个可被 agent 搜到"
        )
        if self.held_out_exams:
            text += f"; 留出体检 {', '.join(self.held_out_exams)}"
        return text


# ============================================================
# health_app_user
# ============================================================


async def ensure_user(email: str, *, name: str = "", tz: str = "UTC") -> str:
    """upsert 合成用户，返回它的数字 id（字符串形式）。

    ``health_app_user.is_del`` 是 NOT NULL 且无默认值，所以显式写入。重复 seed 会把
    之前软删掉的行救活，而不是撞唯一键。
    """
    from mirobody.utils import execute_query

    rows = await execute_query(
        """
        INSERT INTO health_app_user (is_del, email, name, tz, lang)
        VALUES (false, :email, :name, :tz, 'en')
        ON CONFLICT (email) DO UPDATE SET
            is_del    = false,
            name      = EXCLUDED.name,
            tz        = EXCLUDED.tz,
            update_at = CURRENT_TIMESTAMP
        RETURNING id
        """,
        {"email": email, "name": name or email.split("@")[0], "tz": tz},
    )
    if not rows:
        raise RuntimeError(f"upsert health_app_user 失败: {email}")
    return str(rows[0]["id"])


# ============================================================
# th_series_data
# ============================================================

_INSERT_SERIES = """
INSERT INTO th_series_data (
    user_id, indicator, value, start_time, end_time, source_table,
    source_table_id, comment, indicator_id, source, task_id,
    fhir_id, fhir_mapping_info, create_time, update_time, deleted
) VALUES (
    :user_id, :indicator, :value, :start_time, :end_time, :source_table,
    :source_table_id, encrypt_content(:comment), :indicator_id, :source, :task_id,
    :fhir_id, :fhir_mapping_info, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0
)
ON CONFLICT (user_id, indicator, start_time, end_time)
DO UPDATE SET
    value             = EXCLUDED.value,
    source            = EXCLUDED.source,
    source_table      = EXCLUDED.source_table,
    source_table_id   = EXCLUDED.source_table_id,
    comment           = EXCLUDED.comment,
    fhir_mapping_info = EXCLUDED.fhir_mapping_info,
    deleted           = 0,
    update_time       = CURRENT_TIMESTAMP
"""


async def insert_series(rows: list[dict]) -> int:
    """插一批 ``th_series_data``，返回提交的行数。"""
    from mirobody.utils import execute_query

    if not rows:
        return 0
    await execute_query(_INSERT_SERIES, rows, log_sql=False)
    return len(rows)


# ============================================================
# th_files
# ============================================================

_INSERT_FILE = """
INSERT INTO th_files (
    user_id, query_user_id, file_name, file_type, file_key,
    file_content, scene, created_source, created_source_id,
    original_text, text_length, content_hash,
    is_del, created_at, updated_at
) VALUES (
    :user_id, :user_id, encrypt_content(:file_name), 'text/markdown', :file_key,
    encrypt_content(:file_content), :scene, :created_source, :created_source_id,
    encrypt_content(:original_text), :text_length, :content_hash,
    false, now(), now()
)
ON CONFLICT (file_key) DO UPDATE SET
    file_name     = EXCLUDED.file_name,
    file_content  = EXCLUDED.file_content,
    original_text = EXCLUDED.original_text,
    text_length   = EXCLUDED.text_length,
    content_hash  = EXCLUDED.content_hash,
    is_del        = false,
    updated_at    = now()
"""


async def insert_document(user_id: str, user_dir: str, file_name: str, body: str, *, scene: str) -> None:
    """登记一个 markdown 文档。

    ``file_key`` 由用户目录 + 文件名派生，重复 seed 更新同一行而不是堆副本。
    ``original_text`` 必须非空 —— DeepAgent 的 ``/library/`` 镜像正是按这一列筛的，
    空的话文档对 agent 不存在。
    """
    from mirobody.utils import execute_query

    if not body.strip():
        raise ValueError(f"拒绝登记空文档: {file_name}")

    await execute_query(
        _INSERT_FILE,
        {
            "user_id": user_id,
            "file_name": file_name,
            "file_key": f"{SOURCE_TABLE}:{user_dir}:{file_name}",
            "file_content": json.dumps({"source": SOURCE_TABLE, "user_dir": user_dir}),
            "scene": scene,
            "created_source": SOURCE_TABLE,
            "created_source_id": user_dir,
            "original_text": body,
            "text_length": len(body),
            "content_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        },
        log_sql=False,
    )


# ============================================================
# 指标维表 + embedding
# ============================================================


def _embedding_columns() -> tuple[str, str]:
    """当前 provider 对应的 ``(th_series_dim 列, fhir_indicators 列)``。

    两张表列名规则不同，而且**都不能从 provider 名推**：``openrouter`` 这个 provider
    在两张表里分别落到 ``embedding_qwen3_8b``（按模型命名，不按 provider）。所以两边
    各走引擎自己的 helper。

    这里原先是 ``f"embedding_{provider}"`` 手工拼的 —— gemini 和 qwen 恰好对得上，
    于是错误一直没暴露；等 1.2.1 把 openrouter 设为默认 provider，它就会去查一个不
    存在的列。
    """
    from mirobody.indicator.fhir.common import (
        resolve_dim_embedding_column,
        resolve_fhir_embedding_column,
    )

    _, dim_col = resolve_dim_embedding_column()
    _, fhir_col = resolve_fhir_embedding_column()
    return dim_col, fhir_col


async def searchability(user_id: str) -> tuple[int, int, list[str]]:
    """这个用户有多少指标是 agent 真能搜到的。

    返回 ``(总数, 可搜到数, 搜不到的指标名)``，镜像 ``FhirAdapter.search`` 的两条分支：
    某指标只要**任一行**满足「已映射且 ``fhir_indicators`` 有向量」或
    「未映射且 ``th_series_dim`` 有向量」就算可达。按行而非按指标判断，是因为一个指标
    的行可以部分映射，而任一条分支单独成立就够 search 把它捞出来。

    只统计本 seeder 写的行 —— 灌到一个已有真实数据的账号上时，不该被那些数据的
    未 embedding 指标绊住，那不是 seed 该修的。
    """
    from mirobody.utils import execute_query

    dim_col, fhir_col = _embedding_columns()

    rows = (
        await execute_query(
            f"""
        SELECT
            sd.indicator,
            bool_or(sd.fhir_id IS NOT NULL AND fi.{fhir_col} IS NOT NULL)  AS via_fhir,
            bool_or(sd.fhir_id IS NULL     AND dim.{dim_col} IS NOT NULL)  AS via_dim
        FROM th_series_data sd
        LEFT JOIN fhir_indicators fi ON fi.id = sd.fhir_id
        LEFT JOIN th_series_dim dim  ON dim.original_indicator = sd.indicator
        WHERE sd.user_id = :user_id
          AND sd.deleted = 0
          AND sd.source_table = :source_table
        GROUP BY sd.indicator
        """,
            {"user_id": user_id, "source_table": SOURCE_TABLE},
            log_sql=False,
        )
        or []
    )

    unsearchable = [r["indicator"] for r in rows if not (r["via_fhir"] or r["via_dim"])]
    return len(rows), len(rows) - len(unsearchable), unsearchable


async def sync_indicators(user_id: str) -> tuple[int, int, list[str]]:
    """就地跑完 ``IndicatorSyncTask`` 漏斗，直到 embedding 不再有进展。

    就地跑而不是 ``enqueue``：seed 结束时部署应当立刻可用，而一个入队信号会让用户
    干等 worker，还没法判断它到底跑没跑。

    返回最后一轮的 :func:`searchability`。
    """
    from mirobody.task.indicator_sync import IndicatorSyncTask

    await IndicatorSyncTask.backfill_from_registry()
    await IndicatorSyncTask.backfill_from_history()
    await IndicatorSyncTask.backfill_from_dominant()
    await IndicatorSyncTask.insert()

    total, searchable, unsearchable = await searchability(user_id)
    for sweep in range(1, _MAX_EMBED_SWEEPS + 1):
        if not unsearchable:
            break
        await IndicatorSyncTask.embed()
        total, searchable, remaining = await searchability(user_id)
        if len(remaining) >= len(unsearchable):
            # 没有进展：再扫一轮也只是拿同一批行去打同一个失败的 provider。
            unsearchable = remaining
            logger.warning("[seed] 第 %d 轮 embedding 无进展，停止", sweep)
            break
        unsearchable = remaining

    return total, searchable, unsearchable


# ============================================================
# 编排
# ============================================================


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def held_out_exam_dates(exams: list, count: int) -> set[str]:
    """最近 *count* 次体检的日期 —— seed 要跳过的那些。

    给 demo 用：把最新一次面板留出来，让用户以 PDF 形式上传，亲眼看它并入历史。
    ``count=0``（默认）是常规路径，什么都不留。
    """
    if count <= 0:
        return set()
    dates = sorted({str(e.get("exam_date")) for e in exams if e.get("exam_date")})
    return set(dates[-count:])


async def seed_user(user_root: Path, *, require_searchable: bool = True, hold_out_exams: int = 0) -> SeedResult:
    """把一个已下载的用户目录灌进 Postgres。

    *user_root* 是形如 ``user5086_AT_demo`` 的目录，由
    ``generator.eslbench.prepare_data`` 下载得到。

    *require_searchable* 为真时，只要还有指标没拿到 embedding 就抛错 —— 那种部署
    看着像灌好了，回答起来却像没有数据，值得当场失败而不是在 demo 现场才发现。

    *hold_out_exams* 把最近 N 次化验面板从行和文档里都排除，留给后续走正常文件上传
    路径引入。那几天的设备读数照常灌，只留出化验面板。
    """
    user_dir = user_root.name
    email = dir_to_email(user_dir)

    user_id = await ensure_user(email)
    logger.info("[seed] %s → user_id=%s", email, user_id)

    exam_path = user_root / "exam_data.json"
    exams = _load(exam_path) if exam_path.exists() else []
    exams = exams if isinstance(exams, list) else []
    held_out = held_out_exam_dates(exams, hold_out_exams)
    if held_out:
        logger.info("[seed] %s: 留出体检 %s", email, ", ".join(sorted(held_out)))

    # ---- 指标读数 ----
    timeline = _load(user_root / "timeline.json")
    entries = timeline.get("entries") if isinstance(timeline, dict) else timeline
    if not isinstance(entries, list):
        raise ValueError(f"{user_dir}/timeline.json 没有 `entries` 列表")

    def wanted(entry: dict) -> bool:
        if not held_out or entry.get("entry_type") != "exam_indicator":
            return True
        return str(entry.get("time") or "")[:10] not in held_out

    stats = MappingStats()
    batch: list[dict] = []
    inserted = 0
    for row in series_rows((e for e in entries if wanted(e)), user_id=user_id, user_dir=user_dir, stats=stats):
        batch.append(row)
        if len(batch) >= _BATCH:
            inserted += await insert_series(batch)
            batch.clear()
    inserted += await insert_series(batch)
    logger.info("[seed] %s: %s（提交 %d 行）", email, stats.summary(), inserted)

    # ---- 文档 ----
    documents: list[tuple[str, str, str]] = [
        ("events.md", events_document(entries, email), "others"),
        ("profile.md", profile_document(_load(user_root / "profile.json"), email), "others"),
    ]
    documents += [
        (name, body, "report")
        for name, body in exam_documents([e for e in exams if str(e.get("exam_date")) not in held_out], email)
    ]

    # entries 只有 events 文档还要用；解引用，让 ~75k 条的 timeline 在 embedding
    # 之前能被回收。
    del timeline, entries

    for name, body, scene in documents:
        await insert_document(user_id, user_dir, name, body, scene=scene)
    logger.info("[seed] %s: 登记 %d 个文档", email, len(documents))

    # ---- 让指标能被搜到 ----
    total, searchable, unsearchable = await sync_indicators(user_id)

    result = SeedResult(
        email=email,
        user_id=user_id,
        mapping=stats,
        documents=len(documents),
        indicators=total,
        embedded=searchable,
        unembedded=unsearchable,
        held_out_exams=sorted(held_out),
    )

    if unsearchable and require_searchable:
        from mirobody.utils.config import safe_read_cfg

        raise RuntimeError(
            f"{email}: {len(unsearchable)}/{total} 个指标没有 embedding，agent 搜不到它们"
            f"（例如 {', '.join(unsearchable[:5])}）。\n"
            f"检查 EMBEDDING_PROVIDER（当前 {safe_read_cfg('EMBEDDING_PROVIDER', 'gemini')}）"
            f"对应的 API key 是否可用，然后重跑 seed。"
            f"要保留部分结果用于排查，加 --allow-unsearchable。"
        )

    logger.info("[seed] 完成 %s", result.summary())
    return result


# ============================================================
# CLI
# ============================================================


async def _run(args: argparse.Namespace) -> None:
    _require_mirobody()

    from evaluator.utils.mirobody_config import ensure_mirobody_config

    await ensure_mirobody_config()

    data_dir = DATA_DIR
    if args.all:
        user_roots = discover_user_dirs(data_dir)
    else:
        user_roots = [data_dir / e.replace("@", "_AT_") for e in args.users]

    if not user_roots:
        raise RuntimeError(
            f"{data_dir} 下没有用户数据。先跑:\n    python -m generator.eslbench.prepare_data"
        )

    for user_root in user_roots:
        if not user_root.is_dir():
            raise RuntimeError(
                f"{user_root} 不存在。先跑: python -m generator.eslbench.prepare_data"
            )
        result = await seed_user(
            user_root,
            require_searchable=not args.allow_unsearchable,
            hold_out_exams=args.hold_out_exams,
        )
        print(result.summary())
        if result.unembedded:
            print(f"  警告: {len(result.unembedded)} 个指标搜不到: {', '.join(result.unembedded[:10])}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m generator.eslbench.seed_mirobody",
        description="把 ESL-Bench 合成用户灌进自部署的 mirobody",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--users", nargs="+", help="合成用户邮箱，如 user5086@demo")
    group.add_argument(
        "--all",
        action="store_true",
        help="已下载的全部用户。单用户约 67k 行读数、20MB JSON —— 50 个用户是几小时的活，不是 demo",
    )
    parser.add_argument(
        "--hold-out-exams",
        type=int,
        default=0,
        metavar="N",
        help="最近 N 次化验面板不入库，留给文件上传路径引入（配合 labreport 生成 PDF）",
    )
    parser.add_argument(
        "--allow-unsearchable",
        action="store_true",
        help="部分指标没有 embedding 时保留结果。它们对 agent 不可见，仅用于排查",
    )
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
