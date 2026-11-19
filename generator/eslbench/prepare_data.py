"""
ESLBench 数据准备脚本（canonical implementation）

核心函数均接受 benchmark_dir / data_dir / label 参数，不同 benchmark 通过传入
各自的路径复用同一套逻辑。

1. 从 HuggingFace ($ESLBENCH_HF_REPO, 默认 healthmemoryarena/ESL-Bench) 同步 manifest 与题包，
   再下载要准备的用户数据：给了 --dataset 只下这个数据集用到的用户，否则下所有已发布批次
2. 为每个用户创建独立 DuckDB（benchmark/data/{benchmark}/.data/{user_dir}/user.duckdb）
3. [可选] 生成每用户 JSONL + 汇总 full.jsonl（ESLBench 已有预制 JSONL，此步可跳过）

Web UI 启动时只跑 --banks-only（同步 manifest 与题包，几十 MB）；开跑某个数据集前，
缺哪些用户就在后台用 --dataset 补哪些（见 web/app/services/prepare_manager.py）。

目录名即邮箱（_AT_ 替换 @），如 user110_AT_demo → user110@demo。

用法:
    python -m generator.eslbench.prepare_data --dataset sample50-20260331   # 只准备一个数据集
    python -m generator.eslbench.prepare_data                               # 所有已发布批次
    python -m generator.eslbench.prepare_data --force                       # 强制重新下载并重建
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from evaluator.utils import paths

HF_REPO = os.getenv("ESLBENCH_HF_REPO", "healthmemoryarena/ESL-Bench")

# ==================== ESLBench 默认路径 ====================

BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "data" / "eslbench"
# DATA_DIR: prod 下读 HOLYEVAL_USER_DATA_DIR/eslbench；dev 下回退 BENCHMARK_DIR/.data
DATA_DIR = paths.user_data_dir("eslbench")
# 题包落盘用的子目录名 —— 与读取侧共用一处定义，见 paths.BANKS_DIRNAME
BANKS_DIRNAME = paths.BANKS_DIRNAME

# ==================== 共享工具函数 ====================


def dir_to_email(dir_name: str) -> str:
    """目录名 → 邮箱: user110_AT_demo → user110@demo"""
    return dir_name.replace("_AT_", "@")


def discover_user_dirs(data_dir: Path) -> list[Path]:
    """自动发现 .data/ 下的用户目录（排除隐藏目录和 `_banks` 这类以下划线开头的目录）"""
    if not data_dir.is_dir():
        return []
    return sorted(d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith((".", "_")))


def find_timeline(user_dir: Path) -> Path | None:
    """在用户目录中查找 timeline.json（兼容不同命名约定）"""
    candidates = list(user_dir.glob("*_timeline.json"))
    return candidates[0] if candidates else None


def write_jsonl(items: list[dict], path: Path) -> None:
    """写入 JSONL 文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ==================== Step 1: HuggingFace 下载（manifest 驱动） ====================
#
# 本地 `.manifest.json` 是最近一次同步到的远端 manifest 原样副本。不变式：`.data/` 下
# 存在的每个用户目录，都是按这份副本里它所在批次的 checksum 下载的。远端某一批的
# checksum 变了，同步时就删掉本地这一批的用户目录，等哪个数据集用到再重新下载。
# 所以"目录在"就等于"数据是新的"，不必另记哪些用户下过。


def _load_local_manifest(data_dir: Path) -> dict:
    """读取本地 manifest（不存在则返回空 dict）"""
    manifest_file = data_dir / ".manifest.json"
    if manifest_file.exists():
        return json.loads(manifest_file.read_text(encoding="utf-8"))
    return {}


def _save_local_manifest(manifest: dict, data_dir: Path) -> None:
    """保存 manifest 到本地"""
    manifest_file = data_dir / ".manifest.json"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _fetch_remote_manifest(label: str) -> dict | None:
    """从 HuggingFace 拉取 manifest.json（单文件，极快）"""
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(repo_id=HF_REPO, filename="manifest.json", repo_type="dataset")
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[{label}] 警告: 无法获取远端 manifest ({e})")
        return None


def _roster(manifest: dict) -> dict[str, str]:
    """用户目录名 → 所在批次"""
    return {u: batch_id for batch_id, info in (manifest.get("batches") or {}).items() for u in info.get("users", [])}


def dataset_users(path: Path) -> set[str]:
    """数据集里的题目指向的用户目录名（user5062@demo → user5062_AT_demo）"""
    users: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            for target in ((item.get("user") or {}).get("target_overrides") or {}).values():
                email = target.get("user_email") or (target.get("tool_context") or {}).get("user_email")
                if email:
                    users.add(email.replace("@", "_AT_"))
    return users


def _dataset_path(benchmark: str, dataset: str) -> Path:
    from evaluator.utils.benchmark_reader import resolve_dataset

    path, _ = resolve_dataset(benchmark, dataset)
    if not path.is_file():
        raise FileNotFoundError(f"数据集不存在: {benchmark}/{dataset}")
    return path


def _unknown_users_message(benchmark: str, dataset: str, unknown: set[str]) -> str:
    return f"{benchmark}/{dataset} 用到的用户不在任何已发布批次里，下载不到: {', '.join(sorted(unknown))}"


def _user_ready(user_dir: Path) -> bool:
    return (user_dir / "profile.json").is_file() and (user_dir / "user.duckdb").is_file()


def missing_users(benchmark: str, dataset: str) -> list[str]:
    """跑 *dataset* 之前还要准备的用户；空列表表示可以直接跑。

    只看本地，不联网：用户目录里有 profile.json 和 user.duckdb 就算就绪（见上面的
    不变式）。Web 在开跑前调用它，决定要不要先在后台准备数据。数据集用到的用户不在
    任何已发布批次里时抛 ValueError：这种数据下载不到，应当直接说，而不是去下载再失败。
    """
    needed = dataset_users(_dataset_path(benchmark, dataset))
    manifest = _load_local_manifest(DATA_DIR)
    if manifest:
        unknown = needed - _roster(manifest).keys()
        if unknown:
            raise ValueError(_unknown_users_message(benchmark, dataset, unknown))
    return sorted(u for u in needed if not _user_ready(DATA_DIR / u))


def sync_manifest(data_dir: Path, remote: dict, *, label: str = "eslbench", force: bool = False) -> None:
    """把本地对齐到远端 manifest：变更批次的本地用户目录删掉，新增或变更批次的题包拉下来。"""
    import shutil

    from huggingface_hub import snapshot_download

    local_batches = _load_local_manifest(data_dir).get("batches") or {}
    for batch_id, info in (remote.get("batches") or {}).items():
        old = local_batches.get(batch_id)
        changed = not old or old.get("checksum") != info.get("checksum")
        if not (changed or force):
            continue

        if old and changed:
            stale = [data_dir / u for u in old.get("users", []) if (data_dir / u).is_dir()]
            for d in stale:
                shutil.rmtree(d)
            if stale:
                print(f"[{label}] 批次 {batch_id} 已更新, 删除 {len(stale)} 个过期的用户目录（用到时重新下载）")

        # 题包在批次根目录。题包保留 batch 层级：目录名就是批次号，读取侧据此把分数锚到
        # manifest 的 batch checksum，不必再维护一份「数据集 → 批次」索引。
        snapshot_dir = snapshot_download(
            repo_id=HF_REPO, repo_type="dataset", allow_patterns=[f"data/{batch_id}/*.jsonl"]
        )
        banks = 0
        bank_dir = data_dir / BANKS_DIRNAME / batch_id
        for src_file in sorted((Path(snapshot_dir) / "data" / batch_id).glob("*.jsonl")):
            bank_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src_file), str(bank_dir / src_file.name))
            banks += 1
        print(f"[{label}] 批次 {batch_id}: {banks} 个题包")

    _save_local_manifest(remote, data_dir)


def fetch_users(
    data_dir: Path, manifest: dict, users: set[str], *, label: str = "eslbench", force: bool = False
) -> set[str]:
    """下载 *users* 的数据（本地已有的跳过，除非 force）。返回这次实际下载了的用户。"""
    import shutil

    from huggingface_hub import snapshot_download

    roster = _roster(manifest)
    by_batch: dict[str, list[str]] = defaultdict(list)
    for u in sorted(users):
        if force or not (data_dir / u / "profile.json").is_file():
            by_batch[roster[u]].append(u)

    fetched: set[str] = set()
    for batch_id, batch_users in sorted(by_batch.items()):
        print(f"[{label}] 批次 {batch_id}: 下载 {len(batch_users)} 个用户...")
        snapshot_dir = snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            allow_patterns=[f"data/{batch_id}/{u}/*.json" for u in batch_users],
            # 题目答案不进本地 .data/（见 purge_ground_truth），也就不必下载
            ignore_patterns=["*/kg_evaluation_queries*"],
        )
        for u in batch_users:
            src_dir = Path(snapshot_dir) / "data" / batch_id / u
            if not src_dir.is_dir():
                print(f"[{label}] 警告: {batch_id}/{u} 下载后目录不存在, 跳过")
                continue
            dst_dir = data_dir / u
            dst_dir.mkdir(parents=True, exist_ok=True)
            for src_file in src_dir.iterdir():
                if not (src_file.is_file() and src_file.suffix == ".json"):
                    continue
                if src_file.name.startswith("kg_evaluation_queries"):
                    continue  # HF 缓存里可能留着以前整批下载时的答案文件，同样不复制
                shutil.copy2(str(src_file), str(dst_dir / src_file.name))
            fetched.add(u)
    return fetched


def purge_ground_truth(data_dir: Path, *, label: str = "eslbench") -> int:
    """删除 .data/ 下所有用户目录中的 kg_evaluation_queries*.json（ground-truth）"""
    removed = 0
    for user_dir in discover_user_dirs(data_dir):
        for f in user_dir.glob("kg_evaluation_queries*.json"):
            f.unlink()
            removed += 1
    if removed:
        print(f"[{label}] 清理 ground-truth 文件: {removed} 个")
    return removed


# ==================== Step 2: Per-user DuckDB ====================


def bulk_insert_ndjson(con: "duckdb.DuckDBPyConnection", table: str, columns: list[str], rows: list[dict]) -> None:
    """通过临时 NDJSON 文件批量导入数据到 DuckDB（比 executemany 快 200 倍以上）"""
    import tempfile

    if not rows:
        return

    col_select = ", ".join(columns)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".ndjson", delete=False, encoding="utf-8")
    try:
        for row in rows:
            tmp.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.close()
        con.execute(f"INSERT INTO {table} SELECT {col_select} FROM read_ndjson_auto('{tmp.name}')")
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def create_user_duckdb(user_dir: Path, *, force: bool = False) -> None:
    """为单个用户创建 DuckDB 数据库"""
    import duckdb

    db_path = user_dir / "user.duckdb"
    dir_name = user_dir.name
    email = dir_to_email(dir_name)

    if db_path.exists() and not force:
        return

    if db_path.exists():
        db_path.unlink()

    con = duckdb.connect(str(db_path))

    # 建表
    con.execute("""
        CREATE TABLE device_indicators (
            user_id VARCHAR, time TIMESTAMP, indicator VARCHAR,
            device_type VARCHAR, value VARCHAR, unit VARCHAR
        )
    """)
    con.execute("""
        CREATE TABLE exam_indicators (
            user_id VARCHAR, time TIMESTAMP, indicator VARCHAR,
            exam_type VARCHAR, exam_location VARCHAR, value VARCHAR, unit VARCHAR
        )
    """)
    con.execute("""
        CREATE TABLE events (
            user_id VARCHAR, time TIMESTAMP, end_time TIMESTAMP,
            event_id VARCHAR, event_type VARCHAR, event_name VARCHAR,
            start_date DATE, duration_days INTEGER,
            interrupted BOOLEAN, interruption_date DATE
        )
    """)
    # event_indicators 表已移除：含 expected_change/impact_level，会泄露 Explanation 题答案
    # evaluation_queries 表已移除：答案库不应写入 DuckDB，避免被 query_duckdb 直接读取

    # 加载 timeline（自动查找 *_timeline.json）
    tl_path = find_timeline(user_dir)
    if tl_path:
        with open(tl_path, encoding="utf-8") as f:
            data = json.load(f)

        device_rows, exam_rows, event_rows = [], [], []
        for e in data.get("entries", []):
            if e["entry_type"] == "device_indicator":
                raw_val = e["value"]
                device_rows.append(
                    {
                        "user_id": email,
                        "time": e["time"],
                        "indicator": e["indicator"],
                        "device_type": e.get("device_type"),
                        "value": None if raw_val is None else str(raw_val),
                        "unit": e.get("unit"),
                    }
                )
            elif e["entry_type"] == "exam_indicator":
                raw_val = e["value"]
                exam_rows.append(
                    {
                        "user_id": email,
                        "time": e["time"],
                        "indicator": e["indicator"],
                        "exam_type": e.get("exam_type"),
                        "exam_location": e.get("exam_location"),
                        "value": None if raw_val is None else str(raw_val),
                        "unit": e.get("unit"),
                    }
                )
            elif e["entry_type"] == "event":
                ev = e["event"]
                event_rows.append(
                    {
                        "user_id": email,
                        "time": e["time"],
                        "end_time": e.get("end_time"),
                        "event_id": ev["event_id"],
                        "event_type": ev["event_type"],
                        "event_name": ev["event_name"],
                        "start_date": ev["start_date"],
                        "duration_days": ev.get("duration_days"),
                        "interrupted": ev.get("interrupted"),
                        "interruption_date": ev.get("interruption_date"),
                    }
                )

        bulk_insert_ndjson(
            con,
            "device_indicators",
            ["user_id", "time::TIMESTAMP AS time", "indicator", "device_type", "value", "unit"],
            device_rows,
        )
        bulk_insert_ndjson(
            con,
            "exam_indicators",
            ["user_id", "time::TIMESTAMP AS time", "indicator", "exam_type", "exam_location", "value", "unit"],
            exam_rows,
        )
        bulk_insert_ndjson(
            con,
            "events",
            [
                "user_id",
                "time::TIMESTAMP AS time",
                "end_time::TIMESTAMP AS end_time",
                "event_id",
                "event_type",
                "event_name",
                "start_date::DATE AS start_date",
                "duration_days::INTEGER AS duration_days",
                "interrupted::BOOLEAN AS interrupted",
                "interruption_date::DATE AS interruption_date",
            ],
            event_rows,
        )

    # event_indicators 已移除，不写入 DuckDB（含 expected_change 会泄题）
    # evaluation_queries 已移除，不写入 DuckDB

    # 创建索引
    con.execute("CREATE INDEX idx_device_time ON device_indicators (user_id, time)")
    con.execute("CREATE INDEX idx_device_ind ON device_indicators (indicator)")
    con.execute("CREATE INDEX idx_exam_time ON exam_indicators (user_id, time)")
    con.execute("CREATE INDEX idx_event_user ON events (user_id)")
    # event_indicators 索引已随表移除

    con.close()


def build_all_duckdb(
    data_dir: Path, users: list[str], *, rebuild: set[str] = frozenset(), label: str = "eslbench"
) -> None:
    """为 *users* 创建独立 DuckDB；已有的跳过，*rebuild* 里的（数据刚重新下载过）重建"""
    created = 0
    skipped = 0

    for user in users:
        user_dir = data_dir / user
        if not (user_dir / "profile.json").is_file():
            continue
        if (user_dir / "user.duckdb").exists() and user not in rebuild:
            skipped += 1
            continue

        create_user_duckdb(user_dir, force=True)
        created += 1

    print(f"[{label}] DuckDB 创建完成: {created} 个新建, {skipped} 个已存在跳过")


# ==================== Step 3: 生成 JSONL 评测集（可选） ====================


def build_datasets(benchmark_dir: Path, data_dir: Path, *, label: str = "eslbench", force: bool = False) -> None:
    """生成每用户 JSONL + 汇总 full.jsonl"""
    from generator.eslbench.converter import convert_queries

    full_path = benchmark_dir / "full.jsonl"
    if full_path.exists() and full_path.stat().st_size > 0 and not force:
        print(f"[{label}] 评测集 JSONL 已存在, 跳过生成")
        return

    user_dirs = discover_user_dirs(data_dir)
    all_items: list[dict] = []
    user_count = 0

    for user_dir in user_dirs:
        qa_file = user_dir / "kg_evaluation_queries.json"
        if not qa_file.exists():
            print(f"[{label}] 警告: {user_dir.name} 缺少 kg_evaluation_queries.json, 跳过")
            continue

        dir_name = user_dir.name
        email = dir_to_email(dir_name)

        items = convert_queries(qa_file, user_email=email)
        if not items:
            continue

        # 写入每用户 JSONL（文件名 = 目录名.jsonl）
        user_path = benchmark_dir / f"{dir_name}.jsonl"
        write_jsonl(items, user_path)
        print(f"[{label}] 写入 {len(items):>4} 条 -> {dir_name}.jsonl")

        all_items.extend(items)
        user_count += 1

    # 写入汇总 full.jsonl
    write_jsonl(all_items, full_path)
    print(f"[{label}] 写入 {len(all_items):>4} 条 -> full.jsonl ({user_count} 个用户)")

    # 写入 sample.jsonl: 从前 5 个用户的数据合并（从 id 提取用户标识）
    import random

    def _user_from_id(item_id: str) -> str:
        """从 id 提取用户标识，如 user101_AT_demo_Q065 → user101_AT_demo"""
        parts = item_id.rsplit("_Q", 1)
        return parts[0] if len(parts) == 2 else item_id

    sample_users = sorted(set(_user_from_id(item["id"]) for item in all_items))[:5]
    sample_items = [item for item in all_items if _user_from_id(item["id"]) in sample_users]
    random.Random(42).shuffle(sample_items)
    sample_path = benchmark_dir / "sample.jsonl"
    write_jsonl(sample_items, sample_path)
    print(f"[{label}] 写入 {len(sample_items):>4} 条 -> sample.jsonl ({len(sample_users)} 个用户)")


# ==================== ESLBench Main ====================


def main():
    parser = argparse.ArgumentParser(description="ESLBench 数据准备")
    parser.add_argument("--dataset", help="只准备这个数据集用到的用户（不给则准备所有已发布批次）")
    parser.add_argument(
        "--benchmark", default="eslbench", help="--dataset 所属的 benchmark（eslbench_distractor 与 eslbench 共用数据）"
    )
    parser.add_argument("--banks-only", action="store_true", help="只同步 manifest 和题包，不下载用户数据")
    parser.add_argument("--force", action="store_true", help="强制重新下载并重建要准备的用户")
    args = parser.parse_args()
    if args.dataset and args.banks_only:
        parser.error("--dataset 与 --banks-only 不能同时给")

    print("=" * 60)
    print("ESLBench 数据准备")
    print("=" * 60)

    # Step 1: 同步 manifest 与题包，再下载要准备的用户
    print("\n--- Step 1: 同步 HuggingFace 数据 ---")
    remote = _fetch_remote_manifest("eslbench")
    if remote is not None:
        sync_manifest(DATA_DIR, remote, label="eslbench", force=args.force)
        manifest = remote
    else:
        manifest = _load_local_manifest(DATA_DIR)
        if not manifest:
            sys.exit("[eslbench] 错误: 无法获取远端 manifest，本地也没有数据")
        print("[eslbench] 无法连接远端, 只用本地已有数据")

    if args.banks_only:
        print("[eslbench] 只同步了 manifest 与题包")
        print("=" * 60)
        return

    roster = _roster(manifest)
    if args.dataset:
        users = dataset_users(_dataset_path(args.benchmark, args.dataset))
        unknown = users - roster.keys()
        if unknown:
            sys.exit(f"[eslbench] 错误: {_unknown_users_message(args.benchmark, args.dataset, unknown)}")
        print(f"[eslbench] {args.benchmark}/{args.dataset} 用到 {len(users)} 个用户")
    else:
        users = set(roster)

    fetched = set()
    if remote is not None:
        fetched = fetch_users(DATA_DIR, manifest, users, label="eslbench", force=args.force)
    absent = sorted(u for u in users if not (DATA_DIR / u / "profile.json").is_file())
    if absent:
        sys.exit(f"[eslbench] 错误: {len(absent)} 个用户的数据没能下载: {', '.join(absent)}")

    # 清理 ground-truth（兼容旧数据: 新下载已跳过复制，此处清理历史遗留文件）
    purge_ground_truth(DATA_DIR, label="eslbench")

    # Step 2: 创建每用户 DuckDB（数据刚重新下载过的重建）
    print("\n--- Step 2: 创建每用户 DuckDB ---")
    build_all_duckdb(DATA_DIR, sorted(users), rebuild=fetched, label="eslbench")

    # 汇总
    print("\n" + "=" * 60)
    print("准备完成!")
    print(f"  数据目录: {DATA_DIR}/")
    print(f"  本次准备的用户: {len(users)} 个（其中新下载 {len(fetched)} 个）")
    print("=" * 60)


if __name__ == "__main__":
    main()
