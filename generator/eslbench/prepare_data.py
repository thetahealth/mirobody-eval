"""
ESLBench 数据准备脚本（canonical implementation）

核心函数均接受 benchmark_dir / data_dir / label 参数，不同 benchmark 通过传入
各自的路径复用同一套逻辑。

Web UI 启动时通过 PrepareManager 自动执行，完成以下步骤:
1. 从 HuggingFace ($ESLBENCH_HF_REPO, 默认 healthmemoryarena/ESL-Bench) 下载用户数据（基于 manifest.json 增量更新）
2. 为每个用户创建独立 DuckDB（benchmark/data/{benchmark}/.data/{user_dir}/user.duckdb）
3. [可选] 生成每用户 JSONL + 汇总 full.jsonl（ESLBench 已有预制 JSONL，此步可跳过）

目录名即邮箱（_AT_ 替换 @），如 user110_AT_demo → user110@demo。
自动发现用户目录，无需硬编码用户列表。

变更检测: 通过 manifest.json 的 version + batch checksum 实现增量更新，
只下载新增或变更的批次，避免全量重下载。

用法:
    python -m generator.eslbench.prepare_data
    python -m generator.eslbench.prepare_data --force   # 强制重建
"""

import argparse
import json
import os
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
    """自动发现 .data/ 下的用户目录（排除隐藏目录）"""
    if not data_dir.is_dir():
        return []
    return sorted(d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith("."))


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


def _check_download_complete(data_dir: Path) -> bool:
    """检查是否有至少一个用户数据已下载完整"""
    user_dirs = discover_user_dirs(data_dir)
    if not user_dirs:
        return False
    for d in user_dirs:
        if not (d / "profile.json").exists():
            return False
    return True


def download_from_hf(data_dir: Path, *, label: str = "eslbench", force: bool = False) -> tuple[Path, bool]:
    """基于 manifest.json 增量下载用户数据到 .data/ 目录。返回 (data_dir, data_changed)"""
    import shutil

    from huggingface_hub import snapshot_download

    # 拉取远端 manifest
    remote_manifest = _fetch_remote_manifest(label)
    if remote_manifest is None:
        if _check_download_complete(data_dir) and not force:
            print(f"[{label}] 无法连接远端, 使用本地已有数据")
            return data_dir, False
        print(f"[{label}] 错误: 无法获取远端 manifest 且本地无数据")
        return data_dir, False

    local_manifest = _load_local_manifest(data_dir)

    # 快速判断: version 相同且非强制 → 无变更
    if not force and remote_manifest.get("version") == local_manifest.get("version"):
        if _check_download_complete(data_dir):
            print(f"[{label}] manifest version 一致, 跳过下载")
            return data_dir, False

    # 找出需要下载的批次（新增或 checksum 变更）
    batches_to_download: list[str] = []
    remote_batches = remote_manifest.get("batches", {})
    local_batches = local_manifest.get("batches", {})

    for batch_id, batch_info in remote_batches.items():
        local_batch = local_batches.get(batch_id)
        if force or not local_batch or local_batch.get("checksum") != batch_info.get("checksum"):
            batches_to_download.append(batch_id)

    if not batches_to_download and _check_download_complete(data_dir):
        print(f"[{label}] 所有批次 checksum 一致, 跳过下载")
        _save_local_manifest(remote_manifest, data_dir)
        return data_dir, False

    # 逐批次下载
    total_copied = 0
    for batch_id in batches_to_download:
        batch_info = remote_batches[batch_id]
        user_count = batch_info.get("user_count", "?")
        print(f"[{label}] 下载批次 {batch_id} ({user_count} 个用户)...")

        snapshot_dir = snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            # `*/*.json` is the per-user data surface. `*.jsonl` at the batch
            # root is the question bank, and it was NOT fetched before: the
            # leaderboard's dataset for a batch (`manifest.eval_dataset`) could
            # therefore not be run at all, and what did run came from whatever
            # copy this repo happened to vendor — with no version relationship
            # to the manifest the score would be quoted against.
            allow_patterns=[f"data/{batch_id}/*/*.json", f"data/{batch_id}/*.jsonl"],
        )
        snapshot_batch = Path(snapshot_dir) / "data" / batch_id

        if not snapshot_batch.exists():
            print(f"[{label}] 警告: 批次 {batch_id} 下载后目录不存在, 跳过")
            continue

        # 复制到 .data/{user_dir}/ 扁平结构（去掉 batch 层级）
        copied = 0
        for src_dir in sorted(snapshot_batch.iterdir()):
            if not src_dir.is_dir() or src_dir.name.startswith("."):
                continue
            dst_dir = data_dir / src_dir.name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for src_file in src_dir.iterdir():
                if src_file.is_file() and src_file.suffix == ".json":
                    if src_file.name.startswith("kg_evaluation_queries"):
                        continue  # ground-truth 不复制到本地 .data/
                    shutil.copy2(str(src_file), str(dst_dir / src_file.name))
                    copied += 1

        # 题包保留 batch 层级：目录名就是批次号，读取侧据此把分数锚到
        # manifest 的 batch checksum，不必再维护一份「数据集 → 批次」索引。
        banks = 0
        bank_dir = data_dir / BANKS_DIRNAME / batch_id
        for src_file in sorted(snapshot_batch.glob("*.jsonl")):
            bank_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src_file), str(bank_dir / src_file.name))
            banks += 1

        total_copied += copied + banks
        print(f"[{label}] 批次 {batch_id}: {copied} 个用户文件, {banks} 个题包")

    # 保存 manifest
    _save_local_manifest(remote_manifest, data_dir)
    print(f"[{label}] 下载完成: {len(batches_to_download)} 个批次, {total_copied} 个文件")
    return data_dir, total_copied > 0


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


def build_all_duckdb(data_dir: Path, *, label: str = "eslbench", force: bool = False) -> None:
    """为所有用户创建独立 DuckDB"""
    user_dirs = discover_user_dirs(data_dir)
    created = 0
    skipped = 0

    for user_dir in user_dirs:
        db_path = user_dir / "user.duckdb"
        if db_path.exists() and not force:
            skipped += 1
            continue

        create_user_duckdb(user_dir, force=force)
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


def _check_local_complete_eslbench() -> bool:
    """快速本地完整性检查（ESLBench）: manifest + 用户数据 + DuckDB 全部就绪"""
    manifest_file = DATA_DIR / ".manifest.json"
    if not manifest_file.exists():
        return False
    user_dirs = discover_user_dirs(DATA_DIR)
    if not user_dirs:
        return False
    for d in user_dirs:
        if not (d / "profile.json").exists():
            return False
        if not (d / "user.duckdb").exists():
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="ESLBench 数据准备")
    parser.add_argument("--force", action="store_true", help="强制重建（忽略已存在的文件）")
    args = parser.parse_args()

    force = args.force

    print("=" * 60)
    print("ESLBench 数据准备")
    print("=" * 60)

    # 快速路径: 本地数据完整 + manifest version 一致 → 秒退
    if not force and _check_local_complete_eslbench():
        remote_manifest = _fetch_remote_manifest("eslbench")
        local_manifest = _load_local_manifest(DATA_DIR)
        if remote_manifest and remote_manifest.get("version") == local_manifest.get("version"):
            print("[eslbench] 本地数据完整且 manifest version 一致, 跳过准备")
            print("=" * 60)
            return

    # Step 1: 从 HuggingFace 下载（检测远端变更）
    print("\n--- Step 1: 下载 HuggingFace 数据 ---")
    _, data_changed = download_from_hf(DATA_DIR, label="eslbench", force=force)

    # 清理 ground-truth（兼容旧数据: 新下载已跳过复制，此处清理历史遗留文件）
    purge_ground_truth(DATA_DIR, label="eslbench")

    # 远端数据有变更时，级联重建 DuckDB
    rebuild = force or data_changed
    if data_changed and not force:
        print("[eslbench] 远端数据有更新, 将级联重建 DuckDB")

    # Step 2: 创建每用户 DuckDB
    print("\n--- Step 2: 创建每用户 DuckDB ---")
    build_all_duckdb(DATA_DIR, label="eslbench", force=rebuild)

    # 汇总
    print("\n" + "=" * 60)
    print("准备完成!")
    print(f"  数据目录: {DATA_DIR}/")
    user_dirs = discover_user_dirs(DATA_DIR)
    db_count = sum(1 for d in user_dirs if (d / "user.duckdb").exists())
    print(f"  用户数据库: {db_count}/{len(user_dirs)} 个")
    print("=" * 60)


if __name__ == "__main__":
    main()
