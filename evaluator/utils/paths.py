"""路径解析 — 优先读 HOLYEVAL_* 环境变量，未设置时回退仓库内默认路径。

prod 模式（设 env，挂 PVC）：
    $HOLYEVAL_REPORT_DIR/<bench>/<filename>
    $HOLYEVAL_USER_DATA_DIR/<bench>/<user_dir>/{profile,timeline,exam_data}.json + user.duckdb
    $HOLYEVAL_USER_DATA_DIR/<bench>/.hippo_work_dirs/<user_dir>/...

dev 模式（不设 env）：
    <repo>/benchmark/report/<bench>/<filename>
    <repo>/benchmark/data/<bench>/.data/<user_dir>/...
    <repo>/benchmark/data/<bench>/.data/.hippo_work_dirs/<user_dir>/...   ← 跟 user_dir 同父级

注意：dev 模式下 .{hippo,dyg,naive_rag}_work_dirs 位置较老版本下移一层
（从 <bench>/.X_work_dirs/ → <bench>/.data/.X_work_dirs/）。RAG target 第一次跑
会重建索引（OpenIE/embedding 调用一次性开销），之后命中本地缓存。

启动时 main 调用一次 log_resolved_paths()，把最终解析结果打到日志，便于排查
PVC 是否真挂上、.env 是否真生效。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[2]


def report_dir() -> Path:
    """评测报告根目录 — runtime 写入。"""
    env = os.environ.get("HOLYEVAL_REPORT_DIR")
    return Path(env) if env else _REPO_ROOT / "benchmark" / "report"


def user_data_dir(bench: str) -> Path:
    """某 benchmark 的 per-user 数据目录（profile/timeline/exam/duckdb 落盘位置）。

    prod: $HOLYEVAL_USER_DATA_DIR/<bench>/      （扁平，PVC root 下直接是 bench 子目录）
    dev:  <repo>/benchmark/data/<bench>/.data/  （兼容当前结构）
    """
    env = os.environ.get("HOLYEVAL_USER_DATA_DIR")
    return (Path(env) / bench) if env else (_REPO_ROOT / "benchmark" / "data" / bench / ".data")


#: Question banks fetched from the dataset host, kept under the batch they came
#: from: `<user_data_dir>/_banks/<batch>/<dataset>.jsonl`. The batch stays in the
#: path on purpose — it is what lets a score be anchored to that batch's manifest
#: checksum without a second "dataset → batch" index to keep in step.
#:
#: Both the writer (`generator.eslbench.prepare_data`) and the reader
#: (`evaluator.utils.benchmark_reader`) import this rather than spelling the
#: directory twice; a literal in two places is how the two halves drift.
BANKS_DIRNAME = "_banks"


def fetched_bank(bench: str, dataset: str) -> tuple[Path, str] | None:
    """The fetched question bank for *dataset*, and the batch it came from.

    Returns None when nothing was fetched for it — the caller then falls back to
    the copy vendored in the repo, which still works offline but cannot say which
    release it corresponds to.
    """
    root = user_data_dir(bench) / BANKS_DIRNAME
    if not root.is_dir():
        return None
    for batch_dir in sorted(root.iterdir(), reverse=True):
        candidate = batch_dir / f"{dataset}.jsonl"
        if candidate.is_file():
            return candidate, batch_dir.name
    return None


def log_resolved_paths() -> None:
    """启动横幅 — 打印两个 env 的解析结果，便于排查 PVC / .env 配置问题。"""
    rep = report_dir()
    user_env = os.environ.get("HOLYEVAL_USER_DATA_DIR")
    user_root_path = Path(user_env) if user_env else _REPO_ROOT / "benchmark" / "data"
    logger.info("=" * 60)
    logger.info(
        "[paths] HOLYEVAL_REPORT_DIR    = %s (env=%s, exists=%s)",
        rep,
        "set" if os.environ.get("HOLYEVAL_REPORT_DIR") else "fallback",
        rep.is_dir(),
    )
    logger.info(
        "[paths] HOLYEVAL_USER_DATA_DIR = %s (env=%s, exists=%s)",
        user_root_path,
        "set" if user_env else "fallback",
        user_root_path.is_dir(),
    )
    logger.info("=" * 60)
