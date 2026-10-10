"""读取 benchmark/data/ 目录结构 — 列表、详情、加载、路径解析"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List

from evaluator.core.schema import BenchmarkSummary, CaseSummary, DatasetDetail, DatasetInfo, TargetSpec

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "data"


def resolve_dataset(benchmark: str, dataset: str) -> tuple[Path, dict[str, Any]]:
    """Locate a dataset's JSONL, preferring the copy fetched from the dataset host.

    Two copies of a question bank can exist:

    - the one `prepare_data` fetched, under `<user_data>/_banks/<batch>/`. It came
      from the release the leaderboard quotes, and the batch is in its path, so a
      score can be anchored to that batch's manifest checksum.
    - the one vendored in this repo. It works offline — which is why CI can gate
      on it — but carries no version relationship to any release.

    The fetched copy wins when present. The returned dict is provenance: it says
    which of the two was read, and for the fetched one, the batch and checksum a
    reader would need to reproduce the number.
    """
    from evaluator.utils import paths

    fetched = paths.fetched_bank(benchmark, dataset)
    if fetched:
        path, batch = fetched
        prov: dict[str, Any] = {"source": "fetched", "batch": batch}
        local_manifest = paths.user_data_dir(benchmark) / ".manifest.json"
        if local_manifest.is_file():
            try:
                manifest = json.loads(local_manifest.read_text(encoding="utf-8"))
                prov["manifest_version"] = manifest.get("version")
                prov["batch_checksum"] = (manifest.get("batches") or {}).get(batch, {}).get("checksum")
            except (OSError, json.JSONDecodeError):
                logger.warning("本地 manifest 读取失败，来源信息不含 checksum: %s", local_manifest)
        return path, prov

    return _DATA_DIR / benchmark / f"{dataset}.jsonl", {"source": "vendored"}


def _unavailable_reason(benchmark: str, evaluator: str, provenance: dict[str, Any]) -> str:
    """Why a listed dataset cannot be run, in the words a reader needs.

    A dataset with no `evaluator` fails at load. For the newest published batch
    that is not a defect: the questions ship and the answers follow when the next
    batch does, so the field is absent by design. Saying which of the two it is
    matters — one is "wait for the next release", the other is "this file is
    broken" — so the reason comes from the manifest's `released_answers`, not
    from guessing.
    """
    if evaluator:
        return ""

    batch = provenance.get("batch")
    if not batch:
        return "no evaluator declared — cannot be scored"

    from evaluator.utils import paths

    local_manifest = paths.user_data_dir(benchmark) / ".manifest.json"
    if local_manifest.is_file():
        try:
            manifest = json.loads(local_manifest.read_text(encoding="utf-8"))
            entry = (manifest.get("batches") or {}).get(batch, {})
            if not entry.get("released_answers"):
                return f"answers for batch {batch} are not released yet"
        except (OSError, json.JSONDecodeError):
            pass
    return "no evaluator declared — cannot be scored"


def _fetched_dataset_names(benchmark: str) -> set[str]:
    """Dataset names available only because they were fetched.

    The "dataset does not exist" message listed the vendored files alone, so a
    reader who had just fetched the release they wanted was told it was not
    there — with a list that omitted it.
    """
    from evaluator.utils import paths

    root = paths.user_data_dir(benchmark) / paths.BANKS_DIRNAME
    if not root.is_dir():
        return set()
    return {p.stem for batch in root.iterdir() if batch.is_dir() for p in batch.glob("*.jsonl")}


# ============================================================
# 内部辅助
# ============================================================


def _count_jsonl_lines(path: Path) -> int:
    """快速统计 JSONL 文件行数（跳过空行）"""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _peek_evaluator(path: Path) -> str:
    """从 JSONL 首行读取 eval.evaluator（用于列表页展示，不加载完整数据）"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                return (obj.get("eval") or {}).get("evaluator", "")
    except Exception:
        pass
    return ""


def _resolve_refs(raw: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """解析 JSONL 条目中的 $ref 引用，替换为 params 中的实际值

    仅扫描顶层字段：如果某个字段值为 {"$ref": "key"}（且仅含此键），
    则用 params[key] 替换。未匹配的 key 保持原值并输出警告。
    """
    if not params:
        return raw
    for k, v in raw.items():
        if isinstance(v, dict) and len(v) == 1 and "$ref" in v:
            ref_key = v["$ref"]
            if ref_key in params:
                raw[k] = params[ref_key]
            else:
                logger.warning("$ref 引用未找到: '%s'（字段: %s，用例: %s）", ref_key, k, raw.get("id", "?"))
    return raw


def _read_metadata(bench_dir: Path) -> dict[str, Any]:
    """从 metadata.json 读取完整元数据"""
    meta_path = bench_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_metadata_description(bench_dir: Path) -> str:
    """从 metadata.json 读取 description"""
    return _read_metadata(bench_dir).get("description", "")


def _parse_single_target_spec(raw: dict[str, Any]) -> TargetSpec | None:
    """解析单个 target 配置 dict 为 TargetSpec

    支持新格式（含 fields）和旧格式（平铺 TargetInfo dict）的兼容。
    """
    if not raw or "type" not in raw:
        return None
    # 新格式：有 fields 字段
    if "fields" in raw:
        return TargetSpec(**raw)
    # 旧格式兼容：平铺的 TargetInfo dict → 自动转换为 TargetSpec
    target_type = raw["type"]
    fields = {}
    for k, v in raw.items():
        if k in ("type", "target_configurable"):
            continue
        fields[k] = {"default": v, "editable": True, "required": v is not None}
    logger.warning("metadata.json 使用旧格式 target，建议升级为 TargetSpec 格式")
    return TargetSpec(type=target_type, fields=fields)


def _parse_target_specs(raw: Any) -> list[TargetSpec]:
    """解析 metadata.json 中的 target 配置为 TargetSpec 列表

    支持三种格式:
    - list[dict] → 逐个解析（新格式，多 target）
    - dict → 单对象解析，返回单元素列表（兼容旧格式）
    - None → 返回空列表

    自动过滤掉未注册的 target type（插件不存在时静默跳过）。
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        specs = []
        for item in raw:
            spec = _parse_single_target_spec(item)
            if spec:
                specs.append(spec)
        return _filter_registered_specs(specs)
    if isinstance(raw, dict):
        spec = _parse_single_target_spec(raw)
        return _filter_registered_specs([spec]) if spec else []
    return []


def _filter_registered_specs(specs: list[TargetSpec]) -> list[TargetSpec]:
    """过滤掉未注册的 target type（插件不存在时静默跳过）"""
    try:
        from evaluator.core.interfaces.abstract_target_agent import AbstractTargetAgent

        registered = set(AbstractTargetAgent._registry.keys())
        filtered = [s for s in specs if s.type in registered]
        skipped = [s.type for s in specs if s.type not in registered]
        if skipped:
            logger.debug("跳过未注册的 target type: %s", ", ".join(skipped))
        return filtered
    except Exception:
        return specs


def _extract_title(obj: dict[str, Any]) -> str:
    """从 BenchItem 对象中提取一行摘要文本"""
    title = obj.get("title", "")
    if title:
        return title[:100]
    inputs = (obj.get("user") or {}).get("strict_inputs", [])
    if inputs and isinstance(inputs[0], str):
        return inputs[0][:100]
    goal = (obj.get("user") or {}).get("goal", "")
    if goal:
        return goal[:100]
    return ""


# ============================================================
# 列表 / 浏览（Web UI + CLI 共享）
# ============================================================


def list_benchmarks() -> list[BenchmarkSummary]:
    """列出所有 benchmark 及其 dataset"""
    if not _DATA_DIR.is_dir():
        return []

    result: list[BenchmarkSummary] = []
    for bench_dir in sorted(_DATA_DIR.iterdir()):
        if not bench_dir.is_dir() or bench_dir.name.startswith((".", "_")):
            continue

        # Fetched banks belong in the listing, not just in the by-name lookup: a
        # dataset that can be run but never appears in the picker is a dataset
        # nobody finds. Resolved per name so a fetched copy shadows a vendored
        # one of the same name, exactly as running it would.
        names = {p.stem for p in bench_dir.glob("*.jsonl")} | _fetched_dataset_names(bench_dir.name)
        datasets: list[DatasetInfo] = []
        for name in sorted(names):
            jsonl_file, prov = resolve_dataset(bench_dir.name, name)
            if not jsonl_file.is_file():
                continue
            evaluator = _peek_evaluator(jsonl_file)
            datasets.append(
                DatasetInfo(
                    name=name,
                    case_count=_count_jsonl_lines(jsonl_file),
                    file_size_kb=round(jsonl_file.stat().st_size / 1024, 1),
                    evaluator=evaluator,
                    unavailable_reason=_unavailable_reason(bench_dir.name, evaluator, prov),
                )
            )

        metadata = _read_metadata(bench_dir)
        target_spec = _parse_target_specs(metadata.get("target"))
        result.append(
            BenchmarkSummary(
                name=bench_dir.name,
                description=metadata.get("description", ""),
                datasets=datasets,
                target=target_spec,
            )
        )
    return result


def get_dataset_detail(benchmark: str, dataset: str, preview_limit: int = 10) -> DatasetDetail:
    """获取 dataset 详情（含预览用例 + 全量轻量摘要）"""
    jsonl_path, _ = resolve_dataset(benchmark, dataset)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"数据集不存在: {benchmark}/{dataset}")

    cases_preview: list[dict[str, Any]] = []
    case_summaries: list[CaseSummary] = []
    tag_counts: dict[str, int] = {}
    total = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)
            if total <= preview_limit:
                cases_preview.append(obj)
            # BenchItem 没有顶层 target，从 user.target_overrides 推断 target_type
            user = obj.get("user") or {}
            overrides = user.get("target_overrides", {})
            if isinstance(overrides, dict):
                target_type = next(iter(overrides), "")
            elif isinstance(overrides, list) and overrides:
                target_type = overrides[0].get("type", "")  # 兼容旧 list 格式
            else:
                target_type = ""
            case_summaries.append(
                CaseSummary(
                    id=obj.get("id", f"case_{total}"),
                    title=_extract_title(obj),
                    user_type=user.get("type", ""),
                    target_type=target_type,
                    evaluator=(obj.get("eval") or {}).get("evaluator", ""),
                )
            )
            for tag in obj.get("tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

    metadata = _read_metadata(_DATA_DIR / benchmark)

    # 从 case_summaries 提取主流评估器
    eval_types = [s.evaluator for s in case_summaries if s.evaluator]
    evaluator = eval_types[0] if eval_types else ""

    return DatasetDetail(
        benchmark=benchmark,
        dataset=dataset,
        case_count=total,
        cases_preview=cases_preview,
        case_summaries=case_summaries,
        tag_distribution=dict(sorted(tag_counts.items(), key=lambda x: -x[1])),
        description=metadata.get("description", ""),
        target=_parse_target_specs(metadata.get("target")),
        evaluator=evaluator,
    )


def get_case_by_id(benchmark: str, dataset: str, case_id: str) -> dict[str, Any]:
    """从 JSONL 中按 id 查找单条 case 完整数据"""
    jsonl_path, _ = resolve_dataset(benchmark, dataset)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"数据集不存在: {benchmark}/{dataset}")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("id") == case_id:
                return obj

    raise KeyError(f"用例不存在: {case_id}")


# ============================================================
# 加载（runner / task_manager 使用）
# ============================================================


def load_bench_items(
    jsonl_path: str | Path,
    params: dict[str, Any] | None = None,
) -> List["BenchItem"]:  # noqa: F821
    """从 JSONL 文件加载 BenchItem 列表

    Args:
        jsonl_path: JSONL 文件路径
        params:     metadata.json 中的 params 字典，用于解析 $ref 引用
    """
    from evaluator.core.bench_schema import BenchItem

    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")

    items: List[BenchItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                if params:
                    raw = _resolve_refs(raw, params)
                items.append(BenchItem(**raw))
            except Exception as e:
                raise ValueError(f"{path.name} 第 {i} 行解析失败: {e}") from e

    logger.info("加载 %d 条 BenchItem: %s", len(items), path.name)
    return items


def load_benchmark(benchmark: str, dataset: str) -> "BenchMark":  # noqa: F821
    """加载完整的 BenchMark 数据集（metadata.json + JSONL）

    metadata.json 中的 target 通过 Pydantic Discriminated Union 自动解析为 TargetInfo。
    """
    from evaluator.core.bench_schema import BenchMark

    bench_dir = _DATA_DIR / benchmark
    if not bench_dir.is_dir():
        available = sorted(p.name for p in _DATA_DIR.iterdir() if p.is_dir())
        raise FileNotFoundError(f"评测类型不存在: {benchmark}\n可用评测: {available or '(空)'}")

    # 1. 读取 metadata.json（Pydantic 自动根据 type 字段解析 TargetInfo）
    metadata = _read_metadata(bench_dir)

    # 2. 读取 JSONL（传入 params 用于 $ref 解析）
    jsonl_path, provenance = resolve_dataset(benchmark, dataset)
    if not jsonl_path.exists():
        available = sorted({p.stem for p in bench_dir.glob("*.jsonl")} | _fetched_dataset_names(benchmark))
        raise FileNotFoundError(f"数据集不存在: {benchmark}/{dataset}\n[{benchmark}] 可用数据集: {available or '(空)'}")

    params = metadata.get("params", {})
    items = load_bench_items(jsonl_path, params=params or None)

    target_spec = _parse_target_specs(metadata.get("target"))

    bench = BenchMark(
        name=f"{benchmark}/{dataset}",
        description=metadata.get("description", ""),
        target=target_spec,
        items=items,
        max_concurrency=metadata.get("max_concurrency", 0),
    )

    logger.info("加载 BenchMark: %s (%d 条用例)", bench.name, bench.total_count)
    return bench


def resolve_data_path(benchmark: str, dataset: str) -> Path:
    """将 benchmark + dataset 解析为 JSONL 文件路径"""
    bench_dir = _DATA_DIR / benchmark
    if not bench_dir.is_dir():
        available = sorted(p.name for p in _DATA_DIR.iterdir() if p.is_dir())
        raise FileNotFoundError(f"评测类型不存在: {benchmark}\n可用评测: {available or '(空)'}")

    path, _ = resolve_dataset(benchmark, dataset)
    if not path.exists():
        available = sorted({p.stem for p in bench_dir.glob("*.jsonl")} | _fetched_dataset_names(benchmark))
        raise FileNotFoundError(f"数据集不存在: {benchmark}/{dataset}\n[{benchmark}] 可用数据集: {available or '(空)'}")
    return path


# ============================================================
# BenchItem 过滤
# ============================================================


def filter_bench_items(
    items: List["BenchItem"],  # noqa: F821
    ids: str | None = None,
    limit: int | None = None,
) -> list:
    """按 ID 和/或数量过滤 BenchItem

    过滤顺序: 先按 ID 筛选，再应用 limit 截断。
    """
    if ids is not None:
        id_set = {s.strip() for s in ids.split(",") if s.strip()}
        original_count = len(items)
        items = [c for c in items if c.id in id_set]
        matched_ids = {c.id for c in items}
        unmatched = id_set - matched_ids
        if unmatched:
            logger.warning("以下 ID 未在数据集中找到: %s", ", ".join(sorted(unmatched)))
        logger.info("ID 过滤: %d → %d 条", original_count, len(items))

    if limit is not None and limit > 0:
        original_count = len(items)
        items = items[:limit]
        if original_count > limit:
            logger.info("数量限制: %d → %d 条", original_count, len(items))

    return items
