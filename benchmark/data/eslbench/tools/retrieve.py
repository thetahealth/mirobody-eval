"""retrieve — 用户数据 JSON 检索与 DuckDB 查询工具组

通过 ToolRuntime 注入运行时上下文（user_email），context 由 ainvoke(context=...) 透传。

工具:
- list_files   列出 JSON 文件
- read_file    读取 JSON（支持 path）
- query_json   路径检索 JSON（推荐替代整文件读取）
- search_file  结构化关键词搜索（返回 path + 片段）
- lookup_indicator 指标检索（按日期+指标关键词，优先）
- lookup_event   事件检索（按事件名/日期，优先）
- query_duckdb DuckDB 只读 SQL 查询
"""

import json
import logging
import os
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cached_property
from pathlib import Path
from typing import Any

from langchain.tools import tool
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolRuntime

logger = logging.getLogger(__name__)

_DOT_DATA = Path(__file__).resolve().parents[1] / ".data"
_ALLOWED_EXTENSIONS = {".json"}
_BLOCKED_FILES = {"events.json"}
# 前缀屏蔽：kg_evaluation_queries 是评测题库（包含 GT），"." 屏蔽 .openie_cache.json
# 等历史 RAG 评测产物（包含 OpenIE 三元组，等于已剥离的 KG 关系字段）。
_BLOCKED_PREFIXES = ("kg_evaluation_queries", ".")
_FORBIDDEN_SQL = {"INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"}
_BLOCKED_TABLES = {"evaluation_queries"}
_UNICODE_SQL_HINT = (
    "查询含非法 Unicode 字节（可能是被截断的多字节字符），"
    "请用完整、干净的 UTF-8 重写 WHERE 条件后重试，或改用 lookup_indicator 工具。"
)

_MAX_READ_FILE_CHARS = 16000
_MAX_SEARCH_OUTPUT_CHARS = 4000
_MAX_QUERY_ROWS = 200
_MAX_QUERY_CELL_CHARS = 300
_MAX_QUERY_OUTPUT_CHARS = 16000
_MAX_QUERY_SECONDS = 5.0
_CACHE_DB_VERSION = 3

_MAX_JSON_LRU = 4
_MAX_JSON_CACHE_BYTES = 64 * 1024 * 1024  # 64MB 总上限
_MAX_JSON_PATH_RESULTS = 20
_MAX_JSON_PATH_NODE_EXPAND = 120
_MAX_JSON_SEARCH_HITS = 10
_MAX_JSON_SEARCH_NODES = 120000
_MAX_JSON_SEARCH_LIST_SCAN = 6000
_MAX_JSON_VALUE_CHARS = 360
_MAX_JSON_PREVIEW_ITEMS = 5
_MAX_JSON_PREVIEW_DEPTH = 3
_MAX_JSON_FILTER_SCAN = 20000
_MAX_LOOKUP_RESULTS = 20
_FILTER_KEY_ALIASES = {
    "date": ("date", "exam_date", "start_date", "time", "timestamp"),
}

try:
    import orjson
except Exception:  # pragma: no cover - 环境缺少时回退 json
    orjson = None

try:
    import jmespath
except Exception:  # pragma: no cover - 可选依赖
    jmespath = None

_JSON_CACHE: OrderedDict[str, tuple[int, int, Any]] = OrderedDict()
_json_cache_bytes: int = 0  # 当前缓存的总文件大小（近似）


@dataclass
class ToolContext:
    """工具运行时上下文 — 从 user_email 推导数据路径"""

    user_email: str

    @cached_property
    def user_dir_name(self) -> str:
        return self.user_email.replace("@", "_AT_")

    @cached_property
    def data_dir(self) -> Path:
        return _DOT_DATA / self.user_dir_name

    @cached_property
    def db_path(self) -> Path:
        return self.data_dir / "user.duckdb"


def _ctx(runtime: ToolRuntime[ToolContext]) -> ToolContext:
    c = runtime.context
    return c if isinstance(c, ToolContext) else ToolContext(**c)


def _truncate_text(text: str, max_len: int) -> tuple[str, bool]:
    """按字符上限截断文本，返回 (文本, 是否截断)。"""
    if len(text) <= max_len:
        return text, False
    return text[:max_len], True


def _render_truncated(text: str, max_len: int, note_prefix: str = "truncated") -> tuple[str, bool]:
    """截断并追加说明，返回 (渲染结果, 是否截断)。"""
    clipped, truncated = _truncate_text(text, max_len)
    if not truncated:
        return text, False
    return f"{clipped}... ({note_prefix}, total {len(text)} chars)", True


def _dumps_json(data: Any, *, indent: bool = True) -> str:
    """统一 JSON 序列化（优先 orjson）。"""
    if orjson is None:
        return json.dumps(data, ensure_ascii=False, indent=2 if indent else None)
    option = orjson.OPT_INDENT_2 if indent else 0
    return orjson.dumps(data, option=option).decode("utf-8")


def _loads_json(raw: bytes) -> Any:
    """统一 JSON 反序列化（优先 orjson）。"""
    if orjson is None:
        return json.loads(raw.decode("utf-8"))
    return orjson.loads(raw)


def _resolve_json_file(ctx: ToolContext, filename: str) -> tuple[Path | None, str | None]:
    """校验并解析文件路径。"""
    if ".." in filename or "/" in filename or "\\" in filename:
        return None, "错误: 文件名不能包含路径分隔符"
    if filename in _BLOCKED_FILES or filename.startswith(_BLOCKED_PREFIXES):
        return None, f"错误: 文件 {filename} 不可访问"
    fp = ctx.data_dir / filename
    if fp.suffix not in _ALLOWED_EXTENSIONS:
        return None, f"错误: 只能读取 {_ALLOWED_EXTENSIONS} 类型的文件"
    if not fp.exists():
        return None, f"错误: 文件 {filename} 不存在"
    return fp, None


def _load_json_cached(fp: Path) -> Any:
    """加载并缓存 JSON（按文件 mtime/size 自动失效，含总大小上限）。"""
    global _json_cache_bytes
    stat = fp.stat()
    cache_key = str(fp)
    cached = _JSON_CACHE.get(cache_key)
    if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
        _JSON_CACHE.move_to_end(cache_key)
        return cached[2]

    # 缓存未命中 — 先驱逐旧条目（如果 key 已存在则先减去旧 size）
    if cache_key in _JSON_CACHE:
        _json_cache_bytes -= _JSON_CACHE[cache_key][1]
        del _JSON_CACHE[cache_key]

    raw = fp.read_bytes()
    data = _loads_json(raw)
    _JSON_CACHE[cache_key] = (stat.st_mtime_ns, stat.st_size, data)
    _json_cache_bytes += stat.st_size
    _JSON_CACHE.move_to_end(cache_key)

    # 驱逐：条目数超限 或 总大小超限
    while _JSON_CACHE and (len(_JSON_CACHE) > _MAX_JSON_LRU or _json_cache_bytes > _MAX_JSON_CACHE_BYTES):
        _, (_, old_size, _) = _JSON_CACHE.popitem(last=False)
        _json_cache_bytes -= old_size

    return data


def _preview_json(data: Any, *, depth: int = _MAX_JSON_PREVIEW_DEPTH, items: int = _MAX_JSON_PREVIEW_ITEMS) -> Any:
    """对大 JSON 值做结构化预览，防止工具返回过长内容。"""
    if depth <= 0:
        return f"<{type(data).__name__}>"
    if isinstance(data, dict):
        out: dict[str, Any] = {}
        for i, (k, v) in enumerate(data.items()):
            if i >= items:
                out["..."] = f"+{len(data) - items} keys"
                break
            out[str(k)] = _preview_json(v, depth=depth - 1, items=items)
        return out
    if isinstance(data, list):
        preview = [_preview_json(v, depth=depth - 1, items=items) for v in data[:items]]
        if len(data) > items:
            preview.append(f"... +{len(data) - items} items")
        return preview
    text = str(data)
    text, truncated = _truncate_text(text, _MAX_JSON_VALUE_CHARS)
    if truncated:
        return f"{text}... (value truncated)"
    return data


def _parse_json_path(path: str) -> list[str | int]:
    """解析简化 JSON Path:
    - 支持: $.a.b[0].c
    - 支持: entries[*].indicator
    - 不支持复杂过滤表达式
    """
    p = (path or "").strip()
    if not p or p in {"$", "."}:
        return []
    if p.startswith("$"):
        p = p[1:]
    if p.startswith("."):
        p = p[1:]

    tokens: list[str | int] = []
    buf: list[str] = []
    i = 0
    while i < len(p):
        ch = p[i]
        if ch == ".":
            if buf:
                tokens.append("".join(buf))
                buf = []
            i += 1
            continue
        if ch == "[":
            if buf:
                tokens.append("".join(buf))
                buf = []
            j = p.find("]", i + 1)
            if j == -1:
                raise ValueError(f"JSON path 语法错误，缺少 ']': {path}")
            inner = p[i + 1 : j].strip()
            if inner == "*":
                tokens.append("*")
            elif inner.lstrip("-").isdigit():
                tokens.append(int(inner))
            elif (inner.startswith('"') and inner.endswith('"')) or (inner.startswith("'") and inner.endswith("'")):
                tokens.append(inner[1:-1])
            else:
                raise ValueError(f"JSON path 不支持的索引: [{inner}]")
            i = j + 1
            continue
        buf.append(ch)
        i += 1
    if buf:
        tokens.append("".join(buf))
    return tokens


def _query_json_path(
    data: Any,
    tokens: list[str | int],
    *,
    max_results: int = _MAX_JSON_PATH_RESULTS,
    offset: int = 0,
) -> tuple[list[tuple[str, Any]], int]:
    """按路径 token 查询 JSON，支持数组下标与通配符 *。返回 (matches, total)。"""
    nodes: list[tuple[str, Any]] = [("$", data)]
    for tok in tokens:
        next_nodes: list[tuple[str, Any]] = []
        for base_path, value in nodes:
            if tok == "*":
                if isinstance(value, list):
                    for i, item in enumerate(value[:_MAX_JSON_PATH_NODE_EXPAND]):
                        next_nodes.append((f"{base_path}[{i}]", item))
                elif isinstance(value, dict):
                    for i, (k, v) in enumerate(value.items()):
                        if i >= _MAX_JSON_PATH_NODE_EXPAND:
                            break
                        next_nodes.append((f"{base_path}.{k}", v))
                continue

            if isinstance(tok, int):
                if isinstance(value, list):
                    idx = tok if tok >= 0 else len(value) + tok
                    if 0 <= idx < len(value):
                        next_nodes.append((f"{base_path}[{idx}]", value[idx]))
                continue

            if isinstance(value, dict) and tok in value:
                next_nodes.append((f"{base_path}.{tok}", value[tok]))
                continue

            # 允许在 list[dict] 上投影键名: entries.indicator 等价 entries[*].indicator
            if isinstance(value, list):
                for i, item in enumerate(value[:_MAX_JSON_PATH_NODE_EXPAND]):
                    if isinstance(item, dict) and tok in item:
                        next_nodes.append((f"{base_path}[{i}].{tok}", item[tok]))

        if not next_nodes:
            return [], 0
        nodes = next_nodes[: (offset + max_results) * 6]
    total = len(nodes)
    return nodes[offset : offset + max_results], total


def _build_file_summary(filename: str, data: Any) -> dict[str, Any]:
    """构建 JSON 文件结构摘要。"""
    summary: dict[str, Any] = {"file": filename, "root_type": type(data).__name__}
    if isinstance(data, dict):
        summary["top_keys"] = list(data.keys())[:20]
        summary["key_count"] = len(data)
    elif isinstance(data, list):
        summary["length"] = len(data)
        if data:
            summary["item_type"] = type(data[0]).__name__
            summary["item_preview"] = _preview_json(data[0], depth=2, items=6)
    summary["hint"] = (
        "建议优先使用 lookup_indicator/lookup_event 做结构化检索；"
        "或使用 query_json(filename, where_key/where_value 或 path) 精确读取字段"
    )
    return summary


def _extract_fields(item: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    """按字段列表提取 dict 子集，不存在字段返回 None。"""
    return {f: item.get(f) for f in fields}


def _query_json_by_filter(
    data: Any,
    *,
    where_key: str,
    where_value: str,
    select_fields: list[str] | None = None,
    max_results: int = _MAX_JSON_PATH_RESULTS,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """在 list[dict] 或 dict[list[dict]] 中按键值过滤。返回 (results, total_matches)。"""
    where_key = (where_key or "").strip()
    where_value = (where_value or "").strip()
    if not where_key:
        return [], 0

    candidates: list[dict[str, Any]] = []
    # root list[dict]
    if isinstance(data, list):
        for item in data[:_MAX_JSON_FILTER_SCAN]:
            if isinstance(item, dict):
                candidates.append(item)
    # root dict: 取一层 list[dict] 候选（例如 timeline.entries）
    elif isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                for item in v[:_MAX_JSON_FILTER_SCAN]:
                    if isinstance(item, dict):
                        candidates.append(item)

    matched: list[dict[str, Any]] = []
    expected = where_value.casefold()
    for item in candidates:
        values: list[str] = []
        for key in (where_key, *_FILTER_KEY_ALIASES.get(where_key, ())):
            if key not in item:
                continue
            value = item[key]
            if where_key == "date":
                values.append(_date_prefix(value))
            else:
                values.append("" if value is None else str(value))

        if not values:
            continue
        if expected and not any(expected in actual.casefold() for actual in values):
            continue
        matched.append(item)

    total = len(matched)
    page = matched[offset : offset + max_results]
    results: list[dict[str, Any]] = []
    for item in page:
        if select_fields:
            results.append(_extract_fields(item, select_fields))
        else:
            results.append(_preview_json(item, depth=3, items=6))
    return results, total


def _search_json(
    data: Any,
    keyword: str,
    *,
    base_path: str = "$",
    max_hits: int = _MAX_JSON_SEARCH_HITS,
    max_nodes: int = _MAX_JSON_SEARCH_NODES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """在 JSON 中做结构化搜索，返回 (hits, scan_meta)。

    scan_meta 用于把工具的实际扫描状态透明返回给调用方：
    - truncated_lists: 因 _MAX_JSON_SEARCH_LIST_SCAN 被截断的 list 路径列表
    - nodes_budget_exhausted: True 时节点预算耗尽，"未命中" 可能是假阴性
    - hit_cap_reached: True 时命中数达 max_hits，可能存在更多匹配
    """
    scan_meta: dict[str, Any] = {
        "truncated_lists": [],
        "nodes_budget_exhausted": False,
        "hit_cap_reached": False,
    }
    kw = keyword.casefold().strip()
    if not kw:
        return [], scan_meta

    hits: list[dict[str, Any]] = []
    visited = 0
    stack: deque[tuple[str, Any]] = deque([(base_path, data)])

    while stack and len(hits) < max_hits and visited < max_nodes:
        path, value = stack.pop()
        visited += 1

        if isinstance(value, dict):
            for k, v in value.items():
                child_path = f"{path}.{k}"
                if kw in str(k).casefold() and len(hits) < max_hits:
                    hits.append(
                        {
                            "path": child_path,
                            "match": "key",
                            "value_preview": _preview_json(v, depth=2, items=4),
                        }
                    )
                stack.append((child_path, v))
            continue

        if isinstance(value, list):
            remaining_budget = max_nodes - visited
            max_scan = min(len(value), _MAX_JSON_SEARCH_LIST_SCAN, max(remaining_budget, 0))
            if max_scan < len(value):
                scan_meta["truncated_lists"].append(
                    {"path": path, "scanned": max_scan, "total": len(value)}
                )
            for i, item in enumerate(value[:max_scan]):
                stack.append((f"{path}[{i}]", item))
            continue

        text = str(value)
        if kw in text.casefold():
            text, truncated = _truncate_text(text, _MAX_JSON_VALUE_CHARS)
            if truncated:
                text = f"{text}... (value truncated)"
            hits.append({"path": path, "match": "value", "value_preview": text})

    if len(hits) >= max_hits:
        scan_meta["hit_cap_reached"] = True
    if visited >= max_nodes:
        scan_meta["nodes_budget_exhausted"] = True

    return hits, scan_meta


def _norm_text(value: Any) -> str:
    """统一文本归一化（用于大小写不敏感匹配）。"""
    return "" if value is None else str(value).casefold()


def _contains(haystack: Any, needle: str) -> bool:
    """大小写不敏感包含匹配。"""
    n = (needle or "").strip().casefold()
    if not n:
        return True
    return n in _norm_text(haystack)


def _match_indicator_fields(query: str, *values: Any) -> bool:
    """匹配指标名/键名，兼容大小写、连字符和缩写。"""
    q = (query or "").strip()
    if not q:
        return True

    normalized = "".join(ch for ch in q.casefold() if ch.isalnum())
    for value in values:
        text = "" if value is None else str(value)
        if _contains(text, q):
            return True
        compact = "".join(ch for ch in text.casefold() if ch.isalnum())
        if normalized and normalized in compact:
            return True
    return False


def _date_prefix(value: Any) -> str:
    """将日期/时间统一裁剪到 YYYY-MM-DD。"""
    text = "" if value is None else str(value).strip()
    return text[:10] if len(text) >= 10 else text


def _ensure_utf8_sql(sql_clean: str) -> str | None:
    """校验 SQL 为干净的 UTF-8；不干净则返回 hint，否则返回 None。

    模型写出的 SQL 若含被截断的多字节字符（孤立代理项等），在 con.execute
    时会触发 UnicodeEncodeError。此处提前拦截，便于在不依赖 DB / @tool /
    runtime 图的情况下单元测试。
    """
    try:
        sql_clean.encode("utf-8")
        return None
    except UnicodeError:
        return _UNICODE_SQL_HINT


def _is_unicode_error(e: BaseException) -> bool:
    """判断异常是否为（编解码或 DuckDB 报出的）非法 Unicode 错误。"""
    return (
        isinstance(e, UnicodeError)
        or "Invalid unicode" in str(e)
        or "byte sequence mismatch" in str(e)
    )


def _execute_duckdb_with_timeout(
    con: Any,
    sql: str,
    *,
    timeout_seconds: float = _MAX_QUERY_SECONDS,
) -> tuple[list[str], list[tuple[Any, ...]]]:
    """在超时控制下执行 DuckDB SQL，超时后主动 interrupt。"""

    def _run() -> tuple[list[str], list[tuple[Any, ...]]]:
        cur = con.execute(sql)
        columns = [d[0] for d in (cur.description or [])]
        rows = cur.fetchall()
        return columns, rows

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError as exc:
            try:
                con.interrupt()
            except Exception:
                pass
            raise TimeoutError(
                f"SQL 执行超时（>{timeout_seconds:.1f}s），请添加更精确的 WHERE 条件或缩小时间范围"
            ) from exc


def _render_query_table(columns: list[str], rows: list[tuple[Any, ...]]) -> tuple[str, bool]:
    """将 SQL 结果渲染为文本表格，返回 (文本, 是否有单元格截断)。"""
    if not columns:
        return "查询执行成功（无可显示列）", False

    cell_truncated = False

    def _cell(value: Any) -> str:
        nonlocal cell_truncated
        text = "NULL" if value is None else str(value)
        clipped, truncated = _truncate_text(text, _MAX_QUERY_CELL_CHARS)
        if truncated:
            cell_truncated = True
            return f"{clipped}... (cell truncated, total {len(text)} chars)"
        return clipped

    rendered_rows: list[list[str]] = []
    for row in rows:
        rendered_rows.append([_cell(v) for v in row])

    widths: list[int] = []
    for i, col in enumerate(columns):
        max_cell = max((len(r[i]) for r in rendered_rows), default=0)
        widths.append(min(max(len(col), max_cell), _MAX_QUERY_CELL_CHARS + 40))

    header = " | ".join(col.ljust(widths[i]) for i, col in enumerate(columns))
    sep = "-+-".join("-" * widths[i] for i in range(len(columns)))
    body = [" | ".join(r[i].ljust(widths[i]) for i in range(len(columns))) for r in rendered_rows]
    return "\n".join([header, sep, *body]), cell_truncated


# ==================== 工具 ====================


@tool(description="列出用户数据目录下的 JSON 文件名与大小")
def list_files(runtime: ToolRuntime[ToolContext]) -> str:
    """列出用户数据目录下的 JSON 文件名与大小。"""
    ctx = _ctx(runtime)
    if not ctx.data_dir.is_dir():
        return f"错误: 用户数据目录不存在 ({ctx.user_dir_name})"
    files = []
    for f in sorted(ctx.data_dir.iterdir()):
        if f.is_file() and f.suffix in _ALLOWED_EXTENSIONS and f.name not in _BLOCKED_FILES and not f.name.startswith(_BLOCKED_PREFIXES):
            size_kb = f.stat().st_size / 1024
            tag = " [推荐 lookup_* 或 query_json/search_file]" if size_kb > 100 else ""
            files.append(f"{f.name} ({size_kb:.1f} KB){tag}")
    return "\n".join(files) if files else "目录为空或无可读文件"


@tool(
    description=(
        "读取 JSON 文件。支持 path（简化 JSON Path，如 $.entries[0].indicator 或 entries[*].value）。"
        "不传 path 时返回结构摘要；传 path 时返回该路径的值（自动预览/截断）。"
    )
)
def read_file(
    filename: str,
    path: str = "",
    max_items: int = _MAX_JSON_PREVIEW_ITEMS,
    runtime: ToolRuntime[ToolContext] = None,
) -> str:
    """读取 JSON 文件并返回摘要或路径结果。"""
    ctx = _ctx(runtime)
    fp, err = _resolve_json_file(ctx, filename)
    if err:
        return err

    try:
        data = _load_json_cached(fp)
    except Exception as e:
        return f"错误: 解析 JSON 失败: {e}"

    if not path:
        summary = _build_file_summary(filename, data)
        rendered = _dumps_json(summary, indent=True)
        rendered, _ = _render_truncated(rendered, _MAX_READ_FILE_CHARS)
        return rendered

    try:
        tokens = _parse_json_path(path)
    except ValueError as e:
        return f"错误: {e}"

    matches, _ = _query_json_path(data, tokens, max_results=max(1, min(max_items, _MAX_JSON_PATH_RESULTS)))
    if not matches:
        if tokens and isinstance(tokens[0], int) and isinstance(data, list):
            return f"未找到路径: {path}（根数组长度={len(data)}）"
        return f"未找到路径: {path}"

    if len(matches) == 1:
        _, value = matches[0]
        rendered = _dumps_json(_preview_json(value, items=max_items), indent=True)
        rendered, _ = _render_truncated(rendered, _MAX_READ_FILE_CHARS)
        return rendered

    payload = [{"path": p, "value": _preview_json(v, items=max_items)} for p, v in matches]
    rendered = _dumps_json(payload, indent=True)
    rendered, _ = _render_truncated(rendered, _MAX_READ_FILE_CHARS)
    return rendered


@tool(
    description=(
        "查询 JSON（推荐）。支持两种方式：\n"
        "0) jmes 模式：传 jmes（JMESPath 表达式，若环境已安装 jmespath）\n"
        "1) path 模式：传 path（简化 JSONPath，支持负索引，如 $[-1].date）\n"
        "2) filter 模式：传 where_key + where_value，按键值过滤 list[dict]，可配合 select 输出字段\n"
        "path/filter 模式均支持 offset 翻页：结果含 total_matches；若有更多，返回 has_more + next_offset。\n"
        "示例:\n"
        "- query_json('timeline.json', where_key='entry_type', where_value='event')\n"
        "- query_json('timeline.json', path='$.entries[0]')\n"
        "- query_json('exam_data.json', path='$[0].indicators')"
    )
)
def query_json(
    filename: str,
    jmes: str = "",
    path: str = "",
    where_key: str = "",
    where_value: str = "",
    select: str = "",
    max_results: int = _MAX_JSON_PATH_RESULTS,
    offset: int = 0,
    runtime: ToolRuntime[ToolContext] = None,
) -> str:
    """查询 JSON（path 或 filter 模式）。"""
    ctx = _ctx(runtime)
    fp, err = _resolve_json_file(ctx, filename)
    if err:
        return err

    try:
        data = _load_json_cached(fp)
    except Exception as e:
        return f"错误: {e}"

    cap = max(1, min(max_results, _MAX_JSON_PATH_RESULTS))

    # jmes 模式（成熟表达式）
    if jmes.strip():
        if jmespath is None:
            return (
                "错误: 当前环境未安装 jmespath。"
                "可改用 where_key/where_value 过滤模式，或安装 jmespath 后使用 jmes 表达式。"
            )
        try:
            result = jmespath.search(jmes, data)
        except Exception as e:
            return f"错误: jmes 表达式执行失败: {e}"
        rendered = _dumps_json(
            {
                "mode": "jmes",
                "expression": jmes,
                "result": _preview_json(result, depth=3, items=6),
            },
            indent=True,
        )
        rendered, _ = _render_truncated(rendered, _MAX_READ_FILE_CHARS)
        return rendered

    # filter 模式（更适合 LLM，避免猜索引）
    if where_key.strip():
        select_fields = [s.strip() for s in select.split(",") if s.strip()] if select else None
        page_offset = max(0, offset)
        filtered, total = _query_json_by_filter(
            data,
            where_key=where_key,
            where_value=where_value,
            select_fields=select_fields,
            max_results=cap,
            offset=page_offset,
        )
        if not filtered:
            return f"未找到匹配记录: where_key={where_key!r}, where_value={where_value!r}"
        payload: dict[str, Any] = {
            "mode": "filter",
            "where_key": where_key,
            "where_value": where_value,
            "count": len(filtered),
            "total_matches": total,
            "results": filtered,
        }
        if page_offset + len(filtered) < total:
            payload["has_more"] = True
            payload["next_offset"] = page_offset + len(filtered)
            payload["hint"] = f"还有 {total - page_offset - len(filtered)} 条结果，使用 offset={payload['next_offset']} 获取下一页"
        rendered = _dumps_json(payload, indent=True)
        rendered, _ = _render_truncated(rendered, _MAX_READ_FILE_CHARS)
        return rendered

    # path 模式
    if not path.strip():
        return "错误: 需要提供 path，或使用 where_key/where_value 过滤模式"
    try:
        tokens = _parse_json_path(path)
    except Exception as e:
        return f"错误: path 解析失败: {e}"

    page_offset = max(0, offset)
    matches, total = _query_json_path(data, tokens, max_results=cap, offset=page_offset)
    if not matches:
        if tokens and isinstance(tokens[0], int) and isinstance(data, list):
            return f"未找到路径: {path}（根数组长度={len(data)}）"
        return f"未找到路径: {path}"

    items = [{"path": p, "value": _preview_json(v)} for p, v in matches]
    result_payload: dict[str, Any] = {"mode": "path", "count": len(items), "total_matches": total, "results": items}
    if page_offset + len(items) < total:
        result_payload["has_more"] = True
        result_payload["next_offset"] = page_offset + len(items)
        result_payload["hint"] = f"还有 {total - page_offset - len(items)} 条结果，使用 offset={result_payload['next_offset']} 获取下一页"
    rendered = _dumps_json(result_payload, indent=True)
    rendered, _ = _render_truncated(rendered, _MAX_READ_FILE_CHARS)
    return rendered


@tool(
    description=(
        "在指定 JSON 文件中做结构化关键词检索，返回命中 path + 值片段。"
        "可选 path 用于先定位子树再检索（例如 '$.entries'）。"
    )
)
def search_file(
    filename: str,
    keyword: str,
    path: str = "$",
    max_hits: int = _MAX_JSON_SEARCH_HITS,
    runtime: ToolRuntime[ToolContext] = None,
) -> str:
    """在 JSON 文件中做结构化关键词检索。"""
    ctx = _ctx(runtime)
    fp, err = _resolve_json_file(ctx, filename)
    if err:
        return err
    if not keyword.strip():
        return "错误: keyword 不能为空"

    try:
        data = _load_json_cached(fp)
    except Exception as e:
        return f"错误: 解析 JSON 失败: {e}"

    base_data = data
    base_path = "$"
    if path and path not in {"$", "."}:
        try:
            tokens = _parse_json_path(path)
            matched, _ = _query_json_path(data, tokens, max_results=1)
        except Exception as e:
            return f"错误: path 解析失败: {e}"
        if not matched:
            return f"未找到路径: {path}"
        base_path, base_data = matched[0]

    effective_max_hits = max(1, min(max_hits, _MAX_JSON_SEARCH_HITS))
    hits, scan_meta = _search_json(
        base_data,
        keyword,
        base_path=base_path,
        max_hits=effective_max_hits,
    )

    truncation_notes: list[str] = []
    for tl in scan_meta["truncated_lists"][:2]:
        truncation_notes.append(f"{tl['path']} 仅扫描前 {tl['scanned']}/{tl['total']} 条")
    if scan_meta["nodes_budget_exhausted"]:
        truncation_notes.append(f"总扫描节点数达上限 {_MAX_JSON_SEARCH_NODES}")
    truncation_text = "；".join(truncation_notes)

    if not hits:
        if truncation_text:
            return (
                f"未找到匹配内容（搜索范围受限：{truncation_text}；"
                "此为假阴性可能；建议改用 query_json 的 jmes/path/filter 模式定位子树后再搜索）"
            )
        return "未找到匹配内容"

    result: dict[str, Any] = {
        "file": filename,
        "keyword": keyword,
        "base_path": base_path,
        "hits": hits,
    }
    notes = list(truncation_notes)
    if scan_meta["hit_cap_reached"]:
        notes.append(f"命中数达上限 {effective_max_hits}，可能存在更多匹配（缩小关键词或加 path 子树定位）")
    if notes:
        result["scan_note"] = "；".join(notes)
    rendered = _dumps_json(result, indent=True)
    rendered, _ = _render_truncated(rendered, _MAX_SEARCH_OUTPUT_CHARS)
    return rendered


@tool(
    description=(
        "高召回指标检索工具（推荐优先于手写 SQL/路径）：按指标名（可模糊）和可选日期检索。"
        "source 可选 auto/device/exam。返回结构化结果（date, indicator_name, value, unit 等）。"
        "结果包含 total_matches 总匹配数；若有更多结果，返回 has_more=true 和 next_offset，传入 offset 翻页。"
    )
)
def lookup_indicator(
    indicator: str,
    date: str = "",
    source: str = "auto",
    max_results: int = 8,
    offset: int = 0,
    runtime: ToolRuntime[ToolContext] = None,
) -> str:
    """按指标关键词 + 可选日期检索 device/exam 数据。支持 offset 翻页。"""
    if not indicator.strip():
        return "错误: indicator 不能为空"
    source = (source or "auto").strip().lower()
    if source not in {"auto", "device", "exam"}:
        return "错误: source 仅支持 auto/device/exam"

    ctx = _ctx(runtime)
    limit = max(1, min(max_results, _MAX_LOOKUP_RESULTS))
    page_offset = max(0, offset)
    date = (date or "").strip()
    kw = indicator.strip()
    matched: list[dict[str, Any]] = []

    # timeline.json（真实 device 指标流）
    if source in {"auto", "device"}:
        fp = ctx.data_dir / "timeline.json"
        if fp.exists():
            try:
                timeline = _load_json_cached(fp)
                entries = timeline.get("entries") if isinstance(timeline, dict) else timeline
                if isinstance(entries, list):
                    for item in entries:
                        if not isinstance(item, dict) or item.get("entry_type") != "device_indicator":
                            continue
                        item_date = _date_prefix(item.get("time"))
                        if date and item_date != date:
                            continue
                        indicator_name = item.get("indicator")
                        if not _match_indicator_fields(kw, indicator_name):
                            continue
                        matched.append(
                            {
                                "source": "device",
                                "date": item_date,
                                "device_type": item.get("device_type"),
                                "indicator_name": indicator_name,
                                "indicator_key": indicator_name,
                                "value": item.get("value"),
                                "unit": item.get("unit"),
                                "timestamp": item.get("time"),
                            }
                        )
            except Exception:
                logger.warning("[lookup_indicator] 读取 timeline.json 失败", exc_info=True)

    # exam_data.json
    if source in {"auto", "exam"}:
        fp = ctx.data_dir / "exam_data.json"
        if fp.exists():
            try:
                exam_data = _load_json_cached(fp)
                if isinstance(exam_data, list):
                    for rec in exam_data:
                        if not isinstance(rec, dict):
                            continue
                        rec_date = str(rec.get("exam_date") or rec.get("date") or "")
                        if date and rec_date != date:
                            continue
                        indicators = rec.get("indicators")
                        if isinstance(indicators, dict):
                            indicator_items = indicators.items()
                        elif isinstance(indicators, list):
                            indicator_items = [
                                (item.get("indicator_name") or item.get("indicator_key"), item)
                                for item in indicators
                                if isinstance(item, dict)
                            ]
                        else:
                            continue
                        for indicator_key, item in indicator_items:
                            if not isinstance(item, dict):
                                continue
                            indicator_name = item.get("indicator_name") or indicator_key
                            raw_key = item.get("indicator_key") or indicator_key
                            if not _match_indicator_fields(kw, indicator_name, raw_key, indicator_key):
                                continue
                            matched.append(
                                {
                                    "source": "exam",
                                    "date": rec_date,
                                    "exam_type": rec.get("exam_type"),
                                    "exam_location": rec.get("exam_location"),
                                    "indicator_name": indicator_name,
                                    "indicator_key": raw_key,
                                    "status": item.get("status"),
                                    "value": item.get("value"),
                                    "unit": item.get("unit"),
                                    "reference_range": item.get("reference_range"),
                                    "timestamp": item.get("timestamp"),
                                }
                            )
            except Exception:
                logger.warning("[lookup_indicator] 读取 exam_data.json 失败", exc_info=True)

    total = len(matched)
    if total == 0:
        hint = (
            "未找到匹配指标。可尝试："
            "1) 放宽 indicator 关键词；"
            "2) 去掉 date 仅按指标检索；"
            "3) 用 query_duckdb 做聚合/排序。"
        )
        return hint

    hits = matched[page_offset : page_offset + limit]
    result: dict[str, Any] = {
        "mode": "lookup_indicator",
        "indicator": indicator,
        "date": date or None,
        "source": source,
        "count": len(hits),
        "total_matches": total,
        "results": hits,
    }
    if page_offset + len(hits) < total:
        result["has_more"] = True
        result["next_offset"] = page_offset + len(hits)
        result["hint"] = f"还有 {total - page_offset - len(hits)} 条结果，使用 offset={result['next_offset']} 获取下一页"
    rendered = _dumps_json(result, indent=True)
    rendered, _ = _render_truncated(rendered, _MAX_READ_FILE_CHARS)
    return rendered


@tool(
    description=(
        "高召回事件检索工具：按 event_name（可模糊）和可选日期检索 events.json，"
        "同时会匹配 description 与 medications.name 文本，"
        "返回事件核心字段（event_name/start_date/duration_days/medications）。"
        "结果包含 total_matches 总匹配数；若有更多结果，返回 has_more=true 和 next_offset，传入 offset 翻页。"
    )
)
def lookup_event(
    event_name: str = "",
    date: str = "",
    max_results: int = 8,
    offset: int = 0,
    runtime: ToolRuntime[ToolContext] = None,
) -> str:
    """按事件名/日期检索 events.json。支持 offset 翻页。"""
    if not event_name.strip() and not date.strip():
        return "错误: event_name 和 date 至少提供一个"

    ctx = _ctx(runtime)
    fp = ctx.data_dir / "events.json"
    if not fp.exists():
        return "错误: events.json 不存在"

    try:
        data = _load_json_cached(fp)
    except Exception as e:
        return f"错误: 解析 events.json 失败: {e}"

    if not isinstance(data, list):
        return "错误: events.json 结构异常（预期 list）"

    limit = max(1, min(max_results, _MAX_LOOKUP_RESULTS))
    page_offset = max(0, offset)
    event_name = event_name.strip()
    date = date.strip()

    # 先收集所有匹配项
    matched: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if event_name:
            medication_blob = ""
            medications = item.get("medications")
            if medications:
                medication_blob = _dumps_json(medications, indent=False)
            event_match = (
                _contains(item.get("event_name"), event_name)
                or _contains(item.get("description"), event_name)
                or _contains(medication_blob, event_name)
            )
            if not event_match:
                continue
        if date:
            date_ok = (
                _contains(item.get("start_date"), date)
                or _contains(item.get("time"), date)
                or _contains(item.get("end_time"), date)
            )
            if not date_ok:
                continue
        matched.append(item)

    total = len(matched)
    if total == 0:
        return "未找到匹配事件。可尝试放宽 event_name 关键词或仅用 date 检索。"

    # 翻页截取
    page = matched[page_offset : page_offset + limit]
    hits: list[dict[str, Any]] = []
    for item in page:
        # affected_indicators: 提取 indicator_name 列表（比嵌套预览更有用）
        ai_raw = item.get("affected_indicators")
        if isinstance(ai_raw, list):
            ai_names = [i.get("indicator_name", i.get("indicator_key", "?")) for i in ai_raw if isinstance(i, dict)]
            ai_summary = {"count": len(ai_names), "indicator_names": ai_names}
        else:
            ai_summary = ai_raw

        hits.append(
            {
                "event_id": item.get("event_id"),
                "event_name": item.get("event_name"),
                "event_type": item.get("event_type"),
                "start_date": item.get("start_date"),
                "duration_days": item.get("duration_days"),
                "interrupted": item.get("interrupted"),
                "interruption_date": item.get("interruption_date"),
                "medications": _preview_json(item.get("medications"), depth=3, items=6),
                "affected_indicators": ai_summary,
            }
        )

    result: dict[str, Any] = {
        "mode": "lookup_event",
        "event_name": event_name or None,
        "date": date or None,
        "count": len(hits),
        "total_matches": total,
        "results": hits,
    }
    if page_offset + len(hits) < total:
        result["has_more"] = True
        result["next_offset"] = page_offset + len(hits)
        result["hint"] = f"还有 {total - page_offset - len(hits)} 条结果，使用 offset={result['next_offset']} 获取下一页"
    rendered = _dumps_json(result, indent=True)
    rendered, _ = _render_truncated(rendered, _MAX_READ_FILE_CHARS)
    return rendered


def _split_terms(raw: str) -> list[str]:
    parts = []
    for chunk in (raw or "").replace("\n", ",").replace(";", ",").split(","):
        text = chunk.strip()
        if text:
            parts.append(text)
    return parts


def _load_events(ctx: ToolContext) -> list[dict[str, Any]]:
    # 禁止使用 events.json 中的事件-指标影响信息
    return []


def _event_matches_query(item: dict[str, Any], query: str) -> bool:
    medication_blob = ""
    medications = item.get("medications")
    if medications:
        medication_blob = _dumps_json(medications, indent=False)
    return (
        _contains(item.get("event_name"), query)
        or _contains(item.get("description"), query)
        or _contains(medication_blob, query)
    )


def _iter_matching_events(ctx: ToolContext, event_name: str = "", date: str = "") -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    for item in _load_events(ctx):
        if event_name and not _event_matches_query(item, event_name):
            continue
        if date:
            date_ok = (
                _contains(item.get("start_date"), date)
                or _contains(item.get("time"), date)
                or _contains(item.get("end_time"), date)
            )
            if not date_ok:
                continue
        matched.append(item)
    return matched


def _pick_latest_events(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(matches) <= 1:
        return matches
    ordered = sorted(matches, key=lambda item: str(item.get("start_date") or ""))
    return [ordered[-1]]


def _event_interval(item: dict[str, Any]) -> tuple[datetime | None, datetime | None]:
    start_date = item.get("start_date")
    if not start_date:
        return None, None
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
    except Exception:
        return None, None

    try:
        duration_days = int(item.get("duration_days") or 0)
    except Exception:
        duration_days = 0
    if duration_days <= 0:
        end = start
    else:
        end = start + timedelta(days=duration_days - 1)

    interruption_date = item.get("interruption_date")
    if interruption_date:
        try:
            interrupted_end = datetime.strptime(str(interruption_date), "%Y-%m-%d")
            if interrupted_end < end:
                end = interrupted_end
        except Exception:
            pass
    return start, end


def _indicator_names_on_event(item: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for indicator in item.get("affected_indicators") or []:
        if not isinstance(indicator, dict):
            continue
        names.append(str(indicator.get("indicator_name") or indicator.get("indicator_key") or ""))
    return [name for name in names if name]


@tool(
    description=(
        "按日期列出指标名（优先用于“哪些指标在某天被测量/记录”）。"
        "source 支持 auto/device/exam。返回唯一指标名列表与 count。"
    )
)
def list_indicators_on_date(
    date: str,
    source: str = "auto",
    runtime: ToolRuntime[ToolContext] = None,
) -> str:
    """列出某一天 device/exam 中出现过的唯一指标名。"""
    date = (date or "").strip()
    source = (source or "auto").strip().lower()
    if not date:
        return "错误: date 不能为空"
    if source not in {"auto", "device", "exam"}:
        return "错误: source 仅支持 auto/device/exam"

    ctx = _ctx(runtime)
    names: set[str] = set()

    if source in {"auto", "device"}:
        timeline_path = ctx.data_dir / "timeline.json"
        if timeline_path.exists():
            timeline = _load_json_cached(timeline_path)
            entries = timeline.get("entries") if isinstance(timeline, dict) else timeline
            if isinstance(entries, list):
                for item in entries:
                    if not isinstance(item, dict) or item.get("entry_type") != "device_indicator":
                        continue
                    if _date_prefix(item.get("time")) != date:
                        continue
                    indicator_name = item.get("indicator")
                    if indicator_name:
                        names.add(str(indicator_name))

    if source in {"auto", "exam"}:
        exam_path = ctx.data_dir / "exam_data.json"
        if exam_path.exists():
            exam_data = _load_json_cached(exam_path)
            if isinstance(exam_data, list):
                for rec in exam_data:
                    if not isinstance(rec, dict):
                        continue
                    rec_date = str(rec.get("exam_date") or rec.get("date") or "")
                    if rec_date != date:
                        continue
                    indicators = rec.get("indicators") or {}
                    if isinstance(indicators, dict):
                        for indicator_name, item in indicators.items():
                            if isinstance(item, dict):
                                names.add(str(item.get("indicator_name") or indicator_name))
                    elif isinstance(indicators, list):
                        for item in indicators:
                            if isinstance(item, dict):
                                indicator_name = item.get("indicator_name") or item.get("indicator_key")
                                if indicator_name:
                                    names.add(str(indicator_name))

    result = {
        "mode": "list_indicators_on_date",
        "date": date,
        "source": source,
        "count": len(names),
        "indicators": sorted(names),
    }
    rendered = _dumps_json(result, indent=True)
    rendered, _ = _render_truncated(rendered, _MAX_READ_FILE_CHARS)
    return rendered


@tool(
    description=(
        "统计某个指标被多少个事件影响（优先用于“How many events have affected indicator X?”）。"
        "可直接用题面中的指标名，如 NumberofAwakenings-NA / SleepLatency-SL。"
    )
)
def count_events_by_indicator(
    indicator: str,
    runtime: ToolRuntime[ToolContext] = None,
) -> str:
    """统计影响某指标的事件数。"""
    indicator = (indicator or "").strip()
    if not indicator:
        return "错误: indicator 不能为空"

    ctx = _ctx(runtime)
    matched_events: list[dict[str, Any]] = []
    for item in _load_events(ctx):
        for affected in item.get("affected_indicators") or []:
            if not isinstance(affected, dict):
                continue
            if _match_indicator_fields(indicator, affected.get("indicator_name"), affected.get("indicator_key")):
                matched_events.append(item)
                break

    result = {
        "mode": "count_events_by_indicator",
        "indicator": indicator,
        "count": len({item.get('event_id') or (item.get('event_name'), item.get('start_date')) for item in matched_events}),
        "matched_events_preview": [
            f"{item.get('event_name')} ({item.get('start_date')})"
            for item in sorted(matched_events, key=lambda x: (str(x.get("start_date") or ""), str(x.get("event_name") or "")))[:12]
        ],
    }
    return _dumps_json(result, indent=True)


@tool(
    description=(
        "统计某个药物相关的 health_event 数量（优先用于“What is the health-event count related to X medication/vaccine?”）。"
        "默认先做药物名精确匹配，若没有精确结果再回退到包含匹配。"
    )
)
def count_events_by_medication(
    medication: str,
    runtime: ToolRuntime[ToolContext] = None,
) -> str:
    """统计某药物关联的 health_event 数量。"""
    medication = (medication or "").strip()
    if not medication:
        return "错误: medication 不能为空"

    ctx = _ctx(runtime)
    exact_matches: list[dict[str, Any]] = []
    fuzzy_matches: list[dict[str, Any]] = []
    query_norm = medication.casefold()
    for item in _load_events(ctx):
        if item.get("event_type") != "health_event":
            continue
        meds = item.get("medications") or []
        for med in meds:
            if not isinstance(med, dict):
                continue
            name = str(med.get("name") or "")
            if not name:
                continue
            if name.casefold() == query_norm:
                exact_matches.append(item)
                break
            if _contains(name, medication):
                fuzzy_matches.append(item)
                break

    matched_events = exact_matches or fuzzy_matches
    result = {
        "mode": "count_events_by_medication",
        "medication": medication,
        "match_type": "exact" if exact_matches else "contains",
        "count": len({item.get('event_id') or (item.get('event_name'), item.get('start_date')) for item in matched_events}),
        "matched_events": [
            {
                "event_id": item.get("event_id"),
                "event_name": item.get("event_name"),
                "start_date": item.get("start_date"),
            }
            for item in sorted(matched_events, key=lambda x: (str(x.get("start_date") or ""), str(x.get("event_name") or "")))[:20]
        ],
    }
    return _dumps_json(result, indent=True)


@tool(
    description=(
        "统计某个事件影响了多少个指标（优先用于“How many indicators does event X affect?”）。"
        "date 可选，用于同名事件去歧义。"
    )
)
def count_event_indicators(
    event_name: str,
    date: str = "",
    runtime: ToolRuntime[ToolContext] = None,
) -> str:
    """统计事件影响指标数量。"""
    event_name = (event_name or "").strip()
    date = (date or "").strip()
    if not event_name:
        return "错误: event_name 不能为空"

    ctx = _ctx(runtime)
    matches = _iter_matching_events(ctx, event_name=event_name, date=date)
    if not matches:
        return "未找到匹配事件"
    if not date:
        matches = _pick_latest_events(matches)

    payload = []
    for item in matches[:8]:
        indicator_names = sorted(set(_indicator_names_on_event(item)))
        payload.append(
            {
                "event_id": item.get("event_id"),
                "event_name": item.get("event_name"),
                "start_date": item.get("start_date"),
                "indicator_count": len(indicator_names),
                "indicator_names": indicator_names,
            }
        )
    return _dumps_json(
        {
            "mode": "count_event_indicators",
            "event_name": event_name,
            "date": date or None,
            "count": len(payload),
            "results": payload,
        },
        indent=True,
    )


@tool(
    description=(
        "按 1 个或多个指标查共同相关事件（优先用于“Which events affect both A and B”/“jointly related to X and Y”）。"
        "indicators 用逗号分隔；mode=all 表示事件必须同时匹配所有指标，mode=any 表示匹配任一指标。"
    )
)
def find_events_by_indicators(
    indicators: str,
    mode: str = "all",
    max_results: int = 200,
    offset: int = 0,
    runtime: ToolRuntime[ToolContext] = None,
) -> str:
    """按多个指标查找相关事件。"""
    terms = _split_terms(indicators)
    mode = (mode or "all").strip().lower()
    if not terms:
        return "错误: indicators 不能为空"
    if mode not in {"all", "any"}:
        return "错误: mode 仅支持 all/any"

    ctx = _ctx(runtime)
    matched: list[dict[str, Any]] = []
    for item in _load_events(ctx):
        hits = set()
        for affected in item.get("affected_indicators") or []:
            if not isinstance(affected, dict):
                continue
            for term in terms:
                if _match_indicator_fields(term, affected.get("indicator_name"), affected.get("indicator_key")):
                    hits.add(term)
        if mode == "all" and len(hits) != len(terms):
            continue
        if mode == "any" and not hits:
            continue
        matched.append(
            {
                "event_id": item.get("event_id"),
                "event_name": item.get("event_name"),
                "start_date": item.get("start_date"),
                "event_type": item.get("event_type"),
                "matched_indicators": sorted(hits),
            }
        )

    matched.sort(key=lambda item: (str(item.get("start_date") or ""), str(item.get("event_name") or "")))
    total = len(matched)
    page = matched[max(0, offset) : max(0, offset) + max(1, min(max_results, 200))]
    result: dict[str, Any] = {
        "mode": "find_events_by_indicators",
        "indicators": terms,
        "match_mode": mode,
        "count": len(page),
        "total_matches": total,
        "results": page,
    }
    if offset + len(page) < total:
        result["has_more"] = True
        result["next_offset"] = offset + len(page)
    rendered = _dumps_json(result, indent=True)
    rendered, _ = _render_truncated(rendered, _MAX_READ_FILE_CHARS)
    return rendered


@tool(
    description=(
        "查找与某个事件时间区间重叠的其他事件（优先用于“which other events overlap with X”）。"
        "date 可选，用于同名事件去歧义。返回 event name + start date。"
    )
)
def find_overlapping_events(
    event_name: str,
    date: str = "",
    max_results: int = 200,
    offset: int = 0,
    runtime: ToolRuntime[ToolContext] = None,
) -> str:
    """查找与给定事件重叠的其他事件。"""
    event_name = (event_name or "").strip()
    date = (date or "").strip()
    if not event_name:
        return "错误: event_name 不能为空"

    ctx = _ctx(runtime)
    targets = _iter_matching_events(ctx, event_name=event_name, date=date)
    if not targets:
        return "未找到匹配事件"
    if not date:
        targets = _pick_latest_events(targets)

    overlaps: dict[tuple[str, str], dict[str, Any]] = {}
    all_events = _load_events(ctx)
    for target in targets:
        target_start, target_end = _event_interval(target)
        if not target_start or not target_end:
            continue
        target_key = (target.get("event_id"), target.get("start_date"))
        for item in all_events:
            item_key = (item.get("event_id"), item.get("start_date"))
            if item_key == target_key:
                continue
            item_start, item_end = _event_interval(item)
            if not item_start or not item_end:
                continue
            if item_start <= target_end and item_end >= target_start:
                overlaps[(str(item.get("event_name") or ""), str(item.get("start_date") or ""))] = {
                    "event_id": item.get("event_id"),
                    "event_name": item.get("event_name"),
                    "start_date": item.get("start_date"),
                    "event_type": item.get("event_type"),
                }

    ordered = sorted(overlaps.values(), key=lambda item: (str(item.get("start_date") or ""), str(item.get("event_name") or "")))
    total = len(ordered)
    page = ordered[max(0, offset) : max(0, offset) + max(1, min(max_results, 200))]
    result: dict[str, Any] = {
        "mode": "find_overlapping_events",
        "event_name": event_name,
        "date": date or None,
        "count": len(page),
        "total_matches": total,
        "results": page,
    }
    if offset + len(page) < total:
        result["has_more"] = True
        result["next_offset"] = offset + len(page)
    rendered = _dumps_json(result, indent=True)
    rendered, _ = _render_truncated(rendered, _MAX_READ_FILE_CHARS)
    return rendered


@tool(
    description=(
        "计算两个事件时间区间的重叠天数（优先用于“overlap length in days between event A and B”）。"
        "两个事件都可传可选 date 用于去歧义。重叠按自然日交集计算，含首尾。"
    )
)
def overlap_days_between_events(
    event_name_a: str,
    event_name_b: str,
    date_a: str = "",
    date_b: str = "",
    runtime: ToolRuntime[ToolContext] = None,
) -> str:
    """计算两个事件区间的重叠天数。"""
    event_name_a = (event_name_a or "").strip()
    event_name_b = (event_name_b or "").strip()
    date_a = (date_a or "").strip()
    date_b = (date_b or "").strip()
    if not event_name_a or not event_name_b:
        return "错误: event_name_a 和 event_name_b 不能为空"

    ctx = _ctx(runtime)
    matches_a = _iter_matching_events(ctx, event_name=event_name_a, date=date_a)
    matches_b = _iter_matching_events(ctx, event_name=event_name_b, date=date_b)
    if not matches_a or not matches_b:
        return "未找到匹配事件"
    if not date_a:
        matches_a = _pick_latest_events(matches_a)
    if not date_b:
        matches_b = _pick_latest_events(matches_b)

    results = []
    for item_a in matches_a[:8]:
        start_a, end_a = _event_interval(item_a)
        if not start_a or not end_a:
            continue
        for item_b in matches_b[:8]:
            start_b, end_b = _event_interval(item_b)
            if not start_b or not end_b:
                continue
            overlap_start = max(start_a, start_b)
            overlap_end = min(end_a, end_b)
            overlap_days = 0
            if overlap_start <= overlap_end:
                overlap_days = (overlap_end - overlap_start).days + 1
            results.append(
                {
                    "event_a": f"{item_a.get('event_name')} ({item_a.get('start_date')})",
                    "event_b": f"{item_b.get('event_name')} ({item_b.get('start_date')})",
                    "overlap_days": overlap_days,
                }
            )

    return _dumps_json(
        {
            "mode": "overlap_days_between_events",
            "event_name_a": event_name_a,
            "date_a": date_a or None,
            "event_name_b": event_name_b,
            "date_b": date_b or None,
            "results": results,
        },
        indent=True,
    )


def _cache_db_path(ctx: ToolContext) -> Path:
    return ctx.data_dir / f".holyeval_cache.v{_CACHE_DB_VERSION}.duckdb"


def _source_user_id(ctx: ToolContext) -> str:
    timeline_path = ctx.data_dir / "timeline.json"
    if timeline_path.exists():
        try:
            timeline = _load_json_cached(timeline_path)
            if isinstance(timeline, dict) and timeline.get("user_id"):
                return str(timeline["user_id"])
        except Exception:
            logger.warning("[retrieve] 读取 timeline.json user_id 失败", exc_info=True)
    return ctx.user_dir_name


def _compute_end_time(start_date: str | None, duration_days: Any) -> str | None:
    if not start_date:
        return None
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        days = int(duration_days)
    except Exception:
        return f"{start_date} 00:00:00"
    return (start + timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")


def _hydrate_device_indicators(con: Any, ctx: ToolContext) -> None:
    rows: list[tuple[Any, ...]] = []
    timeline_path = ctx.data_dir / "timeline.json"
    if timeline_path.exists():
        timeline = _load_json_cached(timeline_path)
        entries = timeline.get("entries") if isinstance(timeline, dict) else timeline
        user_id = _source_user_id(ctx)
        if isinstance(entries, list):
            for item in entries:
                if not isinstance(item, dict) or item.get("entry_type") != "device_indicator":
                    continue
                rows.append(
                    (
                        user_id,
                        item.get("time"),
                        item.get("indicator"),
                        item.get("device_type"),
                        None if item.get("value") is None else str(item.get("value")),
                        item.get("unit"),
                    )
                )

    con.execute("DROP TABLE IF EXISTS device_indicators")
    con.execute(
        """
        CREATE TABLE device_indicators(
            user_id VARCHAR,
            time TIMESTAMP,
            indicator VARCHAR,
            device_type VARCHAR,
            value VARCHAR,
            unit VARCHAR
        )
        """
    )
    if rows:
        con.executemany("INSERT INTO device_indicators VALUES (?, ?, ?, ?, ?, ?)", rows)


def _hydrate_exam_indicators(con: Any, ctx: ToolContext) -> None:
    rows: list[tuple[Any, ...]] = []
    exam_path = ctx.data_dir / "exam_data.json"
    if exam_path.exists():
        exam_data = _load_json_cached(exam_path)
        user_id = _source_user_id(ctx)
        if isinstance(exam_data, list):
            for rec in exam_data:
                if not isinstance(rec, dict):
                    continue
                rec_date = rec.get("exam_date") or rec.get("date")
                indicators = rec.get("indicators") or {}
                if not isinstance(indicators, dict):
                    continue
                for indicator_name, item in indicators.items():
                    if not isinstance(item, dict):
                        continue
                    timestamp = item.get("timestamp") or (f"{rec_date} 00:00:00" if rec_date else None)
                    rows.append(
                        (
                            user_id,
                            timestamp,
                            item.get("indicator_name") or indicator_name,
                            rec.get("exam_type"),
                            rec.get("exam_location"),
                            None if item.get("value") is None else str(item.get("value")),
                            item.get("unit"),
                        )
                    )

    con.execute("DROP TABLE IF EXISTS exam_indicators")
    con.execute(
        """
        CREATE TABLE exam_indicators(
            user_id VARCHAR,
            time TIMESTAMP,
            indicator VARCHAR,
            exam_type VARCHAR,
            exam_location VARCHAR,
            value VARCHAR,
            unit VARCHAR
        )
        """
    )
    if rows:
        con.executemany("INSERT INTO exam_indicators VALUES (?, ?, ?, ?, ?, ?, ?)", rows)


def _hydrate_events(con: Any, ctx: ToolContext) -> None:
    con.execute("DROP TABLE IF EXISTS events")


def _hydrate_event_indicators(con: Any, ctx: ToolContext) -> None:
    # 不暴露 expected_change/impact_level 等已剥离的语义关系列名
    con.execute("DROP TABLE IF EXISTS event_indicators")


def _hydrate_event_medications(con: Any, ctx: ToolContext) -> None:
    con.execute("DROP TABLE IF EXISTS event_medications")


def _ensure_hydrated_duckdb(ctx: ToolContext) -> Path:
    import shutil
    import tempfile

    # 数据没下载时早报错、并说清怎么办。
    #
    # 下面的 `mkstemp(dir=ctx.data_dir)` 在目录不存在时抛的是
    # `FileNotFoundError: .../user5026_AT_demo/.holyeval_tmp_xxxx.duckdb` ——
    # 一个关于临时文件的报错，指不到真实原因（这个用户的数据面还没拉下来）。
    # ESLBench 的数据不随仓库走，而 README 的命令清单里 eslbench 那条排在数据
    # 准备说明之前,所以这是新用户几乎一定会先撞上的一处。
    if not ctx.data_dir.is_dir():
        raise FileNotFoundError(
            f"用户数据目录不存在: {ctx.data_dir}\n"
            f"ESLBench 的数据不随仓库分发，先拉一次:\n"
            f"    python -m generator.eslbench.prepare_data"
        )

    cache_path = _cache_db_path(ctx)
    source_paths = [
        ctx.db_path,
        ctx.data_dir / "timeline.json",
        ctx.data_dir / "exam_data.json",
    ]
    newest_source = max((p.stat().st_mtime_ns for p in source_paths if p.exists()), default=0)
    cache_mtime = cache_path.stat().st_mtime_ns if cache_path.exists() else -1
    if cache_path.exists() and cache_mtime >= newest_source:
        return cache_path

    # 原子缓存更新：写临时文件 → rename，避免并发读写冲突
    import duckdb

    tmp_fd, tmp_path_str = tempfile.mkstemp(
        suffix=".duckdb", dir=str(ctx.data_dir), prefix=".holyeval_tmp_"
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_path_str)
    try:
        if ctx.db_path.exists():
            shutil.copy2(ctx.db_path, tmp_path)

        con = duckdb.connect(str(tmp_path))
        try:
            _hydrate_device_indicators(con, ctx)
            _hydrate_exam_indicators(con, ctx)
            _hydrate_events(con, ctx)
            _hydrate_event_indicators(con, ctx)
            _hydrate_event_medications(con, ctx)
        finally:
            con.close()

        # POSIX rename 是原子操作，不会影响正在读 cache_path 的连接
        tmp_path.rename(cache_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return cache_path


@tool(
    description=(
        "对用户 DuckDB 执行只读 SQL 查询（SELECT/CTE/DESCRIBE，最多返回 30 行，超时 5 秒）。\n"
        "注意：这是 DuckDB 语法（非 PostgreSQL），JSON 字段存为 VARCHAR 非 JSONB。\n\n"
        "== 表结构 ==\n"
        "device_indicators(user_id VARCHAR, time TIMESTAMP, indicator VARCHAR, device_type VARCHAR, value VARCHAR, unit VARCHAR)\n"
        "exam_indicators(user_id VARCHAR, time TIMESTAMP, indicator VARCHAR, exam_type VARCHAR, exam_location VARCHAR, value VARCHAR, unit VARCHAR)\n\n"
        "== DuckDB 语法提示 ==\n"
        "- 查看表结构: DESCRIBE device_indicators\n"
        "- 日期过滤: WHERE time::DATE = '2025-10-04'\n"
        "- 日期范围: WHERE start_date BETWEEN '2025-01-01' AND '2025-03-31'\n"
        "- 聚合: SELECT indicator, COUNT(*) FROM device_indicators GROUP BY indicator\n"
        "- 模糊匹配: WHERE event_name ILIKE '%menstruation%'\n"
        "- 注意: value 列是 VARCHAR，数值比较需 CAST(value AS DOUBLE)，可能含 NULL 值需用 WHERE value IS NOT NULL 过滤\n"
        "- 禁止: 不要使用 read_json_auto / read_csv 等文件函数，只能查询已建表。JSON 文件查询请用 query_json/read_file 工具"
    )
)
def query_duckdb(sql: str, runtime: ToolRuntime[ToolContext]) -> str:
    """对用户 DuckDB 执行只读 SQL。"""
    ctx = _ctx(runtime)
    hydrated_db = _ensure_hydrated_duckdb(ctx)
    if not hydrated_db.exists():
        return f"错误: 数据库文件不存在 ({hydrated_db.name})"
    sql_clean = sql.strip().rstrip(";")
    if not sql_clean:
        return "错误: SQL 不能为空"

    hint = _ensure_utf8_sql(sql_clean)
    if hint:
        return hint

    # Prefix check 前 strip 掉开头的 "-- " 注释行（合法 SQL 注释，不应触发拒绝）
    prefix_check = sql_clean
    while True:
        head = prefix_check.lstrip()
        if not head.startswith("--"):
            prefix_check = head
            break
        nl = head.find("\n")
        if nl < 0:
            prefix_check = ""
            break
        prefix_check = head[nl + 1 :]

    sql_up = sql_clean.upper()
    prefix_up = prefix_check.upper()
    _ALLOWED_PREFIXES = ("SELECT", "WITH", "DESCRIBE", "PRAGMA TABLE_INFO", "PRAGMA SHOW_TABLES")
    if not any(prefix_up.startswith(p) for p in _ALLOWED_PREFIXES):
        return "错误: 仅允许 SELECT/CTE/DESCRIBE 只读查询"

    for kw in _FORBIDDEN_SQL:
        if kw in sql_up:
            return f"错误: 不支持 {kw} 操作，只允许 SELECT 查询"

    for tbl in _BLOCKED_TABLES:
        if tbl.upper() in sql_up:
            return f"错误: 表 {tbl} 不可访问"
    import duckdb

    is_meta = sql_up.startswith("DESCRIBE") or sql_up.startswith("PRAGMA")
    limited_sql = sql_clean if is_meta else f"SELECT * FROM ({sql_clean}) AS _holyeval_q LIMIT {_MAX_QUERY_ROWS + 1}"

    try:
        con = duckdb.connect(str(hydrated_db), read_only=True)
        try:
            columns, rows = _execute_duckdb_with_timeout(con, limited_sql, timeout_seconds=_MAX_QUERY_SECONDS)
        finally:
            con.close()
    except Exception as e:
        if _is_unicode_error(e):
            return _UNICODE_SQL_HINT
        return f"SQL 执行错误: {e}"

    if not rows:
        return "查询返回空结果"

    rows_truncated = len(rows) > _MAX_QUERY_ROWS
    if rows_truncated:
        rows = rows[:_MAX_QUERY_ROWS]

    rendered, cell_truncated = _render_query_table(columns, rows)
    rendered, output_truncated = _render_truncated(
        rendered,
        _MAX_QUERY_OUTPUT_CHARS,
        note_prefix=f"output truncated to {_MAX_QUERY_OUTPUT_CHARS}",
    )

    notes: list[str] = []
    if rows_truncated:
        notes.append(f"rows truncated to {_MAX_QUERY_ROWS}")
    if cell_truncated:
        notes.append(f"long cell values clipped to {_MAX_QUERY_CELL_CHARS} chars")
    if output_truncated:
        notes.append("table text clipped for model context safety")
    if notes:
        rendered += "\n... (" + "; ".join(notes) + ")"

    logger.info(
        "[retrieve.query_duckdb] user=%s rows=%d cols=%d chars=%d notes=%s",
        ctx.user_dir_name,
        len(rows),
        len(columns),
        len(rendered),
        "; ".join(notes) if notes else "none",
    )

    return rendered


# ==================== 入口 ====================

TOOLS: list[BaseTool] = [
    list_files,
    read_file,
    query_json,
    search_file,
    lookup_indicator,
    list_indicators_on_date,
    query_duckdb,
]


def get_tools(**_) -> list[BaseTool]:
    """返回工具列表（ToolRuntime 由 langgraph ainvoke(context=...) 自动注入）。"""
    return TOOLS
