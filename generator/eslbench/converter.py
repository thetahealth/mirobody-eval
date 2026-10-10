"""
KG Evaluation Queries → mirobody-eval BenchItem 转换器

将 KG 评测查询 JSON（kg_evaluation_queries.json）转为 mirobody-eval BenchItem JSONL，
使用 kg_qa 评估器，保留完整 ground truth 元数据。

target_overrides 仅包含 per-case 参数（email / tool_context），
agent / language / tool_group 等公共参数由 metadata.json 默认值提供。

用法:
  python -m generator.eslbench.converter \
    --input /path/to/kg_evaluation_queries.json \
    --output benchmark/data/eslbench/<dataset>.jsonl \
    --user-email user110@demo
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 新版 query generator 使用 basic/intermediate/... 而旧版 + L1-L5 映射使用 direct/single_hop/...
# 转换映射：保留已标准化的旧名称不变，新名称映射到旧名称
_DIFFICULTY_MAP = {
    "basic": "direct",
    "intermediate": "single_hop",
    "advanced": "multi_hop",
    "expert": "multi_hop_temporal",
    "attribution": "attribution",
}


def convert_queries(
    queries_file: Path,
    user_email: str,
    difficulty: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict]:
    """将 kg_evaluation_queries.json 转为 BenchItem dict 列表（使用 kg_qa 评估器）"""
    if not queries_file.exists():
        logger.warning("找不到文件: %s", queries_file)
        return []

    with open(queries_file, encoding="utf-8") as f:
        data = json.load(f)

    queries = data.get("evaluation_queries", [])
    items: list[dict] = []
    for q in queries:
        if difficulty and q.get("difficulty") != difficulty:
            continue
        if limit and len(items) >= limit:
            break

        query_id_raw = q.get("query_id", f"unknown_{len(items)}")
        # 将数字 ID 前缀替换为 email 目录名: 6473_Q001 → user5022_AT_demo_Q001
        dir_name = user_email.replace("@", "_AT_")
        parts = query_id_raw.split("_", 1)
        if len(parts) == 2 and parts[0].isdigit():
            query_id = f"{dir_name}_{parts[1]}"
        else:
            query_id = query_id_raw
        query_text = q.get("query_text", "")
        gt = q.get("ground_truth", {})
        answer_type = gt.get("answer_type", "text")
        # 规范化 answer_type（query generator 可能输出非标准值）
        _ANSWER_TYPE_MAP = {
            "numeric": "numeric_value",
            "number": "numeric_value",
            "date": "text",
            "datetime": "text",
            "string": "text",
            "bool": "boolean",
        }
        answer_type = _ANSWER_TYPE_MAP.get(answer_type, answer_type)
        expected_value = gt.get("expected_value", "")
        key_points = gt.get("key_points", [])
        source_data = gt.get("source_data")
        history = q.get("history")  # 记忆投毒等的伪造助手轮 → BenchItem.history
        raw_diff = q.get("difficulty", "direct")
        diff = _DIFFICULTY_MAP.get(raw_diff, raw_diff)

        eval_config: dict = {
            "evaluator": "kg_qa",
            "answer_type": answer_type,
            "expected_value": expected_value,
            "key_points": key_points,
            "difficulty": diff,
        }
        if source_data:
            eval_config["source_data"] = source_data

        item = {
            "id": query_id,
            "title": f"[{diff}] {query_text[:60]}",
            "description": query_text,
            "user": {
                "type": "manual",
                "strict_inputs": [query_text],
                # 两个键写的是同一个邮箱，因为两个 target 读的位置不同：
                # `llm_api.tool_context.user_email` 是 llm_api 读的位置，也是
                # bench_schema 里几个只认身份的 target（RAG 那几个、hermes）的取值来源；
                # `mirobody.user_email` 是 mirobody target 读的位置。
                "target_overrides": {
                    "llm_api": {"tool_context": {"user_email": user_email}},
                    "mirobody": {"user_email": user_email},
                },
            },
            "eval": eval_config,
            "tags": [f"difficulty:{diff}", f"answer_type:{answer_type}"],
        }
        if history:
            item["history"] = history
        items.append(item)

    logger.info("转换 %d 条 queries (email=%s)", len(items), user_email)
    return items


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="KG QA → mirobody-eval BenchItem")
    parser.add_argument("--input", required=True, help="kg_evaluation_queries.json 路径")
    parser.add_argument("--output", required=True, help="输出 JSONL 路径")
    parser.add_argument("--user-email", required=True, help="用户 email（如 user110@demo）")
    parser.add_argument("--difficulty", help="过滤难度 (direct/single_hop/multi_hop/multi_hop_temporal/attribution)")
    parser.add_argument("--limit", type=int, help="最多条数")
    args = parser.parse_args()

    items = convert_queries(
        Path(args.input),
        user_email=args.user_email,
        difficulty=args.difficulty,
        limit=args.limit,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("写入 %d 条 → %s", len(items), output_path)


if __name__ == "__main__":
    main()
