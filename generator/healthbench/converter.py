"""
HealthBench → mirobody-eval 数据转换器

将 HealthBench JSONL 转换为 mirobody-eval BenchItem JSONL，使其可以直接
通过 `python -m benchmark.basic_runner healthbench <dataset>` 执行。

转换映射:
  HealthBench prompt (多轮)  →  history (最后一条 user message 之前的对话) + strict_inputs (最后一条 user message)
  HealthBench rubrics        →  eval.rubrics (直接透传)
  HealthBench example_tags   →  tags

BenchItem 不含 target 字段（由运行时 --target-type/--target-model 决定）。

用法:
  python -m generator.healthbench.converter input.jsonl output.jsonl

数据集下载:
  https://huggingface.co/datasets/openai/healthbench
  文件: 2025-05-07-06-14-12_oss_eval.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _extract_theme(tags: List[str]) -> Optional[str]:
    """从 example_tags 中提取 theme 标签"""
    for tag in tags:
        if tag.startswith("theme:"):
            return tag.split(":", 1)[1]
    return None


def _convert_single(entry: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    """将单条 HealthBench 数据转换为 mirobody-eval BenchItem dict

    Args:
        entry: HealthBench JSONL 中的一条记录
        index: 序号（用于 id 生成）

    Returns:
        mirobody-eval BenchItem dict，转换失败返回 None
    """
    prompt_id = entry.get("prompt_id", f"unknown_{index}")
    example_tags = entry.get("example_tags", [])
    prompt = entry.get("prompt", [])
    rubrics = entry.get("rubrics", [])

    if not prompt:
        logger.warning("跳过空 prompt 的条目: %s", prompt_id)
        return None
    if not rubrics:
        logger.warning("跳过无 rubrics 的条目: %s", prompt_id)
        return None

    # 找到最后一条 user message 作为 strict_inputs
    last_user_idx = None
    for i in range(len(prompt) - 1, -1, -1):
        if prompt[i].get("role") == "user":
            last_user_idx = i
            break

    if last_user_idx is None:
        logger.warning("跳过无 user message 的条目: %s", prompt_id)
        return None

    last_user_content = prompt[last_user_idx]["content"]

    # 最后一条 user message 之前的对话 → history
    history: List[Dict[str, str]] = []
    for msg in prompt[:last_user_idx]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role and content:
            history.append({"role": role, "content": content})

    # 提取 theme 作为 title 的一部分
    theme = _extract_theme(example_tags)
    title = f"HealthBench — {theme}" if theme else "HealthBench evaluation"

    # 构建 BenchItem
    bench_item: Dict[str, Any] = {
        "id": f"hb_{prompt_id}",
        "title": title,
        "description": f"HealthBench rubric evaluation ({len(rubrics)} criteria)",
        "user": {
            "type": "manual",
            "strict_inputs": [last_user_content],
        },
        "eval": {
            "evaluator": "healthbench",
            "rubrics": [
                {
                    "criterion": r["criterion"],
                    "points": r["points"],
                    "tags": r.get("tags", []),
                }
                for r in rubrics
            ],
        },
        "tags": example_tags,
    }

    # 仅在有历史对话时添加 history 字段
    if history:
        bench_item["history"] = history

    return bench_item


def convert(
    input_path: str | Path,
    output_path: str | Path,
    limit: int | None = None,
) -> int:
    """将 HealthBench JSONL 转换为 mirobody-eval BenchItem JSONL

    Args:
        input_path:    HealthBench 源 JSONL 路径
        output_path:   输出 mirobody-eval BenchItem JSONL 路径
        limit:         最大转换条数（None 表示全部）

    Returns:
        成功转换的用例数
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue

            if limit is not None and converted >= limit:
                break

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("第 %d 行 JSON 解析失败: %s", i + 1, e)
                skipped += 1
                continue

            bench_item = _convert_single(entry, i)
            if bench_item is None:
                skipped += 1
                continue

            fout.write(json.dumps(bench_item, ensure_ascii=False) + "\n")
            converted += 1

    logger.info("转换完成: %d 条成功, %d 条跳过, 输出: %s", converted, skipped, output_path)
    return converted


def main() -> None:
    """CLI 入口"""
    parser = argparse.ArgumentParser(
        description="将 HealthBench JSONL 转换为 mirobody-eval BenchItem JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python -m generator.healthbench.converter data/healthbench.jsonl benchmark/data/healthbench/full.jsonl\n"
            "  python -m generator.healthbench.converter data/healthbench.jsonl output.jsonl --limit 100\n"
        ),
    )
    parser.add_argument("input", help="HealthBench 源 JSONL 文件路径")
    parser.add_argument("output", help="输出 mirobody-eval BenchItem JSONL 文件路径")
    parser.add_argument("--limit", type=int, default=None, help="最大转换条数")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出详细日志")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    count = convert(args.input, args.output, limit=args.limit)
    print(f"转换完成: {count} 条 BenchItem → {args.output}")


if __name__ == "__main__":
    main()
