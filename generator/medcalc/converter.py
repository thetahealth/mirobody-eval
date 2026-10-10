"""
MedCalc-Bench → mirobody-eval 数据转换器

将 MedCalc-Bench CSV 测试数据转换为 mirobody-eval BenchItem JSONL，使其可以直接
通过 `python -m benchmark.basic_runner medcalc <dataset>` 执行。

转换映射:
  Patient Note + Question  →  strict_inputs (合并为单条用户输入)
  Ground Truth Answer      →  eval.ground_truth
  Lower/Upper Limit        →  eval.lower_limit / eval.upper_limit (decimal 类型)
  Output Type              →  eval.output_type
  Calculator Name          →  title / tags
  Category                 →  tags

BenchItem 不含 target 字段（由运行时 --target-type/--target-model 决定）。

用法:
  python -m generator.medcalc.converter test_data.csv output.jsonl
  python -m generator.medcalc.converter test_data.csv output.jsonl --limit 100 -v

数据集下载:
  https://github.com/ncbi-nlp/MedCalc-Bench/blob/main/datasets/test_data.csv
"""

import argparse
import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# MedCalc-Bench Output Type 到 mirobody-eval output_type 的映射
# 原始 CSV 中 Output Type 字段值: "decimal", "integer", "date"
# Calculator ID 69 是特殊的 weeks_days 类型
_WEEKS_DAYS_CALCULATOR_ID = 69


def _normalize_calculator_name(name: str) -> str:
    """将 Calculator Name 标准化为 tag 友好格式

    "Creatinine Clearance (Cockcroft-Gault Equation)" → "creatinine_clearance_cockcroft_gault_equation"
    """
    name = re.sub(r"[^\w\s]", " ", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name.lower()


def _resolve_output_type(row: Dict[str, str]) -> str:
    """解析 output_type，处理 weeks_days 特殊情况"""
    calc_id = int(row.get("Calculator ID", "0"))
    if calc_id == _WEEKS_DAYS_CALCULATOR_ID:
        return "weeks_days"

    output_type = row.get("Output Type", "").strip().lower()
    if output_type in ("decimal", "integer", "date"):
        return output_type

    # fallback: 尝试从 ground truth 推断
    gt = row.get("Ground Truth Answer", "")
    if "/" in gt:
        return "date"
    if "." in gt:
        return "decimal"
    return "integer"


def _convert_single(row: Dict[str, str], index: int) -> Optional[Dict[str, Any]]:
    """将单行 CSV 数据转换为 mirobody-eval BenchItem dict

    Args:
        row: CSV DictReader 的一行
        index: 行号（用于 id 生成）

    Returns:
        mirobody-eval BenchItem dict，转换失败返回 None
    """
    row_number = row.get("Row Number", str(index + 1)).strip()
    calculator_name = row.get("Calculator Name", "").strip()
    category = row.get("Category", "").strip()
    patient_note = row.get("Patient Note", "").strip()
    question = row.get("Question", "").strip()
    ground_truth = row.get("Ground Truth Answer", "").strip()
    lower_limit = row.get("Lower Limit", "").strip()
    upper_limit = row.get("Upper Limit", "").strip()
    explanation = row.get("Ground Truth Explanation", "").strip()

    if not patient_note:
        logger.warning("跳过空 Patient Note 的条目: row=%s", row_number)
        return None
    if not question:
        logger.warning("跳过空 Question 的条目: row=%s", row_number)
        return None
    if not ground_truth:
        logger.warning("跳过空 Ground Truth 的条目: row=%s", row_number)
        return None

    output_type = _resolve_output_type(row)

    # decimal 类型需要 lower/upper limit
    if output_type == "decimal" and (not lower_limit or not upper_limit):
        logger.warning("decimal 类型缺少容差范围: row=%s, calculator=%s", row_number, calculator_name)
        return None

    # 构建用户输入：Patient Note + Question
    user_input = f"{patient_note}\n\n{question}"

    # 构建 tags
    calc_tag = f"calculator:{_normalize_calculator_name(calculator_name)}" if calculator_name else None
    category_tag = f"category:{category.lower().replace(' ', '_')}" if category else None
    type_tag = f"output_type:{output_type}"
    tags = [t for t in [calc_tag, category_tag, type_tag] if t]

    bench_item: Dict[str, Any] = {
        "id": f"mc_{row_number}",
        "title": f"MedCalc — {calculator_name}" if calculator_name else f"MedCalc — #{row_number}",
        "description": f"医疗计算评测: {calculator_name} ({output_type})",
        "user": {
            "type": "manual",
            "strict_inputs": [user_input],
        },
        "eval": {
            "evaluator": "medcalc",
            "ground_truth": ground_truth,
            "lower_limit": lower_limit if output_type == "decimal" else "",
            "upper_limit": upper_limit if output_type == "decimal" else "",
            "output_type": output_type,
            "explanation": explanation,
        },
        "tags": tags,
    }

    return bench_item


def convert(
    input_path: str | Path,
    output_path: str | Path,
    limit: int | None = None,
) -> int:
    """将 MedCalc-Bench CSV 转换为 mirobody-eval BenchItem JSONL

    Args:
        input_path:    MedCalc-Bench 源 CSV 路径
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
        reader = csv.DictReader(fin)

        for i, row in enumerate(reader):
            if limit is not None and converted >= limit:
                break

            bench_item = _convert_single(row, i)
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
        description="将 MedCalc-Bench CSV 转换为 mirobody-eval BenchItem JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python -m generator.medcalc.converter test_data.csv benchmark/data/medcalc/full.jsonl\n"
            "  python -m generator.medcalc.converter test_data.csv output.jsonl --limit 100 -v\n"
        ),
    )
    parser.add_argument("input", help="MedCalc-Bench 源 CSV 文件路径")
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
