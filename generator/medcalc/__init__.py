"""medcalc — MedCalc-Bench 数据集转换工具

将 MedCalc-Bench CSV 数据集转换为 mirobody-eval BenchItem JSONL 格式，
支持在 mirobody-eval 框架中直接运行医疗计算评测。

数据集来源: https://github.com/ncbi-nlp/MedCalc-Bench
论文: https://arxiv.org/abs/2406.12036
"""

from generator.medcalc.converter import convert

__all__ = ["convert"]
