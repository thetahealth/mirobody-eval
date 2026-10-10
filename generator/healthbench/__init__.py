"""healthbench — HealthBench 数据集转换工具

将 OpenAI HealthBench 数据集（JSONL）转换为 mirobody-eval TestCase 格式，
支持在 mirobody-eval 框架中直接运行 HealthBench 评测。

数据集来源: https://huggingface.co/datasets/openai/healthbench
论文: https://arxiv.org/abs/2505.08775
"""

from generator.healthbench.converter import convert

__all__ = ["convert"]
