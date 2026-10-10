"""
benchmark — 评测集管理与跑分执行

目录结构:
  data/{benchmark}/{dataset}.jsonl  — 按评测类型分目录的 JSONL 数据集
  report/{benchmark}/{dataset}_{date}.json — 评测报告（镜像 data/ 子目录结构）
  basic_runner.py — 跑分执行器

用法:
  python -m benchmark.basic_runner <benchmark> <dataset> [--limit N] [--ids x,y] [-p N] [-v]

示例:
  python -m benchmark.basic_runner healthbench sample
  python -m benchmark.basic_runner healthbench full --limit 100
  python -m benchmark.basic_runner medcalc sample -p 5
"""
