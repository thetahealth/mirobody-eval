"""API response models"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel

# Shared data models (re-exported from evaluator core layer)
from evaluator.core.bench_schema import ApiCallResult  # noqa: F401
from evaluator.core.schema import (  # noqa: F401
    AgentInfo,
    BenchmarkSummary,
    CaseSummary,
    DatasetDetail,
    DatasetInfo,
    ReportEntry,
    TargetInfo,
)


# ==================== Task (Web-specific) ====================


class TaskCreateRequest(BaseModel):
    benchmark: str
    dataset: str
    target_type: str | None = None  # Target system type (specify for multi-target, e.g. "llm_api")
    target: dict[str, Any] | None = None  # Target config from frontend (overrides editable fields)
    ids: str | None = None
    limit: int | None = None
    max_concurrency: int = 3


class TaskSummary(BaseModel):
    task_id: str
    benchmark: str
    dataset: str
    target_type: str | None = None  # Target system type (None in eval-only mode)
    target_model: str | None = None  # Target system model (optional)
    status: str  # pending | running | completed | cancelled | error
    total: int
    completed: int
    created_at: datetime


class TaskDetail(BaseModel):
    task_id: str
    benchmark: str
    dataset: str
    runtime_target: dict[str, Any] | None = None  # Runtime target config (None in eval-only mode)
    status: str
    total: int
    completed: int
    cancelled: bool
    created_at: datetime
    cases: dict[str, Any]
    stats_by_tag: dict[str, Any] | None = None
    report_path: str | None = None
    report_summary: dict[str, Any] | None = None  # Report summary after eval-only completion


# ==================== Eval-Only (Web-specific) ====================


class EvalCheckRequest(BaseModel):
    benchmark: str
    dataset: str
    results: list[ApiCallResult]


class EvalCheckResponse(BaseModel):
    benchmark: str
    dataset: str
    benchmark_description: str = ""
    dataset_total: int  # Total cases in dataset
    submitted: int  # Number of submitted results
    matched: int  # Number of ID matches
    missed_ids: list[str] = []  # IDs in dataset but missing from submission (max 50)
    extra_ids: list[str] = []  # IDs in submission but not in dataset
    tag_distribution: dict[str, int] = {}  # Tag distribution of matched cases


class EvalOnlyRequest(BaseModel):
    benchmark: str
    dataset: str
    results: list[ApiCallResult]
    max_concurrency: int = 5
    # 通用 webhook 回调 (调用方可选; Python 完全不感知具体订阅者)
    callback_url: str | None = None  # 任务完成/失败时 POST 到此 URL
    callback_secret: str | None = None  # 作为 X-Webhook-Secret header 透传给订阅方校验


class EvalOnlyResponse(BaseModel):
    task_id: str
    status: str
    total: int  # Number of matched evaluation cases
    dataset_total: int  # Total cases in dataset
    submitted: int  # Number of submitted results
    matched: int  # Number of ID matches
    missed_ids: list[str] = []  # IDs in dataset but missing from submission (max 50)
    extra_ids: list[str] = []  # IDs in submission but not in dataset
    created_at: datetime


# ==================== Report (Web-specific) ====================


class ReportListResponse(BaseModel):
    reports: list[ReportEntry]

