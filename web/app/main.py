"""FastAPI app factory + route registration"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import sentry_sdk
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Trigger plugin registration
import evaluator.plugin.eval_agent  # noqa: F401
import evaluator.plugin.target_agent  # noqa: F401
import evaluator.plugin.test_agent  # noqa: F401
from evaluator.utils import paths

from web.app.api import agents, benchmarks, guides, reports, tasks
from web.app.services.prepare_manager import prepare_manager
from web.app.services.task_manager import task_manager

from fastapi.responses import JSONResponse

_WEB_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(_WEB_DIR / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动横幅：把 HOLYEVAL_* env 解析结果打到日志，便于排查 PVC / .env 配置
    paths.log_resolved_paths()
    # On startup: scan and run benchmark prepare scripts (non-blocking, background)
    await prepare_manager.start_all()
    yield
    # On shutdown: cancel all active tasks and prepare scripts
    task_manager.cancel_all()
    prepare_manager.cancel_all()


def create_app() -> FastAPI:
    dsn = os.environ.get("HOLYEVAL_PY_SENTRY_DSN")
    if dsn:
        sentry_sdk.init(
            dsn=dsn,
            environment=(os.environ.get("ENV") or "").upper() or None,
            traces_sample_rate=1.0,
        )

    app = FastAPI(title="mirobody-eval Web UI", lifespan=lifespan)

    # Static files
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR / "static")), name="static")

    # K8s health check probe
    @app.get("/api/health")
    async def health():
        return JSONResponse({"status": "ok"})

    # API routes
    app.include_router(benchmarks.router, prefix="/api")
    app.include_router(agents.router, prefix="/api")
    app.include_router(tasks.router, prefix="/api")
    app.include_router(reports.router, prefix="/api")
    app.include_router(guides.router, prefix="/api")

    # Page routes
    @app.get("/")
    async def index(request: Request):
        return templates.TemplateResponse("tasks/index.html", {"request": request, "nav": "tasks"})

    @app.get("/tasks")
    async def tasks_page(request: Request):
        return templates.TemplateResponse("tasks/index.html", {"request": request, "nav": "tasks"})

    @app.get("/tasks/{task_id}")
    async def task_detail_page(request: Request, task_id: str):
        return templates.TemplateResponse("tasks/detail.html", {"request": request, "nav": "tasks", "task_id": task_id})

    @app.get("/benchmarks")
    async def benchmarks_page(request: Request):
        return templates.TemplateResponse("benchmarks/index.html", {"request": request, "nav": "benchmarks"})

    @app.get("/benchmarks/{benchmark}/{dataset}")
    async def benchmark_detail_page(request: Request, benchmark: str, dataset: str):
        return templates.TemplateResponse(
            "benchmarks/detail.html",
            {"request": request, "nav": "benchmarks", "benchmark": benchmark, "dataset": dataset},
        )

    @app.get("/virtual-users")
    async def virtual_users_page(request: Request):
        return templates.TemplateResponse("virtual_users/index.html", {"request": request, "nav": "virtual_users"})

    @app.get("/agents/target")
    async def agents_target_page(request: Request):
        return templates.TemplateResponse("agents/target.html", {"request": request, "nav": "agents_target"})

    @app.get("/agents/eval")
    async def agents_eval_page(request: Request):
        return templates.TemplateResponse("agents/eval.html", {"request": request, "nav": "agents_eval"})

    @app.get("/agents/test")
    async def agents_test_page(request: Request):
        return templates.TemplateResponse("agents/test.html", {"request": request, "nav": "agents_test"})

    @app.get("/reports/{benchmark}/{filename}")
    async def report_detail_page(request: Request, benchmark: str, filename: str):
        return templates.TemplateResponse(
            "reports/detail.html",
            {"request": request, "nav": "tasks", "benchmark": benchmark, "filename": filename},
        )

    @app.get("/guides/{guide_name}")
    async def guide_page(request: Request, guide_name: str):
        title = next((g["title"] for g in guides.GUIDE_LIST if g["name"] == guide_name), guide_name)
        return templates.TemplateResponse(
            "guides/detail.html",
            {"request": request, "nav": f"guide_{guide_name}", "guide_name": guide_name, "guide_title": title},
        )

    return app
