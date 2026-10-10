"""Generic task-lifecycle webhook delivery.

设计原则:
- 通用机制, 不感知任何具体订阅者
- 调用方在创建任务时声明 callback_url + callback_secret, 由此函数透明投递
- Fire-and-forget: 失败仅记日志, 不影响主流程 (eval session)
- 重试: 3 次指数退避 (1s / 3s / 9s); 每次 5s 超时
- 签名: X-Webhook-Secret header (与常见 hf-dataset webhook 约定一致, 简单可靠)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from web.app.services.task_manager import TaskEntry

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_SEC = 5.0
_RETRY_DELAYS_SEC = (1.0, 3.0, 9.0)
_HEADER_SECRET = "X-Webhook-Secret"


def _build_payload(entry: TaskEntry) -> dict[str, Any]:
    return {
        "task_id": entry.task_id,
        "status": entry.status,  # "completed" | "error" | "cancelled"
        "benchmark": entry.benchmark,
        "dataset": entry.dataset,
        "total": entry.total,
        "completed": entry.completed,
        "report_path": entry.report_path,
        "error": entry.error,
        "completed_at": datetime.now().isoformat(),
    }


async def fire_task_webhook(entry: TaskEntry) -> None:
    """Fire-and-forget webhook delivery for a task.

    No-op when entry.callback_url 为空。失败仅记日志, 永不向上抛。
    """
    if not entry.callback_url:
        return

    url = entry.callback_url
    secret = entry.callback_secret or ""
    payload = _build_payload(entry)
    headers = {_HEADER_SECRET: secret} if secret else {}

    last_err: Exception | None = None
    for attempt, delay in enumerate(_RETRY_DELAYS_SEC, start=1):
        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SEC) as client:
                resp = await client.post(url, json=payload, headers=headers)
                if 200 <= resp.status_code < 300:
                    logger.info(
                        "[webhook] %s task=%s status=%s attempt=%d",
                        url, entry.task_id, entry.status, attempt,
                    )
                    return
                # 4xx 视为终态, 不重试 (例: 401 secret 不对、404 端点不存在)
                if 400 <= resp.status_code < 500:
                    logger.warning(
                        "[webhook] %s task=%s 4xx, no retry: code=%d body=%s",
                        url, entry.task_id, resp.status_code, resp.text[:200],
                    )
                    return
                # 5xx 重试
                last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except (httpx.HTTPError, asyncio.TimeoutError) as e:
            last_err = e

        if attempt < len(_RETRY_DELAYS_SEC):
            logger.warning(
                "[webhook] %s task=%s attempt=%d failed (%s), retry in %ss",
                url, entry.task_id, attempt, last_err, delay,
            )
            await asyncio.sleep(delay)

    logger.error(
        "[webhook] %s task=%s 投递最终失败, 放弃: %s",
        url, entry.task_id, last_err,
    )
