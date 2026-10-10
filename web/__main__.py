"""python -m web — start mirobody-eval Web UI"""

import logging
import os
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import uvicorn

LOG_FILE = "web.log"


def _setup_logging():
    """Configure logging to both terminal and file"""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s")

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    root.addHandler(sh)


class _HealthHandler(BaseHTTPRequestHandler):
    """独立线程的健康检查，不经过 asyncio 事件循环"""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def log_message(self, format, *args):
        pass  # 静默日志，避免刷屏


def _start_health_server(port: int):
    """在独立线程启动健康检查服务器"""
    server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logging.getLogger(__name__).info("Health check server started on port %d", port)


def main():
    _setup_logging()
    logging.getLogger(__name__).info("Web UI started, log file: %s", LOG_FILE)

    port = int(os.environ.get("HOLYEVAL_WEB_PORT", 8000))
    health_port = int(os.environ.get("HOLYEVAL_HEALTH_PORT", 8001))

    # 独立线程的健康检查服务器，不受 asyncio 事件循环影响
    _start_health_server(health_port)

    # Disable reload in production/container (files don't change in K8s, reload is unnecessary and affects signal handling)
    reload = os.environ.get("HOLYEVAL_RELOAD", "false").lower() == "true"
    uvicorn.run(
        "web.app.main:create_app",
        factory=True,
        host="0.0.0.0",
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    main()
