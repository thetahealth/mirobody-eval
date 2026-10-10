"""
hermes_setup — Hermes Agent 自动检测、安装与配置

检测 Hermes Agent 的安装和运行状态，必要时自动完成：
1. 安装 Hermes Agent（curl install script）
2. 同步 OPENROUTER_API_KEY 到 ~/.hermes/.env
3. 在 config.yaml 中启用 api_server platform
4. 启动 gateway 服务

用法:
    from evaluator.utils.hermes_setup import ensure_hermes_ready, check_hermes_status

    status = await check_hermes_status()
    if not status.ready:
        await ensure_hermes_ready()
"""

import asyncio
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_HERMES_HOME = Path.home() / ".hermes"
_HERMES_CONFIG = _HERMES_HOME / "config.yaml"
_HERMES_ENV = _HERMES_HOME / ".env"
_DEFAULT_API_URL = "http://127.0.0.1:8642"
_INSTALL_SCRIPT_URL = "https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh"


class HermesSetupError(RuntimeError):
    """Hermes 自动安装/配置失败"""
    pass


@dataclass
class HermesStatus:
    """Hermes Agent 各层就绪状态"""
    installed: bool = False           # hermes CLI 可执行
    config_valid: bool = False        # config.yaml 含 api_server enabled
    api_key_present: bool = False     # .env 含 OPENROUTER_API_KEY（或其他 provider key）
    gateway_running: bool = False     # gateway 进程存活
    api_reachable: bool = False       # HTTP /health 可达
    model_name: str = ""              # 探测到的模型名
    issues: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.installed and self.config_valid and self.api_key_present and self.api_reachable


# ============================================================
# 检测
# ============================================================


def _find_hermes_bin() -> str | None:
    """查找 hermes 可执行文件"""
    # 优先 ~/.local/bin/hermes（install.sh 默认位置）
    local_bin = Path.home() / ".local" / "bin" / "hermes"
    if local_bin.is_file() and os.access(local_bin, os.X_OK):
        return str(local_bin)
    return shutil.which("hermes")


def _check_installed() -> bool:
    """检查 Hermes 是否已安装"""
    return _find_hermes_bin() is not None


def _check_config_valid() -> bool:
    """检查 config.yaml 是否已启用 api_server platform"""
    if not _HERMES_CONFIG.exists():
        return False
    try:
        import yaml
        with open(_HERMES_CONFIG, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        platforms = cfg.get("platforms")
        if not isinstance(platforms, dict):
            return False
        api_server = platforms.get("api_server")
        if not isinstance(api_server, dict):
            return False
        return api_server.get("enabled") is True
    except Exception:
        return False


def _check_api_key() -> bool:
    """检查 ~/.hermes/.env 是否含有效的 provider API key"""
    if not _HERMES_ENV.exists():
        return False
    try:
        content = _HERMES_ENV.read_text(encoding="utf-8")
        # 检查常见 provider key（非注释行）
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("#"):
                continue
            for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
                if line.startswith(f"{key}=") and len(line) > len(f"{key}="):
                    return True
        return False
    except Exception:
        return False


def _check_gateway_running() -> bool:
    """检查 gateway 进程是否存活"""
    pid_file = _HERMES_HOME / "gateway.pid"
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)  # 检查进程是否存在
        return True
    except (ValueError, ProcessLookupError, PermissionError, OSError):
        return False


async def _check_api_reachable(base_url: str) -> tuple[bool, str]:
    """探测 HTTP API 是否可达，返回 (reachable, model_name)"""
    import aiohttp
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get(f"{base_url}/health") as resp:
                if resp.status != 200:
                    return False, ""
            async with session.get(f"{base_url}/v1/models") as resp:
                if resp.status != 200:
                    return True, ""
                data = await resp.json()
                models = data.get("data", [])
                model_name = models[0].get("id", "") if models else ""
                return True, model_name
    except Exception:
        return False, ""


async def check_hermes_status(base_url: str = _DEFAULT_API_URL) -> HermesStatus:
    """全面检测 Hermes Agent 就绪状态"""
    status = HermesStatus()

    status.installed = _check_installed()
    if not status.installed:
        status.issues.append("Hermes Agent 未安装")

    status.config_valid = _check_config_valid()
    if not status.config_valid:
        status.issues.append("config.yaml 未启用 api_server platform")

    status.api_key_present = _check_api_key()
    if not status.api_key_present:
        status.issues.append("~/.hermes/.env 未配置 LLM provider API key")

    status.gateway_running = _check_gateway_running()
    if not status.gateway_running:
        status.issues.append("Gateway 服务未运行")

    status.api_reachable, status.model_name = await _check_api_reachable(base_url)
    if not status.api_reachable:
        status.issues.append(f"API server 不可达 ({base_url})")

    return status


# ============================================================
# 自动安装与配置
# ============================================================


async def _run_shell(cmd: str, *, timeout: int = 300, step_name: str = "") -> str:
    """执行 shell 命令，返回 stdout，失败时抛出 HermesSetupError"""
    logger.info("[Hermes Setup] %s: %s", step_name, cmd)
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise HermesSetupError(f"{step_name} 超时（{timeout}s）: {cmd}")

    output = stdout.decode("utf-8", errors="replace") if stdout else ""
    if proc.returncode != 0:
        raise HermesSetupError(f"{step_name} 失败 (exit={proc.returncode}):\n{output[-500:]}")

    return output


async def _install_hermes() -> None:
    """安装 Hermes Agent"""
    logger.info("[Hermes Setup] Step 1/5: 安装 Hermes Agent...")
    await _run_shell(
        f"curl -fsSL {_INSTALL_SCRIPT_URL} | bash",
        timeout=300,
        step_name="安装 Hermes",
    )
    # 验证
    if not _check_installed():
        raise HermesSetupError("安装完成但 hermes CLI 不可用，请检查 PATH 是否包含 ~/.local/bin")
    logger.info("[Hermes Setup] 安装完成")


def _sync_api_key() -> None:
    """将 mirobody-eval 的 OPENROUTER_API_KEY 同步到 ~/.hermes/.env"""
    logger.info("[Hermes Setup] Step 2/5: 同步 API key...")

    # 从 mirobody-eval 环境获取 key
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise HermesSetupError(
            "mirobody-eval 环境未配置 OPENROUTER_API_KEY，请在 .env 文件中添加后重试"
        )

    _HERMES_ENV.parent.mkdir(parents=True, exist_ok=True)

    # 读取现有内容
    existing = ""
    if _HERMES_ENV.exists():
        existing = _HERMES_ENV.read_text(encoding="utf-8")

    # 检查是否已存在（非注释）
    for line in existing.splitlines():
        stripped = line.strip()
        if stripped.startswith("OPENROUTER_API_KEY=") and not stripped.startswith("#"):
            value = stripped.split("=", 1)[1].strip()
            if value:
                logger.info("[Hermes Setup] OPENROUTER_API_KEY 已存在，跳过")
                return

    # 写入
    new_line = f"OPENROUTER_API_KEY={api_key}\n"
    if existing and not existing.endswith("\n"):
        new_line = "\n" + new_line
    _HERMES_ENV.write_text(existing + new_line, encoding="utf-8")
    logger.info("[Hermes Setup] OPENROUTER_API_KEY 已写入 ~/.hermes/.env")


def _ensure_api_server_config() -> None:
    """确保 config.yaml 中 api_server platform 已启用"""
    logger.info("[Hermes Setup] Step 3/5: 配置 api_server platform...")

    if _check_config_valid():
        logger.info("[Hermes Setup] api_server 已启用，跳过")
        return

    import yaml

    cfg: dict = {}
    if _HERMES_CONFIG.exists():
        with open(_HERMES_CONFIG, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    # 确保 platforms.api_server.enabled = true
    if "platforms" not in cfg or not isinstance(cfg["platforms"], dict):
        cfg["platforms"] = {}
    if "api_server" not in cfg["platforms"] or not isinstance(cfg["platforms"]["api_server"], dict):
        cfg["platforms"]["api_server"] = {}
    cfg["platforms"]["api_server"]["enabled"] = True

    with open(_HERMES_CONFIG, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info("[Hermes Setup] api_server platform 已启用")


async def _ensure_model_config() -> None:
    """确保 config.yaml 中有 model + provider 配置（默认 OpenRouter）"""
    import yaml

    if not _HERMES_CONFIG.exists():
        return

    with open(_HERMES_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    model_cfg = cfg.get("model")
    if isinstance(model_cfg, dict) and model_cfg.get("default"):
        logger.info("[Hermes Setup] 模型已配置: %s", model_cfg.get("default"))
        return

    # 未配置模型，设置默认值
    cfg["model"] = {
        "default": "openai/gpt-5.4-mini",
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions",
    }

    with open(_HERMES_CONFIG, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info("[Hermes Setup] 默认模型配置已写入: openai/gpt-5.4-mini via openrouter")


async def _start_gateway() -> None:
    """启动 Hermes gateway 服务"""
    logger.info("[Hermes Setup] Step 5/5: 启动 gateway...")

    hermes_bin = _find_hermes_bin()
    if not hermes_bin:
        raise HermesSetupError("hermes CLI 不可用")

    await _run_shell(f"{hermes_bin} gateway start", timeout=30, step_name="启动 gateway")

    # 等待 API server 就绪（最多 30 秒）
    for i in range(15):
        await asyncio.sleep(2)
        reachable, model = await _check_api_reachable(_DEFAULT_API_URL)
        if reachable:
            logger.info("[Hermes Setup] Gateway 已就绪 (model=%s)", model or "hermes-agent")
            return

    raise HermesSetupError(
        f"Gateway 已启动但 API server 未就绪 ({_DEFAULT_API_URL})，"
        "请检查 ~/.hermes/logs/gateway.error.log"
    )


async def ensure_hermes_ready(base_url: str = _DEFAULT_API_URL) -> HermesStatus:
    """自动检测并完成 Hermes 安装、配置、启动

    Returns:
        最终的 HermesStatus（ready=True 或抛出 HermesSetupError）
    """
    status = await check_hermes_status(base_url)
    if status.ready:
        logger.info("[Hermes Setup] Hermes 已就绪 (model=%s)", status.model_name)
        return status

    logger.warning("[Hermes Setup] Hermes 未就绪，开始自动配置: %s", status.issues)

    # Step 1: 安装
    if not status.installed:
        await _install_hermes()

    # Step 2: 同步 API key
    if not status.api_key_present:
        _sync_api_key()

    # Step 3: 配置 api_server
    if not status.config_valid:
        _ensure_api_server_config()

    # Step 4: 确保 model 配置
    await _ensure_model_config()

    # Step 5: 启动 gateway（如果不可达）
    if not status.api_reachable:
        # gateway 可能在运行但 api_server 没启用，改了 config 后需要重启
        hermes_bin = _find_hermes_bin()
        if status.gateway_running and hermes_bin:
            logger.info("[Hermes Setup] 重启 gateway 以应用配置变更...")
            await _run_shell(f"{hermes_bin} gateway restart", timeout=30, step_name="重启 gateway")
            await asyncio.sleep(5)
        else:
            await _start_gateway()

    # 最终验证
    final_status = await check_hermes_status(base_url)
    if not final_status.ready:
        raise HermesSetupError(
            f"自动配置完成但 Hermes 仍未就绪: {final_status.issues}\n"
            "请手动检查:\n"
            "  1. hermes gateway status\n"
            "  2. cat ~/.hermes/logs/gateway.error.log\n"
            "  3. curl http://127.0.0.1:8642/health"
        )

    logger.info("[Hermes Setup] 自动配置完成，Hermes 已就绪 (model=%s)", final_status.model_name)
    return final_status
