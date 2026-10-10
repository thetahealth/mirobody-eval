"""
config — 配置管理模块

从 .env 文件和环境变量读取配置。
"""

import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv

# 加载 .env 文件（本地开发用）
load_dotenv()

# Docker 部署时从配置中心加载（有 CONFIG_SERVER 环境变量时生效）
try:
    from evaluator.utils.remote_config import load_remote_config

    load_remote_config()
except Exception:
    pass


# 远端 config 在生产容器里默认推 GOOGLE_GENAI_USE_VERTEXAI=true（搭配 ADC 凭证用 Vertex），
# 但本地评测只有 GOOGLE_API_KEY/GEMINI_API_KEY，没有 ADC，会被 Vertex 端点 401 拒掉。
# 这里复用 evaluator/utils/llm.py:147 的策略，但做在全局加载点，让 dyg vendor、
# google.genai 直连、langchain 包装都自动落到 AI Studio (generativelanguage.googleapis.com)。
# 想强制 Vertex 时显式 export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json 即可保留。
if (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")) and not os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS"
):
    for _k in ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"):
        os.environ.pop(_k, None)


@lru_cache(maxsize=128)
def get_config(key: str, default: Any = None) -> Any:
    """
    获取配置值

    Args:
        key: 配置键名（支持点分隔格式，如 'hermes.base_url'）
        default: 默认值

    Returns:
        配置值
    """
    # 将点分隔格式转换为环境变量格式（大写 + 下划线）
    env_key = key.upper().replace(".", "_")
    return os.environ.get(env_key, default)


def get_agent_llm_timeout() -> int:
    """获取 Agent LLM 调用超时时间（秒），默认 840s（14分钟）"""
    return int(get_config("AGENT_LLM_TIMEOUT", 840))

