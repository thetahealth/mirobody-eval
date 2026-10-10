import json
import numpy as np
import asyncio
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Optional, List, Any, Callable, Dict
from pathlib import Path

import aioboto3
from openai import AsyncOpenAI, AsyncAzureOpenAI, APIConnectionError, RateLimitError

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import os

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage

logger = logging.getLogger("DyG-RAG")

global_openai_async_client = None
global_azure_openai_async_client = None
global_amazon_bedrock_async_client = None
global_gemini_client = None
global_google_genai_raw_client = None
global_multi_key_gemini_client = None  # 多 key Gemini 客户端
_dotenv_loaded = False

# Gemini 全局并发控制（按 GEMINI_CONCURRENCY）
_global_gemini_call_sem: asyncio.Semaphore | None = None
# Gemini embedding 全局并发控制（按 GEMINI_EMBEDDING_CONCURRENCY，默认 10）
_global_gemini_embedding_sem: asyncio.Semaphore | None = None

# Gemini API 调用超时（秒），防止 SSL 连接卡死
GEMINI_CALL_TIMEOUT = float(os.environ.get("GEMINI_CALL_TIMEOUT", "120"))

# 全局线程池，使用 daemon 线程以便超时后不阻塞进程退出
_gemini_thread_pool: ThreadPoolExecutor | None = None


def _get_gemini_thread_pool() -> ThreadPoolExecutor:
    """获取 Gemini API 调用专用的线程池（daemon 线程）"""
    global _gemini_thread_pool
    if _gemini_thread_pool is None:
        # 线程数必须 >= 总并发数（key 数量 × 每 key 并发），否则多余的协程会在线程池排队，变成串行
        max_workers = get_gemini_total_concurrency() + 4
        _gemini_thread_pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="gemini-api-"
        )
    return _gemini_thread_pool


async def _run_in_thread_with_timeout(func: Callable, timeout: float) -> Any:
    """
    在线程池中执行同步函数，支持真正的超时控制。

    注意：超时后线程仍会继续运行直到完成，但调用方不再等待。
    这比 asyncio.to_thread 更可控，因为我们使用有限大小的线程池。
    """
    loop = asyncio.get_running_loop()
    pool = _get_gemini_thread_pool()

    future = pool.submit(func)
    try:
        # 使用 asyncio 友好的方式等待 Future
        result = await asyncio.wait_for(
            loop.run_in_executor(None, future.result, timeout),
            timeout=timeout
        )
        return result
    except (asyncio.TimeoutError, FuturesTimeoutError):
        # 尝试取消 Future（可能已经在执行中，取消会失败但无害）
        future.cancel()
        raise TimeoutError(f"Gemini API 调用超时 ({timeout}s)")


def _get_gemini_concurrency() -> int:
    _load_dotenv_if_present()
    try:
        return max(1, int(os.getenv("GEMINI_CONCURRENCY", "8")))
    except Exception:
        return 8


def _get_gemini_embedding_concurrency() -> int:
    """
    Gemini embedding 专用并发度（单 key 模式）。
    默认每个 key 10 并发，可通过 GEMINI_EMBEDDING_CONCURRENCY 覆盖。
    """
    _load_dotenv_if_present()
    try:
        return max(1, int(os.getenv("GEMINI_EMBEDDING_CONCURRENCY", "10")))
    except Exception:
        return 10


def _get_global_gemini_call_semaphore() -> asyncio.Semaphore:
    """
    统一控制所有 Gemini generate_content 的并行度（包含结构化 JSON 调用）。
    """
    global _global_gemini_call_sem
    if _global_gemini_call_sem is None:
        _global_gemini_call_sem = asyncio.Semaphore(_get_gemini_concurrency())
    return _global_gemini_call_sem


def _get_global_gemini_embedding_semaphore() -> asyncio.Semaphore:
    """
    统一控制 Gemini embed_content 的并行度（单 key 模式）。
    与 generate_content 分开，避免为了 embedding 降并发影响 NER/事件抽取。
    """
    global _global_gemini_embedding_sem
    if _global_gemini_embedding_sem is None:
        _global_gemini_embedding_sem = asyncio.Semaphore(_get_gemini_embedding_concurrency())
    return _global_gemini_embedding_sem


def get_openai_async_client_instance():
    global global_openai_async_client
    if global_openai_async_client is None:
        global_openai_async_client = AsyncOpenAI()
    return global_openai_async_client


def get_azure_openai_async_client_instance():
    global global_azure_openai_async_client
    if global_azure_openai_async_client is None:
        global_azure_openai_async_client = AsyncAzureOpenAI()
    return global_azure_openai_async_client


def get_amazon_bedrock_async_client_instance():
    global global_amazon_bedrock_async_client
    if global_amazon_bedrock_async_client is None:
        global_amazon_bedrock_async_client = aioboto3.Session()
    return global_amazon_bedrock_async_client


def _import_google_genai():
    """
    兼容不同版本的 Google GenAI Python SDK 导入方式。

    - 新 SDK 常见：from google import genai
    - 部分环境：import google.genai as genai
    """
    genai = None
    types = None

    try:
        # 推荐写法（python-genai）
        from google import genai as _genai  # type: ignore

        genai = _genai
    except Exception:
        try:
            import google.genai as _genai  # type: ignore

            genai = _genai
        except Exception as e:
            raise ImportError(
                "未安装 Google GenAI SDK。请安装依赖：pip install google-genai"
            ) from e

    try:
        # types 模块用于 GenerateContentConfig / ThinkingConfig / EmbedContentConfig
        from google.genai import types as _types  # type: ignore

        types = _types
    except Exception:
        # types 不存在时，仍可尝试不带 config 调用 generate_content
        types = None

    return genai, types


def _create_http_options():
    """
    创建 Google GenAI Client 的 http_options 配置。

    注意：不要在 HttpOptions 上设置 timeout，SDK 会将其误解为 gRPC deadline 导致请求失败。
    但 httpx.Client 自身的 timeout 和 limits 是安全的，用于控制底层连接行为。
    """
    import httpx
    genai, types = _import_google_genai()

    proxy = os.getenv("GEMINI_PROXY", "")

    _timeout = httpx.Timeout(connect=30.0, read=300.0, write=60.0, pool=30.0)
    _limits = httpx.Limits(max_connections=20, max_keepalive_connections=10, keepalive_expiry=30)

    if proxy:
        http_options = types.HttpOptions(
            httpxClient=httpx.Client(proxy=proxy, timeout=_timeout, limits=_limits),
            httpxAsyncClient=httpx.AsyncClient(proxy=proxy, timeout=_timeout, limits=_limits),
        )
    else:
        http_options = types.HttpOptions(
            httpxClient=httpx.Client(timeout=_timeout, limits=_limits),
            httpxAsyncClient=httpx.AsyncClient(timeout=_timeout, limits=_limits),
        )

    return http_options


def _load_dotenv_if_present():
    """
    从仓库根目录加载 .env（如果存在）。
    说明：
    - 依赖 python-dotenv（requirements.txt 已包含）
    - 不强制要求 .env 存在；不存在就忽略
    """
    global _dotenv_loaded
    if _dotenv_loaded:
        return

    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        _dotenv_loaded = True
        return

    try:
        # 以本文件所在目录为基准，向上找 repo 根（这里假设 graphrag/ 在根目录下）
        repo_root = Path(__file__).resolve().parents[1]
        env_path = repo_root / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=str(env_path), override=False)
    finally:
        _dotenv_loaded = True


@dataclass
class GeminiClient:
    """
    Gemini LLM 客户端（异步 + 并发控制 + 指数退避重试）

    说明：
    - 这里用 asyncio.to_thread 包装同步 SDK 调用，避免阻塞事件循环
    - 并发既可以由 GraphRAG 的 limit_async_func_call 控制，也可以由这里的 semaphore 控制
    """

    api_key: str
    model: str = "gemini-3-pro-preview"
    thinking_level: str = "high"  # minimal, low, medium, high
    concurrency: int = 8
    max_retries: int = 3
    retry_delay: float = 1.0

    _client: Any = field(default=None, init=False, repr=False)
    _sem: asyncio.Semaphore = field(init=False, repr=False)

    def __post_init__(self):
        self._sem = asyncio.Semaphore(max(1, int(self.concurrency)))

    def _get_client(self):
        if self._client is None:
            genai, _ = _import_google_genai()
            # 统一通过 api_key 显式初始化，避免环境变量歧义
            self._client = genai.Client(api_key=self.api_key, http_options=_create_http_options())
        return self._client

    @staticmethod
    def _extract_text(resp: Any) -> str:
        """
        从 generate_content 返回对象中提取文本。
        尽量只拼接 text parts，规避 thought_signature 之类的告警/字段。
        """
        # 部分 SDK 直接提供 resp.text
        if hasattr(resp, "text") and isinstance(resp.text, str) and resp.text:
            return resp.text

        out = ""
        try:
            if resp and getattr(resp, "candidates", None):
                cand0 = resp.candidates[0]
                content = getattr(cand0, "content", None)
                parts = getattr(content, "parts", None) if content else None
                if parts:
                    for part in parts:
                        txt = getattr(part, "text", None)
                        if isinstance(txt, str) and txt:
                            out += txt
        except Exception:
            # 兜底：返回空串，让上层决定是否重试/报错
            return ""

        return out

    async def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> str:
        async with self._sem:
            genai, types = _import_google_genai()
            client = self._get_client()

            # 默认补 thinking_config；但如果调用方显式传 config（尤其是 response_schema 场景），
            # 则必须"原样透传"，避免 SDK 忽略/冲突导致结构化输出失效。
            if config is None:
                cfg = {"thinking_config": {"thinking_level": self.thinking_level}}
            else:
                cfg = config

            last_error: Exception | None = None
            for attempt in range(self.max_retries + 1):
                try:
                    # 使用原生异步 API，避免同步线程池阻塞
                    resp = await asyncio.wait_for(
                        client.aio.models.generate_content(
                            model=model or self.model, contents=prompt, config=cfg
                        ),
                        timeout=GEMINI_CALL_TIMEOUT,
                    )
                    return self._extract_text(resp)
                except asyncio.TimeoutError:
                    last_error = TimeoutError(f"Gemini API 调用超时 ({GEMINI_CALL_TIMEOUT}s)")
                    logger.warning(f"Gemini API 超时 (attempt {attempt + 1}/{self.max_retries + 1})")
                    if attempt < self.max_retries:
                        delay = self.retry_delay * (2**attempt)
                        await asyncio.sleep(delay)
                    else:
                        break
                except Exception as e:
                    last_error = e
                    err_type = type(e).__name__
                    err_msg = str(e) or repr(e)
                    logger.warning(f"Gemini API 错误 [{err_type}]: {err_msg} (attempt {attempt + 1}/{self.max_retries + 1})")
                    if attempt < self.max_retries:
                        delay = self.retry_delay * (2**attempt)
                        await asyncio.sleep(delay)
                    else:
                        break

            raise last_error if last_error is not None else RuntimeError("Gemini 调用失败")


def _load_all_gemini_api_keys() -> List[str]:
    """
    从环境变量加载所有 Gemini API keys。
    支持两种格式：
    1. 单个 key: GEMINI_API_KEY=xxx
    2. 多个 key: GEMINI_API_KEY_1=xxx, GEMINI_API_KEY_2=xxx, ...
    """
    _load_dotenv_if_present()
    keys = []

    # 先尝试加载多 key 格式（GEMINI_API_KEY_*）
    for key, value in os.environ.items():
        if key.startswith("GEMINI_API_KEY_"):
            if value and value.strip():
                keys.append(value.strip())

    # 如果找到多 key，清除 GOOGLE_API_KEY 防止 google-genai SDK 劫持
    # （SDK 发现 GOOGLE_API_KEY 会忽略传入的 api_key 参数）
    if keys and os.environ.get("GOOGLE_API_KEY"):
        logger.info("检测到多 key 模式，清除 GOOGLE_API_KEY 防止 SDK 劫持")
        os.environ.pop("GOOGLE_API_KEY", None)

    # 如果没有找到多 key，则使用单个 GEMINI_API_KEY 或 GOOGLE_API_KEY
    if not keys:
        single_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if single_key and single_key.strip():
            keys.append(single_key.strip())

    return keys


@dataclass
class MultiKeyGeminiClient:
    """
    多 API Key Gemini 客户端，支持轮询负载均衡。

    每个 key 独立维护自己的并发控制（Semaphore），
    通过轮询方式分配请求到不同的 key，实现总并发数 = key 数量 × 单 key 并发数。
    """
    api_keys: List[str]
    model: str = "gemini-3-pro-preview"
    thinking_level: str = "high"
    concurrency_per_key: int = 5  # 每个 key 的并发数
    max_retries: int = 3
    retry_delay: float = 1.0

    _clients: List[GeminiClient] = field(default=None, init=False, repr=False)
    _raw_clients: List[Any] = field(default=None, init=False, repr=False)
    _raw_sems: List[asyncio.Semaphore] = field(default=None, init=False, repr=False)  # raw_client 的并发控制
    _embed_raw_sems: List[asyncio.Semaphore] = field(default=None, init=False, repr=False)  # embedding 专用并发控制
    _current_index: int = field(default=0, init=False, repr=False)
    _lock: asyncio.Lock = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not self.api_keys:
            raise ValueError("至少需要一个 API key")

        self._clients = []
        self._raw_clients = []
        self._raw_sems = []
        self._embed_raw_sems = []

        # embedding 默认每个 key 10 并发，可通过环境变量覆盖
        try:
            embed_conc_per_key = max(1, int(os.getenv("GEMINI_EMBEDDING_CONCURRENCY", "10")))
        except Exception:
            embed_conc_per_key = 10

        for key in self.api_keys:
            client = GeminiClient(
                api_key=key,
                model=self.model,
                thinking_level=self.thinking_level,
                concurrency=self.concurrency_per_key,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
            )
            self._clients.append(client)

            # 为每个 key 创建原生 client（用于 embedding 等）
            genai, _ = _import_google_genai()
            raw_client = genai.Client(api_key=key, http_options=_create_http_options())
            self._raw_clients.append(raw_client)
            # 为每个 raw_client 创建独立的 semaphore
            self._raw_sems.append(asyncio.Semaphore(self.concurrency_per_key))
            # 为 embedding 创建独立的 semaphore（避免影响 generate_content）
            self._embed_raw_sems.append(asyncio.Semaphore(embed_conc_per_key))

        self._current_index = 0
        self._lock = asyncio.Lock()

        import logging
        logging.getLogger(__name__).info(
            f"MultiKeyGeminiClient 初始化完成: {len(self.api_keys)} 个 key, "
            f"每个 key 并发 {self.concurrency_per_key}, 总并发 {len(self.api_keys) * self.concurrency_per_key}"
        )

    @property
    def total_concurrency(self) -> int:
        """总并发数 = key 数量 × 单 key 并发数"""
        return len(self.api_keys) * self.concurrency_per_key

    def _get_next_index(self) -> int:
        """获取下一个索引（轮询）"""
        idx = self._current_index
        self._current_index = (self._current_index + 1) % len(self._clients)
        return idx

    def _get_next_client(self) -> GeminiClient:
        """轮询获取下一个 client"""
        idx = self._get_next_index()
        return self._clients[idx]

    def _get_next_raw_client_with_sem(self) -> tuple[Any, asyncio.Semaphore]:
        """轮询获取下一个原生 client 及其 semaphore"""
        idx = self._current_index
        self._current_index = (self._current_index + 1) % len(self._raw_clients)
        return self._raw_clients[idx], self._raw_sems[idx]

    def _get_next_raw_client_with_embed_sem(self) -> tuple[Any, asyncio.Semaphore]:
        """轮询获取下一个原生 client 及其 embedding semaphore"""
        idx = self._current_index
        self._current_index = (self._current_index + 1) % len(self._raw_clients)
        return self._raw_clients[idx], self._embed_raw_sems[idx]

    async def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> str:
        """异步生成，自动轮询选择 client"""
        async with self._lock:
            client = self._get_next_client()
        return await client.generate(prompt, model=model, config=config)

    def get_raw_client(self) -> Any:
        """获取原生 client（用于 embedding 等非异步场景）- 已废弃，请用 get_raw_client_with_sem"""
        raw_client = self._raw_clients[self._current_index]
        self._current_index = (self._current_index + 1) % len(self._raw_clients)
        return raw_client

    async def get_raw_client_with_sem(self) -> tuple[Any, asyncio.Semaphore]:
        """异步获取原生 client 及其 semaphore（线程安全）"""
        async with self._lock:
            return self._get_next_raw_client_with_sem()

    async def get_raw_client_with_embed_sem(self) -> tuple[Any, asyncio.Semaphore]:
        """异步获取原生 client 及其 embedding semaphore（线程安全）"""
        async with self._lock:
            return self._get_next_raw_client_with_embed_sem()

    @property
    def total_concurrency(self) -> int:
        """总并发数"""
        return len(self.api_keys) * self.concurrency_per_key


def get_multi_key_gemini_client_instance(
    *,
    api_keys: Optional[List[str]] = None,
    model: Optional[str] = None,
    thinking_level: Optional[str] = None,
    concurrency_per_key: Optional[int] = None,
    max_retries: Optional[int] = None,
    retry_delay: Optional[float] = None,
) -> MultiKeyGeminiClient:
    """
    获取全局 MultiKeyGeminiClient 实例（懒加载）。

    优先使用多 key 配置，如果没有则回退到单 key。
    """
    global global_multi_key_gemini_client
    _load_dotenv_if_present()

    if global_multi_key_gemini_client is None:
        resolved_keys = api_keys or _load_all_gemini_api_keys()
        if not resolved_keys:
            raise ValueError(
                "缺少 Gemini API Key：请设置环境变量 GEMINI_API_KEY 或 GEMINI_API_KEY_1 / GEMINI_API_KEY_2 ..."
            )

        global_multi_key_gemini_client = MultiKeyGeminiClient(
            api_keys=resolved_keys,
            model=model or os.getenv("GEMINI_MODEL", "gemini-3-pro-preview"),
            thinking_level=thinking_level or os.getenv("GEMINI_THINKING_LEVEL", "high"),
            concurrency_per_key=concurrency_per_key
            if concurrency_per_key is not None
            else int(os.getenv("GEMINI_CONCURRENCY", "5")),
            max_retries=max_retries
            if max_retries is not None
            else int(os.getenv("GEMINI_MAX_RETRIES", "3")),
            retry_delay=retry_delay
            if retry_delay is not None
            else float(os.getenv("GEMINI_RETRY_DELAY", "1.0")),
        )

    return global_multi_key_gemini_client


def get_gemini_total_concurrency() -> int:
    """
    获取 Gemini 总并发数。

    如果使用多 key 模式，返回 key 数量 × 单 key 并发数。
    否则返回单 key 的并发数。
    """
    _load_dotenv_if_present()

    # 获取并发配置
    concurrency_per_key = int(os.getenv("GEMINI_CONCURRENCY", "5"))

    # 尝试加载多 key
    api_keys = _load_all_gemini_api_keys()
    if api_keys:
        return len(api_keys) * concurrency_per_key

    # 单 key 模式
    return concurrency_per_key


def get_gemini_client_instance(
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    thinking_level: Optional[str] = None,
    concurrency: Optional[int] = None,
    max_retries: Optional[int] = None,
    retry_delay: Optional[float] = None,
) -> GeminiClient:
    """
    获取全局 GeminiClient 实例（懒加载）。

    约定环境变量：
    - GEMINI_API_KEY: 必填
    - GEMINI_MODEL: 默认模型名
    - GEMINI_THINKING_LEVEL: 思考强度
    - GEMINI_CONCURRENCY / GEMINI_MAX_RETRIES / GEMINI_RETRY_DELAY: 可选
    """
    global global_gemini_client
    _load_dotenv_if_present()

    if global_gemini_client is None:
        resolved_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "缺少 Gemini API Key：请设置环境变量 GEMINI_API_KEY / GOOGLE_API_KEY 或传入 api_key"
            )

        global_gemini_client = GeminiClient(
            api_key=resolved_key,
            model=model or os.getenv("GEMINI_MODEL", "gemini-3-pro-preview"),
            thinking_level=thinking_level
            or os.getenv("GEMINI_THINKING_LEVEL", "high"),
            concurrency=concurrency
            if concurrency is not None
            else int(os.getenv("GEMINI_CONCURRENCY", "8")),
            max_retries=max_retries
            if max_retries is not None
            else int(os.getenv("GEMINI_MAX_RETRIES", "3")),
            retry_delay=retry_delay
            if retry_delay is not None
            else float(os.getenv("GEMINI_RETRY_DELAY", "1.0")),
        )

    return global_gemini_client


def get_google_genai_raw_client_instance(*, api_key: Optional[str] = None):
    """
    获取 google.genai.Client 原生实例（用于 embeddings 等能力）。
    """
    global global_google_genai_raw_client
    _load_dotenv_if_present()

    if global_google_genai_raw_client is None:
        resolved_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError("缺少 GEMINI_API_KEY / GOOGLE_API_KEY（用于 Gemini embeddings）")
        genai, _ = _import_google_genai()
        global_google_genai_raw_client = genai.Client(api_key=resolved_key, http_options=_create_http_options())

    return global_google_genai_raw_client


def _pack_chat_to_text(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Any] | None = None,
) -> str:
    """
    将 OpenAI 风格 messages 转成纯文本，便于 Gemini 的 contents 直接输入。
    """
    parts: List[str] = []
    if system_prompt:
        parts.append(f"[SYSTEM]\n{system_prompt}\n")

    if history_messages:
        for m in history_messages:
            try:
                role = m.get("role", "unknown")
                content = m.get("content", "")
            except Exception:
                role, content = "unknown", str(m)

            # content 可能是 list/dict（例如 Bedrock 兼容格式），简单兜底成字符串
            if isinstance(content, (dict, list)):
                content = json.dumps(content, ensure_ascii=False)

            parts.append(f"[{str(role).upper()}]\n{str(content)}\n")

    parts.append(f"[USER]\n{prompt}\n")
    return "\n".join(parts).strip()


async def gemini_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Any] = [],
    **kwargs,
) -> str:
    """
    Gemini 文本生成（可选 KV 缓存）。

    兼容 GraphRAG 的调用签名：支持 system_prompt / history_messages / **kwargs
    - 会忽略 OpenAI 专有的参数（如 response_format），避免报错
    """
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)

    # 忽略 OpenAI 专有/不通用参数（保持兼容，不影响调用方）
    kwargs.pop("response_format", None)

    # 调用方可以显式指定 thinking_level（如 cheap model 传 "minimal"），None 则用 client 默认值
    thinking_level = kwargs.pop("thinking_level", None)

    # Gemini 结构化输出参数（如 response_schema）
    # 官方字段名：response_json_schema（pydantic: model_json_schema()）
    response_json_schema = kwargs.pop("response_json_schema", None)
    # 兼容旧字段名（内部统一转成 response_json_schema）
    response_schema = kwargs.pop("response_schema", None)
    if response_json_schema is None and response_schema is not None:
        response_json_schema = response_schema
    response_mime_type = kwargs.pop("response_mime_type", None)
    temperature = kwargs.pop("temperature", None)

    # 目前仅支持把对话拼成纯文本喂给 Gemini
    packed = _pack_chat_to_text(
        prompt=prompt, system_prompt=system_prompt, history_messages=history_messages
    )

    if hashing_kv is not None:
        # 注意：schema/config 也要参与缓存 key，避免不同 schema 复用同一缓存
        cfg_for_hash = {
            "response_mime_type": response_mime_type,
            "response_json_schema": response_json_schema,
            "temperature": temperature,
        }
        args_hash = compute_args_hash("gemini", model, packed, json.dumps(cfg_for_hash, ensure_ascii=False, sort_keys=True))
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    # 使用全局客户端（模型名通过这里传入，保证 cache 的 model 一致）
    gen_cfg: dict = {}
    if response_mime_type is not None:
        gen_cfg["response_mime_type"] = response_mime_type
    if response_json_schema is not None:
        gen_cfg["response_json_schema"] = response_json_schema
    if temperature is not None:
        gen_cfg["temperature"] = temperature
    # 结构化输出（response_json_schema）与 thinking_config 组合会导致 Gemini 极慢/超时，跳过 thinking
    if thinking_level is not None and thinking_level.lower() not in ("none", "") and response_json_schema is None:
        gen_cfg["thinking_config"] = {"thinking_level": thinking_level}

    # 优先使用多 key client（如果配置了多个 key）
    try:
        multi_client = get_multi_key_gemini_client_instance()
        use_multi_key = len(multi_client.api_keys) > 1
    except Exception:
        use_multi_key = False
        multi_client = None

    # 结构化输出（response_json_schema/response_mime_type）必须走原生 generate_content(config=dict)
    # 不做任何稳健处理：只要模型没按 schema 输出，后续 json.loads 就会直接报错。
    if gen_cfg:
        if use_multi_key and multi_client is not None:
            # 多 key 模式：轮询选择 raw_client，每个 client 有自己的 semaphore 并发控制
            raw_client, raw_sem = await multi_client.get_raw_client_with_sem()
        else:
            raw_client = get_google_genai_raw_client_instance()
            raw_sem = None

        async def _call_raw_async() -> str:
            # 使用原生异步 API，避免同步线程池阻塞（超时后线程不释放导致线程池耗尽）
            import time as _time
            _t0 = _time.monotonic()
            resp = await asyncio.wait_for(
                raw_client.aio.models.generate_content(
                    model=model,
                    contents=packed,
                    config=gen_cfg,
                ),
                timeout=GEMINI_CALL_TIMEOUT,
            )
            _elapsed = _time.monotonic() - _t0
            logger.info(f"[DIAG] generate_content returned in {_elapsed:.1f}s (prompt {len(packed)} chars)")
            # 直接取 text（schema + application/json 应保证是纯 JSON 文本）
            if hasattr(resp, "text") and isinstance(resp.text, str) and resp.text:
                return resp.text
            return GeminiClient._extract_text(resp)

        # 多 key 模式使用各自的 semaphore，单 key 模式使用全局 semaphore
        import time as _time
        _sem_wait_start = _time.monotonic()
        try:
            if use_multi_key and raw_sem is not None:
                async with raw_sem:
                    _sem_waited = _time.monotonic() - _sem_wait_start
                    if _sem_waited > 1.0:
                        logger.warning(f"[DIAG] Waited {_sem_waited:.1f}s for raw_sem")
                    result = await _call_raw_async()
            else:
                sem = _get_global_gemini_call_semaphore()
                async with sem:
                    _sem_waited = _time.monotonic() - _sem_wait_start
                    if _sem_waited > 1.0:
                        logger.warning(f"[DIAG] Waited {_sem_waited:.1f}s for global_sem")
                    result = await _call_raw_async()
        except asyncio.TimeoutError:
            _total_waited = _time.monotonic() - _sem_wait_start
            raise TimeoutError(f"Gemini API 调用超时 ({GEMINI_CALL_TIMEOUT}s, total_waited={_total_waited:.1f}s)")
    else:
        # 若调用方显式指定了 thinking_level，构造 config 覆盖 client 默认值
        _skip_thinking = not thinking_level or thinking_level.lower() in ("none", "")
        plain_config = {} if _skip_thinking else {"thinking_config": {"thinking_level": thinking_level}}
        if use_multi_key and multi_client is not None:
            # 多 key 模式：通过 MultiKeyGeminiClient 的 generate 方法（内部轮询 + 并发控制）
            result = await multi_client.generate(packed, model=model, config=plain_config)
        else:
            client = get_gemini_client_instance(model=model)
            sem = _get_global_gemini_call_semaphore()
            async with sem:
                result = await client.generate(packed, model=model, config=plain_config)

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": result, "model": model, "provider": "gemini"}}
        )
        await hashing_kv.index_done_callback()

    return result


def create_gemini_complete_function(model_id: str) -> Callable:
    """
    工厂函数：按指定 model_id 生成 Gemini completion function。
    用法与 Bedrock 的 create_* 保持一致。
    """

    async def gemini_complete(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: List[Any] = [],
        **kwargs,
    ) -> str:
        return await gemini_complete_if_cache(
            model_id,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )

    gemini_complete.__name__ = f"{model_id}_complete"
    return gemini_complete


async def gemini_best_complete(
    prompt: str, system_prompt: Optional[str] = None, history_messages: List[Any] = [], **kwargs
) -> str:
    """
    默认 best LLM：Gemini（模型名从环境变量 GEMINI_MODEL 读取）
    """
    model = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview")
    return await gemini_complete_if_cache(
        model,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gemini_cheap_complete(
    prompt: str, system_prompt: Optional[str] = None, history_messages: List[Any] = [], **kwargs
) -> str:
    """
    默认 cheap LLM：Gemini（模型名从环境变量 GEMINI_CHEAP_MODEL 读取）
    """
    model = os.getenv("GEMINI_CHEAP_MODEL", "gemini-3-flash-preview")
    # 便宜模型默认用 low 思考强度，可通过 GEMINI_CHEAP_THINKING_LEVEL 覆盖
    cheap_thinking = os.getenv("GEMINI_CHEAP_THINKING_LEVEL", "none")
    kwargs.setdefault("thinking_level", cheap_thinking)
    return await gemini_complete_if_cache(
        model,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=2048)
async def gemini_embedding(texts: list[str]) -> np.ndarray:
    """
    Gemini embeddings（默认用 gemini-embedding-001）。

    - 默认输出维度设为 1536，以兼容本项目现有向量库/时间编码逻辑
    - 模型名可通过环境变量 GEMINI_EMBEDDING_MODEL 覆盖
    - 支持多 key 并发
    """
    _load_dotenv_if_present()
    genai, types = _import_google_genai()

    # 优先使用多 key client
    try:
        multi_client = get_multi_key_gemini_client_instance()
        use_multi_key = len(multi_client.api_keys) > 1
    except Exception:
        use_multi_key = False
        multi_client = None

    if use_multi_key and multi_client is not None:
        # embedding 使用独立 semaphore：默认每个 key 仅 1 并发，不影响 generate_content 并发
        client, raw_sem = await multi_client.get_raw_client_with_embed_sem()
    else:
        client = get_google_genai_raw_client_instance()
        raw_sem = None

    model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    output_dim = int(os.getenv("GEMINI_EMBEDDING_DIM", "1536"))

    def _batch_stats() -> dict:
        # 仅做轻量统计，避免额外 tokenization 带来的开销
        lens = [len(t) for t in texts if isinstance(t, str)]
        if not lens:
            return {"n": 0}
        lens_sorted = sorted(lens)
        return {
            "n": len(lens),
            "min_len": lens_sorted[0],
            "p50_len": lens_sorted[len(lens_sorted) // 2],
            "p95_len": lens_sorted[max(0, int(len(lens_sorted) * 0.95) - 1)],
            "max_len": lens_sorted[-1],
        }

    def _extract_error_details(err: Exception) -> str:
        """
        尝试从不同 SDK/异常类型中提取可诊断信息（HTTP code、message、body 摘要等）。
        超时类异常通常拿不到 HTTP 响应，只能看到 timeout 本身。
        """
        parts: list[str] = []
        err_type = type(err).__name__
        err_msg = str(err) or repr(err)
        parts.append(f"[{err_type}] {err_msg}")

        # 常见字段兜底（不同库字段名不一）
        for attr in ("status_code", "code", "status", "http_status"):
            v = getattr(err, attr, None)
            if v is not None:
                parts.append(f"{attr}={v!r}")

        resp = getattr(err, "response", None)
        if resp is not None:
            try:
                sc = getattr(resp, "status_code", None)
                if sc is not None:
                    parts.append(f"response.status_code={sc!r}")
                # response text/body 可能很大，只取前 500 字符
                body = getattr(resp, "text", None) or getattr(resp, "content", None)
                if body is not None:
                    body_s = body.decode("utf-8", "ignore") if isinstance(body, (bytes, bytearray)) else str(body)
                    body_s = body_s.strip()
                    if body_s:
                        parts.append(f"response.body[:500]={body_s[:500]!r}")
            except Exception:
                pass

        return " | ".join(parts)

    def _call() -> list[list[float]]:
        cfg = None
        if types is not None:
            try:
                cfg = types.EmbedContentConfig(output_dimensionality=output_dim)
            except Exception:
                cfg = None

        try:
            if cfg is not None:
                resp = client.models.embed_content(model=model, contents=texts, config=cfg)
            else:
                resp = client.models.embed_content(model=model, contents=texts)
        except Exception as e:
            # 把 batch/模型信息一起打出来，便于判断是限流/输入问题/并发问题
            meta = {
                "model": model,
                "output_dim": output_dim,
                "use_multi_key": use_multi_key,
                "batch": _batch_stats(),
            }
            raise RuntimeError(f"Gemini embed_content failed: {meta} | {_extract_error_details(e)}") from e

        # 兼容不同 SDK 返回结构
        embs = getattr(resp, "embeddings", None)
        if embs is None:
            # 有的实现可能叫 embedding / values
            single = getattr(resp, "embedding", None)
            if single is not None:
                values = getattr(single, "values", None) or getattr(single, "value", None)
                if values is None:
                    raise RuntimeError("Gemini embedding 返回缺少 values 字段")
                return [list(values)]
            raise RuntimeError("Gemini embedding 返回缺少 embeddings 字段")

        vectors: list[list[float]] = []
        for e in embs:
            values = getattr(e, "values", None) or getattr(e, "value", None)
            if values is None:
                # 兜底：如果 e 本身就是 list
                if isinstance(e, list):
                    values = e
                else:
                    raise RuntimeError("Gemini embedding 返回项缺少 values")
            vectors.append(list(values))

        return vectors

    EMB_MAX_RETRIES = 4
    EMB_RETRY_BASE_DELAY = 5.0

    async def _call_with_retry() -> list[list[float]]:
        for attempt in range(EMB_MAX_RETRIES + 1):
            try:
                # 用统一的线程池 + 超时实现，避免卡死且超时信息更明确
                return await _run_in_thread_with_timeout(_call, GEMINI_CALL_TIMEOUT)
            except Exception as e:
                if attempt < EMB_MAX_RETRIES:
                    # 加一点随机抖动，避免所有任务在同一时间点同时重试造成“重试风暴”
                    jitter = random.uniform(0.5, 1.5)
                    delay = EMB_RETRY_BASE_DELAY * (2 ** attempt) * jitter
                    err_type = type(e).__name__
                    err_msg = str(e) or repr(e)
                    meta = {
                        "model": model,
                        "output_dim": output_dim,
                        "use_multi_key": use_multi_key,
                        "batch": _batch_stats(),
                        "gemini_call_timeout_s": GEMINI_CALL_TIMEOUT,
                    }
                    logger.warning(
                        f"Embedding failed (attempt {attempt + 1}/{EMB_MAX_RETRIES + 1}), "
                        f"retrying in {delay:.0f}s: [{err_type}] {err_msg} | meta={meta}"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

    # 多 key 模式使用 embedding 专用 semaphore，单 key 模式使用 embedding 专用全局 semaphore
    if use_multi_key and raw_sem is not None:
        t0 = time.monotonic()
        async with raw_sem:
            waited = time.monotonic() - t0
            if waited > 1.0:
                logger.info(f"Embedding waited {waited:.2f}s for per-key semaphore (multi-key)")
            vectors = await _call_with_retry()
    else:
        sem = _get_global_gemini_embedding_semaphore()
        t0 = time.monotonic()
        async with sem:
            waited = time.monotonic() - t0
            if waited > 1.0:
                logger.info(f"Embedding waited {waited:.2f}s for global embedding semaphore (single-key)")
            vectors = await _call_with_retry()

    arr = np.array(vectors, dtype=np.float32)
    return arr


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = get_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def amazon_bedrock_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    amazon_bedrock_async_client = get_amazon_bedrock_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    messages.extend(history_messages)
    messages.append({"role": "user", "content": [{"text": prompt}]})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    inference_config = {
        "temperature": 0,
        "maxTokens": 4096 if "max_tokens" not in kwargs else kwargs["max_tokens"],
    }

    async with amazon_bedrock_async_client.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "us-east-1")
    ) as bedrock_runtime:
        if system_prompt:
            response = await bedrock_runtime.converse(
                modelId=model, messages=messages, inferenceConfig=inference_config,
                system=[{"text": system_prompt}]
            )
        else:
            response = await bedrock_runtime.converse(
                modelId=model, messages=messages, inferenceConfig=inference_config,
            )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response["output"]["message"]["content"][0]["text"], "model": model}}
        )
        await hashing_kv.index_done_callback()
    return response["output"]["message"]["content"][0]["text"]


def create_amazon_bedrock_complete_function(model_id: str) -> Callable:
    """
    Factory function to dynamically create completion functions for Amazon Bedrock

    Args:
        model_id (str): Amazon Bedrock model identifier (e.g., "us.anthropic.claude-3-sonnet-20240229-v1:0")

    Returns:
        Callable: Generated completion function
    """
    async def bedrock_complete(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: List[Any] = [],
        **kwargs
    ) -> str:
        return await amazon_bedrock_complete_if_cache(
            model_id,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs
        )
    
    # Set function name for easier debugging
    bedrock_complete.__name__ = f"{model_id}_complete"
    
    return bedrock_complete


async def gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-3.5-turbo",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def amazon_bedrock_embedding(texts: list[str]) -> np.ndarray:
    amazon_bedrock_async_client = get_amazon_bedrock_async_client_instance()

    async with amazon_bedrock_async_client.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "us-east-1")
    ) as bedrock_runtime:
        embeddings = []
        for text in texts:
            body = json.dumps(
                {
                    "inputText": text,
                    "dimensions": 1024,
                }
            )
            response = await bedrock_runtime.invoke_model(
                modelId="amazon.titan-embed-text-v2:0", body=body,
            )
            response_body = await response.get("body").read()
            embeddings.append(json.loads(response_body))
    return np.array([dp["embedding"] for dp in embeddings])


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_embedding(texts: list[str]) -> np.ndarray:
    openai_async_client = get_openai_async_client_instance()
    response = await openai_async_client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_complete_if_cache(
    deployment_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    azure_openai_client = get_azure_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(deployment_name, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await azure_openai_client.chat.completions.create(
        model=deployment_name, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {
                args_hash: {
                    "return": response.choices[0].message.content,
                    "model": deployment_name,
                }
            }
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content


async def azure_gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await azure_openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def azure_gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await azure_openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_embedding(texts: list[str]) -> np.ndarray:
    azure_openai_client = get_azure_openai_async_client_instance()
    response = await azure_openai_client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])
