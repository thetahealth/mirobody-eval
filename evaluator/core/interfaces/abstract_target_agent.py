"""
AbstractTargetAgent — 被测目标代理抽象基类

职责：
- 封装被测系统（HTTP API / GUI / 其他系统）
- 将被测系统的接口适配为统一的 TargetAgentReaction
- 自动管理被测系统侧的对话记忆

注册机制（__init_subclass__）：
    class LlmApiTargetAgent(AbstractTargetAgent, name="llm_api"):
        ...
    查询: AbstractTargetAgent.get("llm_api")

生命周期：
- 创建：通过 TargetInfo 初始化
- 运行：在对话循环中响应测试代理行为
- 终止：当对话循环结束
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from datetime import datetime
from typing import ClassVar, Dict, List, Optional, Type

from evaluator.core.schema import (
    SessionInfo,
    TargetAgentMemory,
    TargetAgentReaction,
    TargetInfo,
    TestAgentAction,
)
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


class _PerUserRateLimiter:
    """Per-key sliding window rate limiter — 同一 key 在 window 内最多 max_requests 次"""

    def __init__(self, max_requests: int = 5, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window = window_seconds
        self._timestamps: dict[str, deque[float]] = defaultdict(deque)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def acquire(self, key: str) -> None:
        while True:
            async with self._locks[key]:
                now = time.monotonic()
                ts = self._timestamps[key]
                while ts and now - ts[0] > self.window:
                    ts.popleft()
                if len(ts) < self.max_requests:
                    ts.append(now)
                    return
                wait = self.window - (now - ts[0]) + 0.1
            logger.debug("[RateLimiter] key=%s 达到限速 (%d/%ds)，等待 %.1fs", key, self.max_requests, int(self.window), wait)
            await asyncio.sleep(wait)


# 全局单例 — 同一 user 60s 内最多 5 次请求
_RATE_LIMITER = _PerUserRateLimiter(max_requests=5, window_seconds=60.0)


class AbstractTargetAgent(ABC):
    """被测目标代理抽象基类"""

    _registry: ClassVar[Dict[str, Type["AbstractTargetAgent"]]] = {}
    _params_registry: ClassVar[Dict[str, type]] = {}  # name → TargetInfo config model

    def __init_subclass__(
        cls,
        *,
        name: str | None = None,
        params_model: type | None = None,
        **kwargs: object,
    ) -> None:
        super().__init_subclass__(**kwargs)
        if name is not None:
            if name in AbstractTargetAgent._registry:
                logger.warning(
                    "TargetAgent %r 重复注册，已覆盖: %s -> %s",
                    name,
                    AbstractTargetAgent._registry[name].__name__,
                    cls.__name__,
                )
            AbstractTargetAgent._registry[name] = cls
            if params_model is not None:
                AbstractTargetAgent._params_registry[name] = params_model

    @classmethod
    def get(cls, name: str) -> Type["AbstractTargetAgent"]:
        """根据名称获取已注册的 TargetAgent 类"""
        if name not in cls._registry:
            registered = list(cls._registry.keys())
            raise ValueError(f"未注册的 TargetAgent 类型: {name!r}，已注册: {registered or '(空)'}")
        return cls._registry[name]

    @classmethod
    def get_all(cls) -> Dict[str, Type["AbstractTargetAgent"]]:
        """获取所有已注册的 TargetAgent 类"""
        return dict(cls._registry)

    @classmethod
    def has(cls, name: str) -> bool:
        """检查指定名称的 TargetAgent 是否已注册"""
        return name in cls._registry

    def __init__(self, target_config: TargetInfo, history: list[BaseMessage] | None = None):
        self.target_config = target_config
        self.history: list[BaseMessage] = history or []
        self.memory_list: List[TargetAgentMemory] = []

    @property
    def rate_limit_key(self) -> str | None:
        """返回限速 key（如 user email），None 表示不限速。子类按需 override。"""
        return None

    @abstractmethod
    async def _generate_next_reaction(self, test_action: Optional[TestAgentAction]) -> TargetAgentReaction:
        """生成被测系统的响应（子类实现）

        实现要点：
        1. 将 TestAgentAction 转换为被测系统的输入
        2. 调用被测系统
        3. 将被测系统的输出转换为 TargetAgentReaction
        """
        ...

    async def do_generate(self, test_action: Optional[TestAgentAction]) -> TargetAgentReaction:
        """对外暴露的生成方法（框架实现，子类不需要重写）

        职责：
        1. 自动管理 memory（补全上一轮的 test_response）
        2. per-user 限速（rate_limit_key 非 None 时生效）
        3. 调用 _generate_next_reaction
        4. 记录新一轮 memory
        """
        # 补全上一轮 memory 的 test_response
        if self.memory_list and test_action is not None:
            last_memory = self.memory_list[-1]
            last_memory.test_response = test_action
            last_memory.test_response_time = datetime.now()

        # per-user 限速
        key = self.rate_limit_key
        if key is not None:
            await _RATE_LIMITER.acquire(key)

        # 生成目标响应
        target_reaction = await self._generate_next_reaction(test_action)

        # 记录新 memory
        self.memory_list.append(
            TargetAgentMemory(
                target_reaction=target_reaction,
                target_reaction_time=datetime.now(),
            )
        )

        return target_reaction

    def get_session_info(self) -> SessionInfo:
        """暴露给 EvalAgent 的会话元数据，子类按需 override"""
        return SessionInfo()
