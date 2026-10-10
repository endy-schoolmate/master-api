"""
平台适配器基类 - v9.0 多Provider支持

平台抽象模式：
- PlatformAdapter 接口定义统一操作契约
- BasePlatformAdapter 提供通用 HTTP 能力
- 具体适配器只需实现平台特有逻辑
"""
import time
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import httpx

logger = logging.getLogger("acu.platform")


# ========== 数据结构 ==========

@dataclass
class ModelInfo:
    """模型信息"""
    id: str
    object: str = "model"
    owned_by: str = ""
    display_name: str = ""
    context_length: int = 0
    max_output_tokens: int = 0
    capabilities: List[str] = field(default_factory=list)


@dataclass
class ChatResponse:
    """统一聊天响应"""
    id: str
    model: str
    choices: List[Dict]
    usage: Dict[str, int] = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    created: int = 0
    system_fingerprint: str = ""
    provider: str = ""


@dataclass
class StreamChunk:
    """流式chunk"""
    data: str  # SSE data line content
    is_done: bool = False
    usage: Optional[Dict] = None


@dataclass
class UpstreamHealth:
    """上游健康状态"""
    available: bool = True
    latency_ms: float = 0.0
    error: str = ""
    checked_at: float = 0.0


# ========== 统一异常层次 ==========

class UpstreamError(Exception):
    """上游错误基类"""
    def __init__(self, message: str, status_code: int = 500, code: str = "upstream_error"):
        self.message = message
        self.status_code = status_code
        self.code = code
        super().__init__(self.message)


class RateLimitError(UpstreamError):
    """速率限制错误 (429)"""
    def __init__(self, message: str = "上游限流", retry_after: int = 5):
        super().__init__(message, status_code=429, code="rate_limit_exceeded")
        self.retry_after = retry_after


class AuthenticationError(UpstreamError):
    """认证错误 (401/403)"""
    def __init__(self, message: str = "认证失败"):
        super().__init__(message, status_code=403, code="authentication_error")


class InsufficientQuotaError(UpstreamError):
    """配额不足 (402/403)"""
    def __init__(self, message: str = "配额不足"):
        super().__init__(message, status_code=402, code="insufficient_quota")


class BadRequestError(UpstreamError):
    """请求参数错误 (400/422)"""
    def __init__(self, message: str = "请求参数错误", code: str = "bad_request"):
        super().__init__(message, status_code=400, code=code)


class InternalServerError(UpstreamError):
    """上游服务器错误 (500)"""
    def __init__(self, message: str = "上游服务器错误"):
        super().__init__(message, status_code=500, code="upstream_internal_error")


class TimeoutError_(UpstreamError):
    """超时错误 (504)"""
    def __init__(self, message: str = "上游响应超时"):
        super().__init__(message, status_code=504, code="upstream_timeout")


# ========== 适配器接口 ==========

class PlatformAdapter(ABC):
    """平台适配器抽象基类"""

    @abstractmethod
    async def chat_completions(
        self, body: dict, api_key: str, base_url: str
    ) -> ChatResponse:
        """聊天补全（非流式）"""
        ...

    @abstractmethod
    async def chat_completions_stream(
        self, body: dict, api_key: str, base_url: str
    ):
        """聊天补全（流式） - 返回异步生成器"""
        ...

    @abstractmethod
    async def models(self, api_key: str, base_url: str) -> List[ModelInfo]:
        """获取模型列表"""
        ...

    @abstractmethod
    async def health_check(self, api_key: str, base_url: str) -> UpstreamHealth:
        """健康检查"""
        ...

    @abstractmethod
    def normalize_error(self, status_code: int, response_body: dict) -> UpstreamError:
        """将上游错误映射为统一格式"""
        ...

    @abstractmethod
    def normalize_model_id(self, model_id: str) -> str:
        """统一模型ID格式"""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """提供商名称"""
        ...

    @property
    @abstractmethod
    def default_base_url(self) -> str:
        """默认基础URL"""
        ...


class BaseHTTPAdapter(PlatformAdapter):
    """提供通用HTTP能力的抽象适配器基类"""

    def __init__(self):
        self._pool: Optional[httpx.AsyncClient] = None

    async def _get_pool(self) -> httpx.AsyncClient:
        if self._pool is None or self._pool.is_closed:
            self._pool = httpx.AsyncClient(
                timeout=httpx.Timeout(600.0, connect=30.0),
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            )
        return self._pool

    async def _request(
        self, method: str, url: str, headers: dict, json_body: dict = None,
        timeout: float = 60.0, stream: bool = False
    ) -> httpx.Response:
        """发送HTTP请求"""
        pool = await self._get_pool()
        kwargs = {"headers": headers}
        if json_body:
            kwargs["json"] = json_body
        if stream:
            req = pool.build_request(method, url, **kwargs, timeout=httpx.Timeout(timeout))
            return await pool.send(req, stream=True)
        return await pool.request(method, url, **kwargs, timeout=httpx.Timeout(timeout))

    async def close(self):
        if self._pool and not self._pool.is_closed:
            await self._pool.aclose()

    def normalize_error(self, status_code: int, response_body: dict) -> UpstreamError:
        """默认错误映射 - OpenAI格式"""
        err = response_body.get("error", response_body) if isinstance(response_body, dict) else {}
        msg = err.get("message", "") if isinstance(err, dict) else str(response_body)

        if status_code == 429:
            retry_after = 5
            if isinstance(response_body, dict) and "error" in response_body:
                e = response_body["error"]
                if isinstance(e, dict):
                    retry_after = int(e.get("retry_after", 5))
            return RateLimitError(msg, retry_after=retry_after)
        elif status_code in (401, 403):
            return AuthenticationError(msg)
        elif status_code == 402:
            return InsufficientQuotaError(msg)
        elif 400 <= status_code < 500:
            return BadRequestError(msg)
        elif status_code >= 500:
            return InternalServerError(msg)
        return UpstreamError(msg, status_code)

    def normalize_model_id(self, model_id: str) -> str:
        return model_id


# ========== 适配器注册中心 ==========

class PlatformAdapterRegistry:
    """平台适配器注册中心"""

    _adapters: Dict[str, PlatformAdapter] = {}

    @classmethod
    def register(cls, name: str, adapter: PlatformAdapter):
        """注册适配器"""
        cls._adapters[name] = adapter
        logger.info(f"平台适配器已注册: {name} ({adapter.__class__.__name__})")

    @classmethod
    def get(cls, name: str) -> Optional[PlatformAdapter]:
        """获取适配器"""
        return cls._adapters.get(name)

    @classmethod
    def get_or_default(cls, name: str) -> PlatformAdapter:
        """获取适配器，未注册则返回默认"""
        adapter = cls._adapters.get(name)
        if adapter is None:
            # 未注册时尝试用名称匹配provider
            for key, a in cls._adapters.items():
                if name.startswith(key) or key.startswith(name):
                    return a
            # 返回 NvidiaAdapter 作为默认
            return cls._adapters.get("nvidia", list(cls._adapters.values())[0] if cls._adapters else None)
        return adapter

    @classmethod
    def list_providers(cls) -> List[str]:
        """列出所有已注册的提供商"""
        return list(cls._adapters.keys())

    @classmethod
    async def close_all(cls):
        """关闭所有适配器连接池"""
        for name, adapter in cls._adapters.items():
            if hasattr(adapter, 'close'):
                try:
                    await adapter.close()
                except Exception as e:
                    logger.warning(f"关闭适配器 {name} 连接池失败: {e}")
