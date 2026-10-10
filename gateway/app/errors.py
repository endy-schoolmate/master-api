"""
统一错误处理系统 - v9.0

核心设计：
1. 统一异常层次结构（继承自 UpstreamError）
2. Provider 特定错误映射函数
3. 流式错误处理（含部分内容恢复）
4. Retry-After 头解析
5. 标准 OpenAI 格式错误响应
"""
import json
import re
import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger("acu.errors")


# ========== 统一异常层次结构 ==========

@dataclass
class ErrorDetails:
    """错误详情（OpenAI标准格式）"""
    message: str
    type: str = "invalid_request_error"
    code: str = "unknown"
    param: Optional[str] = None
    retry_after: Optional[int] = None
    provider: str = ""
    model: str = ""
    request_id: str = ""


class GatewayError(Exception):
    """网关错误基类"""
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        code: str = "gateway_error",
        error_type: str = "api_error",
        retry_after: Optional[int] = None,
        provider: str = "",
        model: str = "",
    ):
        self.message = message
        self.status_code = status_code
        self.code = code
        self.error_type = error_type
        self.retry_after = retry_after
        self.provider = provider
        self.model = model
        super().__init__(self.message)

    def to_openai_error(self) -> dict:
        """转换为 OpenAI 标准错误格式"""
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "code": self.code,
                "param": None,
            }
        }

    def to_response(self) -> tuple[dict, int]:
        """生成 (body, status_code) 响应"""
        return self.to_openai_error(), self.status_code


# 具体异常类型

class RateLimitError(GatewayError):
    """速率限制 (429) - 可重试"""
    def __init__(self, message: str = "请求过频", retry_after: int = 5, **kwargs):
        super().__init__(message, status_code=429, code="rate_limit_exceeded",
                         error_type="rate_limit_error", retry_after=retry_after, **kwargs)


class AuthenticationError(GatewayError):
    """认证失败 (401) - 不可重试"""
    def __init__(self, message: str = "认证失败", **kwargs):
        super().__init__(message, status_code=401, code="invalid_api_key",
                         error_type="invalid_request_error", **kwargs)


class InsufficientQuotaError(GatewayError):
    """配额不足 (402) - 不可重试"""
    def __init__(self, message: str = "配额不足", **kwargs):
        super().__init__(message, status_code=402, code="insufficient_quota",
                         error_type="insufficient_quota", **kwargs)


class ContentFilterError(GatewayError):
    """内容过滤 (400) - 不可重试"""
    def __init__(self, message: str = "内容被过滤", **kwargs):
        super().__init__(message, status_code=400, code="content_filter",
                         error_type="invalid_request_error", **kwargs)


class ContextWindowExceededError(GatewayError):
    """上下文超长 (400) - 视为不可重试"""
    def __init__(self, message: str = "上下文长度超限", **kwargs):
        super().__init__(message, status_code=400, code="context_length_exceeded",
                         error_type="invalid_request_error", **kwargs)


class TimeoutError_(GatewayError):
    """超时 (504) - 可重试"""
    def __init__(self, message: str = "上游响应超时", **kwargs):
        super().__init__(message, status_code=504, code="upstream_timeout",
                         error_type="timeout_error", **kwargs)


class UpstreamUnavailableError(GatewayError):
    """上游不可用 (503) - 可重试"""
    def __init__(self, message: str = "上游服务暂不可用", **kwargs):
        super().__init__(message, status_code=503, code="service_unavailable",
                         error_type="service_unavailable_error", **kwargs)


class BadRequestError(GatewayError):
    """请求参数错误 (400) - 不可重试"""
    def __init__(self, message: str = "请求参数错误", code: str = "bad_request", **kwargs):
        super().__init__(message, status_code=400, code=code,
                         error_type="invalid_request_error", **kwargs)


class AllKeysExhaustedError(GatewayError):
    """所有密钥耗尽 (503) - 可重试"""
    def __init__(self, message: str = "所有上游密钥暂不可用", **kwargs):
        super().__init__(message, status_code=503, code="all_keys_exhausted",
                         error_type="service_unavailable_error", **kwargs)


# 可重试状态码
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 408, 409, 425}

# 不可重试状态码
NON_RETRYABLE_STATUS_CODES = {400, 401, 402, 403, 404, 405, 406, 422}


# ========== Retry-After 头解析 ==========

RETRY_AFTER_HEADERS = [
    "retry-after-ms",
    "x-ms-retry-after-ms",
    "retry-after",
    "x-ratelimit-reset",
    "ratelimit-reset",
]


def parse_retry_after(response_headers) -> Optional[int]:
    """
    解析 Retry-After 头，返回毫秒数。
    """
    headers_lower = {}
    for k, v in (response_headers or {}).items():
        headers_lower[k.lower()] = v

    for header_name in RETRY_AFTER_HEADERS:
        value = headers_lower.get(header_name)
        if value is None:
            continue
        try:
            value_str = str(value).strip()
            if header_name == "retry-after":
                # 可能是秒数或HTTP日期
                try:
                    seconds = int(value_str)
                    return min(seconds * 1000, 60000)  # 最多60秒
                except ValueError:
                    # 尝试解析HTTP日期
                    from email.utils import parsedate_to_datetime
                    dt = parsedate_to_datetime(value_str)
                    delta = (dt - datetime.now(timezone.utc)).total_seconds()
                    return max(0, min(int(delta * 1000), 60000))
            elif header_name in ("retry-after-ms", "x-ms-retry-after-ms"):
                ms = int(value_str)
                return min(ms, 60000)
            else:
                # x-ratelimit-reset 可能是Unix秒
                reset_val = int(value_str)
                return min(reset_val * 1000, 60000)
        except (ValueError, TypeError):
            continue

    return None


# ========== 上游错误映射 ==========

def map_upstream_error(
    status_code: int,
    response_body: dict,
    provider: str = "nvidia",
    model: str = "",
) -> GatewayError:
    """
    将上游错误映射为统一 GatewayError。

    双重匹配策略：
    1. 先对错误消息进行特征字符串匹配（更精确）
    2. 再用 HTTP 状态码兜底
    """
    error_obj = response_body.get("error", response_body) if isinstance(response_body, dict) else {}
    error_message = ""
    if isinstance(error_obj, dict):
        error_message = error_obj.get("message", "") or error_obj.get("msg", "")
    elif isinstance(error_obj, str):
        error_message = error_obj
    else:
        error_message = str(response_body)

    error_message_lower = error_message.lower()

    # === 第一轮：HTTP 状态码快速匹配 ===
    if status_code == 429:
        return RateLimitError(error_message, provider=provider, model=model)

    # === 第二轮：字符串特征匹配（精确） ===
    # 上下文超长
    if any(kw in error_message_lower for kw in [
        "context length", "maximum context", "too many tokens",
        "prompt is too long", "max_tokens", "context_length_exceeded",
        "token limit", "window is too small",
    ]):
        return ContextWindowExceededError(error_message, provider=provider, model=model)

    # 内容过滤
    if any(kw in error_message_lower for kw in [
        "content_filter", "content filter", "safety", "harmful",
        "content_policy_violation", "inappropriate", "moderation",
    ]):
        return ContentFilterError(error_message, provider=provider, model=model)

    # 配额不足（排除已处理的429错误）
    if any(kw in error_message_lower for kw in [
        "insufficient_quota", "insufficient quota", "quota exceeded",
        "out of credits", "exceeded your current quota",
    ]):
        return InsufficientQuotaError(error_message, provider=provider, model=model)

    # 模型不存在
    if any(kw in error_message_lower for kw in [
        "model not found", "model_not_found", "not supported model",
        "does not exist", "no available model",
    ]):
        return BadRequestError(error_message, code="model_not_found", provider=provider, model=model)

    # === 第三轮：HTTP 状态码匹配（兜底） ===
    elif status_code in (401, 403):
        return AuthenticationError(error_message, provider=provider, model=model)
    elif status_code == 402:
        return InsufficientQuotaError(error_message, provider=provider, model=model)
    elif status_code in (502, 503):
        return UpstreamUnavailableError(error_message, provider=provider, model=model)
    elif status_code == 504:
        return TimeoutError_(error_message, provider=provider, model=model)
    elif status_code in (408, 409, 425):
        return RateLimitError(error_message, provider=provider, model=model)
    elif 400 <= status_code < 500:
        return BadRequestError(error_message, provider=provider, model=model)
    elif status_code >= 500:
        return GatewayError(error_message, status_code=status_code,
                           code="upstream_error", error_type="api_error",
                           provider=provider, model=model)

    return GatewayError(error_message, status_code=status_code,
                       code="upstream_error", error_type="api_error",
                       provider=provider, model=model)


# ========== 错误分类工具 ==========

def is_retryable_error(e: Exception) -> bool:
    """检查错误是否可重试"""
    if isinstance(e, GatewayError):
        return e.status_code in RETRYABLE_STATUS_CODES
    return True  # 连接级错误默认可重试


def is_non_retryable_error(e: Exception) -> bool:
    """检查错误是否不可重试"""
    if isinstance(e, GatewayError):
        return e.status_code in NON_RETRYABLE_STATUS_CODES
    return False


# ========== 流式错误处理 ==========

@dataclass
class PartialStreamResult:
    """部分流式结果（流式传输中断时保留已生成内容）"""
    generated_content: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    is_pre_first_chunk: bool = True
    chunks_received: int = 0


def should_stream_fallback(status_code: int, is_pre_first_chunk: bool) -> bool:
    """
    判断是否应该进行流式回退。

    原则：
    - 第一块之前的所有错误 → 可回退
    - 第一块之后的 429/5xx → 可回退（但会丢失部分内容）
    - 第一块之后的 4xx（非429）→ 不可回退（客户端错误）
    """
    if is_pre_first_chunk:
        return True  # 第一块之前，可以完整重试

    # 第一块之后
    if status_code == 429:
        return True  # 限流，可以重试
    if status_code >= 500:
        return True  # 服务端错误，可以重试

    return False  # 客户端错误，不可重试


def format_openai_chunk(data: dict) -> str:
    """格式化 OpenAI SSE chunk"""
    return f"data: {json.dumps(data)}\n\n"
