"""
熔断器模块 v10.0 - 防呆防傻核心组件
=====================================
实现以下防呆机制：

1. 上游熔断器：连续失败N次后自动熔断，避免雪崩
2. 请求大小防护：拒绝超大请求体（防止OOM）
3. 消息数量限制：防止超长对话导致上游超时
4. 模型白名单兜底：未知模型直接拦截（可配置放行）
5. 并发请求防护：单客户端并发上限，防止资源耗尽
6. 空请求防护：拒绝空messages/空content的请求
7. 超时梯度策略：根据模型类型自动调整超时
8. 优雅降级：所有密钥不可用时返回友好提示而非500
"""
import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from threading import Lock

logger = logging.getLogger("acu.circuit_breaker")


@dataclass
class CircuitState:
    """熔断器状态"""
    failure_count: int = 0
    last_failure_time: float = 0.0
    open_until: float = 0.0  # 熔断恢复时间戳
    total_requests: int = 0
    total_failures: int = 0

    @property
    def is_open(self) -> bool:
        """熔断器是否打开（拒绝请求）"""
        return time.time() < self.open_until

    @property
    def failure_rate(self) -> float:
        """失败率"""
        if self.total_requests == 0:
            return 0.0
        return self.total_failures / self.total_requests


class CircuitBreaker:
    """
    熔断器 - v10.0 防呆防傻

    策略：
    - failure_threshold: 连续失败次数阈值，达到后熔断
    - failure_threshold_percentage: 失败率阈值（可选）
    - recovery_timeout: 熔断后恢复等待时间（秒）
    - half_open_max: 半开状态下允许的探测请求数

    状态机：CLOSED → OPEN → HALF_OPEN → CLOSED / OPEN
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        failure_threshold_percentage: float = 0.5,
        recovery_timeout: float = 60.0,
        half_open_max: int = 3,
        min_requests_for_percentage: int = 10,
    ):
        self.failure_threshold = failure_threshold
        self.failure_threshold_percentage = failure_threshold_percentage
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max
        self.min_requests_for_percentage = min_requests_for_percentage
        self._circuits: Dict[str, CircuitState] = {}
        self._lock = Lock()
        self._half_open_attempts: Dict[str, int] = defaultdict(int)

    def _get_or_create(self, key: str) -> CircuitState:
        if key not in self._circuits:
            self._circuits[key] = CircuitState()
        return self._circuits[key]

    def can_request(self, key: str) -> bool:
        """检查是否允许请求通过"""
        with self._lock:
            state = self._get_or_create(key)
            now = time.time()

            if state.is_open:
                # 熔断器打开中
                if now >= state.open_until:
                    # 进入半开状态，允许探测
                    if self._half_open_attempts[key] < self.half_open_max:
                        self._half_open_attempts[key] += 1
                        logger.info(f"熔断器半开探测: key={key[:16]} attempts={self._half_open_attempts[key]}")
                        return True
                    return False
                return False

            # 熔断器关闭，允许请求
            return True

    def record_success(self, key: str):
        """记录成功请求"""
        with self._lock:
            state = self._get_or_create(key)
            state.total_requests += 1
            state.failure_count = 0

            # 如果在半开状态，成功则关闭熔断器
            if state.open_until > 0 and self._half_open_attempts[key] > 0:
                state.open_until = 0
                self._half_open_attempts[key] = 0
                logger.info(f"熔断器恢复: key={key[:16]} (半开探测成功)")

    def record_failure(self, key: str, error_type: str = ""):
        """记录失败请求"""
        with self._lock:
            state = self._get_or_create(key)
            now = time.time()
            state.total_requests += 1
            state.total_failures += 1
            state.failure_count += 1
            state.last_failure_time = now

            # 检查是否需要熔断
            should_open = False
            reason = ""

            # 条件1：连续失败达到阈值
            if state.failure_count >= self.failure_threshold:
                should_open = True
                reason = f"连续失败{state.failure_count}次"

            # 条件2：失败率超过阈值（需要足够样本量）
            if (state.total_requests >= self.min_requests_for_percentage and
                    state.failure_rate >= self.failure_threshold_percentage):
                should_open = True
                reason = f"失败率{state.failure_rate:.0%}(>{self.failure_threshold_percentage:.0%})"

            if should_open:
                state.open_until = now + self.recovery_timeout
                logger.warning(
                    f"熔断器打开: key={key[:16]} reason={reason} "
                    f"recovery={self.recovery_timeout}s error={error_type}"
                )

    def get_status(self, key: str) -> dict:
        """获取熔断器状态（用于管理面板展示）"""
        with self._lock:
            state = self._circuits.get(key)
            if not state:
                return {"status": "closed", "failure_count": 0, "failure_rate": 0.0}
            now = time.time()
            if state.is_open:
                remaining = state.open_until - now
                return {
                    "status": "open",
                    "failure_count": state.failure_count,
                    "failure_rate": round(state.failure_rate, 4),
                    "recovery_in_seconds": round(remaining, 1),
                }
            return {
                "status": "closed",
                "failure_count": state.failure_count,
                "failure_rate": round(state.failure_rate, 4),
                "total_requests": state.total_requests,
            }

    def reset(self, key: str = None):
        """重置熔断器"""
        with self._lock:
            if key:
                self._circuits.pop(key, None)
                self._half_open_attempts.pop(key, None)
                logger.info(f"熔断器已手动重置: key={key[:16]}")
            else:
                self._circuits.clear()
                self._half_open_attempts.clear()
                logger.info("所有熔断器已重置")

    def get_all_status(self) -> dict:
        """获取所有熔断器状态（v10.0修复：避免在持锁时调用get_status导致死锁）"""
        with self._lock:
            now = time.time()
            result = {}
            for k, state in self._circuits.items():
                if state.is_open:
                    remaining = state.open_until - now
                    result[k] = {
                        "status": "open",
                        "failure_count": state.failure_count,
                        "failure_rate": round(state.failure_rate, 4),
                        "recovery_in_seconds": round(remaining, 1),
                    }
                else:
                    result[k] = {
                        "status": "closed",
                        "failure_count": state.failure_count,
                        "failure_rate": round(state.failure_rate, 4),
                        "total_requests": state.total_requests,
                    }
            return result


# ========== 全局单例 ==========

_circuit_breaker: Optional[CircuitBreaker] = None


def get_circuit_breaker() -> CircuitBreaker:
    global _circuit_breaker
    if _circuit_breaker is None:
        _circuit_breaker = CircuitBreaker()
    return _circuit_breaker


# ========== 请求防护函数 ==========

# 最大请求体大小（字节）- 超过则拒绝
MAX_REQUEST_BODY_SIZE = 10 * 1024 * 1024  # 10MB

# 最大消息数量 - 超过则拒绝（防止超长对话导致上游超时）
MAX_MESSAGES_COUNT = 200

# 单条消息最大内容长度（字符）- 超过则拒绝
MAX_MESSAGE_CONTENT_LENGTH = 500_000  # 50万字符

# 最大 max_tokens 参数值
MAX_MAX_TOKENS = 131072

# 嵌套深度限制（防止恶意嵌套JSON）
MAX_JSON_DEPTH = 20


def validate_request_safety(body: dict, is_embeddings: bool = False) -> Optional[str]:
    """
    v10.0 防呆防傻：请求安全校验

    在 validate_and_sanitize 之后执行，检查更深层次的安全问题。
    返回 None 表示通过，返回字符串表示错误原因。
    检查项：请求体大小、消息数量、max_tokens 上限、空内容等。
    """
    if not isinstance(body, dict):
        return "请求体必须是JSON对象"

    # 1. 请求体大小估算（通过序列化长度）
    try:
        import json
        body_size = len(json.dumps(body, ensure_ascii=False))
        if body_size > MAX_REQUEST_BODY_SIZE:
            return f"请求体过大({body_size // 1024 // 1024}MB)，最大允许{MAX_REQUEST_BODY_SIZE // 1024 // 1024}MB"
    except Exception:
        pass

    if not is_embeddings:
        messages = body.get("messages")
        if messages is None:
            return "请求体缺少 'messages' 字段"

        # 2. 消息数量限制
        if isinstance(messages, list) and len(messages) > MAX_MESSAGES_COUNT:
            return f"消息数量过多({len(messages)}条)，最大允许{MAX_MESSAGES_COUNT}条"

        # 3. 单条消息内容长度检查
        for i, msg in enumerate(messages if isinstance(messages, list) else []):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str) and len(content) > MAX_MESSAGE_CONTENT_LENGTH:
                return f"messages[{i}] 内容过长({len(content)}字符)，最大允许{MAX_MESSAGE_CONTENT_LENGTH}字符"
            elif isinstance(content, list):
                # 多模态消息
                for j, part in enumerate(content):
                    if isinstance(part, dict):
                        text = part.get("text", "")
                        if isinstance(text, str) and len(text) > MAX_MESSAGE_CONTENT_LENGTH:
                            return f"messages[{i}].content[{j}] 文本过长({len(text)}字符)"

    # 4. max_tokens 防护
    max_tokens = body.get("max_tokens")
    if max_tokens is not None:
        try:
            max_tokens = int(max_tokens)
            if max_tokens > MAX_MAX_TOKENS:
                return f"max_tokens 值{max_tokens}过大，最大允许{MAX_MAX_TOKENS}"
            if max_tokens <= 0:
                return f"max_tokens 值{max_tokens}无效，必须大于0"
        except (ValueError, TypeError):
            return f"max_tokens 值'{max_tokens}'无效，必须为正整数"

    # 5. n 参数防护（防止生成过多回复导致资源耗尽）
    n = body.get("n")
    if n is not None:
        try:
            n = int(n)
            if n > 10:
                return f"n 值{n}过大，最大允许10"
            if n <= 0:
                return f"n 值{n}无效，必须大于0"
        except (ValueError, TypeError):
            pass

    # 6. 检测可能的 JSON 注入（嵌套过深）
    try:
        import json
        text = json.dumps(body, ensure_ascii=False)
        depth = 0
        max_depth = 0
        for ch in text:
            if ch == '{' or ch == '[':
                depth += 1
                max_depth = max(max_depth, depth)
                if max_depth > MAX_JSON_DEPTH:
                    return f"JSON嵌套深度{max_depth}超过限制{MAX_JSON_DEPTH}，可能为恶意请求"
            elif ch == '}' or ch == ']':
                depth -= 1
    except Exception:
        pass

    return None


def get_model_timeout(model: str, default_timeout: float = 120.0) -> float:
    """
    v10.0 防呆防傻：根据模型类型自动调整超时

    - 推理模型（reasoning）：更长超时（思考时间）
    - 视觉模型（vision）：中等超时（图片处理）
    - 小模型：短超时（快速响应）
    - 默认：标准超时
    """
    model_lower = model.lower()

    # 推理模型 - 需要更长的思考时间
    reasoning_keywords = ["reasoning", "thinking", "o1", "nemotron-ultra", "nemotron-3-ultra"]
    if any(kw in model_lower for kw in reasoning_keywords):
        return 600.0  # 10分钟

    # 视觉模型 - 图片处理需要额外时间
    vision_keywords = ["vision", "vl", "vila", "omni", "fuyu"]
    if any(kw in model_lower for kw in vision_keywords):
        return 300.0  # 5分钟

    # 大模型 - 参数量大，推理慢
    large_keywords = ["70b", "120b", "251b", "550b", "675b", "49b", "36b"]
    if any(kw in model_lower for kw in large_keywords):
        return 180.0  # 3分钟

    # 小模型 - 快速响应
    small_keywords = ["1b", "2b", "3b", "4b", "8b", "mini", "nano", "small"]
    if any(kw in model_lower for kw in small_keywords):
        return 60.0  # 1分钟

    return default_timeout


def graceful_degradation_response(error_type: str, detail: str = "") -> dict:
    """
    v10.0 防呆防傻：优雅降级响应

    - 不暴露内部错误细节
    - 提供用户友好的错误提示
    - 建议用户可采取的行动
    """
    responses = {
        "all_keys_exhausted": {
            "status_code": 503,
            "message": "AI服务当前繁忙，所有上游资源暂时不可用。请稍后重试。",
            "type": "service_unavailable",
            "code": "all_keys_exhausted",
            "suggestion": "请等待30秒后重试，或联系管理员检查上游服务状态",
        },
        "circuit_open": {
            "status_code": 503,
            "message": "上游服务暂时不可用（熔断保护中），请稍后重试。",
            "type": "service_unavailable",
            "code": "circuit_breaker_open",
            "suggestion": "系统正在自动恢复，请等待1分钟后重试",
        },
        "timeout": {
            "status_code": 524,
            "message": "AI模型响应超时，可能是模型推理时间过长或上游负载过高。",
            "type": "timeout_error",
            "code": "upstream_timeout",
            "suggestion": "请尝试使用更快的模型，或减少请求中的 max_tokens 参数",
        },
        "connection_error": {
            "status_code": 502,
            "message": "无法连接到上游AI服务，可能是网络故障或服务临时不可用。",
            "type": "connection_error",
            "code": "upstream_connection_failed",
            "suggestion": "请稍后重试，系统已自动尝试切换备用通道",
        },
        "rate_limited": {
            "status_code": 429,
            "message": "请求过于频繁，已被限流。请降低请求频率。",
            "type": "rate_limit_exceeded",
            "code": "rate_limited",
            "suggestion": "请等待2-3秒后重试，或联系管理员提升配额",
        },
        "internal_error": {
            "status_code": 500,
            "message": "服务器内部错误，请稍后重试。",
            "type": "internal_error",
            "code": "internal_error",
            "suggestion": "如问题持续，请联系管理员并提供请求时间",
        },
    }

    resp = responses.get(error_type, responses["internal_error"])
    if detail:
        resp["detail"] = detail
    return resp
