"""
公开API路由 - OpenAI兼容端点

端点列表：
- POST /v1/chat/completions    OpenAI兼容聊天补全（支持流式）
- POST /api/v1/chat/completions /api/v1前缀兼容路径
- GET  /v1/models              OpenAI兼容模型列表
- GET  /api/v1/models          /api/v1前缀兼容路径
- GET  /api/public/models      公开模型列表（无需认证）
- POST /v1/embeddings          OpenAI兼容向量化

关键流程：
1. 客户端认证（下游密钥验证）
2. 调度器选择密钥（select_key）
3. 上游NVIDIA API调用
4. 流式SSE响应 / 非流式响应
5. 错误处理（429自动切换密钥，tenacity自动重试）
6. 请求日志记录

 流式稳定性优化：
- 流式per-chunk空闲超时检测（替代全局超时）
- SSE keepalive心跳（30秒间隔发送": ping"注释行）
- 流式中途断连优雅处理（发送错误SSE事件 + [DONE]）
- 差异化403/429/timeout冷却（配合scheduler.py的差异化策略）
"""
import asyncio
import json
import re
import threading
import time
import traceback
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse, Response
from pydantic import BaseModel, ConfigDict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.database import (
    fetch_one, fetch_all, execute, get_setting, insert_audit, utcnow, localnow,
)
from app.security import (
    decrypt_secret, decrypt_upstream_key, hash_secret, mask_secret,
    verify_admin_token,
)
from app.scheduler import (
    get_scheduler, get_threshold_for_model, get_timeout_for_model,
    apply_model_defaults, REASONING_MODELS, SLOW_MODELS,
    STREAM_FIRST_BYTE_TIMEOUT, STREAM_IDLE_TIMEOUT,
)
from app.middleware import is_cc_switch_request, get_client_ip
from app.request_validator import (
    validate_and_correct_model, build_model_error_suggestion,
    clean_and_validate_api_key, normalize_base_url,
    validate_and_sanitize, refresh_verified_models,
)
from app.circuit_breaker import (
    get_circuit_breaker,
    get_model_timeout, graceful_degradation_response,
)
from app.responses_bridge import (
    OAUTH_PROVIDER_ID, RESPONSES_PROVIDER_IDS, chat_to_responses, responses_to_chat, translate_sse_lines,
)
from app.openai_oauth import parse_envelope, ensure_fresh_access, OAuthError, get_proxied_client

logger = logging.getLogger("acu.api")

router = APIRouter()

# ========== 多 Provider 上游路由辅助 ==========
# 原项目把所有上游都打到 admin_settings.upstream_base_url 一个固定地址（默认 NVIDIA NIM）。
# 这里加 per-key provider + base_url 解析：每个 upstream_keys 行可独立声明走哪个厂商、用哪个 base_url。
# 适配器（NvidiaAdapter / OpenAIAdapter）已写好 provider_name/default_base_url，
# 此函数把 PlatformAdapterRegistry 透出供调用方选择适配器。

def _get_adapter_registry():
    """惰性获取平台适配器注册中心（避免循环引用）"""
    try:
        from app.platforms.base import PlatformAdapterRegistry
        return PlatformAdapterRegistry
    except Exception as e:
        logger.warning(f"获取适配器注册中心失败: {e}")
        return None


def _join_upstream_url(base_url: str, original_url: str) -> str:
    """保留原 url 的"服务路径后缀"（如 /chat/completions），拼到新 base_url 后面。

    原 url 是用全局 admin_settings.upstream_base_url (例 NVIDIA: .../v1) 拼出来的，
    形如 https://integrate.api.nvidia.com/v1/chat/completions
    新 base_url 例 https://open.bigmodel.cn/api/paas/v4
    要拼成     https://open.bigmodel.cn/api/paas/v4/chat/completions
    ——所以不能直接套用原 url 全 path（会出现 /v4/v1/chat/completions），
    而应只保留"服务后缀"（端点名那段）。

    实现策略：从 original_url 取 path 段，剥掉 schema://host 后剩余 — 然后用
    "/chat/completions"/"/embeddings"/"/models" 等"服务路径标识"作为分界，
    取标识及其之后的子串。

    若分界识别不到（兼容新协议端点），就退回到 base_url 直接拼末尾 `/chat/completions`。
    """
    if not original_url:
        return base_url
    try:
        scheme_end = original_url.find("://") + 3
        path_slash = original_url.find("/", scheme_end)
        if path_slash == -1:
            suffix = "/chat/completions"
        else:
            path = original_url[path_slash:]
            # 找服务路径标识出现位置；优先 chat_completions，再 embeddings，再 models
            for marker in ("/chat/completions", "/completions", "/embeddings",
                           "/generateContent", "/messages", "/models"):
                idx = path.find(marker)
                if idx != -1:
                    suffix = path[idx:]
                    break
            else:
                # 没识别到，退回 base_url 末尾拼个 /chat/completions 兜底
                suffix = "/chat/completions"
    except Exception:
        return original_url

    base = base_url.rstrip("/")
    return f"{base}{suffix}"

# ========== 流式稳定性配置 ==========
# SSE keepalive心跳间隔（秒）- 在流式传输空闲期间发送": ping"注释行
# 防止客户端/Nginx因长时间无数据而断开连接
SSE_KEEPALIVE_INTERVAL = 15  # 15秒发送一次心跳
# 流式per-chunk空闲超时（秒）- 两个SSE事件之间的最大间隔
# 超过此时间未收到上游数据则视为超时
STREAM_CHUNK_IDLE_TIMEOUT = 180  # 3分钟（推理模型思考时间可能很长）


# ========== 自定义异常 ==========

class UpstreamRetryableError(Exception):
    """上游瞬时/密钥级错误；可换 key 或切换聚合候选。

    kind 是错误分级标签，聚合链据此决定"同一候选模型内还重试几次"：
    - timeout（读/响应超时，等待成本高）与 no_key（无可用密钥，重选无意义）
      立即切换下一个候选；
    - 其余（429/401/403/5xx/网络/空响应等）保留换 key 重试机会，容忍偶发抖动。
    """

    def __init__(self, message: str, status_code: int | None = None, kind: str = "upstream_error"):
        self.status_code = status_code
        self.kind = kind
        super().__init__(message)


class UpstreamRequestError(Exception):
    """请求级终态错误，切换 key 或模型也不会改善本次请求。"""

    def __init__(self, status_code: int, error_body: dict):
        self.status_code = status_code
        self.error_body = error_body
        super().__init__(_safe_error_message(error_body))


class UpstreamModelUnavailableError(Exception):
    """确定的模型级错误；聚合链应跳到下一个候选模型。"""

    def __init__(self, status_code: int, error_body: dict):
        self.status_code = status_code
        self.error_body = error_body
        super().__init__(_safe_error_message(error_body))


def _safe_error_message(error_body: object) -> str:
    """提取有长度限制且不含密钥的上游错误摘要。"""
    if isinstance(error_body, dict):
        error = error_body.get("error", error_body)
        if isinstance(error, dict):
            text = error.get("message") or error.get("detail") or "上游请求失败"
        else:
            text = error
    else:
        text = error_body
    text = str(text or "上游请求失败").replace("\r", " ").replace("\n", " ")
    import re
    text = re.sub(
        r"(?i)(bearer\s+|api[_-]?key[=:]\s*|token[=:]\s*|secret[=:]\s*|password[=:]\s*)[^\s,;\"']+",
        "[REDACTED]",
        text,
    )
    return text[:240]


def _parse_upstream_error_body(error_text: str) -> dict:
    try:
        body = json.loads(error_text)
    except Exception:
        body = {"error": {"message": error_text}}
    if not isinstance(body, dict):
        body = {"error": {"message": str(body)}}
    if "error" not in body or not isinstance(body["error"], dict):
        body = {"error": {"message": _safe_error_message(body)}}
    body["error"]["message"] = _safe_error_message(body["error"])
    return body


def _is_model_not_found(status_code: int, error_body: dict) -> bool:
    if status_code != 404:
        return False
    message = _safe_error_message(error_body).lower()
    return "model" in message and any(
        marker in message for marker in ("not found", "not exist", "unknown", "不存在", "未部署")
    )


def _should_failover_after_request_error(error: UpstreamRequestError) -> bool:
    """Return whether a 4xx response may be specific to one chain candidate."""
    if error.status_code != 400:
        return False

    message = _safe_error_message(error.error_body).lower()
    terminal_markers = (
        "content policy",
        "content_filter",
        "content filter",
        "policy violation",
        "safety",
        "moderation",
        "malformed",
        "invalid json",
        "missing required",
        "messages must",
        "request body",
        "鍐呭",
        "绛栫暐",
        "瀹夊叏",
    )
    return not any(marker in message for marker in terminal_markers)


def _attempt_row_status(error: Exception) -> int:
    """候选尝试行的状态码：无上游状态时按错误分级映射（与链尾返回给客户端的口径一致）。"""
    kind = getattr(error, "kind", "") or ""
    status = getattr(error, "status_code", None)
    if isinstance(status, int) and status > 0:
        return status
    if kind == "timeout":
        return 504
    if kind == "no_key":
        return 503
    return 502  # network/empty/upstream_error/其他：上游侧故障


def _log_chain_attempt_row(request: Request | None, model_ref: str, error: Exception, attempt_started: float | None, body: dict | None = None) -> None:
    """把一次候选尝试独立落一行请求日志（log_category=chain_attempt）。

    聚合模型的一次客户端请求会展开成多个真实模型尝试，只有行级记录
    才能在"请求日志"里直接看到每个备用模型的具体报错（而不是只看到
    虚拟模型 ID 的最终一行）。失败写入放后台线程，不阻塞链路切换。
    """
    try:
        if isinstance(error, UpstreamRetryableError):
            kind = getattr(error, "kind", "") or "upstream_error"
            reason = {
                "timeout": "超时",
                "connect_timeout": "连接超时",
                "network": "网络错误",
                "no_key": "无可用密钥",
                "rate_limited": "上游限流",
                "auth": "密钥认证失败",
                "forbidden": "上游拒绝",
                "server": "上游服务错误",
                "empty": "空响应",
                "oauth": "OAuth凭据失败",
            }.get(kind, "瞬时失败")
            text = _safe_error_message(str(error))
        elif isinstance(error, UpstreamModelUnavailableError):
            kind, reason, text = "model_unavailable", "模型不可用", _safe_error_message(error.error_body)
        elif isinstance(error, UpstreamRequestError):
            kind, reason, text = "request_error", "请求级错误", _safe_error_message(error.error_body)
        else:
            kind, reason, text = type(error).__name__, "异常", str(error)

        now = time.time()
        rt = max(0.0, now - attempt_started) if attempt_started else 0.0
        state = getattr(request, "state", None)
        try:
            client_ip = get_client_ip(request) if request is not None else ""
        except Exception:
            client_ip = ""
        try:
            ua = (request.headers.get("user-agent", "") or "")[:512] if request is not None else ""
        except Exception:
            ua = ""
        # 该候选实际发出的请求体：脱敏（messages 等省略）+ 截断，供详情核对参数是否被剥离
        attempt_body = ""
        if body:
            try:
                from app.middleware import RequestLoggingMiddleware
                attempt_body = RequestLoggingMiddleware._sanitize_request_body(
                    json.dumps(body, ensure_ascii=False)
                )[:4096]
            except Exception:
                attempt_body = ""
        _log_request_sync(
            client_id=getattr(state, "log_client_id", "") or "",
            key_id="",
            model=model_ref,
            status_code=_attempt_row_status(error),
            rt=rt,
            prompt_tokens=0, completion_tokens=0, total_tokens=0,
            is_stream=bool(getattr(state, "log_is_stream", False)),
            error_msg=f"{reason}: {text}"[:512],
            start_ts=attempt_started,
            error_type="ChainAttempt",
            error_detail=f"{reason}: {text}"[:1024],
            business_code=kind,
            request_path=str(getattr(getattr(request, "url", None), "path", "") or ""),
            http_method=getattr(request, "method", ""),
            client_ip=client_ip,
            user_agent=ua,
            request_body=attempt_body,
            include_errors=True,
            force_log_category="chain_attempt",
        )
    except Exception:
        logger.debug("记录候选尝试日志失败", exc_info=True)


# ========== 候选参数自适应（上游 400 "unsupported parameter" 剥离与学习） ==========
# ZCode 等客户端会给所有请求附带 reasoning / enable_thinking / reasoning_effort 等
# 非标准参数，不同上游接受度不一。某候选报"参数不支持"时：解析出参数名 →
# 从请求体剥离 → 同候选原地重试，并把参数记入该候选的黑名单，
# 同一次请求中排在后面的候选会预防性应用同一黑名单，不再白白消耗候选。

_PARAM_BLOCKLIST_MAX_MODELS = 256
_param_blocklist: dict[str, set[str]] = {}  # real_model_ref -> 该上游不接受的参数名
_param_blocklist_lock = threading.Lock()

_ESSENTIAL_BODY_KEYS = {"model", "messages", "stream", "stream_options", "tools", "tool_choice"}

_UNSUPPORTED_PARAM_RE = re.compile(
    r"unsupported parameter|unrecognized request argument|unknown (?:parameter|argument)"
    r"|invalid parameter|does not support parameter|is not supported|not supported|不支持参数|参数不支持",
    re.I,
)
_PARAM_NAME_RE = re.compile(r"[`'\"“”]([a-zA-Z_][a-zA-Z0-9_]{0,40})[`'\"“”]|supplied:\s*([a-zA-Z_][a-zA-Z0-9_]{0,40})")


def _match_unsupported_params(message: str) -> list[str]:
    """从上游错误消息里解析"不被支持的参数名"列表（最多6个）。"""
    essential_lower = {k.lower() for k in _ESSENTIAL_BODY_KEYS}
    names: list[str] = []

    # 优先尝试作为 JSON 解析，读取 param 或 error.param 字段
    try:
        data = json.loads(message)
        if isinstance(data, dict):
            p = data.get("param") or (data.get("error", {}).get("param") if isinstance(data.get("error"), dict) else None)
            if p and isinstance(p, str) and p.lower() not in essential_lower and p not in names:
                names.append(p)
    except Exception:
        pass

    msg_str = str(message or "")
    if "unexpected reasoning effort" in msg_str.lower() or "reasoning effort" in msg_str.lower():
        if "reasoning_effort" not in names:
            names.append("reasoning_effort")

    if not names and not _UNSUPPORTED_PARAM_RE.search(msg_str):
        return []

    for mt in _PARAM_NAME_RE.finditer(msg_str):
        name = mt.group(1) or mt.group(2)
        if name and name.lower() not in essential_lower and name not in names:
            names.append(name)
        if len(names) >= 6:
            break
    return names


def _adapt_reasoning_effort_for_provider(body: dict, provider: str = "", model_ref: str = "", base_url: str = "") -> None:
    """自适应调整 reasoning_effort 枚举值，彻底解决 AMD / OpenAI 规范不兼容导致的 502/400 报错。

    AMD / Qwen 节点仅支持: xhigh (default), medium, low（不支持 high，不支持 none/minimal）
    OpenAI / 标准节点支持: high, medium, low, none（不支持 xhigh）
    """
    if "reasoning_effort" not in body:
        return
    val = str(body["reasoning_effort"]).strip().lower()
    is_amd = (
        (provider or "").lower() == "amd"
        or "amd" in (model_ref or "").lower()
        or "amd.com" in (base_url or "").lower()
        or "qwen" in (model_ref or "").lower()
    )
    if is_amd:
        if val in ("high", "ultra", "max", "highest", "xhigh"):
            body["reasoning_effort"] = "xhigh"
        elif val in ("none", "minimal", "disabled", "false"):
            body.pop("reasoning_effort", None)
        elif val not in ("xhigh", "medium", "low"):
            body["reasoning_effort"] = "xhigh"
    else:
        if val in ("xhigh", "ultra", "max", "highest"):
            body["reasoning_effort"] = "high"
        elif val in ("minimal",):
            body["reasoning_effort"] = "low"


def _strip_blocked_params(real_model: str, sub_body: dict) -> list[str]:
    """按学习到的黑名单预防性剥离参数；返回实际剥离的名单。"""
    with _param_blocklist_lock:
        blocked = _param_blocklist.get(real_model)
    if not blocked:
        return []
    stripped = [k for k in list(sub_body.keys()) if k in blocked]
    for k in stripped:
        sub_body.pop(k, None)
    return stripped


def _learn_param_blocklist(real_model: str, names: list[str]) -> None:
    """把解析出的不支持参数记入该候选的黑名单（跨请求生效，供预防性剥离）。"""
    if not names:
        return
    with _param_blocklist_lock:
        blk = _param_blocklist.setdefault(real_model, set())
        blk.update(names)
        if len(_param_blocklist) > _PARAM_BLOCKLIST_MAX_MODELS:
            _param_blocklist.pop(next(iter(_param_blocklist)))


def _strip_params_from_body(sub_body: dict, names: list[str]) -> list[str]:
    """从请求体剥离指定参数；返回实际剥离的名单。"""
    stripped = [n for n in names if n in sub_body]
    for n in stripped:
        sub_body.pop(n, None)
    return stripped


def _chain_failure_status(last_kind: str, last_status: int | None) -> tuple[int, str, str]:
    """把链尾最后一个候选的失败类别映射为返回给客户端的状态码/业务码。

    一律返回 503 会让"请求日志里的错误码"与"客户端实际收到的"对不上
    （例如上游超时客户端却收到 503）。映射规则：超时→504、限流→429、
    上游5xx→502，其余（全冷却/无密钥/请求级错误）→503。
    """
    if last_kind == "timeout":
        return 504, "virtual_model_upstream_timeout", "timeout_error"
    if last_kind == "rate_limited" or last_status == 429:
        return 429, "virtual_model_rate_limited", "rate_limit_exceeded"
    if last_kind in ("server", "connect_timeout") or (isinstance(last_status, int) and last_status >= 500):
        return 502, "virtual_model_upstream_error", "bad_gateway"
    return 503, "virtual_model_all_failed", "service_unavailable"


def _record_upstream_attempt(request: Request | None, model_ref: str, error: Exception, attempt_started: float | None = None, body: dict | None = None) -> None:
    """将受限的候选失败上下文交给中间件写入管理员日志。"""
    # 每个候选尝试独立一行（chain_attempt），聚合失败时请求日志可直接看到各备用模型报错
    import threading as _threading
    _threading.Thread(
        target=_log_chain_attempt_row, args=(request, model_ref, error, attempt_started, body), daemon=True
    ).start()

    if request is None or not hasattr(request, "state"):
        return
    provider, model = _split_provider_model_ref(model_ref)
    attempts = getattr(request.state, "upstream_attempts", [])
    if len(attempts) >= 16:
        return
    status = getattr(error, "status_code", None)
    if isinstance(error, UpstreamRetryableError):
        text, reason = _safe_error_message(str(error)), getattr(error, "kind", "") or "retryable"
    elif isinstance(error, UpstreamModelUnavailableError):
        text, reason = _safe_error_message(error.error_body), "model_unavailable"
    elif isinstance(error, UpstreamRequestError):
        text, reason = _safe_error_message(error.error_body), "request_error"
    else:
        text, reason = _safe_error_message(str(error)), type(error).__name__
    attempts.append({
        "provider": provider,
        "model": model,
        "upstream_status": status,
        "reason": reason,
        "error": text,
    })
    request.state.upstream_attempts = attempts


# ========== 请求模型 ==========

class ChatRequest(BaseModel):
    """聊天请求 - 全量透传，支持所有模型参数"""
    model: str
    messages: list
    stream: Optional[bool] = False
    model_config = ConfigDict(extra="allow")


# ========== 客户端认证 ==========

def authenticate_client(request: Request) -> dict:
    """
    验证下游客户端API密钥

    支持两种认证方式：
    1. Authorization: Bearer sk-xxx
    2. x-api-key: sk-xxx
    """
    # 提取密钥
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        raw_token = auth[7:]
    else:
        raw_token = request.headers.get("x-api-key", "")

    # === 管理后台网页对话：管理员登录态直接放行（免 API Key 鉴权） ===
    # 控制台已持有管理员Token（HMAC签名），网页对话用它直调 /v1/*，
    # 无需再配置网关 API Key；仅当 Token 校验通过时放行，否则走下方 Key 校验。
    if raw_token and "." in raw_token and not raw_token.startswith(("acu_", "sk-")):
        _admin_secret = get_setting("gateway_secret")
        if _admin_secret and verify_admin_token(raw_token, _admin_secret):
            return {
                "id": "",
                "client_id": "admin-web-chat",
                "status": "active",
                "client_name": "管理后台网页对话",
                "client_status": "active",
                "user_type": "admin",
            }

    # === 本地 ChatGPT Desktop / Codex 客户端：PROXY_MANAGED / OAuth 放行 ===
    client_ip = get_client_ip(request)
    if (raw_token in ("PROXY_MANAGED", "") or raw_token.startswith("eyJ")) and client_ip in ("127.0.0.1", "::1", "localhost", "testclient"):
        first_key = fetch_one(
            "SELECT ck.id, ck.client_id, ck.status, c.name as client_name, c.status as client_status, "
            "COALESCE(c.user_type, 'old') as user_type "
            "FROM client_api_keys ck "
            "JOIN clients c ON ck.client_id = c.id "
            "WHERE ck.status = 'active' AND c.status = 'active' LIMIT 1"
        )
        if first_key:
            return dict(first_key)
        return {
            "id": "",
            "client_id": "codex-local",
            "status": "active",
            "client_name": "ChatGPT 本地桌面端",
            "client_status": "active",
            "user_type": "admin",
        }

    # === v9.2: API Key 清洗（去除空白/换行/多余引号） ===
    cleaned_token, key_error = clean_and_validate_api_key(raw_token)
    if key_error:
        # 测试/演示key直接拦截
        raise HTTPException(status_code=401, detail={
            "message": key_error,
            "type": "invalid_request_error",
            "code": "invalid_api_key",
        })

    if not cleaned_token or not cleaned_token.startswith(("acu_", "sk-")):
        raise HTTPException(status_code=401, detail={
            "message": "无效的API密钥",
            "type": "invalid_request_error",
            "code": "invalid_api_key",
        })

    # 查询密钥（包含 user_type）— 优先使用缓存
    key_hash = hash_secret(cleaned_token)
    
    # 检查缓存
    from app.scheduler import get_scheduler
    scheduler = get_scheduler()
    cached = scheduler.get_client_key_cache(key_hash)
    if cached is not None:
        return dict(cached)

    key_row = fetch_one(
        "SELECT ck.id, ck.client_id, ck.status, c.name as client_name, c.status as client_status, "
        "COALESCE(c.user_type, 'old') as user_type "
        "FROM client_api_keys ck "
        "JOIN clients c ON ck.client_id = c.id "
        "WHERE ck.key_hash = %s AND ck.status = 'active' AND c.status = 'active'",
        (key_hash,),
    )

    if not key_row:
        raise HTTPException(status_code=401, detail={
            "message": "API密钥无效或已被禁用",
            "type": "invalid_request_error",
            "code": "invalid_api_key",
        })

    # v10.0: last_used_at 更新改为后台线程（避免同步DB写阻塞事件循环）
    try:
        import threading
        _now = utcnow()
        _kid = key_row["id"]
        threading.Thread(
            target=lambda: _async_update_last_used(_kid, _now),
            daemon=True
        ).start()
    except Exception:
        pass

    # 写入缓存
    result = dict(key_row)
    scheduler.set_client_key_cache(key_hash, result)
    return result


def _async_update_last_used(key_id, ts):
    """Background thread: update last_used_at without blocking event loop"""
    try:
        execute("UPDATE client_api_keys SET last_used_at = %s WHERE id = %s", (ts, key_id))
    except Exception:
        pass


# ========== 模型列表缓存 ==========

_models_cache = {"data": None, "expires": 0}

# 经过实际API调用验证的可用模型集合（硬编码白名单+运行时验证）
# NVIDIA NIM的/v1/models端点会列出所有平台模型，但很多实际上无法通过chat/completions调用
# 这里只保留实测可用的模型ID，确保用户看到的每个模型都能实际使用
_VERIFIED_WORKING_MODELS = {
    # === deepseek-ai ===
    "deepseek-ai/deepseek-v4-flash",
    "deepseek-ai/deepseek-v4-pro",
    # === z-ai ===
    "z-ai/glm-5.2",
    # === qwen ===
    "qwen/qwen3.5-397b-a17b",
    "qwen/qwen3-next-80b-a3b-instruct",
    # === minimaxai ===
    "minimaxai/minimax-m3",
    "minimaxai/minimax-m2.7",
    # === stepfun-ai ===
    "stepfun-ai/step-3.5-flash",
    "stepfun-ai/step-3.7-flash",
    # === openai ===
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    # === meta ===
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.1-8b-instruct",
    "meta/llama-3.2-11b-vision-instruct",
    "meta/llama-3.2-1b-instruct",
    "meta/llama-3.2-3b-instruct",
    "meta/llama-3.2-90b-vision-instruct",
    "meta/llama-3.3-70b-instruct",
    "meta/llama-guard-4-12b",
    # === mistralai ===
    "mistralai/mistral-large-3-675b-instruct-2512",
    "mistralai/mistral-medium-3.5-128b",
    "mistralai/mistral-nemotron",
    "mistralai/mistral-small-4-119b-2603",
    "mistralai/mixtral-8x7b-instruct-v0.1",
    # === google ===
    "google/gemma-2-2b-it",
    "google/gemma-3n-e2b-it",
    "google/gemma-3n-e4b-it",
    "google/gemma-4-31b-it",
    "google/gemma-3-12b-it",        # v10.0 新增
    "google/gemma-3-4b-it",         # v10.0 新增
    "google/diffusiongemma-26b-a4b-it",
    # === deepseek-ai 新模型 ===
    "deepseek-ai/deepseek-coder-6.7b-instruct",  # v10.0 新增
    # === meta 新模型 ===
    "meta/llama-4-maverick-17b-128e-instruct",   # v10.0 新增 Llama 4
    # === moonshotai 新模型 ===
    "moonshotai/kimi-k2.6",                      # v10.0 新增 Kimi
    # === nv-mistralai ===
    "nv-mistralai/mistral-nemo-12b-instruct",    # v10.0 新增
    # === nvidia 对话模型 ===
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3-nano-30b-a3b",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "nvidia/nemotron-nano-3-30b-a3b",             # v10.0 新增
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "nvidia/llama-3.3-nemotron-super-49b-v1",
    "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    "nvidia/llama-3.1-nemotron-51b-instruct",     # v10.0 新增
    "nvidia/llama-3.1-nemotron-70b-instruct",     # v10.0 新增
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",    # v10.0 新增
    "nvidia/llama-3.1-nemotron-nano-8b-v1",       # v10.0 新增
    "nvidia/nemotron-mini-4b-instruct",
    "nvidia/nemotron-nano-12b-v2-vl",
    "nvidia/nvidia-nemotron-nano-9b-v2",
    # === nvidia 安全/检测模型 ===
    "nvidia/llama-3.1-nemoguard-8b-content-safety",
    "nvidia/llama-3.1-nemoguard-8b-topic-control",
    "nvidia/llama-3.1-nemotron-safety-guard-8b-v3",
    "nvidia/nemotron-3.5-content-safety",
    "nvidia/gliner-pii",
    "nvidia/ising-calibration-1-35b-a3b",
    # === nvidia 翻译 ===
    "nvidia/riva-translate-4b-instruct-v1.1",
    "nvidia/riva-translate-4b-instruct",           # v10.0 新增
    # === nvidia 视觉/解析 ===
    "nvidia/vila",                                  # v10.0 新增
    "nvidia/nemotron-parse",                        # v10.0 新增
    # === 其他 ===
    "abacusai/dracarys-llama-3.1-70b-instruct",
    "bytedance/seed-oss-36b-instruct",
    "poolside/laguna-xs-2.1",
    "sarvamai/sarvam-m",
    "thinkingmachines/inkling",
    "upstage/solar-10.7b-instruct",
}


async def fetch_upstream_models() -> list:
    """返回对外暴露的模型列表：只包含虚拟聚合模型，不暴露真实上游模型。"""
    now_ts = time.time()
    if _models_cache["data"] and _models_cache["expires"] > now_ts:
        return _models_cache["data"]

    # 只返回虚拟聚合模型，真实上游模型不对外暴露
    models = []
    try:
        from app import virtual_models as vm
        for v in vm.get_virtual_models():
            if not v.get("enabled", True):
                continue
            models.append({
                "id": v["id"],
                "object": "model",
                "owned_by": "masterapi-virtual",
                "display_name": v.get("display_name") or v["id"],
            })
    except Exception as e:
        logger.warning(f"读取虚拟模型列表失败: {e}")

    _models_cache["data"] = models
    _models_cache["expires"] = now_ts + 30  # 30秒缓存，虚拟模型变更后快速生效
    try:
        refresh_verified_models(models)
    except Exception:
        pass
    logger.info(f"模型列表(仅虚拟模型): {len(models)}个: {[m['id'] for m in models]}")
    return models


# ========== 模型列表端点 ==========

def _enrich_model_list(models: list) -> list:
    """为模型列表添加能力标签和友好名称（基于NIM模型目录）"""
    try:
        from app.nim_models import NIM_MODEL_CATALOG, get_model_sort_priority
    except ImportError:
        return models

    priorities = get_model_sort_priority()

    enriched = []
    for m in models:
        model_id = m.get("id", "")
        info = NIM_MODEL_CATALOG.get(model_id)

        enriched_model = dict(m)

        if info:
            # 从NIM模型目录获取详细信息
            enriched_model["display_name"] = info.display_name
            enriched_model["context_length"] = info.context_length
            enriched_model["max_output_tokens"] = info.max_output_tokens

            # 能力标签
            capabilities = []
            if info.supports_images:
                capabilities.append("视觉")
            if info.supports_tools:
                capabilities.append("工具调用")
            if info.context_length >= 1000000:
                capabilities.append("1M上下文")

            # 基于标签推断
            tags_lower = [t.lower() for t in (info.tags or [])]
            if any(kw in tags_lower for kw in ["embedding", "embed"]):
                capabilities.append("嵌入")
            if any(kw in tags_lower for kw in ["safety", "guard", "guardrails"]):
                capabilities.append("安全")
            if any(kw in tags_lower for kw in ["coding", "code generation", "code"]):
                capabilities.append("代码")
            if any(kw in tags_lower for kw in ["ocr", "table extraction"]):
                capabilities.append("OCR")
            if any(kw in tags_lower for kw in ["asr", "tts", "speech"]):
                capabilities.append("语音")
            if any(kw in tags_lower for kw in ["translation", "translate"]):
                capabilities.append("翻译")
            if "deprecated" in tags_lower:
                capabilities.append("已弃用")
            if any(kw in tags_lower for kw in ["reasoning", "agent"]):
                if "推理" not in capabilities:
                    capabilities.append("推理")

            # 如果没有推理/嵌入/安全/代码等标签，默认标记为推理
            if not capabilities and info.model_family in (
                "deepseek", "glm", "qwen", "kimi", "nemotron", "gpt-oss",
                "step", "llama", "mistral", "gemma", "minimax", "yi",
                "jamba", "phi", "granite", "dbrx", "solar", "palmyra",
                "codestral", "mixtral", "codellama",
            ):
                capabilities.append("推理")

            enriched_model["capabilities"] = capabilities
        else:
            # 不在目录中的模型，推断基本能力
            capabilities = []
            model_lower = model_id.lower()
            if any(kw in model_lower for kw in ["vision", "vl", "fuyu", "kosmos", "omni"]):
                capabilities.append("视觉")
            if any(kw in model_lower for kw in ["embed", "bge-m3", "arctic-embed"]):
                capabilities.append("嵌入")
            if any(kw in model_lower for kw in ["safety", "guard", "gliner-pii", "content-safety"]):
                capabilities.append("安全")
            if any(kw in model_lower for kw in ["codestral", "starcoder", "codegemma", "coder"]):
                capabilities.append("代码")
            if any(kw in model_lower for kw in ["ocr", "parse"]):
                capabilities.append("OCR")
            if any(kw in model_lower for kw in ["riva-translate"]):
                capabilities.append("翻译")
            if any(kw in model_lower for kw in ["gemma-3n", "dracarys", "seed-oss"]):
                capabilities.append("已弃用")
            if not capabilities:
                capabilities.append("推理")
            enriched_model["capabilities"] = capabilities

        # 排序优先级
        enriched_model["sort_priority"] = priorities.get(model_id, 999)

        enriched.append(enriched_model)

    # 按排序优先级排序
    enriched.sort(key=lambda m: m.get("sort_priority", 999))

    return enriched


def _append_virtual_models(enriched: list) -> list:
    """把已启用的虚拟聚合模型追加到模型列表末尾（OpenAI 兼容条目）。

    虚拟模型对 agent 表现为一个普通可选模型，实际由网关按 chain failover。
    去重：若虚拟 ID 与已有真实模型同名则跳过（避免遮蔽）。
    """
    try:
        from app import virtual_models as vm
        existing_ids = {m.get("id") for m in enriched if isinstance(m, dict)}
        for v in vm.get_virtual_models():
            if not v.get("enabled", True):
                continue
            vid = v.get("id")
            if not vid or vid in existing_ids:
                continue
            chain = v.get("chain") or []
            enriched.append({
                "id": vid,
                "object": "model",
                "owned_by": "masterapi-virtual",
                "display_name": v.get("display_name") or vid,
                "capabilities": ["聚合"],
                "is_virtual": True,
                "chain": chain,
                "sort_priority": -1,  # 置顶，方便 agent 选到
            })
    except Exception as e:
        logger.warning(f"追加虚拟模型到列表失败: {e}")
    return enriched


@router.get("/v1/models", tags=["公共API"])
@router.get("/api/v1/models", tags=["公共API"])
async def list_models(request: Request):
    """OpenAI兼容模型列表（只返回虚拟聚合模型）"""
    try:
        await asyncio.to_thread(authenticate_client, request)
    except HTTPException:
        pass  # 未认证也允许查看模型列表

    models = await fetch_upstream_models()
    return {"object": "list", "data": models}


@router.get("/api/public/models", tags=["公共API"])
async def public_models():
    """公开模型列表（无需认证，只返回虚拟聚合模型）"""
    models = await fetch_upstream_models()
    enriched = _enrich_model_list(models)
    enriched = _append_virtual_models(enriched)
    return {"object": "list", "data": enriched}

def _extract_usage_tokens(
    usage: dict,
    p_cur: int = 0,
    c_cur: int = 0,
    tot_cur: int = 0,
    cached_cur: int = 0,
    c_write_cur: int = 0
) -> tuple[int, int, int, int, int]:
    """从上游各种格式的 usage 字典中提取 tokens 信息。
    兼容 OpenAI, Anthropic, DeepSeek, Gemini, Qwen/DashScope, Responses API, OneAPI 等。
    """
    if not isinstance(usage, dict):
        return p_cur, c_cur, tot_cur, cached_cur, c_write_cur

    p = usage.get("prompt_tokens") or usage.get("input_tokens") or p_cur
    c = usage.get("completion_tokens") or usage.get("output_tokens") or c_cur
    tot = usage.get("total_tokens") or tot_cur

    ptd = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    if not isinstance(ptd, dict):
        ptd = {}

    cached = (
        ptd.get("cached_tokens")
        or ptd.get("cache_read_tokens")
        or ptd.get("prompt_cache_hit_tokens")
        or usage.get("prompt_cache_hit_tokens")
        or usage.get("cache_read_input_tokens")
        or usage.get("cached_content_token_count")
        or usage.get("cached_tokens")
        or usage.get("cache_hit_tokens")
        or usage.get("cache_read_tokens")
        or cached_cur
    )

    c_write = (
        ptd.get("cache_write_tokens")
        or ptd.get("cache_creation_tokens")
        or ptd.get("cache_creation_input_tokens")
        or usage.get("cache_creation_input_tokens")
        or usage.get("prompt_cache_miss_tokens")
        or usage.get("cache_write_tokens")
        or c_write_cur
    )

    return int(p or 0), int(c or 0), int(tot or 0), int(cached or 0), int(c_write or 0)


# ========== 流式请求处理 ==========

def _record_committed_stream_failure(model: str, reason: str = "", is_severe: bool = False) -> None:
    """流式已提交给客户端后的中途失败：补记模型级软熔断失败。

    提交后链路层早已返回、无法再走 _record_upstream_attempt；
    反复"晚死"的候选靠这里的累计触发软熔断，减少被再次选中的概率。
    """
    try:
        from app import virtual_models as _vm_csf
        _vm_csf.record_model_failure(model, reason=reason, is_severe=is_severe)
    except Exception:
        logger.debug("记录提交后流式失败失败", exc_info=True)


async def _prepend_stream_lines(head: list, source):
    """把探测期缓冲的事件行先交给消费方，再继续透传上游流的剩余部分。"""
    for line in head:
        yield line
    async for line in source:
        yield line


async def _probe_stream_start(source, probe_seconds: float, max_buffered_lines: int = 200):
    """探测聚合链流式候选的"流首段"，健康才允许提交给客户端。

    背景：大量聚合上游（one-api/new-api 类分销）会先返回 HTTP 200，
    再在 SSE 流里输出 error 事件（如"无可用渠道"），或 200 后长时间不出
    首个内容块。一旦把 200 和字节发给客户端就无法再切换模型，客户端只会
    看到中途的 upstream_stream_error。因此在提交前缓冲并检查前几个事件：

    - 看到 choices 内容/角色块 → 健康，返回 (已缓冲行, None)；
    - 看到 error 事件 / 裸错误 JSON → 返回 (缓冲行, (错误摘要, "upstream_error"))；
    - 流未输出任何内容即结束 → 返回 (缓冲行, (摘要, "empty"))；
    - probe_seconds 内没有任何事件 → 返回 (缓冲行, (摘要, "timeout"))；
    - 缓冲超过 max_buffered_lines 行仍未判定 → 视为健康直接提交（兜底防失控）。

    返回后 source 已被部分消费：健康时调用方须用 _prepend_stream_lines
    把缓冲行拼回流的开头再交给生成器。
    """
    buffered: list = []
    deadline = time.time() + probe_seconds
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return buffered, (f"探测期内未收到首个流事件（等待{probe_seconds:.0f}s）", "timeout")
        try:
            line = await asyncio.wait_for(anext(source), timeout=remaining)
        except StopAsyncIteration:
            return buffered, ("上游流未输出任何内容即结束（空流）", "empty")
        except asyncio.TimeoutError:
            return buffered, (f"探测期内未收到首个流事件（等待{probe_seconds:.0f}s）", "timeout")
        except httpx.TimeoutException:
            return buffered, ("上游流读取超时", "timeout")
        except (httpx.NetworkError, httpx.RemoteProtocolError):
            return buffered, ("上游流连接中断", "network")

        buffered.append(line)
        stripped = line.strip()
        if not stripped or stripped.startswith(":"):
            continue  # 空行/SSE注释（keepalive）
        if stripped.startswith("{") and stripped.endswith("}"):
            # 上游直接吐裸 JSON 错误
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if "error" in obj or "detail" in obj:
                err_val = obj.get("error") or obj.get("detail")
                return buffered, (_safe_error_message(err_val), "upstream_error")
            continue
        if stripped.startswith("data:"):
            payload = stripped[5:].strip()
            if payload == "[DONE]":
                return buffered, ("上游流未输出任何内容即结束（[DONE]）", "empty")
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and "error" in data:
                return buffered, (_safe_error_message(data["error"]), "upstream_error")
            if isinstance(data, dict) and data.get("choices"):
                # 必须看到真实内容（文本/推理/工具调用）才算健康：
                # 聚合上游常先发标准角色块（delta:{"role":"assistant"}）再挂起或报错，
                # 仅凭 choices 存在就提交会让客户端挂到流式空闲超时才失败。
                c = data["choices"][0]
                delta = (c.get("delta") if isinstance(c, dict) else {}) or {}
                if (delta.get("content") or delta.get("reasoning_content")
                        or delta.get("reasoning") or delta.get("tool_calls")
                        or delta.get("function_call") or delta.get("text")):
                    return buffered, None  # 健康：真实内容块
            continue
        # 其余（event:/id:/字段行等）继续缓冲
        if len(buffered) >= max_buffered_lines:
            return buffered, None


async def _handle_stream_request(
    scheduler, client_id, key_id, model,
    upstream_url, headers, body, timeout, start_time, request, in_chain=False,
    responses_mode=False, client_override=None, request_start=None
):
    """处理流式SSE请求

     优化：
    - per-chunk空闲超时检测（而非全局超时）
    - SSE keepalive心跳（防止客户端/Nginx断连）
    - 流式中途断连优雅处理

    先发起请求并检查状态码：
    - 429/5xx: 抛出UpstreamRetryableError触发重试
    - 4xx: 直接返回错误响应
    - 200: 返回流式SSE响应
    """
    pool = client_override or await scheduler.get_stream_pool()

    # 预检：发起请求并检查状态码，再决定是重试还是流式返回
    # : 流式请求使用600秒httpx超时（大于per-chunk空闲超时180秒）
    # 实际的空闲检测由stream_generator中的per-chunk逻辑处理
    req = pool.build_request(
        "POST", upstream_url, headers=headers, json=body,
        timeout=httpx.Timeout(600.0, connect=30.0),
    )
    # 聚合链内候选：用链层给的 timeout 封顶"等到响应头"的等待（非末位候选=候选预算，
    # 末位候选=完整模型超时），防止挂死的候选拖住整链切换。流式体阶段的空闲控制
    # 仍由 stream_generator 的 per-chunk 逻辑负责，不受此预算影响。
    send_deadline = max(1.0, float(timeout)) if in_chain else None
    try:
        if send_deadline is not None:
            resp = await asyncio.wait_for(pool.send(req, stream=True), timeout=send_deadline)
        else:
            resp = await pool.send(req, stream=True)
    except asyncio.TimeoutError:
        rt = time.time() - start_time
        scheduler.record_response(key_id, model, False, rt, 0, "timeout")
        scheduler.trigger_hard_cooldown(key_id, model, "timeout")
        raise UpstreamRetryableError(
            f"candidate_headers_timeout: no response headers within {send_deadline:.0f}s",
            kind="timeout",
        )
    except httpx.ConnectTimeout as exc:
        # 连接建立超时（上限30s）：等待成本低，保留换 key 重试价值
        scheduler.record_response(key_id, model, False, time.time() - start_time, 0, "timeout")
        scheduler.trigger_hard_cooldown(key_id, model, "timeout")
        raise UpstreamRetryableError("upstream_connect_timeout", kind="connect_timeout") from exc
    except httpx.TimeoutException:
        scheduler.record_response(key_id, model, False, time.time() - start_time, 0, "timeout")
        scheduler.trigger_hard_cooldown(key_id, model, "timeout")
        raise UpstreamRetryableError("upstream_timeout", kind="timeout")
    except (httpx.NetworkError, httpx.RemoteProtocolError) as exc:
        scheduler.record_response(key_id, model, False, time.time() - start_time, 0, "conn_error")
        scheduler.trigger_isolation(key_id, model, "conn_error")
        raise UpstreamRetryableError("upstream_network_error", kind="network") from exc

    if resp.status_code != 200:
        # 收到响应头前可以安全重试；错误正文只用于脱敏后的诊断。
        try:
            error_bytes = await resp.aread()
            error_text = error_bytes.decode("utf-8", errors="replace")[:500]
        finally:
            await resp.aclose()
        error_body = _parse_upstream_error_body(error_text)
        error_message = _safe_error_message(error_body)
        rt = time.time() - start_time

        if resp.status_code == 429:
            scheduler.record_response(key_id, model, False, rt, 429, "429")
            scheduler.trigger_hard_cooldown(key_id, model, "429")
            raise UpstreamRetryableError(f"upstream_429: {error_message}", status_code=429, kind="rate_limited")
        if resp.status_code >= 500:
            scheduler.record_response(key_id, model, False, rt, resp.status_code, "5xx")
            scheduler.trigger_hard_cooldown(key_id, model, "5xx")
            raise UpstreamRetryableError(f"upstream_{resp.status_code}: {error_message}", status_code=resp.status_code, kind="server")

        scheduler.record_response(key_id, model, False, rt, resp.status_code, "4xx")
        if resp.status_code == 401:
            raise UpstreamRetryableError(f"upstream_401: {error_message}", status_code=401, kind="auth")
        if resp.status_code == 403:
            lower_message = error_message.lower()
            if "model" in lower_message and any(marker in lower_message for marker in ("permission", "access", "not allowed", "无权限")) and in_chain:
                raise UpstreamModelUnavailableError(403, error_body)
            if not any(marker in lower_message for marker in ("content", "policy", "safety", "moderation", "内容", "策略", "安全")):
                scheduler.trigger_hard_cooldown(key_id, model, "403")
                raise UpstreamRetryableError(f"upstream_403: {error_message}", status_code=403, kind="forbidden")
        if _is_model_not_found(resp.status_code, error_body) and in_chain:
            raise UpstreamModelUnavailableError(resp.status_code, error_body)
        if in_chain:
            raise UpstreamRequestError(resp.status_code, error_body)
        scheduler.release_client_request(client_id)
        return JSONResponse(status_code=resp.status_code, content=error_body)

    # 状态码200 - 聚合链候选先"探测"流首段再提交（详见 _probe_stream_start）：
    # 聚合上游常见"先回200再在SSE里报错/长时间不出首块"，一旦先把 200 发给
    # 客户端就无法再切换模型；探测发现异常则静默关闭本候选、抛可重试错误，
    # 由链路层切换下一个候选。非链路路径保持纯透传，行为不变。
    stream_source = resp.aiter_lines()
    if responses_mode:
        stream_source = translate_sse_lines(stream_source, model)
    if in_chain:
        from app import virtual_models as _vm_probe
        probe_budget = min(
            _vm_probe.CHAIN_STREAM_PROBE_SECONDS,
            max(5.0, float(timeout) - (time.time() - start_time)),
        )
        buffered_head, probe_err = await _probe_stream_start(stream_source, probe_budget)
        if probe_err is not None:
            probe_msg, probe_kind = probe_err
            rt = time.time() - start_time
            if probe_kind == "timeout":
                scheduler.record_response(key_id, model, False, rt, 0, "timeout")
                scheduler.trigger_hard_cooldown(key_id, model, "timeout")
            elif probe_kind == "network":
                scheduler.record_response(key_id, model, False, rt, 0, "conn_error")
                scheduler.trigger_isolation(key_id, model, "conn_error")
            else:
                fail_type = "empty_stream" if probe_kind == "empty" else "upstream_stream_error"
                scheduler.record_response(key_id, model, False, rt, 502, fail_type)
                scheduler.trigger_hard_cooldown(key_id, model, fail_type)
            logger.warning(f"[聚合] {model} 流式探测失败({probe_kind}): {probe_msg}，切换下一个候选")
            await resp.aclose()
            raise UpstreamRetryableError(
                f"upstream_stream_error: {probe_msg}",
                kind=probe_kind if probe_kind in ("timeout", "network", "empty") else "upstream_error",
            )
        if buffered_head:
            stream_source = _prepend_stream_lines(buffered_head, stream_source)

    # 供生成器错误路径落库用：脱敏后的请求体（messages 省略），便于核对客户端参数
    try:
        from app.middleware import RequestLoggingMiddleware as _RLM
        _req_body_json = _RLM._sanitize_request_body(json.dumps(body, ensure_ascii=False))[:4096]
    except Exception:
        _req_body_json = ""

    # 客户端视角的请求起点（含网关选 key/调度耗时），用于首字与总使用时长口径
    client_start = request_start or start_time

    # 创建流式生成器（: 带keepalive和per-chunk空闲检测）
    async def stream_generator():
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        cached_tokens = 0
        cache_write_tokens = 0
        first_byte_received = False
        first_content_ts = None  # 首个真实内容/思考/工具调用块的到达时刻
        last_data_time = time.time()  # 上次收到上游数据的时间
        last_keepalive_time = time.time()  # 上次发送keepalive的时间
        chunks_received = 0  # 已接收的SSE事件数
        pending_sse_fields = []  # 已收到、但尚未归属data行的SSE字段行（id:/event:/retry:/注释）

        raw_content_chunks = 0  # 已收到的实际内容/思考chunk数
        seen_upstream_error = ""  # 捕获的上游错误信息
        seen_finish_reason = None  # 记录已收到的结束原因，防止后续usage块冲掉
        seen_done = False  # 记录是否收到[DONE]
        seen_tool_calls = False  # 记录流中是否已有工具调用
        accumulated_visible_content = ""  # 累计正文（用于嵌入工具调用提取与意图自愈）
        has_tools = bool(body.get("tools"))
        from app.think_stripper import StreamThinkStripper, strip_think_tags
        think_stripper = StreamThinkStripper()

        def _timing_values():
            """(首字时长ms, 总使用时长ms)，均为客户端视角。"""
            ttft = (first_content_ts - client_start) * 1000 if first_content_ts else 0.0
            total = (time.time() - client_start) * 1000
            return ttft, total

        def _final_error_payload(err_text: str) -> str:
            """最终错误附完整候选尝试链（控制台详情按 JSON 渲染时间线）。

            流式响应已以 200 提交，中间件不再补写失败行；本行是这次请求
            唯一的最终记录，必须自带"试过哪些候选、各自为何失败"。
            """
            try:
                attempts = getattr(request.state, "upstream_attempts", [])
                if attempts:
                    return json.dumps(
                        {"final": err_text[:500], "attempts": attempts[:16]},
                        ensure_ascii=False,
                    )
            except Exception:
                pass
            return err_text

        try:
            # 使用（探测缓冲拼接后的）行流逐行读取，配合空闲检测
            # OAuth（Responses API）上游：先把 response.* 事件逐行翻成
            # chat.completion.chunk，再走同一套透传/usage 统计/keepalive 逻辑
            source = stream_source
            async for line in source:
                now = time.time()

                # per-chunk空闲超时检测
                idle_time = now - last_data_time
                if idle_time > STREAM_CHUNK_IDLE_TIMEOUT:
                    # 空闲超时 - 优雅终止流式传输
                    logger.warning(
                        f"流式空闲超时: key={key_id[:8]} model={model} "
                        f"idle={idle_time:.0f}s chunks={chunks_received} "
                        f"已接收部分数据，优雅终止"
                    )
                    # 如果已收到部分数据，发送超时错误事件但保留已生成内容
                    rt = now - start_time
                    scheduler.record_response(key_id, model, False, rt, 0, "timeout")
                    scheduler.trigger_hard_cooldown(key_id, model, "timeout")
                    _record_committed_stream_failure(model, reason="empty_stream", is_severe=True)
                    error_data = {"error": {"message": "上游响应超时（空闲时间过长）",
                                            "type": "timeout_error", "code": "upstream_idle_timeout"}}
                    yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    seen_done = True
                    scheduler.release_client_request(client_id)
                    _ttft_err, _total_err = _timing_values()
                    await _log_request(
                        client_id, key_id, model, 504, rt,
                        prompt_tokens, completion_tokens, total_tokens, True,
                        _final_error_payload(f"流式空闲超时(idle={idle_time:.0f}s)"),
                        start_ts=start_time,
                        error_type="IdleTimeout", error_detail=f"空闲{idle_time:.0f}s超时",
                        include_errors=True, request_body=_req_body_json,
                        business_code="upstream_idle_timeout",
                        request_path=str(request.url.path), http_method=request.method,
                        client_ip=get_client_ip(request), user_agent=request.headers.get("user-agent", "")[:512],
                        ttft_ms=_ttft_err, total_duration_ms=_total_err,
                    )
                    return

                if line:
                    last_data_time = now
                    chunks_received += 1
                    if not first_byte_received:
                        first_byte_received = True

                    stripped = line.strip()
                    # 1) 检测上游是否直接吐出裸 JSON 错误（无 data: 前缀）
                    if stripped.startswith("{") and stripped.endswith("}"):
                        try:
                            raw_obj = json.loads(stripped)
                            if "error" in raw_obj or "detail" in raw_obj:
                                err_val = raw_obj.get("error") or raw_obj.get("detail")
                                seen_upstream_error = _safe_error_message(err_val)
                                _learn_param_blocklist(model, _match_unsupported_params(seen_upstream_error))
                                logger.warning(f"上游流式返回裸错误JSON: key={key_id[:8]} model={model}: {seen_upstream_error}")
                                error_data = {"error": {"message": f"上游错误: {seen_upstream_error}", "type": "upstream_error", "code": "upstream_stream_error"}}
                                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                                yield "data: [DONE]\n\n"
                                seen_done = True
                                rt = time.time() - start_time
                                scheduler.record_response(key_id, model, False, rt, 502, "upstream_stream_error")
                                scheduler.trigger_hard_cooldown(key_id, model, "upstream_stream_error")
                                _record_committed_stream_failure(model, reason=seen_upstream_error, is_severe=True)
                                scheduler.release_client_request(client_id)
                                _ttft_err, _total_err = _timing_values()
                                await _log_request(
                                    client_id, key_id, model, 502, rt, prompt_tokens, completion_tokens, total_tokens, True,
                                    _final_error_payload(f"上游流式错误: {seen_upstream_error}"), start_ts=start_time,
                                    error_type="UpstreamStreamError", error_detail=seen_upstream_error,
                                    include_errors=True, request_body=_req_body_json,
                                    business_code="upstream_stream_error",
                                    request_path=getattr(request, 'url', None) and str(request.url.path) or "",
                                    http_method=getattr(request, 'method', ''), client_ip=get_client_ip(request),
                                    user_agent=getattr(request, 'headers', {}).get('user-agent', '')[:512] if hasattr(request, 'headers') else "",
                                    ttft_ms=_ttft_err, total_duration_ms=_total_err,
                                )
                                return
                        except json.JSONDecodeError:
                            pass

                    # 2) SSE字段行（id: / event: / retry: / ": "注释）不能单独成事件。
                    if not line.startswith("data:"):
                        # 过滤上游发出的非标准 event: done，防止破坏客户端默认的 message 事件通道
                        if line.startswith("event:") and line.split(":", 1)[1].strip() == "done":
                            continue
                        pending_sse_fields.append(line)
                        last_keepalive_time = now
                        continue

                    # 解析 [DONE] 标记
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        pending_sse_fields.clear()
                        seen_done = True
                        tail_vis, tail_rsn = think_stripper.finish()
                        if tail_rsn:
                            rsn_chunk = {
                                "id": f"chatcmpl-{int(time.time())}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [{"index": 0, "delta": {"reasoning_content": tail_rsn}, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(rsn_chunk, ensure_ascii=False)}\n\n"
                        if tail_vis:
                            vis_chunk = {
                                "id": f"chatcmpl-{int(time.time())}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [{"index": 0, "delta": {"content": tail_vis}, "finish_reason": seen_finish_reason or None}],
                            }
                            yield f"data: {json.dumps(vis_chunk, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                        break

                    for pending_field in pending_sse_fields:
                        yield f"{pending_field}\n"
                    pending_sse_fields.clear()
                    if payload:
                        try:
                            data = json.loads(payload)
                            if "error" in data:
                                err_val = data["error"]
                                seen_upstream_error = _safe_error_message(err_val)
                                _learn_param_blocklist(model, _match_unsupported_params(seen_upstream_error))
                                logger.warning(f"上游流式返回data: error: key={key_id[:8]} model={model}: {seen_upstream_error}")
                                error_data = {"error": {"message": f"上游错误: {seen_upstream_error}", "type": "upstream_error", "code": "upstream_stream_error"}}
                                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                                yield "data: [DONE]\n\n"
                                seen_done = True
                                rt = time.time() - start_time
                                scheduler.record_response(key_id, model, False, rt, 502, "upstream_stream_error")
                                scheduler.trigger_hard_cooldown(key_id, model, "upstream_stream_error")
                                _record_committed_stream_failure(model, reason=seen_upstream_error, is_severe=True)
                                scheduler.release_client_request(client_id)
                                _ttft_err, _total_err = _timing_values()
                                await _log_request(
                                    client_id, key_id, model, 502, rt, prompt_tokens, completion_tokens, total_tokens, True,
                                    _final_error_payload(f"上游流式错误: {seen_upstream_error}"), start_ts=start_time,
                                    error_type="UpstreamStreamError", error_detail=seen_upstream_error,
                                    include_errors=True, request_body=_req_body_json,
                                    business_code="upstream_stream_error",
                                    request_path=getattr(request, 'url', None) and str(request.url.path) or "",
                                    http_method=getattr(request, 'method', ''), client_ip=get_client_ip(request),
                                    user_agent=getattr(request, 'headers', {}).get('user-agent', '')[:512] if hasattr(request, 'headers') else "",
                                    ttft_ms=_ttft_err, total_duration_ms=_total_err,
                                )
                                return
                            # 提取 usage: 兼容 OpenAI, Azure, Anthropic, Responses API, OneAPI 等多种格式与嵌套位置
                            u = (
                                data.get("usage")
                                or (data.get("response") and isinstance(data["response"], dict) and data["response"].get("usage"))
                                or (data.get("message") and isinstance(data["message"], dict) and data["message"].get("usage"))
                                or (data.get("choices") and isinstance(data["choices"], list) and len(data["choices"]) > 0 and isinstance(data["choices"][0], dict) and data["choices"][0].get("usage"))
                            )
                            if u and isinstance(u, dict):
                                prompt_tokens, completion_tokens, total_tokens, cached_tokens, cache_write_tokens = _extract_usage_tokens(
                                    u, prompt_tokens, completion_tokens, total_tokens, cached_tokens, cache_write_tokens
                                )
                            if "choices" in data:
                                if not data["choices"]:
                                    # AMD 等上游在流末尾发出携带 usage 但 choices: [] 的 chunk
                                    # 规范化：注入空 choice 防止 CC Switch / Codex 下标越界或误判流异常
                                    data["choices"] = [{"index": 0, "delta": {}, "finish_reason": seen_finish_reason or None}]
                                    line = f"data: {json.dumps(data, ensure_ascii=False)}"
                                    yield f"{line}\n\n"
                                    continue
                                else:
                                    c = data["choices"][0]
                                    delta = (c.get("delta") if isinstance(c, dict) else {}) or {}
                                    if delta.get("tool_calls") or delta.get("function_call"):
                                        seen_tool_calls = True

                                    # 剥离并提取思维链标签（<thinking>, <think>, <thought>, <reasoning>）
                                    delta_content = delta.get("content")
                                    rsn_to_emit = ""
                                    if isinstance(delta_content, str) and delta_content:
                                        vis_text, rsn_text = think_stripper.feed(delta_content)
                                        if rsn_text:
                                            rsn_to_emit = rsn_text
                                        if vis_text:
                                            accumulated_visible_content += vis_text
                                            delta["content"] = vis_text
                                        else:
                                            delta.pop("content", None)

                                    # 工具调用或结束时刷新未闭合的思考缓冲（防止发起工具调用前未闭合的 <thinking> 外显）
                                    if delta.get("tool_calls") or delta.get("function_call") or c.get("finish_reason"):
                                        tail_vis, tail_rsn = think_stripper.finish()
                                        if tail_rsn:
                                            rsn_to_emit = (rsn_to_emit + tail_rsn) if rsn_to_emit else tail_rsn
                                        if tail_vis:
                                            existing_vis = delta.get("content") or ""
                                            delta["content"] = existing_vis + tail_vis

                                    # 如果本分块提取到了思考内容：先单独发一个 reasoning_content 分块，让 CC Switch / Codex 正常收纳至思考区
                                    if rsn_to_emit:
                                        rsn_chunk = {
                                            "id": data.get("id") or f"chatcmpl-{int(time.time())}",
                                            "object": "chat.completion.chunk",
                                            "created": data.get("created") or int(time.time()),
                                            "model": data.get("model") or model,
                                            "choices": [{"index": 0, "delta": {"reasoning_content": rsn_to_emit}, "finish_reason": None}],
                                        }
                                        yield f"data: {json.dumps(rsn_chunk, ensure_ascii=False)}\n\n"
                                        raw_content_chunks += 1
                                        if first_content_ts is None:
                                            first_content_ts = now

                                    # 规范化 reasoning -> reasoning_content
                                    if "reasoning" in delta and "reasoning_content" not in delta:
                                        delta["reasoning_content"] = delta.pop("reasoning")

                                    fr = c.get("finish_reason")
                                    if fr:
                                        seen_finish_reason = fr
                                    elif seen_finish_reason:
                                        # 避免后续 usage 块将已设定的 finish_reason 冲刷为 null
                                        c["finish_reason"] = seen_finish_reason

                                    # 若剥离后 delta 已变空且无结束标记，无需发送空 delta
                                    if not delta and not c.get("finish_reason"):
                                        continue

                                    if (delta.get("content") or delta.get("reasoning_content")
                                            or delta.get("reasoning") or delta.get("tool_calls")
                                            or delta.get("function_call") or delta.get("text")):
                                        raw_content_chunks += 1
                                        if first_content_ts is None:
                                            first_content_ts = now

                                    line = f"data: {json.dumps(data, ensure_ascii=False)}"
                        except json.JSONDecodeError:
                            pass
                    yield f"{line}\n\n"
                    last_keepalive_time = now  # 收到数据时重置keepalive计时
                else:
                    # 空行 = SSE事件边界：尚未有data行的字段行到此作废，避免出现
                    # 无data事件（见上）；同时检查是否需要发送keepalive
                    pending_sse_fields.clear()
                    if now - last_keepalive_time >= SSE_KEEPALIVE_INTERVAL:
                        # 发送SSE注释行作为keepalive（客户端会忽略SSE注释）
                        yield ": ping\n\n"
                        last_keepalive_time = now
                        logger.debug(f"流式keepalive: key={key_id[:8]} model={model}")

            # 流式结束：刷新可能残留的 tail
            tail_vis, tail_rsn = think_stripper.finish()
            if tail_rsn:
                rsn_chunk = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {"reasoning_content": tail_rsn}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(rsn_chunk, ensure_ascii=False)}\n\n"
            if tail_vis:
                vis_chunk = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": tail_vis}, "finish_reason": seen_finish_reason or None}],
                }
                yield f"data: {json.dumps(vis_chunk, ensure_ascii=False)}\n\n"

            # 流式结束：防御上游返回空流（0 个内容 chunk）
            if raw_content_chunks == 0:
                raw_pending = " ".join([p for p in pending_sse_fields if not p.startswith(":")])
                err_desc = seen_upstream_error or (_safe_error_message(raw_pending) if raw_pending else "上游未返回任何有效内容（响应为空或连接提前关闭）")
                logger.warning(f"上游流式响应内容为空: key={key_id[:8]} model={model}: {err_desc}")
                error_data = {"error": {"message": err_desc, "type": "upstream_error", "code": "upstream_empty_response"}}
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                if not seen_done:
                    yield "data: [DONE]\n\n"
                rt = time.time() - start_time
                scheduler.record_response(key_id, model, False, rt, 502, "empty_stream")
                scheduler.trigger_hard_cooldown(key_id, model, "empty_stream")
                _record_committed_stream_failure(model, reason="empty_stream", is_severe=True)
                scheduler.release_client_request(client_id)
                _ttft_err, _total_err = _timing_values()
                await _log_request(
                    client_id, key_id, model, 502, rt,
                    prompt_tokens, completion_tokens, total_tokens, True, _final_error_payload(err_desc),
                    start_ts=start_time,
                    error_type="EmptyStream", error_detail=err_desc,
                        include_errors=True, request_body=_req_body_json,
                    business_code="upstream_empty_response",
                    request_path=getattr(request, 'url', None) and str(request.url.path) or "",
                    http_method=getattr(request, 'method', ''),
                    client_ip=get_client_ip(request),
                    user_agent=getattr(request, 'headers', {}).get('user-agent', '')[:512] if hasattr(request, 'headers') else "",
                    ttft_ms=_ttft_err, total_duration_ms=_total_err,
                )
                return

            # 流式完成 - 刷新可能剩余的思考/文本缓冲并发送[DONE]标记
            if not seen_done:
                tail_vis, tail_rsn = think_stripper.finish()
                if tail_rsn or tail_vis:
                    flushed_delta = {}
                    if tail_rsn:
                        flushed_delta["reasoning_content"] = tail_rsn
                    if tail_vis:
                        accumulated_visible_content += tail_vis
                        flushed_delta["content"] = tail_vis
                    flushed_chunk = {
                        "id": f"chatcmpl_{uuid.uuid4().hex[:12]}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{"index": 0, "delta": flushed_delta, "finish_reason": seen_finish_reason or None}]
                    }
                    yield f"data: {json.dumps(flushed_chunk, ensure_ascii=False)}\n\n"

                # 智能工具调用补全与行动意图拦截 (Auto-Nudge / Tool Call Parser)
                if has_tools and not seen_tool_calls and accumulated_visible_content and not seen_upstream_error:
                    try:
                        from app.tool_call_parser import parse_raw_tool_calls, is_unexecuted_action_intent
                    except ImportError:
                        from gateway.app.tool_call_parser import parse_raw_tool_calls, is_unexecuted_action_intent
                    clean_text, extracted_tcs = parse_raw_tool_calls(accumulated_visible_content)
                    if extracted_tcs:
                        logger.info(f"从流式内容中提取到嵌入工具调用: {[t['function']['name'] for t in extracted_tcs]}")
                        tc_chunk = {
                            "id": f"chatcmpl_{uuid.uuid4().hex[:12]}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [{"index": 0, "delta": {"tool_calls": extracted_tcs}, "finish_reason": "tool_calls"}]
                        }
                        yield f"data: {json.dumps(tc_chunk, ensure_ascii=False)}\n\n"
                        seen_tool_calls = True
                    elif is_unexecuted_action_intent(accumulated_visible_content):
                        # 模型只输出了口头行动承诺（如 "Scene complete. Rendering to check."），执行自动呼叫补全
                        try:
                            logger.info(f"检测到模型行动意图未落地工具调用 ('{accumulated_visible_content.strip()[:60]}...')，执行 Auto-Nudge 补全")
                            nudge_msgs = list(body.get("messages", []))
                            nudge_msgs.append({"role": "assistant", "content": accumulated_visible_content.strip()})
                            nudge_msgs.append({
                                "role": "user",
                                "content": f"[System Protocol: You announced an immediate action ('{accumulated_visible_content.strip()[:80]}') but did not issue a tool call. You must invoke the required tool call directly now.]"
                            })
                            nudge_payload = dict(body)
                            nudge_payload["messages"] = nudge_msgs
                            nudge_payload["stream"] = False
                            nudge_payload["tool_choice"] = "auto"

                            nudge_resp = await pool.post(
                                upstream_url,
                                headers=headers,
                                json=nudge_payload,
                                timeout=httpx.Timeout(20.0, connect=10.0),
                            )
                            if nudge_resp.status_code == 200:
                                n_data = nudge_resp.json()
                                if n_data.get("choices") and isinstance(n_data["choices"][0], dict):
                                    n_c0 = n_data["choices"][0]
                                    n_msg = n_c0.get("message") or {}
                                    n_tcs = n_msg.get("tool_calls")
                                    if not n_tcs and n_msg.get("content"):
                                        _, n_tcs = parse_raw_tool_calls(n_msg["content"])
                                    if n_tcs:
                                        logger.info(f"Auto-Nudge 成功补全工具调用: {[t.get('function', {}).get('name') for t in n_tcs]}")
                                        n_chunk = {
                                            "id": f"chatcmpl_{uuid.uuid4().hex[:12]}",
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": model,
                                            "choices": [{"index": 0, "delta": {"tool_calls": n_tcs}, "finish_reason": "tool_calls"}]
                                        }
                                        yield f"data: {json.dumps(n_chunk, ensure_ascii=False)}\n\n"
                                        seen_tool_calls = True
                        except Exception as ne:
                            logger.warning(f"Auto-Nudge 尝试补全失败 (放行): {ne}")

                yield "data: [DONE]\n\n"

            # 正常结束：所有数据已接收完毕
            rt = time.time() - start_time
            scheduler.record_response(key_id, model, True, rt, 200, "")
            scheduler.release_client_request(client_id)

            # v10.0 防呆防傻：记录熔断器成功
            cb = get_circuit_breaker()
            cb.record_success(f"model:{model}")

            # 首字时长（客户端视角）/ 总使用时长（客户端视角）
            ttft_value = (first_content_ts - client_start) * 1000 if first_content_ts else 0.0
            total_value = (time.time() - client_start) * 1000

            # 记录请求日志（全量日志）
            await _log_request(
                client_id, key_id, model, 200, rt,
                prompt_tokens, completion_tokens, total_tokens, True, "",
                start_ts=start_time,
                request_path=getattr(request, 'url', None) and str(request.url.path) or "",
                http_method=getattr(request, 'method', ''),
                client_ip=get_client_ip(request),
                user_agent=getattr(request, 'headers', {}).get('user-agent', '')[:512] if hasattr(request, 'headers') else "",
                cached_tokens=cached_tokens,
                cache_write_tokens=cache_write_tokens,
                ttft_ms=ttft_value,
                total_duration_ms=total_value,
            )
            return  # 正常返回，不执行下面的错误处理

        except httpx.TimeoutException:
            rt = time.time() - start_time
            scheduler.record_response(key_id, model, False, rt, 0, "timeout")
            scheduler.trigger_hard_cooldown(key_id, model, "timeout")
            _record_committed_stream_failure(model, reason="timeout", is_severe=True)
            error_data = {"error": {"message": "上游响应超时",
                                    "type": "timeout_error", "code": "upstream_timeout"}}
            yield f"data: {json.dumps(error_data)}\n\n"
            if not seen_done:
                yield "data: [DONE]\n\n"
            scheduler.release_client_request(client_id)
            _ttft_err, _total_err = _timing_values()
            await _log_request(
                client_id, key_id, model, 504, rt, 0, 0, 0, True, _final_error_payload("上游响应超时"),
                start_ts=start_time,
                error_type="TimeoutException", error_detail="上游响应超时",
                include_errors=True, request_body=_req_body_json,
                business_code="upstream_timeout",
                request_path=str(request.url.path), http_method=request.method,
                client_ip=get_client_ip(request), user_agent=request.headers.get("user-agent", "")[:512],
                ttft_ms=_ttft_err, total_duration_ms=_total_err,
            )
        except httpx.ReadError as e:
            rt = time.time() - start_time
            scheduler.record_response(key_id, model, False, rt, 0, "conn_error")
            logger.warning(f"流式读取错误: key={key_id[:8]} model={model} chunks={chunks_received} error={e}")
            if raw_content_chunks > 0 and seen_finish_reason:
                # 已经收到了正常的结束标记，网络在末尾关闭不属于异常中断
                logger.info(f"流式已正常收到结束标记: key={key_id[:8]} model={model}")
            else:
                scheduler.trigger_hard_cooldown(key_id, model, "timeout")
                _record_committed_stream_failure(model, reason=f"read_error: {e}", is_severe=True)
                error_data = {"error": {"message": f"流式传输中断: {str(e)}",
                                        "type": "connection_error", "code": "stream_interrupted"}}
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            if not seen_done:
                yield "data: [DONE]\n\n"
            scheduler.release_client_request(client_id)
            _ttft_err, _total_err = _timing_values()
            await _log_request(
                client_id, key_id, model, 502, rt,
                prompt_tokens, completion_tokens, total_tokens, True,
                _final_error_payload(f"流式传输中断(已接收{chunks_received}chunks): {str(e)}"),
                start_ts=start_time,
                error_type="ReadError", error_detail=str(e)[:1024],
                include_errors=True, request_body=_req_body_json,
                error_stack=traceback.format_exc()[:4096],
                business_code="stream_interrupted",
                request_path=str(request.url.path), http_method=request.method,
                client_ip=get_client_ip(request), user_agent=request.headers.get("user-agent", "")[:512],
                ttft_ms=_ttft_err, total_duration_ms=_total_err,
            )
        except Exception as e:
            rt = time.time() - start_time
            scheduler.record_response(key_id, model, False, rt, 0, "error")
            err_type = type(e).__name__
            is_sev = ("SSL" in err_type or "Remote" in err_type or "Protocol" in err_type or "Transport" in err_type or "Connection" in err_type)
            scheduler.trigger_hard_cooldown(key_id, model, "conn_error" if is_sev else "error")
            _record_committed_stream_failure(model, reason=f"{err_type}: {e}", is_severe=is_sev)
            logger.warning(f"流式传输异常: key={key_id[:8]} model={model} error={err_type}: {e}")
            error_data = {"error": {"message": f"流式传输错误: {str(e)}",
                                    "type": "internal_error", "code": "stream_error"}}
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"
            scheduler.release_client_request(client_id)
            # : 记录流式异常日志
            _ttft_err, _total_err = _timing_values()
            await _log_request(
                client_id, key_id, model, 500, rt, 0, 0, 0, True, _final_error_payload(f"流式传输错误: {str(e)}"),
                start_ts=start_time,
                error_type=err_type, error_detail=str(e)[:1024],
                include_errors=True, request_body=_req_body_json,
                error_stack=traceback.format_exc()[:4096],
                business_code="stream_error",
                request_path=str(request.url.path), http_method=request.method,
                client_ip=get_client_ip(request), user_agent=request.headers.get("user-agent", "")[:512],
                ttft_ms=_ttft_err, total_duration_ms=_total_err,
            )
        finally:
            await resp.aclose()

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ========== 非流式请求处理 ==========

async def _handle_nonstream_request(
    scheduler, client_id, key_id, model,
    upstream_url, headers, body, timeout, start_time, request=None, in_chain=False,
    responses_mode=False, client_override=None, request_start=None
):
    """处理非流式请求

    返回值：
    - 成功: JSONResponse
    - 4xx: JSONResponse（错误）
    - 429/5xx/超时/连接错误: 抛出UpstreamRetryableError触发重试

    in_chain=True（聚合模型链场景）：4xx（含 400/404 模型不存在等）也抛
    UpstreamRetryableError 以便链路层切换到下一个真实模型，且不释放并发名额
    （由链路层统一释放）。普通单模型路径 in_chain=False，行为保持不变。
    """
    pool = client_override or await scheduler.get_http_pool()

    # 客户端视角起点（含网关调度耗时）
    client_start = request_start or start_time

    try:
        resp = await pool.post(
            upstream_url, headers=headers, json=body,
            timeout=httpx.Timeout(timeout, connect=30.0),
        )
        rt = time.time() - start_time

        if resp.status_code == 200:
            # 成功（v10.0: 防御上游返回空body或200返回error）
            try:
                data = resp.json()
                if data is None:
                    data = {}
            except Exception:
                data = {}
            if responses_mode:
                # OAuth（Responses API）上游：翻译回 chat.completion 结构再返回
                data = responses_to_chat(data, model)
            if isinstance(data, dict) and "error" in data:
                err_msg = _safe_error_message(data["error"])
                scheduler.record_response(key_id, model, False, rt, 502, "upstream_error")
                scheduler.trigger_hard_cooldown(key_id, model, "upstream_error")
                raise UpstreamRetryableError(f"upstream_error: {err_msg}", status_code=502, kind="upstream_error")
            if isinstance(data, dict) and not data.get("choices") and not data.get("id"):
                err_msg = _safe_error_message(data) if data else "上游未返回有效choices内容"
                scheduler.record_response(key_id, model, False, rt, 502, "empty_choices")
                scheduler.trigger_hard_cooldown(key_id, model, "empty_choices")
                raise UpstreamRetryableError(f"upstream_empty: {err_msg}", status_code=502, kind="empty")
            u = (
                data.get("usage")
                or (data.get("response") and isinstance(data["response"], dict) and data["response"].get("usage"))
                or (data.get("message") and isinstance(data["message"], dict) and data["message"].get("usage"))
            ) if isinstance(data, dict) else {}
            prompt_tokens, completion_tokens, total_tokens, cached_tokens, cache_write_tokens = _extract_usage_tokens(u)

            # 剥离并提取思维链标签（<thinking>, <think>, <thought>, <reasoning>）与嵌入工具调用
            if isinstance(data, dict) and data.get("choices"):
                from app.think_stripper import strip_think_tags
                try:
                    from app.tool_call_parser import parse_raw_tool_calls, is_unexecuted_action_intent
                except ImportError:
                    from gateway.app.tool_call_parser import parse_raw_tool_calls, is_unexecuted_action_intent
                for choice in data["choices"]:
                    if isinstance(choice, dict):
                        msg = choice.get("message")
                        if isinstance(msg, dict):
                            content = msg.get("content")
                            if isinstance(content, str) and content:
                                clean_vis, extracted_rsn = strip_think_tags(content)
                                # 检查是否有嵌入的文本格式工具调用
                                clean_vis, extracted_tcs = parse_raw_tool_calls(clean_vis)
                                if extracted_tcs and not msg.get("tool_calls"):
                                    msg["tool_calls"] = extracted_tcs
                                    choice["finish_reason"] = "tool_calls"
                                msg["content"] = clean_vis
                                if extracted_rsn:
                                    existing_rsn = msg.get("reasoning_content") or msg.get("reasoning") or ""
                                    msg["reasoning_content"] = (existing_rsn + "\n" + extracted_rsn).strip() if existing_rsn else extracted_rsn
                            if "reasoning" in msg and "reasoning_content" not in msg:
                                msg["reasoning_content"] = msg.pop("reasoning")

                # 如果模型只有口头行动宣告而未发起工具调用，且客户端传入了 tools
                if bool(body.get("tools")) and data["choices"] and isinstance(data["choices"][0], dict):
                    c0 = data["choices"][0]
                    m0 = c0.get("message") or {}
                    if not m0.get("tool_calls") and is_unexecuted_action_intent(m0.get("content") or ""):
                        try:
                            logger.info(f"非流式: 检测到模型行动意图未落地工具调用 ('{(m0.get('content') or '').strip()[:60]}...')，执行 Auto-Nudge 补全")
                            nudge_msgs = list(body.get("messages", []))
                            nudge_msgs.append({"role": "assistant", "content": (m0.get("content") or "").strip()})
                            nudge_msgs.append({
                                "role": "user",
                                "content": f"[System Protocol: You announced an immediate action ('{(m0.get('content') or '').strip()[:80]}') but did not issue a tool call. You must invoke the required tool call directly now.]"
                            })
                            nudge_payload = dict(body)
                            nudge_payload["messages"] = nudge_msgs
                            nudge_payload["stream"] = False
                            nudge_payload["tool_choice"] = "auto"
                            client = client_override or http_client
                            nudge_resp = await client.post(
                                effective_upstream_url,
                                headers=headers,
                                json=nudge_payload,
                                timeout=httpx.Timeout(20.0, connect=10.0),
                            )
                            if nudge_resp.status_code == 200:
                                n_data = nudge_resp.json()
                                if n_data.get("choices") and isinstance(n_data["choices"][0], dict):
                                    n_c0 = n_data["choices"][0]
                                    n_msg = n_c0.get("message") or {}
                                    n_tcs = n_msg.get("tool_calls")
                                    if not n_tcs and n_msg.get("content"):
                                        _, n_tcs = parse_raw_tool_calls(n_msg["content"])
                                    if n_tcs:
                                        m0["tool_calls"] = n_tcs
                                        c0["finish_reason"] = "tool_calls"
                                        logger.info(f"非流式 Auto-Nudge 成功补全工具调用: {[t.get('function', {}).get('name') for t in n_tcs]}")
                        except Exception as ne:
                            logger.warning(f"非流式 Auto-Nudge 失败 (放行): {ne}")

            scheduler.record_response(key_id, model, True, rt, 200, "")
            scheduler.release_client_request(client_id)

            # v10.0 熔断记录成功
            cb = get_circuit_breaker()
            cb.record_success(f"model:{model}")

            # 记录日志（全量日志）
            await _log_request(
                client_id, key_id, model, 200, rt,
                prompt_tokens, completion_tokens, total_tokens, False, "",
                start_ts=start_time,
                response_body=json.dumps(data, ensure_ascii=False)[:65536],
                request_path=str(request.url.path) if request else "",
                http_method=request.method if request else "",
                client_ip=get_client_ip(request) if request else "",
                user_agent=request.headers.get("user-agent", "")[:512] if request else "",
                cached_tokens=cached_tokens,
                cache_write_tokens=cache_write_tokens,
                ttft_ms=(time.time() - client_start) * 1000,
                total_duration_ms=(time.time() - client_start) * 1000,
            )

            return JSONResponse(status_code=200, content=data)

        elif resp.status_code == 429:
            # 429 - 触发桶级冷却
            error_body = _parse_upstream_error_body(resp.text)
            error_message = _safe_error_message(error_body)
            scheduler.record_response(key_id, model, False, rt, 429, "429")
            scheduler.trigger_hard_cooldown(key_id, model, "429")
            logger.warning(f"上游429: key={key_id[:8]} model={model} 触发桶级冷却")
            raise UpstreamRetryableError(f"upstream_429: {error_message}", status_code=429, kind="rate_limited")

        elif resp.status_code >= 500:
            # 5xx - 上游瞬时故障，可换 key 或切换聚合候选。
            error_body = _parse_upstream_error_body(resp.text)
            error_message = _safe_error_message(error_body)
            scheduler.record_response(key_id, model, False, rt, resp.status_code, "5xx")
            scheduler.trigger_hard_cooldown(key_id, model, "5xx")
            logger.warning(f"上游{resp.status_code}: key={key_id[:8]} model={model} 触发桶级冷却+5xx熔断检测")
            raise UpstreamRetryableError(f"upstream_{resp.status_code}: {error_message}", status_code=resp.status_code, kind="server")

        else:
            error_body = _parse_upstream_error_body(resp.text)
            error_message = _safe_error_message(error_body)
            scheduler.record_response(key_id, model, False, rt, resp.status_code, "4xx")

            # 401 是当前上游 key 失效；403 只有明确的模型权限错误才切模型。
            if resp.status_code == 401:
                raise UpstreamRetryableError(f"upstream_401: {error_message}", status_code=401, kind="auth")
            if resp.status_code == 403:
                lower_message = error_message.lower()
                if "model" in lower_message and any(marker in lower_message for marker in ("permission", "access", "not allowed", "无权限")):
                    if in_chain:
                        raise UpstreamModelUnavailableError(403, error_body)
                elif not any(marker in lower_message for marker in ("content", "policy", "safety", "moderation", "内容", "策略", "安全")):
                    scheduler.trigger_hard_cooldown(key_id, model, "403")
                    raise UpstreamRetryableError(f"upstream_403: {error_message}", status_code=403, kind="forbidden")

            if _is_model_not_found(resp.status_code, error_body) and in_chain:
                raise UpstreamModelUnavailableError(resp.status_code, error_body)

            # 参数、上下文、请求体等错误不会因换 key/模型而改善；聚合链必须立即停止。
            if in_chain:
                raise UpstreamRequestError(resp.status_code, error_body)
            scheduler.release_client_request(client_id)
            return JSONResponse(status_code=resp.status_code, content=error_body)

    except UpstreamRetryableError:
        raise  # 重新抛出给tenacity处理
    except httpx.ConnectTimeout as e:
        # 连接建立超时（上限30s）：等待成本低，保留换 key 重试价值
        rt = time.time() - start_time
        scheduler.record_response(key_id, model, False, rt, 0, "timeout")
        scheduler.trigger_hard_cooldown(key_id, model, "timeout")
        logger.warning(f"上游连接超时: key={key_id[:8]} model={model} rt={rt:.1f}s")
        raise UpstreamRetryableError(
            f"connect_timeout: key={key_id[:8]} model={model}", kind="connect_timeout"
        ) from e
    except httpx.TimeoutException:
        rt = time.time() - start_time
        scheduler.record_response(key_id, model, False, rt, 0, "timeout")
        # 超时只触发冷却，不触发隔离（超时可能是NVIDIA响应慢，不代表密钥故障）
        scheduler.trigger_hard_cooldown(key_id, model, "timeout")
        logger.warning(f"请求超时: key={key_id[:8]} model={model} rt={rt:.1f}s")
        raise UpstreamRetryableError(
            f"timeout: key={key_id[:8]} model={model}", kind="timeout"
        )
    except (httpx.NetworkError, httpx.RemoteProtocolError) as e:
        rt = time.time() - start_time
        scheduler.record_response(key_id, model, False, rt, 0, "conn_error")
        scheduler.trigger_isolation(key_id, model, "conn_error")
        logger.error(f"上游网络错误: key={key_id[:8]} model={model} error={e}")
        raise UpstreamRetryableError("upstream_network_error", kind="network") from e


# ========== 带重试的上游调用（聊天补全，普通单模型路径） ==========

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=16),
    retry=retry_if_exception_type(UpstreamRetryableError),
    reraise=True
)
async def _call_upstream(
    scheduler, client_id, model, upstream_url, body, timeout, is_stream, request,
    dispatch_start_time=None
):
    """普通单模型上游调用：tenacity 自动重试（每次重选 key）。

    行为与改造前完全一致——聚合模型链路不走这里，而是直接调用 _dispatch_once。
    """
    return await _dispatch_once(
        scheduler, client_id, model, upstream_url, body, timeout, is_stream, request,
        dispatch_start_time=dispatch_start_time, in_chain=False,
    )


# ========== 上游调用核心（选 key + 调一次） ==========

async def _dispatch_once(
    scheduler, client_id, model, upstream_url, body, timeout, is_stream, request,
    dispatch_start_time=None, in_chain=False
):
    """选 key 并调用一次上游（聊天补全）。

    遇 429/5xx/超时/连接错误抛 UpstreamRetryableError，供上层（重试或聚合链）处理。

    in_chain=True（聚合模型链场景）：
    - select_key 返回 None 时不直接向客户端返回 503，而是抛 UpstreamRetryableError
      且**不释放**客户端并发名额，交由链路层切换到下一个真实模型；
      并发名额由链路层在整链结束时统一释放，保证记账平衡。
    """
    # : 记录密钥选择耗时（网关调度延迟度量）
    _key_select_start = time.time()
    select_result = await asyncio.to_thread(scheduler.select_key, model)
    _key_select_ms = (time.time() - _key_select_start) * 1000

    # 将调度耗时存入scheduler临时属性和request.state，供日志记录使用
    scheduler._last_dispatch_ms = _key_select_ms
    if request and hasattr(request, 'state'):
        request.state.gateway_dispatch_ms = _key_select_ms
    if not select_result:
        # 聚合链路场景：不在此处 release（并发计数由 chain 层统一管理），
        # 直接抛 UpstreamRetryableError，让上层切到链中下一个模型。
        if in_chain:
            raise UpstreamRetryableError(
                f"no_key: model={model}（该模型暂无可用密钥/熔断中）", kind="no_key"
            )
        scheduler.release_client_request(client_id)
        # 检查是否是模型级熔断（而非真正的密钥不可用）
        now = time.time()
        circuit_breaker_until = scheduler._model_429_circuit_breaker.get(model, 0)
        circuit_breaker_5xx_until = scheduler._model_5xx_circuit_breaker.get(model, 0)
        if now < circuit_breaker_until:
            raise HTTPException(status_code=429, detail={
                "message": "上游限流中，请稍后重试",
                "type": "rate_limit_exceeded",
                "code": "model_rate_limited",
            })
        if now < circuit_breaker_5xx_until:
            raise HTTPException(status_code=503, detail={
                "message": f"模型 '{model}' 上游服务暂时不可用（5xx错误过多），请稍后重试",
                "type": "service_unavailable",
                "code": "model_5xx_circuit_broken",
            })
        raise HTTPException(status_code=503, detail={
            "message": "服务繁忙，所有上游密钥暂不可用，请稍后重试",
            "type": "service_unavailable",
            "code": "all_keys_exhausted",
        })

    key_id, api_key, selected_model, provider, base_url = select_result

    # 按选中 key 的 provider/base_url 解析最终上游 URL：
    # - 优先用 key 自带的 base_url；空则用该 provider 默认 base_url；
    # - 都没拿到再退到全局 admin_settings.upstream_base_url（兼容历史 NVIDIA 默认）
    if base_url:
        effective_base_url = base_url
    else:
        adapters = _get_adapter_registry()
        adapter = adapters.get_or_default(provider) if adapters else None
        effective_base_url = adapter.default_base_url if adapter else None
    if not effective_base_url:
        effective_base_url = get_setting("upstream_base_url") or "https://integrate.api.nvidia.com/v1"
    # 计算最终请求 URL：替换原本传入的 upstream_url
    # （chat_completions 路径在 base_url 后拼 /chat/completions，embeddings 拼 /embeddings）
    effective_upstream_url = _join_upstream_url(effective_base_url, upstream_url)

    logger.info(
        f"上游路由: key={key_id[:8]} provider={provider} "
        f"base={effective_base_url} model={model}"
    )

    body = dict(body)
    _adapt_reasoning_effort_for_provider(body, provider=provider, model_ref=model, base_url=effective_base_url)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # CC Switch适配
    if is_cc_switch_request(request):
        headers["User-Agent"] = request.headers.get("user-agent", "")

    start_time = time.time()

    # ---- Responses API 转发分支 ----
    # - openai-codex（ChatGPT 账号登录）：api_key_ciphertext 存加密 OAuth 信封，转发前
    #   换出新鲜 access token；
    # - openai-responses（手动选择的 API 格式）：普通 API Key 直调上游 /responses。
    # 两者都把 chat/completions 翻成 Responses 协议再转发。
    responses_mode = provider in RESPONSES_PROVIDER_IDS
    client_override = None
    if responses_mode:
        effective_upstream_url = effective_base_url.rstrip("/") + "/responses"
        if parse_envelope(api_key) is not None:
            try:
                access_token = await ensure_fresh_access(key_id, api_key)
            except OAuthError as exc:
                rt = time.time() - start_time
                scheduler.record_response(key_id, model, False, rt, 0, "oauth_refresh_failed")
                if not in_chain:
                    scheduler.release_client_request(client_id)
                    raise HTTPException(status_code=502, detail={
                        "message": f"ChatGPT 账号凭据刷新失败：{exc}",
                        "type": "oauth_refresh_failed",
                        "code": "oauth_refresh_failed",
                    })
                # 聚合链场景：抛可重试错误，让链切换到下一个候选模型
                raise UpstreamRetryableError(f"oauth_refresh_failed: {exc}", kind="oauth")
            if not access_token:
                rt = time.time() - start_time
                scheduler.record_response(key_id, model, False, rt, 0, "oauth_refresh_failed")
                if not in_chain:
                    scheduler.release_client_request(client_id)
                    raise HTTPException(status_code=502, detail={
                        "message": "该上游密钥不是 ChatGPT 账号登录凭据（信封缺失），请删除后用「接入方法→OpenAI (ChatGPT 账号登录)」重新添加",
                        "type": "oauth_refresh_failed",
                        "code": "oauth_refresh_failed",
                    })
                raise UpstreamRetryableError("oauth_refresh_failed: 凭据信封缺失", kind="oauth")
        else:
            # 兼容：普通 OpenAI 平台 Key 也允许直调 /responses
            access_token = api_key
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        body = chat_to_responses(body, body.get("model") or model)
        # 配置了 oauth_proxy 时，OpenAI 访问走专用代理连接池
        try:
            client_override = await get_proxied_client()
        except Exception as exc:
            logger.warning(f"oauth_proxy 连接池创建失败，回退直连: {exc}")
            client_override = None

    try:
        if is_stream:
            return await _handle_stream_request(
                scheduler, client_id, key_id, model,
                effective_upstream_url, headers, body, timeout, start_time, request,
                in_chain=in_chain, responses_mode=responses_mode, client_override=client_override,
                request_start=dispatch_start_time or start_time,
            )
        else:
            return await _handle_nonstream_request(
                scheduler, client_id, key_id, model,
                effective_upstream_url, headers, body, timeout, start_time, request,
                in_chain=in_chain, responses_mode=responses_mode, client_override=client_override,
                request_start=dispatch_start_time or start_time,
            )
    except (UpstreamRetryableError, UpstreamRequestError, UpstreamModelUnavailableError):
        # 链路层负责请求级终止、候选切换以及整链唯一一次并发释放。
        raise
    except Exception as e:
        rt = time.time() - start_time
        scheduler.record_response(key_id, model, False, rt, 0, "error")
        if not in_chain:
            scheduler.release_client_request(client_id)
        logger.error(f"请求异常: key={key_id[:8]} model={model} error={e}", exc_info=True)
        raise HTTPException(status_code=500, detail={
            "message": "网关处理上游响应时发生内部错误",
            "type": "internal_error",
            "code": "internal_error",
        })


# ========== 虚拟聚合模型：Failover 链调用 ==========

def _split_provider_model_ref(model_ref: str) -> tuple[str, str]:
    """将聚合链的 ``provider::model`` 引用拆为上游模型 ID。

    未带 provider 的旧配置保持原样，兼容已有聚合模型。
    """
    if "::" in model_ref:
        provider, model_id = model_ref.split("::", 1)
        if provider and model_id:
            return provider, model_id
    return "", model_ref


async def _call_upstream_chain(
    scheduler, client_id, virtual_id, chain, base_upstream_url, body,
    is_stream, request, dispatch_start_time=None
):
    """按 chain 顺序依次尝试真实模型，实现无缝故障转移。

    切换速度与容错的平衡策略（配合 virtual_models 的自适应软熔断）：

    对每个真实模型：
    - 若该模型正处于软熔断中（近期持续失败），直接跳过；
    - 否则按"错误分级"决定模型内重试次数：超时/无密钥类错误立即换下一个候选
      （等待成本高或重选无变化），429/5xx/401 等瞬时错误保留换 key 重试机会；
    - 链中非末位候选的等待时间封顶为 CHAIN_CANDIDATE_TIMEOUT_SECONDS（快速试错），
      末位兜底候选保留完整模型超时（耐心等完）；
    - 整链超过 CHAIN_TOTAL_BUDGET_SECONDS 后，仅尝试末位兜底候选；
    - 成功则记 success 并直接返回响应（流式一旦开始输出即锁定该模型）；
    - 该模型这轮尝试全失败则记 failure（可能触发软熔断）并切下一个；
      但 no_key（无可用密钥）不算模型失败——密钥冷却 ≠ 模型坏了。

    仅当 chain 内所有模型都失败时，才向客户端返回错误。
    并发名额：整链只在最终结束（成功返回前 / 全失败后）释放一次，
    保证与 record_client_request 的一次登记平衡。
    """
    from app import virtual_models as vm

    # 覆写请求体的 model 为当前真实模型；chat_path 沿用传入的 upstream_url 结构
    last_error = "no_candidate"
    last_kind = ""       # 最后一个失败候选的错误分级（用于映射返回给客户端的状态码）
    last_status = None   # 最后一个失败候选的上游状态码
    tried_any = False
    all_cooling = True
    chain_started = dispatch_start_time if dispatch_start_time else time.time()
    budget_exhausted_logged = False

    # 动态获取当前请求的候选链顺序（支持平滑轮询 round-robin、主备优先 priority、自适应感知 adaptive）
    ordered_chain = vm.get_ordered_chain(virtual_id) or chain
    logger.info(f"[聚合] 虚拟模型 {virtual_id} 调度队列: {ordered_chain}")

    # 预取熔断快照：既用于跳过判断，也用于判断某候选后面是否还有"真正可切"的备选
    cooling_flags = [vm.is_model_cooling(m) for m in ordered_chain]

    for idx, real_model in enumerate(ordered_chain):
        # 软熔断跳过（但记录，若全部在熔断中则最后放宽重试第一个）
        if cooling_flags[idx]:
            rem = vm.cooldown_remaining(real_model)
            logger.info(f"[聚合] 跳过熔断中的模型 {real_model}（剩余 {rem:.0f}s）")
            last_error = f"model_cooling:{real_model}"
            # 跳过类原因不覆盖真实上游错误（保留 504/429/502 的状态码保真）
            if not last_kind:
                last_kind = "model_cooling"
            continue
        all_cooling = False
        tried_any = True

        # 整链预算：超时后跳过"还有备选"的候选，把时间留给链尾兜底模型
        has_fallback = any(not cooling_flags[j] for j in range(idx + 1, len(ordered_chain)))
        if has_fallback and time.time() - chain_started > vm.CHAIN_TOTAL_BUDGET_SECONDS:
            if not budget_exhausted_logged:
                budget_exhausted_logged = True
                logger.warning(
                    f"[聚合] 整链预算 {vm.CHAIN_TOTAL_BUDGET_SECONDS:.0f}s 已耗尽，"
                    "跳过剩余非兜底候选，直接尝试链尾兜底"
                )
            last_error = f"chain_budget_exhausted:{real_model}"
            if not last_kind:
                last_kind = "chain_budget"
            continue

        # 调度器使用完整引用锁定 provider；发送给上游的 model 则必须是原始模型 ID。
        _, upstream_model = _split_provider_model_ref(real_model)
        sub_body = dict(body)
        sub_body["model"] = upstream_model
        sub_body = apply_model_defaults(sub_body)
        pre_stripped = _strip_blocked_params(real_model, sub_body)
        if pre_stripped:
            logger.info(f"[聚合] {real_model} 预防性剥离已知不支持参数 {pre_stripped}")
        _adapt_reasoning_effort_for_provider(sub_body, provider=_, model_ref=real_model, base_url=base_upstream_url)
        if is_stream:
            sub_body.setdefault("stream_options", {})
            if isinstance(sub_body["stream_options"], dict):
                sub_body["stream_options"]["include_usage"] = True

        # 候选超时预算：非末位候选封顶（快速试错）；末位兜底候选用完整模型超时，
        # 但同样受整链预算约束（保底 CHAIN_FINAL_MIN_SECONDS），避免整链无限拖长、
        # 远超客户端/中间代理的超时窗口（那会导致客户端先收到 504/524）。
        model_timeout = get_timeout_for_model(upstream_model)
        if has_fallback and model_timeout > vm.CHAIN_CANDIDATE_TIMEOUT_SECONDS:
            timeout = vm.CHAIN_CANDIDATE_TIMEOUT_SECONDS
            logger.info(
                f"[聚合] {real_model} 为非兜底候选，单次尝试预算封顶 {timeout:.0f}s"
                f"（模型默认超时 {model_timeout:.0f}s）"
            )
        elif not has_fallback:
            remaining_budget = vm.CHAIN_TOTAL_BUDGET_SECONDS - (time.time() - chain_started)
            timeout = min(model_timeout, max(vm.CHAIN_FINAL_MIN_SECONDS, remaining_budget))
        else:
            timeout = model_timeout

        logger.info(f"[聚合] 虚拟模型 {virtual_id} -> 尝试真实模型 {real_model}")

        # 错误分级决定该模型内的重试次数（见 virtual_models.chain_attempts_for_kind）；
        # 请求级错误必须终止整链。
        model_failed = False
        for attempt in range(vm.CHAIN_PER_MODEL_ATTEMPTS):
            attempt_started = time.time()
            try:
                resp = await _dispatch_once(
                    scheduler, client_id, real_model, base_upstream_url, sub_body,
                    timeout, is_stream, request,
                    dispatch_start_time=dispatch_start_time, in_chain=True,
                )
                vm.record_model_success(real_model)
                logger.info(f"[聚合] 真实模型 {real_model} 命中，返回响应")
                return resp
            except UpstreamRequestError as e:
                _record_upstream_attempt(request, real_model, e, attempt_started, sub_body)
                # 参数不被该上游接受（HTTP 400 形式）：剥离后原地重试并学习
                param_names = _match_unsupported_params(_safe_error_message(e.error_body))
                if param_names:
                    _learn_param_blocklist(real_model, param_names)
                    stripped = _strip_params_from_body(sub_body, param_names)
                    if stripped:
                        last_error = _safe_error_message(e.error_body)
                        logger.warning(f"[聚合] {real_model} 不支持参数 {stripped}，已剥离后原地重试")
                        continue
                last_kind, last_status = "request_error", e.status_code
                if _should_failover_after_request_error(e):
                    last_error = _safe_error_message(e.error_body)
                    model_failed = True
                    logger.warning(
                        f"[聚合] {real_model} returned HTTP {e.status_code}; "
                        "trying the next candidate"
                    )
                    break
                scheduler.release_client_request(client_id)
                # 参数、上下文、内容策略等拒绝与候选无关，保留原上游状态即可。
                return JSONResponse(status_code=e.status_code, content=e.error_body)
            except UpstreamModelUnavailableError as e:
                _record_upstream_attempt(request, real_model, e, attempt_started, sub_body)
                last_error = _safe_error_message(e.error_body)
                last_kind, last_status = "model_unavailable", e.status_code
                model_failed = True
                logger.warning(f"[聚合] {real_model} 模型不可用，切换下一个候选")
                break
            except UpstreamRetryableError as e:
                _record_upstream_attempt(request, real_model, e, attempt_started, sub_body)
                last_error = _safe_error_message(str(e))
                kind = getattr(e, "kind", "") or "upstream_error"
                # 参数不被该上游接受（可能以 200+SSE 流内错误形式出现，而非 HTTP 400）：
                # 同样剥离后原地重试并学习——否则每个候选都会带着同样的坏参数全灭
                param_names = _match_unsupported_params(last_error)
                if param_names:
                    _learn_param_blocklist(real_model, param_names)
                    stripped = _strip_params_from_body(sub_body, param_names)
                    if stripped:
                        model_failed = True
                        logger.warning(f"[聚合] {real_model} 不支持参数 {stripped}（{kind}），已剥离后原地重试")
                        continue
                if kind == "no_key":
                    # 该模型当前没有可用密钥（密钥级冷却/无密钥）：立刻重选结果不变，
                    # 且"密钥暂时不可用"不应记成"模型故障"，直接换下一个候选；
                    # 也不覆盖前面候选的真实上游错误（状态码保真）。
                    logger.info(f"[聚合] {real_model} 暂无可用密钥，直接切换下一个候选")
                    break
                last_kind, last_status = kind, e.status_code
                if attempt + 1 >= vm.chain_attempts_for_kind(kind):
                    model_failed = True
                    if kind == "timeout":
                        logger.warning(
                            f"[聚合] {real_model} 等待超时，按策略不再对同一模型重试"
                            "（再等一轮只会更慢），立即切换下一个候选"
                        )
                    else:
                        logger.warning(f"[聚合] {real_model} {kind}失败: {last_error}，切换下一个候选")
                    break
                model_failed = True
                logger.warning(
                    f"[聚合] {real_model} 第{attempt + 1}次瞬时失败({kind}): {last_error}，换 key 重试"
                )
                continue
            except HTTPException as e:
                last_error = (e.detail.get("message") if isinstance(e.detail, dict) else str(e.detail)) or "http_error"
                last_kind, last_status = "http_error", e.status_code
                model_failed = True
                logger.warning(f"[聚合] {real_model} 返回 HTTP {e.status_code}，切换下一个候选")
                break

        # 只有模型不可用或真实瞬时故障才进入聚合软熔断；
        # 请求级错误已在上方返回，no_key 不计入（避免误伤健康模型）。
        if model_failed:
            is_sev = last_kind in ("network", "ssl", "503", "conn_error", "timeout", "upstream_error")
            vm.record_model_failure(real_model, reason=last_kind, is_severe=is_sev)

    # 如果所有模型都在软熔断中（一个都没真正尝试），放宽：强试第一个，避免全链假死
    if not tried_any and all_cooling and ordered_chain:
        forced = ordered_chain[0]
        logger.warning(f"[聚合] 全链熔断中，强制尝试首个模型 {forced}")
        _, upstream_model = _split_provider_model_ref(forced)
        sub_body = dict(body)
        sub_body["model"] = upstream_model
        sub_body = apply_model_defaults(sub_body)
        _adapt_reasoning_effort_for_provider(sub_body, provider=_, model_ref=forced, base_url=base_upstream_url)
        if is_stream:
            sub_body.setdefault("stream_options", {})
            if isinstance(sub_body["stream_options"], dict):
                sub_body["stream_options"]["include_usage"] = True
        timeout = get_timeout_for_model(upstream_model)
        attempt_started = time.time()
        try:
            resp = await _dispatch_once(
                scheduler, client_id, forced, base_upstream_url, sub_body,
                timeout, is_stream, request,
                dispatch_start_time=dispatch_start_time, in_chain=True,
            )
            vm.record_model_success(forced)
            return resp
        except UpstreamRequestError as e:
            _record_upstream_attempt(request, forced, e, attempt_started, sub_body)
            scheduler.release_client_request(client_id)
            return JSONResponse(status_code=e.status_code, content=e.error_body)
        except (UpstreamRetryableError, UpstreamModelUnavailableError, HTTPException) as e:
            _record_upstream_attempt(request, forced, e, attempt_started, sub_body)
            last_error = str(e.detail if isinstance(e, HTTPException) else e)
            last_kind = getattr(e, "kind", "") or ("http_error" if isinstance(e, HTTPException) else last_kind)
            last_status = getattr(e, "status_code", None)

    # 全链失败：释放并发名额，向客户端返回错误。
    # 状态码按最后一个候选的失败类别映射（而非一律 503），
    # 保证请求日志与客户端实际收到的一致：超时→504、限流→429、上游5xx→502。
    scheduler.release_client_request(client_id)
    fail_status, fail_code, fail_type = _chain_failure_status(last_kind, last_status)
    logger.error(f"[聚合] 虚拟模型 {virtual_id} 全链失败({fail_status}): {last_error}")
    raise HTTPException(status_code=fail_status, detail={
        "message": f"聚合模型 '{virtual_id}' 的所有备选模型当前均不可用，请稍后重试。",
        "type": fail_type,
        "code": fail_code,
        "last_error": str(last_error)[:200],
    })


# ========== 聊天补全端点 ==========

@router.post("/v1/chat/completions", tags=["公共API"])
@router.post("/api/v1/chat/completions", tags=["公共API"])
async def chat_completions(request: Request):
    """
    OpenAI兼容聊天补全端点

    流程：
    1. 客户端认证
    2. 解析请求体
    3. 调度器选择密钥
    4. 调用上游NVIDIA API（tenacity自动重试）
    5. 流式/非流式响应
    6. 记录请求日志
    """
    # 1. 客户端认证 (v10.0: 异步调用避免同步DB查询阻塞事件循环)
    try:
        client_info = await asyncio.to_thread(authenticate_client, request)
        client_id = client_info["client_id"]
    except HTTPException as e:
        # : 认证失败日志 - 尝试从key_prefix解析client信息
        _auth_key_prefix = ""
        _auth_client_id = ""
        _auth_header = request.headers.get("authorization", "")
        if _auth_header.startswith("Bearer "):
            _auth_key_prefix = _auth_header[7:20]
            try:
                _key_row = await asyncio.to_thread(
                    fetch_one,
                    "SELECT client_id FROM client_api_keys WHERE key_prefix LIKE %s LIMIT 1",
                    (f"{_auth_key_prefix}%",),
                )
                if _key_row:
                    _auth_client_id = _key_row["client_id"]
            except Exception:
                pass
        # 认证失败的日志行也尽量带上请求的模型名（从请求体解析），便于定位是哪个模型调用被拒
        _auth_model = ""
        _auth_body_sniff = ""
        if request.method == "POST":
            try:
                _auth_body = await request.json()
                if isinstance(_auth_body, dict):
                    _auth_model = str(_auth_body.get("model", "") or "")[:200]
                    _auth_body_sniff = json.dumps(_auth_body, ensure_ascii=False)[:4096]
            except Exception:
                pass
        await _log_request(
            client_id=_auth_client_id, key_id="", model=_auth_model, status_code=e.status_code, rt=0,
            prompt_tokens=0, completion_tokens=0, total_tokens=0,
            is_stream=False, error_msg="API密钥无效或已被禁用",
            error_type="AuthenticationError",
            error_detail=e.detail.get("message", str(e.detail)) if isinstance(e.detail, dict) else str(e.detail),
            business_code=e.detail.get("code", "invalid_api_key") if isinstance(e.detail, dict) else "invalid_api_key",
            request_path=str(request.url.path),
            http_method=request.method,
            client_ip=get_client_ip(request),
            user_agent=request.headers.get("user-agent", "")[:512],
            request_body=_auth_body_sniff,
        )
        raise

    # 2. 解析请求体
    try:
        body = await request.json()
    except Exception:
        # : 记录JSON解析失败
        await _log_request(
            client_id=client_id, key_id="", model="", status_code=400, rt=0,
            prompt_tokens=0, completion_tokens=0, total_tokens=0,
            is_stream=False, error_msg="请求体JSON解析失败",
            error_type="ParseError", error_detail="请求体JSON解析失败",
            business_code="invalid_json",
            request_path=str(request.url.path),
            http_method=request.method,
            client_ip=get_client_ip(request),
            user_agent=request.headers.get("user-agent", "")[:512],
        )
        raise HTTPException(status_code=400, detail={
            "message": "请求体JSON解析失败",
            "type": "invalid_request_error",
            "code": "invalid_json",
        })

    model = body.get("model", "")
    # 中间件不依赖被截断的原始请求体解析模型；失败日志也可关联虚拟模型。
    request.state.log_model = model
    request.state.log_is_stream = bool(body.get("stream"))
    request.state.log_client_id = client_id
    request.state.upstream_attempts = []

    if not model:
        raise HTTPException(status_code=400, detail={
            "message": "缺少model参数",
            "type": "invalid_request_error",
            "code": "missing_model",
        })

    # === 虚拟聚合模型（Failover 链）拦截 ===
    # 必须在 validate_and_correct_model 之前判断：否则虚拟 ID 会被模糊匹配
    # 误改成某个真实模型。命中虚拟模型时，跳过单模型的校验/纠错/熔断逻辑，
    # 直接进入 chain 分发（chain 内的每个真实模型再各自校验）。
    from app import virtual_models as _vm
    _is_virtual = _vm.is_virtual_model(model)
    _virtual_chain = _vm.get_chain(model) if _is_virtual else []
    if _is_virtual and not _virtual_chain:
        raise HTTPException(status_code=503, detail={
            "message": f"聚合模型 '{model}' 未配置任何可用的真实模型",
            "type": "service_unavailable",
            "code": "virtual_model_empty_chain",
        })

    # === 定点路由（provider::model）：客户端（如管理后台网页对话）精确选择某个供应商下的模型 ===
    # 调度器 select_key 原生支持 provider::model 锁定单一供应商；此处需要：
    # 1) 跳过虚拟模型展开与模型ID纠错（"::" 引用不是注册模型ID，纠错会误伤）；
    # 2) 发给上游的 body.model 换成裸模型 ID（与聚合链 _call_upstream_chain 行为一致）。
    _pinned_provider, _pinned_model = _split_provider_model_ref(model)
    _is_pinned = bool(_pinned_provider and _pinned_model)
    if _is_pinned:
        body["model"] = _pinned_model

    # === v9.2: 模型ID智能映射与纠错 ===
    corrected_model, was_corrected = (model, False) if (_is_virtual or _is_pinned) else validate_and_correct_model(model)
    if was_corrected:
        logger.info(f"模型ID已自动修正: '{model}' -> '{corrected_model}'")
        body["model"] = corrected_model
        model = corrected_model
    elif model != corrected_model:
        # 无法自动修正，返回建议
        suggestion = build_model_error_suggestion(model)
        raise HTTPException(status_code=400, detail={
            "message": f"模型ID '{model}' 无效。{suggestion}",
            "type": "invalid_request_error",
            "code": "invalid_model",
        })

    # === v9.2: 检查模型是否存在（对照NIM模型目录 + 上游实时列表） ===
    # 聚合虚拟模型跳过此校验：它不是真实上游模型，真实性由 chain 里各成员保证
    # 定点路由（provider::model）同样跳过：provider 前缀由调度器解析
    if not _is_virtual and not _is_pinned:
        try:
            from app.nim_models import NIM_MODEL_CATALOG
            is_known = model in _VERIFIED_WORKING_MODELS or model in NIM_MODEL_CATALOG
            if not is_known and _models_cache["data"]:
                is_known = any(m.get("id") == model for m in _models_cache["data"] if isinstance(m, dict))
            if not is_known:
                # 尝试从上游实时获取
                try:
                    upstream_models = await fetch_upstream_models()
                    is_known = any(m.get("id") == model for m in upstream_models if isinstance(m, dict))
                except Exception:
                    pass
            if not is_known:
                suggestion = build_model_error_suggestion(model)
                logger.warning(f"模型ID不在可用列表中: '{model}' (可能不存在于上游，已放行)")
                # 不拦截，仅记录日志 - 上游可能有新增模型
        except ImportError:
            pass

    # === v9.2: 请求体格式强制校验 + 参数容错 ===
    validate_and_sanitize(body)

    # === v10.0: 请求安全校验已移除（不再限制消息数量/内容长度等） ===

    is_stream = body.get("stream", False)

    # 自动补全模型默认参数
    body = apply_model_defaults(body)

    # 流式请求注入 stream_options.include_usage
    if is_stream:
        if "stream_options" not in body:
            body["stream_options"] = {}
        body["stream_options"]["include_usage"] = True

    # === v10.0 防呆防傻：熔断器检查 ===
    # 聚合虚拟模型跳过：熔断粒度落在 chain 内各真实模型（_dispatch_once 内的模型级熔断
    # + virtual_models 的模型级软熔断），避免用虚拟 ID 做 key 误伤整条链
    if not _is_virtual:
        cb = get_circuit_breaker()
        cb_key = f"model:{model}"
        if not cb.can_request(cb_key):
            degradation = graceful_degradation_response("circuit_open")
            raise HTTPException(status_code=degradation["status_code"], detail={
                "message": degradation["message"],
                "type": degradation["type"],
                "code": degradation["code"],
                "suggestion": degradation["suggestion"],
            })

    # 3. 调度器初始化
    scheduler = get_scheduler()
    await scheduler._ensure_pools()

    # 记录客户端请求（算法5/6/7数据采集）
    scheduler.record_client_request(client_id, model)

    # 4. 带重试的上游调用
    raw_base_url = get_setting("upstream_base_url") or "https://integrate.api.nvidia.com/v1"
    base_url, url_warning = normalize_base_url(raw_base_url)
    if url_warning:
        logger.info(f"Base URL 标准化: {url_warning}")
    chat_path = get_setting("chat_path") or "/chat/completions"
    upstream_url = f"{base_url}{chat_path}"
    # 定点路由的超时启发式按裸模型 ID 判断（与聚合链取 upstream_model 一致）
    timeout = get_timeout_for_model(_pinned_model if _is_pinned else model)

    _request_start_ts = time.time()
    try:
        if _is_virtual:
            # 聚合模型：按 chain 顺序 failover（每个真实模型内部仍走完整调度）
            return await _call_upstream_chain(
                scheduler, client_id, model, _virtual_chain,
                upstream_url, body, is_stream, request,
                dispatch_start_time=_request_start_ts
            )
        return await _call_upstream(
            scheduler, client_id, model, upstream_url, body, timeout, is_stream, request,
            dispatch_start_time=_request_start_ts
        )
    except HTTPException as e:
        # v10.0 修复：记录服务不可用等HTTP异常日志
        if e.status_code >= 400:
            await _log_request(
                client_id=client_id, key_id="", model=model,
                status_code=e.status_code, rt=time.time() - _request_start_ts,
                prompt_tokens=0, completion_tokens=0, total_tokens=0,
                is_stream=is_stream,
                error_msg=e.detail.get("message", str(e.detail)) if isinstance(e.detail, dict) else str(e.detail),
                error_type="HTTPException",
                error_detail=e.detail.get("message", str(e.detail)) if isinstance(e.detail, dict) else str(e.detail),
                business_code=e.detail.get("code", "http_error") if isinstance(e.detail, dict) else "http_error",
                request_path=str(request.url.path), http_method=request.method,
                client_ip=get_client_ip(request),
                user_agent=request.headers.get("user-agent", "")[:512],
            )
        raise
    except UpstreamRetryableError as e:
        # 重试次数用尽
        scheduler.release_client_request(client_id)
        # v10.0 防呆防傻：记录熔断器失败
        cb = get_circuit_breaker()
        cb_key = f"model:{model}"
        cb.record_failure(cb_key, str(e))

        # v10.0 修复：记录上游重试失败日志
        last_err = str(e)
        if "429" in last_err:
            degradation = graceful_degradation_response("rate_limited")
            _err_type = "RateLimited"
            _biz_code = "upstream_rate_limited"
            if ":" in last_err:
                detail_part = last_err.split(":", 1)[1].strip()
                if detail_part:
                    degradation["message"] = f"上游限流中（{detail_part}），已自动重试所有可用密钥后仍然受限，请稍后重试。"
        elif "timeout" in last_err:
            degradation = graceful_degradation_response("timeout")
            _err_type = "TimeoutError"
            _biz_code = "upstream_timeout"
            degradation["message"] = f"上游服务响应超时（{last_err}），已自动重试所有可用密钥后仍超时，请稍后重试。"
        elif "conn_error" in last_err or "network_error" in last_err:
            degradation = graceful_degradation_response("connection_error")
            _err_type = "ConnectionError"
            _biz_code = "upstream_connection_error"
            degradation["message"] = f"上游连接失败（{last_err}），已自动重试所有可用密钥后仍失败，请检查网络或代理配置。"
        elif "403" in last_err:
            degradation = {
                "status_code": 403,
                "message": f"上游AI模型访问被拒绝（{last_err}）。API密钥权限不足或模型已被禁用，已自动切换到其他可用密钥后仍失败。",
                "type": "permission_error",
                "code": "upstream_access_denied",
                "suggestion": "请联系管理员检查上游密钥权限和模型白名单配置",
            }
            _err_type = "PermissionError"
            _biz_code = "upstream_access_denied"
        else:
            degradation = {
                "status_code": 502,
                "message": f"上游AI模型服务返回错误（{last_err}），已自动重试所有可用密钥后仍然失败。",
                "type": "upstream_error",
                "code": "max_retries_exceeded",
                "suggestion": "请稍后重试，如频繁遇到此错误请联系管理员检查上游服务状态",
            }
            _err_type = "UpstreamError"
            _biz_code = "max_retries_exceeded"
        await _log_request(
            client_id=client_id, key_id="", model=model,
            status_code=degradation["status_code"], rt=time.time() - _request_start_ts,
            prompt_tokens=0, completion_tokens=0, total_tokens=0,
            is_stream=is_stream, error_msg=last_err[:512],
            error_type=_err_type, error_detail=last_err[:1024],
            business_code=_biz_code,
            request_path=str(request.url.path), http_method=request.method,
            client_ip=get_client_ip(request),
            user_agent=request.headers.get("user-agent", "")[:512],
        )
        raise HTTPException(status_code=degradation["status_code"], detail=degradation)


# ========== 请求日志记录 ==========

def _log_request_sync(
    client_id: str, key_id: str, model: str, status_code: int,
    rt: float, prompt_tokens: int, completion_tokens: int,
    total_tokens: int, is_stream: bool, error_msg: str,
    start_ts: float = None,
    error_type: str = "", error_detail: str = "", error_stack: str = "",
    business_code: str = "", request_body: str = "", response_body: str = "",
    request_path: str = "", http_method: str = "",
    client_ip: str = "", user_agent: str = "",
    gateway_dispatch_ms: float = 0.0,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
    include_errors: bool = False,
    force_log_category: str = "",
    ttft_ms: float = 0.0,
    total_duration_ms: float = 0.0,
):
    """记录日志到数据库

    默认只写成功日志（>=400 的失败请求由中间件统一记录）。
    include_errors=True 供聚合链候选尝试日志（chain_attempt）使用，
    用 force_log_category 指定分类，避免失败行被吞掉。
    """
    if status_code >= 400 and not include_errors:
        return

    if gateway_dispatch_ms == 0.0:
        try:
            from app.scheduler import get_scheduler
            gateway_dispatch_ms = getattr(get_scheduler(), '_last_dispatch_ms', 0.0)
        except Exception:
            gateway_dispatch_ms = 0.0

    try:
        from app.database import localnow
        now_str = localnow()
        latency_ms = round(rt * 1000, 3)
        latency_us = int(rt * 1_000_000)

        # 首字时长 / 总使用时长：兼容历史行（未记录新字段时按端到端延迟兜底）
        try:
            ttft_value = round(float(ttft_ms), 3) if ttft_ms and float(ttft_ms) > 0 else None
        except (TypeError, ValueError):
            ttft_value = None
        try:
            total_value = round(float(total_duration_ms), 3) if total_duration_ms and float(total_duration_ms) > 0 else latency_ms
        except (TypeError, ValueError):
            total_value = latency_ms

        started_at = ""
        completed_at = ""
        if start_ts is not None:
            from datetime import datetime, timezone, timedelta
            CST = timezone(timedelta(hours=8))
            start_dt = datetime.fromtimestamp(start_ts, tz=CST)
            end_dt = datetime.fromtimestamp(start_ts + rt, tz=CST)
            started_at = start_dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{start_dt.microsecond // 1000:03d}+08:00"
            completed_at = end_dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{end_dt.microsecond // 1000:03d}+08:00"

        if force_log_category:
            log_category = force_log_category
        else:
            log_category = "normal"
            if status_code == 401:
                log_category = "auth_fail"
            elif status_code == 403 or status_code >= 400 or status_code == 0:
                log_category = "error"

        # 计算请求成本与货币
        cost = 0.0
        currency = "USD"
        try:
            from app.pricing import calculate_request_cost
            c_info = calculate_request_cost(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                cache_write_tokens=cache_write_tokens,
            )
            cost = c_info.get("cost", 0.0)
            currency = c_info.get("currency", "USD")
        except Exception as ce:
            logger.warning(f"Error calculating request cost: {ce}")

        execute(
            """INSERT INTO request_logs
               (id, client_id, upstream_key_id, model, status_code, latency_ms, latency_us, retried,
                prompt_tokens, completion_tokens, total_tokens, is_stream, error_msg, created_at,
                started_at, completed_at,
                request_path, http_method, client_ip, user_agent,
                request_params, request_body, response_body,
                error_type, error_detail, error_stack, business_code, log_category, gateway_dispatch_ms,
                cached_tokens, cache_write_tokens, cost, currency, ttft_ms, total_duration_ms)
               VALUES (%s, %s, %s, %s, %s, %s, %s, 0, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                str(uuid.uuid4()), client_id, key_id, model,
                status_code, latency_ms, latency_us,
                prompt_tokens, completion_tokens, total_tokens,
                1 if is_stream else 0, error_msg, now_str,
                started_at, completed_at,
                request_path, http_method, client_ip, user_agent,
                "", request_body, response_body,
                error_type, error_detail, error_stack, business_code, log_category,
                round(gateway_dispatch_ms, 2),
                cached_tokens, cache_write_tokens, cost, currency,
                ttft_value, total_value,
            ),
        )

    except Exception as e:
        logger.error(f"记录请求日志失败: {e}")


async def _log_request(*args, **kwargs):
    """异步包装器：将同步的 DB 写入操作放到线程池中执行，避免阻塞事件循环"""
    await asyncio.to_thread(_log_request_sync, *args, **kwargs)


# ========== Embeddings带重试上游调用 ==========

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=16),
    retry=retry_if_exception_type(UpstreamRetryableError),
    reraise=True
)
async def _call_upstream_embeddings(scheduler, client_id, model, url, body):
    """带tenacity重试的上游调用 - 向量化

    每次重试会重新选择密钥，遇到429/5xx/连接错误抛出UpstreamRetryableError触发重试。
    """
    select_result = await asyncio.to_thread(scheduler.select_key, model)
    if not select_result:
        if client_id:
            scheduler.release_client_request(client_id)
        raise HTTPException(status_code=503, detail={
            "message": "服务繁忙，请稍后重试",
            "type": "service_unavailable",
            "code": "all_keys_exhausted",
        })

    key_id, api_key, selected_model, provider, base_url = select_result

    # 按 provider/base_url 解析上游 URL（与 chat_completions 一致）
    if base_url:
        effective_base_url = base_url
    else:
        adapters = _get_adapter_registry()
        adapter = adapters.get_or_default(provider) if adapters else None
        effective_base_url = adapter.default_base_url if adapter else None
    if not effective_base_url:
        effective_base_url = get_setting("upstream_base_url") or "https://integrate.api.nvidia.com/v1"
    effective_url = _join_upstream_url(effective_base_url, url)
    logger.info(
        f"embeddings路由: key={key_id[:8]} provider={provider} base={effective_base_url} model={model}"
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    start_time = time.time()
    try:
        pool = await scheduler.get_http_pool()
        resp = await pool.post(effective_url, headers=headers, json=body, timeout=60.0)
        rt = time.time() - start_time

        if resp.status_code == 200:
            scheduler.record_response(key_id, model, True, rt, 200, "")
            # v10.0 防呆防傻：记录熔断器成功
            cb = get_circuit_breaker()
            cb.record_success(f"embeddings:{model}")
            return JSONResponse(status_code=200, content=resp.json())
        elif resp.status_code == 429:
            # 429 - 触发桶级冷却并切换密钥重试
            scheduler.record_response(key_id, model, False, rt, 429, "429")
            scheduler.trigger_hard_cooldown(key_id, model, "429")
            logger.warning(f"embeddings上游429: key={key_id[:8]} model={model}")
            raise UpstreamRetryableError(f"429: key={key_id[:8]} model={model}", kind="rate_limited")
        elif resp.status_code >= 500:
            # 5xx - 记录并切换密钥重试
            scheduler.record_response(key_id, model, False, rt, resp.status_code, "5xx")
            logger.warning(f"embeddings上游{resp.status_code}: key={key_id[:8]} model={model}")
            raise UpstreamRetryableError(f"{resp.status_code}: key={key_id[:8]} model={model}", kind="server")
        else:
            # 4xx（非429）- 直接返回错误
            scheduler.record_response(key_id, model, False, rt, resp.status_code, "4xx")
            try:
                return JSONResponse(status_code=resp.status_code, content=resp.json())
            except Exception:
                return JSONResponse(status_code=resp.status_code, content={"error": {"message": resp.text}})

    except UpstreamRetryableError:
        raise  # 让tenacity处理重试
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        # 连接错误 - 触发隔离并切换密钥重试
        rt = time.time() - start_time
        scheduler.record_response(key_id, model, False, rt, 0, "conn_error")
        scheduler.trigger_isolation(key_id, model, "conn_error")
        logger.warning(f"embeddings连接错误: key={key_id[:8]} model={model} error={e}")
        raise UpstreamRetryableError(f"conn_error: key={key_id[:8]} model={model}", kind="network")
    except Exception as e:
        rt = time.time() - start_time
        scheduler.record_response(key_id, model, False, rt, 0, "error")
        raise HTTPException(status_code=500, detail={
            "message": f"内部错误: {str(e)}",
            "type": "internal_error",
            "code": "internal_error",
        })


# ========== Embeddings端点 ==========

@router.post("/v1/embeddings", tags=["公共API"])
@router.post("/api/v1/embeddings", tags=["公共API"])
async def embeddings(request: Request):
    """OpenAI兼容向量化端点（tenacity自动重试）"""
    client_info = await asyncio.to_thread(authenticate_client, request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail={
            "message": "请求体JSON解析失败",
            "type": "invalid_request_error",
            "code": "invalid_json",
        })

    model = body.get("model", "")
    if not model:
        raise HTTPException(status_code=400, detail={
            "message": "缺少model参数",
            "type": "invalid_request_error",
            "code": "missing_model",
        })

    # === v10.0: embeddings 安全校验已移除 ===

    # v10.0 防呆防傻：embeddings 熔断器检查
    cb = get_circuit_breaker()
    cb_key = f"embeddings:{model}"
    if not cb.can_request(cb_key):
        degradation = graceful_degradation_response("circuit_open")
        raise HTTPException(status_code=degradation["status_code"], detail=degradation)

    scheduler = get_scheduler()
    await scheduler._ensure_pools()

    base_url = get_setting("upstream_base_url") or "https://integrate.api.nvidia.com/v1"
    url = f"{base_url}/embeddings"

    _emb_start_ts = time.time()
    try:
        result = await _call_upstream_embeddings(scheduler, client_info["client_id"], model, url, body)
        cb.record_success(cb_key)
        # v10.0 修复：记录 embeddings 成功日志
        await _log_request(
            client_id=client_info["client_id"], key_id="", model=model,
            status_code=200, rt=time.time() - _emb_start_ts,
            prompt_tokens=0, completion_tokens=0, total_tokens=0,
            is_stream=False, error_msg="",
            request_path=str(request.url.path), http_method=request.method,
            client_ip=get_client_ip(request),
            user_agent=request.headers.get("user-agent", "")[:512],
        )
        return result
    except UpstreamRetryableError as e:
        cb.record_failure(cb_key, "embeddings_retry_exhausted")
        degradation = graceful_degradation_response("all_keys_exhausted")
        # v10.0 修复：记录 embeddings 失败日志
        await _log_request(
            client_id=client_info["client_id"], key_id="", model=model,
            status_code=degradation["status_code"], rt=time.time() - _emb_start_ts,
            prompt_tokens=0, completion_tokens=0, total_tokens=0,
            is_stream=False, error_msg=str(e)[:512],
            error_type="UpstreamRetryableError", error_detail=str(e)[:1024],
            business_code="embeddings_retry_exhausted",
            request_path=str(request.url.path), http_method=request.method,
            client_ip=get_client_ip(request),
            user_agent=request.headers.get("user-agent", "")[:512],
        )
        raise HTTPException(status_code=degradation["status_code"], detail=degradation)


# ========== Responses API 端点 (专门服务 ChatGPT Desktop / Codex CLI) ==========

@router.post("/v1/responses", tags=["公共API"])
@router.post("/api/v1/responses", tags=["公共API"])
async def responses_endpoint(request: Request):
    """
    OpenAI Responses API 兼容端点
    支持 ChatGPT Desktop (Codex) / Codex CLI 等使用 wire_api = "responses" 的客户端。
    将 Responses API 请求转换为内部 Chat Completions 请求，调用聚合模型链/单模型，
    并将输出转换回 Responses API 格式或 SSE 事件流。
    """
    # 1. 客户端认证
    client_info = await asyncio.to_thread(authenticate_client, request)
    client_id = client_info["client_id"]

    # 2. 解析请求体
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail={
            "message": "请求体JSON解析失败",
            "type": "invalid_request_error",
            "code": "invalid_json",
        })

    from app.responses_bridge import (
        responses_request_to_chat,
        chat_response_to_responses_api,
        ChatToResponsesSseAdapter,
    )

    # 3. 将 Responses API 请求转换为 Chat Completions 请求格式
    chat_body = responses_request_to_chat(body)
    model = chat_body.get("model", "")
    is_stream = bool(chat_body.get("stream", False))

    request.state.log_model = model
    request.state.log_is_stream = is_stream
    request.state.log_client_id = client_id
    request.state.upstream_attempts = []

    if not model:
        raise HTTPException(status_code=400, detail={
            "message": "缺少model参数",
            "type": "invalid_request_error",
            "code": "missing_model",
        })

    # === 虚拟聚合模型与定点路由逻辑 ===
    from app import virtual_models as _vm
    _is_virtual = _vm.is_virtual_model(model)
    _virtual_chain = _vm.get_chain(model) if _is_virtual else []
    if _is_virtual and not _virtual_chain:
        raise HTTPException(status_code=503, detail={
            "message": f"聚合模型 '{model}' 未配置任何可用的真实模型",
            "type": "service_unavailable",
            "code": "virtual_model_empty_chain",
        })

    _pinned_provider, _pinned_model = _split_provider_model_ref(model)
    _is_pinned = bool(_pinned_provider and _pinned_model)
    if _is_pinned:
        chat_body["model"] = _pinned_model

    corrected_model, was_corrected = (model, False) if (_is_virtual or _is_pinned) else validate_and_correct_model(model)
    if was_corrected:
        chat_body["model"] = corrected_model
        model = corrected_model
    elif model != corrected_model:
        suggestion = build_model_error_suggestion(model)
        raise HTTPException(status_code=400, detail={
            "message": f"模型ID '{model}' 无效。{suggestion}",
            "type": "invalid_request_error",
            "code": "invalid_model",
        })

    validate_and_sanitize(chat_body)
    chat_body = apply_model_defaults(chat_body)

    if is_stream:
        if "stream_options" not in chat_body:
            chat_body["stream_options"] = {}
        chat_body["stream_options"]["include_usage"] = True

    if not _is_virtual:
        cb = get_circuit_breaker()
        cb_key = f"model:{model}"
        if not cb.can_request(cb_key):
            degradation = graceful_degradation_response("circuit_open")
            raise HTTPException(status_code=degradation["status_code"], detail={
                "message": degradation["message"],
                "type": degradation["type"],
                "code": degradation["code"],
                "suggestion": degradation["suggestion"],
            })

    scheduler = get_scheduler()
    await scheduler._ensure_pools()
    scheduler.record_client_request(client_id, model)

    raw_base_url = get_setting("upstream_base_url") or "https://integrate.api.nvidia.com/v1"
    base_url, _ = normalize_base_url(raw_base_url)
    chat_path = get_setting("chat_path") or "/chat/completions"
    upstream_url = f"{base_url}{chat_path}"
    timeout = get_timeout_for_model(_pinned_model if _is_pinned else model)
    _request_start_ts = time.time()

    if _is_virtual:
        raw_resp = await _call_upstream_chain(
            scheduler, client_id, model, _virtual_chain,
            upstream_url, chat_body, is_stream, request,
            dispatch_start_time=_request_start_ts
        )
    else:
        raw_resp = await _call_upstream(
            scheduler, client_id, model, upstream_url, chat_body, timeout, is_stream, request,
            dispatch_start_time=_request_start_ts
        )

    # 4. 将响应转换回 Responses API 格式
    if not is_stream:
        try:
            content_bytes = raw_resp.body
            chat_data = json.loads(content_bytes.decode("utf-8") if isinstance(content_bytes, bytes) else content_bytes)
            responses_data = chat_response_to_responses_api(chat_data, model)
            return JSONResponse(content=responses_data, status_code=raw_resp.status_code)
        except Exception as e:
            logger.error(f"Responses API 非流式格式转换失败: {e}")
            return raw_resp

    # 流式模式：将 chat.completion.chunk SSE 转换为 Responses API SSE 事件流
    async def responses_event_stream():
        adapter = ChatToResponsesSseAdapter(model)
        async for sse_chunk in adapter.stream_from_chat_iterator(raw_resp.body_iterator):
            yield sse_chunk

    return StreamingResponse(
        responses_event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

