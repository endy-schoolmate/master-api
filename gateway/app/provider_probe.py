"""Small, side-effect-free provider connectivity probes for the admin API.

Probes deliberately bypass the scheduler: they use exactly the supplied provider,
base URL and credential and never update key status, buckets, or circuit breakers.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Optional

import httpx


_SAFE_TEXT_LIMIT = 240
_SECRET_PATTERN = re.compile(
    r"(?i)(?:bearer\s+|api[_-]?key[=:]\s*|token[=:]\s*|password[=:]\s*|secret[=:]\s*)"
    r"[^\s,;\"']+"
)


@dataclass(frozen=True)
class ProviderProbe:
    provider: str
    api_key: str
    base_url: str = ""
    model: Optional[str] = None


def split_provider_model_ref(model_ref: str) -> tuple[str, str]:
    """Return ``(provider, model)`` for a persisted ``provider::model`` ref."""
    if "::" in model_ref:
        provider, model = model_ref.split("::", 1)
        if provider and model:
            return provider, model
    return "", model_ref


def _safe_text(value: object) -> str:
    text = str(value or "").replace("\x00", " ").replace("\r", " ").replace("\n", " ")
    text = _SECRET_PATTERN.sub("[REDACTED]", text)
    return text[:_SAFE_TEXT_LIMIT]


def _error_message(response: httpx.Response) -> str:
    """Best-effort human-readable error from a response; never empty."""
    text = ""
    try:
        payload = response.json()
    except Exception:
        text = _safe_text(response.text)
    else:
        if isinstance(payload, dict):
            error = payload.get("error", payload)
            if isinstance(error, dict):
                text = _safe_text(error.get("message") or error.get("detail") or payload)
            else:
                text = _safe_text(error)
        else:
            text = _safe_text(payload)
    if not text:
        reason = (response.reason_phrase or "").strip()
        text = f"上游返回 {response.status_code} {reason}（无响应体）".strip()
    return text


def _default_base_url(provider: str) -> str:
    """Only return an exact registered adapter default; never guess NVIDIA."""
    try:
        from app.platforms import register_all_adapters
        from app.platforms.base import PlatformAdapterRegistry

        if PlatformAdapterRegistry.get(provider) is None:
            register_all_adapters()
        adapter = PlatformAdapterRegistry.get(provider)
        return (getattr(adapter, "default_base_url", "") or "").rstrip("/") if adapter else ""
    except Exception:
        return ""


def normalize_pasted_key(raw: str) -> str:
    """清理粘贴 key 时的常见误带：'Bearer ' 前缀与所有空白字符。"""
    key = re.sub(r"^bearer\s+", "", (raw or "").strip(), flags=re.IGNORECASE)
    return re.sub(r"\s+", "", key)


def normalize_upstream_base_url(raw: Optional[str]) -> str:
    """清洗 Base URL：去除首尾空白与尾部斜杠，若误填了完整端点（如 /chat/completions 等）则自动剥离保留基础路径。"""
    u = (raw or "").strip().rstrip("/")
    if not u:
        return ""
    return re.sub(r"/(?:chat/completions|responses|messages|models)/?$", "", u, flags=re.IGNORECASE).rstrip("/")


def _headers(api_key: str) -> dict[str, str]:
    # Persisted gateway keys currently use the OpenAI-compatible Bearer contract.
    # A non-library User-Agent avoids naive WAF rules that block python-httpx.
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": _USER_AGENT,
    }


_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MasterAPI/10.0"


_SAMPLE_MODELS_CAP = 500


def _list_request(provider: str, base_url: str, api_key: str) -> tuple[str, dict[str, str], str]:
    """Build a format-aware ``GET /models`` request: (url, headers, flavor)."""
    if provider == "anthropic":
        return f"{base_url}/models", {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
        }, "openai"
    if provider == "gemini":
        return f"{base_url}/models", {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
        }, "gemini"
    return f"{base_url}/models", _headers(api_key), "openai"


def _extract_model_ids(payload: object, flavor: str) -> list[str]:
    """Pull model IDs from an upstream listing payload.

    OpenAI/Anthropic: ``{"data": [{"id": ...}]}``; Gemini: ``{"models": [{"name":
    "models/xxx", "supportedGenerationMethods": [...]}]}``.
    """
    ids: list[str] = []
    if not isinstance(payload, dict):
        return ids
    raw = payload.get("data", payload.get("models", []))
    if not isinstance(raw, list):
        return ids
    for item in raw:
        if isinstance(item, dict):
            if flavor == "gemini":
                mid = str(item.get("name") or "").strip()
                if mid.startswith("models/"):
                    mid = mid.split("/", 1)[1]
                methods = item.get("supportedGenerationMethods")
                if isinstance(methods, list) and methods and "generateContent" not in methods:
                    continue
            else:
                mid = str(item.get("id") or "").strip()
        else:
            mid = str(item or "").strip()
        if mid and mid not in ids:
            ids.append(mid)
        if len(ids) >= _SAMPLE_MODELS_CAP:
            break
    return ids


def _result(*, ok: bool, provider: str, model: Optional[str], status: Optional[int] = None,
            latency_ms: Optional[int] = None, reason: str = "", error: str = "",
            credential_only: bool = False, **extra) -> dict:
    data = {
        "ok": ok,
        "provider": provider,
        "model": model,
        "http_status": status,
        "latency_ms": latency_ms,
        "reason": reason,
        "error": _safe_text(error),
        "credential_only": credential_only,
    }
    data.update(extra)
    return data


async def probe_provider(
    probe: ProviderProbe,
    *,
    client: Optional[httpx.AsyncClient] = None,
) -> dict:
    """Probe a configured provider without mutating gateway runtime state.

    With a model this sends the smallest useful non-streaming chat request. Without
    one it performs credential-only ``GET /models`` and labels the result as such.
    """
    provider = (probe.provider or "").strip()
    ref_provider, model = split_provider_model_ref((probe.model or "").strip())
    if ref_provider:
        if provider and provider != ref_provider:
            return _result(ok=False, provider=provider, model=model, reason="provider_mismatch",
                           error="模型引用的 provider 与密钥配置不一致")
        provider = ref_provider
    base_url = (probe.base_url or _default_base_url(provider)).rstrip("/")
    # 剥离多余的服务端点后缀（容错处理：避免用户或调用方传入完整路径导致双重拼接）
    base_url = re.sub(r"/(?:chat/completions|responses|messages|models)/?$", "", base_url, flags=re.IGNORECASE).rstrip("/")
    if not provider:
        return _result(ok=False, provider="", model=model or None, reason="no_provider", error="缺少 provider")
    if not base_url:
        return _result(ok=False, provider=provider, model=model or None, reason="no_base_url",
                       error="该 provider 未配置 base_url，也没有已注册的默认地址")
    if not probe.api_key:
        return _result(ok=False, provider=provider, model=model or None, reason="no_api_key", error="缺少 API Key")
    if re.search(r"\s", probe.api_key):
        return _result(ok=False, provider=provider, model=model or None, reason="invalid_key_whitespace",
                       error="API Key 中含有空格/换行（通常为复制不完整），请在编辑供应商时重新粘贴完整 Key")

    owns_client = client is None
    if owns_client and provider in ("openai-codex", "openai-responses"):
        # Responses API 上游：配置了 oauth_proxy 时走专用代理连接池（共享，不关闭）
        try:
            from app.openai_oauth import get_proxied_client
            client = await get_proxied_client()
            if client is not None:
                owns_client = False
        except Exception:
            client = None
    if owns_client:
        client = httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=10.0))
    assert client is not None
    started = time.perf_counter()
    try:
        if model:
            if provider == "openai-codex" or provider == "openai-responses":
                # Responses API 上游（ChatGPT 账号登录 / openai-responses 格式）
                response = await client.post(
                    f"{base_url}/responses",
                    headers=_headers(probe.api_key),
                    json={
                        "model": model,
                        "input": "Hi",
                        "max_output_tokens": 16,
                        "stream": False,
                        "store": False,
                    },
                    timeout=httpx.Timeout(20.0, connect=10.0),
                )
                latency_ms = round((time.perf_counter() - started) * 1000)
                if response.status_code != 200:
                    return _result(ok=False, provider=provider, model=model, status=response.status_code,
                                   latency_ms=latency_ms, reason="http_error", error=_error_message(response))
                try:
                    payload = response.json()
                except Exception:
                    payload = {}
                output = payload.get("output") if isinstance(payload, dict) else None
                if not isinstance(output, list):
                    return _result(ok=False, provider=provider, model=model, status=200, latency_ms=latency_ms,
                                   reason="invalid_response", error="响应未包含有效 output")
                return _result(ok=True, provider=provider, model=model, status=200, latency_ms=latency_ms)

            response = await client.post(
                f"{base_url}/chat/completions",
                headers=_headers(probe.api_key),
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 1,
                    "stream": False,
                },
            )
            latency_ms = round((time.perf_counter() - started) * 1000)
            if response.status_code != 200:
                return _result(ok=False, provider=provider, model=model, status=response.status_code,
                               latency_ms=latency_ms, reason="http_error", error=_error_message(response))
            try:
                payload = response.json()
            except Exception:
                payload = {}
            choices = payload.get("choices") if isinstance(payload, dict) else None
            if not choices:
                return _result(ok=False, provider=provider, model=model, status=200, latency_ms=latency_ms,
                               reason="invalid_response", error="响应未包含有效 choices")
            return _result(ok=True, provider=provider, model=model, status=200, latency_ms=latency_ms)

        url, list_headers, flavor = _list_request(provider, base_url, probe.api_key)
        response = await client.get(url, headers=list_headers, timeout=httpx.Timeout(20.0, connect=10.0))
        latency_ms = round((time.perf_counter() - started) * 1000)
        if response.status_code != 200:
            return _result(ok=False, provider=provider, model=None, status=response.status_code,
                           latency_ms=latency_ms, reason="http_error", error=_error_message(response),
                           credential_only=True, requested_url=url)
        try:
            payload = response.json()
            sample_models = _extract_model_ids(payload, flavor)
            model_count = len(sample_models)
        except Exception:
            sample_models = []
            model_count = 0
        return _result(ok=True, provider=provider, model=None, status=200, latency_ms=latency_ms,
                       credential_only=True, model_count=model_count, sample_models=sample_models,
                       requested_url=url)
    except httpx.TimeoutException:
        return _result(ok=False, provider=provider, model=model or None, reason="timeout", error="探测超时")
    except httpx.ConnectError as exc:
        return _result(ok=False, provider=provider, model=model or None, reason="connect_error", error=exc)
    except httpx.HTTPError as exc:
        return _result(ok=False, provider=provider, model=model or None, reason="network_error", error=exc)
    finally:
        if owns_client:
            await client.aclose()
