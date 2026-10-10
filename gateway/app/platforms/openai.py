"""
OpenAI 平台适配器 - v9.0 多Provider支持

支持 OpenAI、Azure OpenAI、以及所有兼容 OpenAI 格式的平台
"""
import json
import time
import logging
from typing import List, Optional, AsyncGenerator

from app.platforms.base import (
    BaseHTTPAdapter, ModelInfo, ChatResponse, StreamChunk,
    UpstreamHealth, UpstreamError, RateLimitError,
)

logger = logging.getLogger("acu.platform.openai")


class OpenAIAdapter(BaseHTTPAdapter):
    """OpenAI 兼容平台适配器"""

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def default_base_url(self) -> str:
        return "https://api.openai.com/v1"

    async def chat_completions(
        self, body: dict, api_key: str, base_url: str
    ) -> ChatResponse:
        """OpenAI 非流式聊天补全"""
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        resp = await self._request("POST", url, headers, body, timeout=180.0)

        if resp.status_code != 200:
            try:
                err_body = resp.json()
            except Exception:
                err_body = {"error": {"message": resp.text[:500]}}
            raise self._normalize_error(resp.status_code, err_body)

        data = resp.json()
        usage = data.get("usage", {})
        return ChatResponse(
            id=data.get("id", ""),
            model=data.get("model", body.get("model", "")),
            choices=data.get("choices", []),
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            created=data.get("created", int(time.time())),
            provider="openai",
        )

    async def chat_completions_stream(self, body: dict, api_key: str, base_url: str) -> AsyncGenerator[StreamChunk, None]:
        """OpenAI 流式聊天补全"""
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        resp = await self._request("POST", url, headers, body, timeout=600.0, stream=True)

        if resp.status_code != 200:
            error_bytes = await resp.aread()
            await resp.aclose()
            try:
                err_body = json.loads(error_bytes)
            except Exception:
                err_body = {"error": {"message": error_bytes.decode("utf-8", errors="replace")[:500]}}
            raise self._normalize_error(resp.status_code, err_body)

        try:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        yield StreamChunk(data="[DONE]", is_done=True)
                        return
                    try:
                        data = json.loads(data_str)
                        usage = data.get("usage")
                        yield StreamChunk(data=data_str, usage=usage)
                    except json.JSONDecodeError:
                        yield StreamChunk(data=data_str)
        except Exception as e:
            logger.error(f"OpenAI流式读取错误: {e}")
            raise
        finally:
            await resp.aclose()

    async def models(self, api_key: str, base_url: str) -> List[ModelInfo]:
        """获取OpenAI模型列表"""
        url = f"{base_url}/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = await self._request("GET", url, headers, timeout=30.0)

        if resp.status_code != 200:
            logger.error(f"获取OpenAI模型列表失败: {resp.status_code}")
            return []

        data = resp.json()
        raw = data.get("data", [])
        return [
            ModelInfo(
                id=m.get("id", ""),
                owned_by=m.get("owned_by", "openai"),
            )
            for m in raw if isinstance(m, dict)
        ]

    async def health_check(self, api_key: str, base_url: str) -> UpstreamHealth:
        """健康检查"""
        start = time.time()
        try:
            models = await self.models(api_key, base_url)
            latency = (time.time() - start) * 1000
            return UpstreamHealth(
                available=len(models) > 0,
                latency_ms=latency,
                checked_at=time.time(),
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return UpstreamHealth(
                available=False, latency_ms=latency,
                error=str(e), checked_at=time.time(),
            )

    def _normalize_error(self, status_code: int, response_body: dict) -> UpstreamError:
        """OpenAI 格式错误映射"""
        err = response_body.get("error", {}) if isinstance(response_body, dict) else {}
        if isinstance(err, str):
            msg = err
        elif isinstance(err, dict):
            msg = err.get("message", "")
            code = err.get("code", "")
            if code == "insufficient_quota":
                from app.platforms.base import InsufficientQuotaError
                return InsufficientQuotaError(msg)
        else:
            msg = str(response_body)

        if status_code == 429:
            return RateLimitError(msg)
        elif status_code in (401, 403):
            from app.platforms.base import AuthenticationError
            return AuthenticationError(msg)
        elif status_code >= 500:
            from app.platforms.base import InternalServerError
            return InternalServerError(msg)
        return UpstreamError(msg, status_code)
