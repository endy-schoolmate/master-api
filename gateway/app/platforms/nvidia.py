"""
NVIDIA NIM 平台适配器 - v10.0 Agent 框架完全兼容版

- reasoning_effort / reasoning_budget（Agent 推理预算）
- Embedding 支持（NVIDIA NIM 原生支持）
- 完整 OpenAI Chat Completions 兼容（工具调用、流式、视觉）
- OpenAI 标准错误格式映射
"""
import json
import time
import logging
from typing import List, Optional, AsyncGenerator, Dict, Any

from app.platforms.base import (
    BaseHTTPAdapter, ModelInfo, ChatResponse, StreamChunk,
    UpstreamHealth, UpstreamError,
)

logger = logging.getLogger("acu.platform.nvidia")


class NvidiaAdapter(BaseHTTPAdapter):
    """NVIDIA NIM 平台适配器（v10.0: 完全兼容 OpenAI Chat Completions + Agent 框架）"""

    @property
    def provider_name(self) -> str:
        return "nvidia"

    @property
    def default_base_url(self) -> str:
        return "https://integrate.api.nvidia.com/v1"

    # ---- NVIDIA 特有参数白名单 ----
    NVIDIA_EXTRA_PARAMS = {
        "reasoning_effort",     # none | medium | high
        "reasoning_budget",     # max tokens for reasoning
        "chat_template_kwargs", # 高级模板控制
    }

    def _build_request_body(self, body: dict) -> dict:
        """构建请求体：提取 NVIDIA 特有参数到顶层"""
        nv_params = {}
        for key in self.NVIDIA_EXTRA_PARAMS:
            if key in body:
                nv_params[key] = body.pop(key)
        request_body = {**body, **nv_params}
        return request_body

    async def chat_completions(
        self, body: dict, api_key: str, base_url: str
    ) -> ChatResponse:
        """NVIDIA 非流式聊天补全（v10.0: 支持 reasoning_effort 等 Agent 参数）"""
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        request_body = self._build_request_body(dict(body))
        resp = await self._request("POST", url, headers, request_body, timeout=180.0)

        if resp.status_code != 200:
            try:
                err_body = resp.json()
            except Exception:
                err_body = {"error": {"message": resp.text[:500]}}
            raise self.normalize_error(resp.status_code, err_body)

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
            provider="nvidia",
        )

    async def chat_completions_stream(self, body: dict, api_key: str, base_url: str) -> AsyncGenerator[StreamChunk, None]:
        """NVIDIA 流式聊天补全（v10.0: 支持 streaming + reasoning 事件流）"""
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        request_body = self._build_request_body(dict(body))
        request_body["stream"] = True
        if "stream_options" not in request_body:
            request_body["stream_options"] = {"include_usage": True}

        resp = await self._request("POST", url, headers, request_body, timeout=600.0, stream=True)

        if resp.status_code != 200:
            error_bytes = await resp.aread()
            await resp.aclose()
            try:
                err_body = json.loads(error_bytes)
            except Exception:
                err_body = {"error": {"message": error_bytes.decode("utf-8", errors="replace")[:500]}}
            raise self.normalize_error(resp.status_code, err_body)

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
            logger.error(f"NVIDIA 流式读取错误: {e}")
            raise
        finally:
            await resp.aclose()

    async def embeddings(
        self, body: dict, api_key: str, base_url: str
    ) -> Dict[str, Any]:
        """NVIDIA Embedding API - 兼容 OpenAI /v1/embeddings"""
        url = f"{base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        resp = await self._request("POST", url, headers, body, timeout=60.0)

        if resp.status_code != 200:
            try:
                err_body = resp.json()
            except Exception:
                err_body = {"error": {"message": resp.text[:500]}}
            raise self.normalize_error(resp.status_code, err_body)

        return resp.json()

    async def models(self, api_key: str, base_url: str) -> List[ModelInfo]:
        """获取NVIDIA模型列表"""
        url = f"{base_url}/models"
        headers = {"Authorization": f"Bearer {api_key}"}

        resp = await self._request("GET", url, headers, timeout=30.0)

        if resp.status_code != 200:
            logger.error(f"获取NVIDIA模型列表失败: {resp.status_code}")
            return []

        data = resp.json()
        raw_models = data.get("data", [])
        if isinstance(raw_models, list):
            return [
                ModelInfo(
                    id=m.get("id", ""),
                    owned_by=m.get("owned_by", "nvidia"),
                )
                for m in raw_models
            ]
        return []

    async def health_check(self, api_key: str, base_url: str) -> UpstreamHealth:
        """健康检查 - 调用轻量模型列表接口"""
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
                available=False,
                latency_ms=latency,
                error=str(e),
                checked_at=time.time(),
            )

    def normalize_error(self, status_code: int, response_body: dict) -> UpstreamError:
        """NVIDIA NIM 错误映射（兼容 OpenAI 错误格式）"""
        err = response_body.get("error", response_body) if isinstance(response_body, dict) else {}
        msg = err.get("message", "") if isinstance(err, dict) else str(response_body)
        code = err.get("code", "") if isinstance(err, dict) else ""

        if status_code == 429:
            return UpstreamError(msg, 429, code or "rate_limit_exceeded")
        elif status_code in (401, 403):
            return UpstreamError(msg, 403, "authentication_error")
        elif status_code in (400, 422):
            return UpstreamError(msg, 400, code or "bad_request")
        elif status_code == 202:
            return UpstreamError(msg, 202, "pending")
        elif status_code >= 500:
            return UpstreamError(msg, 500, code or "upstream_error")
        return UpstreamError(msg, status_code, code)
