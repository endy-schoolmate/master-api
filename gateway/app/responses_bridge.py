"""Chat Completions ↔ OpenAI Responses API 协议桥接。

供"OpenAI (ChatGPT 账号登录)"上游使用：用户 access token 只能调
api.openai.com 的 Responses API（/v1/responses，订阅计费），而网关对客户端
与上游的统一契约是 chat/completions。本模块做双向翻译：

- chat_to_responses(): 请求翻译（messages/tools/max_tokens → input/tools/max_output_tokens）
- responses_to_chat(): 非流式响应翻译（output[] → choices[0].message）
- translate_sse_lines(): 流式 SSE 逐行翻译（response.* 事件 → chat.completion.chunk 行）

翻译策略：store:false、max_output_tokens 下限 16、system → developer、订阅模型不吃
temperature/top_p（一律丢弃）、reasoning_effort → reasoning.effort。
"""

from __future__ import annotations

import json
import time
from typing import AsyncIterator, Iterable, Optional

OAUTH_PROVIDER_ID = "openai-codex"

# 走 Responses API 转发的 provider（ChatGPT 账号登录 + 手动选 openai-responses 格式的上游）
RESPONSES_PROVIDER_IDS = {OAUTH_PROVIDER_ID, "openai-responses"}

_MIN_OUTPUT_TOKENS = 16


def _input_text(text: str) -> dict:
    return {"type": "input_text", "text": text}


def _output_text(text: str) -> dict:
    return {"type": "output_text", "text": text, "annotations": []}


def _content_parts(content, text_type: str) -> list[dict]:
    """chat content（字符串或多模态数组）→ responses content parts。"""
    if isinstance(content, str):
        return [{"type": text_type, "text": content}] if content else []
    parts: list[dict] = []
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                if part:
                    parts.append({"type": text_type, "text": str(part)})
                continue
            ptype = part.get("type")
            if ptype in ("text", "input_text", "output_text"):
                parts.append({"type": text_type, "text": part.get("text", "")})
            elif ptype == "image_url":
                image_url = part.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                if url:
                    parts.append({"type": "input_image", "image_url": url, "detail": "auto"})
            # 其他媒体类型（audio 等）网关不做桥接，静默丢弃
    return parts


def _convert_tools(tools) -> Optional[list[dict]]:
    """chat tools（包一层 function{}）→ responses tools（平铺）。"""
    out: list[dict] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") in (None, "function"):
            fn = tool.get("function") or {}
            name = fn.get("name") or tool.get("name")
            if not name:
                continue
            item = {
                "type": "function",
                "name": name,
                "strict": False,
            }
            if fn.get("description") or tool.get("description"):
                item["description"] = fn.get("description") or tool.get("description")
            params = fn.get("parameters")
            if isinstance(params, dict) and params:
                item["parameters"] = params
            out.append(item)
    return out or None


def _convert_tool_choice(tool_choice) -> Optional[dict]:
    if not isinstance(tool_choice, dict):
        return None  # "auto"/"none"/"required" 原样直传
    fn = tool_choice.get("function") or {}
    name = fn.get("name") or tool_choice.get("name")
    return {"type": "function", "name": name} if name else None


def chat_to_responses(body: dict, model: str) -> dict:
    """chat/completions 请求体 → /v1/responses 请求体。"""
    messages = body.get("messages") or []
    input_items: list[dict] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role in ("system", "developer"):
            text = msg.get("content")
            if isinstance(text, list):
                text = "\n".join(p.get("text", "") for p in text if isinstance(p, dict))
            if text:
                input_items.append({"role": role if role == "developer" else "developer", "content": str(text)})
        elif role == "user":
            parts = _content_parts(msg.get("content"), "input_text")
            if parts:
                input_items.append({"role": "user", "content": parts})
        elif role == "assistant":
            content_parts = _content_parts(msg.get("content"), "output_text")
            if content_parts:
                input_items.append({"role": "assistant", "content": content_parts})
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                call_id = tc.get("id") or fn.get("name") or ""
                input_items.append({
                    "type": "function_call",
                    "call_id": call_id,
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments") or "{}",
                })
        elif role == "tool":
            output = msg.get("content")
            if isinstance(output, list):
                output = "\n".join(p.get("text", "") for p in output if isinstance(p, dict))
            input_items.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id") or "",
                "output": str(output if output is not None else ""),
            })

    payload: dict = {
        "model": model,
        "input": input_items,
        "stream": bool(body.get("stream", False)),
        "store": False,
    }
    # 订阅模型不吃采样参数：全部丢弃，仅保留安全可映射的项
    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
    if isinstance(max_tokens, int) and max_tokens > 0:
        payload["max_output_tokens"] = max(max_tokens, _MIN_OUTPUT_TOKENS)
    tools = _convert_tools(body.get("tools"))
    if tools:
        payload["tools"] = tools
    tool_choice = _convert_tool_choice(body.get("tool_choice"))
    if tool_choice:
        payload["tool_choice"] = tool_choice
    elif body.get("tool_choice") in ("auto", "none", "required"):
        payload["tool_choice"] = body["tool_choice"]
    if isinstance(body.get("parallel_tool_calls"), bool):
        payload["parallel_tool_calls"] = body["parallel_tool_calls"]
    effort = body.get("reasoning_effort")
    if effort in ("minimal", "low", "medium", "high"):
        payload["reasoning"] = {"effort": effort}
    rf = body.get("response_format")
    if isinstance(rf, dict) and rf.get("type") == "json_object":
        payload["text"] = {"format": {"type": "json_object"}}
    elif isinstance(rf, dict) and rf.get("type") == "json_schema":
        schema = rf.get("json_schema") or {}
        payload["text"] = {"format": {"type": "json_schema", "name": schema.get("name") or "response",
                                      "schema": schema.get("schema") or {}, "strict": False}}
    return payload


def _usage_to_chat(usage: dict) -> dict:
    usage = usage or {}
    prompt = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    completion = usage.get("output_tokens") or usage.get("completion_tokens") or 0
    out_details = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
    in_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}

    cached = (
        in_details.get("cached_tokens") or
        in_details.get("cache_read_tokens") or
        usage.get("cache_read_input_tokens") or
        usage.get("prompt_cache_hit_tokens") or
        usage.get("cached_tokens") or
        0
    )
    cache_write = (
        in_details.get("cache_write_tokens") or
        in_details.get("cache_creation_tokens") or
        usage.get("cache_creation_input_tokens") or
        0
    )

    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": usage.get("total_tokens", prompt + completion) or (prompt + completion),
        "prompt_tokens_details": {
            "cached_tokens": cached,
            "cache_write_tokens": cache_write,
        },
        "completion_tokens_details": {"reasoning_tokens": out_details.get("reasoning_tokens", 0) or 0},
    }


def responses_to_chat(data: dict, model: str) -> dict:
    """非流式 /v1/responses 响应 → chat.completion 响应。"""
    content_parts: list[str] = []
    tool_calls: list[dict] = []
    finish_reason = "stop"
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    content_parts.append(part.get("text", ""))
                elif isinstance(part, dict) and part.get("type") == "refusal":
                    content_parts.append(part.get("refusal", ""))
        elif item_type == "function_call":
            tool_calls.append({
                "id": item.get("call_id") or item.get("id") or f"call_{len(tool_calls)}",
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments") or "{}",
                },
            })
        # reasoning 等条目不进 chat 消息
    if tool_calls:
        finish_reason = "tool_calls"
    if data.get("status") == "incomplete":
        finish_reason = "length"

    message: dict = {"role": "assistant", "content": "".join(content_parts) or None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage = _usage_to_chat(data.get("usage") or {})
    created = data.get("created_at") or int(time.time())
    resp_id = data.get("id") or ""
    return {
        "id": f"chatcmpl-{resp_id}" if resp_id else f"chatcmpl-{int(created)}",
        "object": "chat.completion",
        "created": created,
        "model": data.get("model") or model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage,
    }


class _SseChunkBuilder:
    """把 Responses 流事件累积成 chat.completion.chunk 行（单 choice，index 0）。"""

    def __init__(self, model: str):
        self.model = model
        self.resp_id = ""
        self.created = int(time.time())
        self.tool_slot: dict[str, int] = {}       # item_id → chat tool_calls index
        self.tool_args_len: dict[str, int] = {}   # item_id → 已按 delta 发出的参数长度
        self.tool_count = 0
        self.has_tool_calls = False
        self.role_sent = False
        self.usage_sent = False

    def _chunk(self, delta: dict, finish_reason: Optional[str] = None, usage: Optional[dict] = None) -> str:
        return "data: " + json.dumps({
            "id": self.resp_id or f"chatcmpl-{self.created}",
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            **({"usage": usage} if usage else {}),
        }, ensure_ascii=False)

    def handle_event(self, ev: dict) -> list[str]:
        etype = ev.get("type", "")
        out: list[str] = []

        if etype == "response.created":
            resp = ev.get("response") or {}
            self.resp_id = resp.get("id") or ""
            self.model = resp.get("model") or self.model
            if resp.get("created_at"):
                self.created = resp["created_at"]
            if not self.role_sent:
                self.role_sent = True
                out.append(self._chunk({"role": "assistant", "content": ""}))
            return out

        if etype in ("response.in_progress", "response.queued", "ping"):
            return out

        if etype == "response.output_text.delta":
            text = ev.get("delta") or ""
            if text:
                out.append(self._chunk({"content": text}))
            return out

        if etype == "response.output_item.added":
            item = ev.get("item") or {}
            if item.get("type") == "function_call":
                item_id = item.get("id") or f"item_{ev.get('output_index')}"
                idx = self.tool_count
                self.tool_slot[item_id] = idx
                self.tool_count += 1
                self.has_tool_calls = True
                out.append(self._chunk({"tool_calls": [{
                    "index": idx,
                    "id": item.get("call_id") or item_id,
                    "type": "function",
                    "function": {"name": item.get("name", ""), "arguments": item.get("arguments") or ""},
                }]}))
            return out

        if etype == "response.function_call_arguments.delta":
            item_id = ev.get("item_id") or ""
            idx = self.tool_slot.get(item_id)
            delta = ev.get("delta") or ""
            if idx is not None and delta:
                self.tool_args_len[item_id] = self.tool_args_len.get(item_id, 0) + len(delta)
                out.append(self._chunk({"tool_calls": [{"index": idx, "function": {"arguments": delta}}]}))
            return out

        if etype == "response.output_item.done":
            item = ev.get("item") or {}
            if item.get("type") == "function_call":
                item_id = item.get("id") or ""
                idx = self.tool_slot.get(item_id)
                if idx is None:
                    # 全程没有 added/arguments.delta，参数只出现在 done 里：整块补发
                    idx = self.tool_slot[item_id] = self.tool_count
                    self.tool_count += 1
                    self.has_tool_calls = True
                    out.append(self._chunk({"tool_calls": [{
                        "index": idx,
                        "id": item.get("call_id") or item_id,
                        "type": "function",
                        "function": {"name": item.get("name", ""), "arguments": item.get("arguments") or "{}"},
                    }]}))
                else:
                    # 已按增量发过参数：done 里的完整 arguments 不再重发，避免客户端翻倍
                    args = item.get("arguments")
                    if args and self.tool_args_len.get(item_id, 0) == 0:
                        out.append(self._chunk({"tool_calls": [{"index": idx, "function": {"arguments": args}}]}))
            return out

        if etype == "response.failed":
            resp = ev.get("response") or {}
            err = resp.get("error") or {}
            out.append("data: " + json.dumps({
                "error": {"message": err.get("message") or "上游 Responses 请求失败",
                          "type": "upstream_error", "code": err.get("code") or "response_failed"},
            }, ensure_ascii=False))
            return out

        if etype == "error":
            err = ev.get("error") or {}
            out.append("data: " + json.dumps({
                "error": {"message": err.get("message") or "上游流式错误",
                          "type": err.get("type") or "upstream_error", "code": err.get("code") or ""},
            }, ensure_ascii=False))
            return out

        if etype in ("response.completed", "response.done", "response.incomplete"):
            resp = ev.get("response") or {}
            self.resp_id = resp.get("id") or self.resp_id
            usage = _usage_to_chat(resp.get("usage") or {})
            finish = "tool_calls" if self.has_tool_calls else (
                "length" if etype == "response.incomplete" else "stop")
            if not self.usage_sent:
                self.usage_sent = True
                out.append(self._chunk({}, finish_reason=finish, usage=usage))
            return out

        # response.output_text.done / reasoning summary 事件等：不产生输出
        return out


async def translate_sse_lines(lines: AsyncIterator[str], model: str) -> AsyncIterator[str]:
    """把 Responses API 的 SSE 行流逐行翻成 chat.completion.chunk 的 `data: {...}` 行。

    输入是 aiter_lines() 的输出（已去换行）：`event:` 行丢弃，`data:` 行解析，
    空行忽略。输出行不带尾随空行——调用方（stream_generator）会统一补 "\\n\\n"。
    """
    builder = _SseChunkBuilder(model)
    async for line in lines:
        line = line.strip()
        if not line or line.startswith(":") or line.startswith("event:"):
            continue
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            ev = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        for out_line in builder.handle_event(ev):
            yield out_line


def stream_request_is_translatable(body: dict) -> bool:
    """防御：仅当请求体是标准 chat 结构时才翻译。"""
    return isinstance(body, dict) and isinstance(body.get("messages"), list)
