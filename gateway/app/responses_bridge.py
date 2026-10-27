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
import uuid
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
        payload["max_output_tokens"] = max(max_tokens, 65536)
    else:
        payload["max_output_tokens"] = 65536
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


# ========== 下游客户端（ChatGPT Desktop / Codex CLI）反向桥接 ==========

def responses_request_to_chat(body: dict) -> dict:
    """把客户端发来的 /v1/responses 请求体翻译为 /v1/chat/completions 请求体。"""
    messages = []
    # 如果客户端直接传了 messages（防御性支持兼容混合请求）
    if isinstance(body.get("messages"), list):
        messages = list(body["messages"])
    else:
        raw_input = body.get("input") or []
        if isinstance(raw_input, str):
            messages.append({"role": "user", "content": raw_input})
        elif isinstance(raw_input, list):
            for item in raw_input:
                if isinstance(item, str):
                    messages.append({"role": "user", "content": item})
                elif isinstance(item, dict):
                    role = item.get("role")
                    itype = item.get("type")
                    if itype in ("function_call", "custom_tool_call"):
                        call_id = item.get("call_id") or item.get("id") or ""
                        args = item.get("arguments")
                        if not args and "input" in item:
                            inp = item["input"]
                            if isinstance(inp, dict):
                                args = json.dumps(inp, ensure_ascii=False)
                            else:
                                args = json.dumps({"input": inp}, ensure_ascii=False)
                        messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": item.get("name", ""),
                                    "arguments": args or "{}",
                                }
                            }]
                        })
                    elif itype in ("function_call_output", "custom_tool_call_output"):
                        call_id = item.get("call_id") or item.get("id") or ""
                        out_val = item.get("output", "")
                        if isinstance(out_val, (dict, list)):
                            out_val = json.dumps(out_val, ensure_ascii=False)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": str(out_val),
                        })
                    elif role:
                        chat_role = "system" if role in ("system", "developer") else role
                        content = item.get("content")
                        if isinstance(content, str):
                            messages.append({"role": chat_role, "content": content})
                        elif isinstance(content, list):
                            text_parts = []
                            for p in content:
                                if isinstance(p, str):
                                    text_parts.append(p)
                                elif isinstance(p, dict):
                                    ptype = p.get("type", "")
                                    if ptype in ("text", "input_text", "output_text"):
                                        text_parts.append(p.get("text", ""))
                                    elif ptype == "image_url":
                                        text_parts.append(p.get("text", ""))
                            messages.append({"role": chat_role, "content": "".join(text_parts)})

    chat_body: dict = {
        "model": body.get("model", ""),
        "messages": messages,
        "stream": bool(body.get("stream", False)),
    }

    # 工具转换
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        chat_tools = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            t_type = t.get("type")
            if t_type in ("function", "custom"):
                custom_data = t.get("custom") if isinstance(t.get("custom"), dict) else {}
                name = custom_data.get("name") or t.get("name") or ""
                desc = custom_data.get("description") or t.get("description", "")
                params = custom_data.get("parameters") or t.get("parameters")
                if "function" in t and isinstance(t["function"], dict):
                    fn = dict(t["function"])
                    if fn.get("name") == "apply_patch" and (not fn.get("parameters") or not fn["parameters"].get("properties")):
                        fn["parameters"] = {
                            "type": "object",
                            "properties": {
                                "input": {
                                    "type": "string",
                                    "description": "The complete diff patch text starting with *** Begin Patch and ending with *** End Patch"
                                }
                            },
                            "required": ["input"]
                        }
                    chat_tools.append({"type": "function", "function": fn})
                else:
                    if name == "apply_patch" and (not params or not isinstance(params, dict) or not params.get("properties")):
                        params = {
                            "type": "object",
                            "properties": {
                                "input": {
                                    "type": "string",
                                    "description": "The complete diff patch text starting with *** Begin Patch and ending with *** End Patch"
                                }
                            },
                            "required": ["input"]
                        }
                    elif not params or not isinstance(params, dict):
                        params = {"type": "object", "properties": {}}
                    chat_tools.append({
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": desc,
                            "parameters": params,
                        }
                    })
            else:
                chat_tools.append(t)
        if chat_tools:
            chat_body["tools"] = chat_tools

    tool_choice = body.get("tool_choice")
    if tool_choice:
        chat_body["tool_choice"] = tool_choice

    # 推理参数映射：ultra/high/medium/low -> reasoning_effort
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if effort:
            effort_map = {"ultra": "high", "max": "high", "xhigh": "high"}
            chat_body["reasoning_effort"] = effort_map.get(effort, effort)
    elif body.get("reasoning_effort"):
        effort = body["reasoning_effort"]
        effort_map = {"ultra": "high", "max": "high", "xhigh": "high"}
        chat_body["reasoning_effort"] = effort_map.get(effort, effort)

    # 防御性：清除顶层 thinking 参数，防止部分上游报 400 错误
    if "thinking" in body:
        if "reasoning_effort" not in chat_body:
            chat_body["reasoning_effort"] = "high"

    # max_output_tokens -> max_tokens：保底至少 65536，防止输出截断
    max_tokens = body.get("max_output_tokens") or body.get("max_tokens")
    if isinstance(max_tokens, int) and max_tokens > 0:
        chat_body["max_tokens"] = max(max_tokens, 65536)
    else:
        chat_body["max_tokens"] = 65536

    # 注入 Agent ReAct 协议：防止模型在有 tools 时仅输出行动叙述而漏发工具调用
    try:
        try:
            from app.tool_call_parser import inject_agent_react_instructions
        except ImportError:
            from gateway.app.tool_call_parser import inject_agent_react_instructions
        if inject_agent_react_instructions:
            chat_body = inject_agent_react_instructions(chat_body)
    except Exception:
        pass

    return chat_body


def chat_response_to_responses_api(data: dict, model: str) -> dict:
    """非流式 chat.completion 响应 → /v1/responses 响应。"""
    resp_id = data.get("id") or f"resp_{uuid.uuid4().hex[:24]}"
    if resp_id.startswith("chatcmpl-"):
        resp_id = f"resp_{resp_id[9:]}"
    created = data.get("created") or int(time.time())

    output_items = []
    choices = data.get("choices") or []
    if choices and isinstance(choices[0], dict):
        c0 = choices[0]
        msg = c0.get("message") or {}
        content = msg.get("content")
        tool_calls = msg.get("tool_calls") or []
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")

        if not reasoning and content and isinstance(content, str):
            from app.think_stripper import strip_think_tags
            content, reasoning = strip_think_tags(content)

        if not tool_calls and content and isinstance(content, str):
            try:
                from app.tool_call_parser import parse_raw_tool_calls
            except ImportError:
                from gateway.app.tool_call_parser import parse_raw_tool_calls
            clean_c, extracted_tcs = parse_raw_tool_calls(content)
            if extracted_tcs:
                content = clean_c
                tool_calls.extend(extracted_tcs)

        if reasoning:
            output_items.append({
                "id": f"rs_{uuid.uuid4().hex[:24]}",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": str(reasoning)}]
            })

        if not (content and str(content).strip()) and not tool_calls and reasoning:
            content = str(reasoning)

        if content and (isinstance(content, str) and content.strip() or not isinstance(content, str)):
            output_items.append({
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": str(content)}]
            })

        for i, tc in enumerate(tool_calls):
            if isinstance(tc, dict):
                fn = tc.get("function") or {}
                output_items.append({
                    "id": f"item_{uuid.uuid4().hex[:24]}",
                    "type": "function_call",
                    "call_id": tc.get("id") or f"call_{i}",
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments") or "{}",
                })

    usage = data.get("usage") or {}
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    responses_usage = {
        "input_tokens": pt,
        "output_tokens": ct,
        "total_tokens": usage.get("total_tokens", pt + ct),
    }

    return {
        "id": resp_id,
        "object": "response",
        "created_at": created,
        "status": "completed",
        "model": data.get("model") or model,
        "output": output_items,
        "usage": responses_usage,
    }


class ChatToResponsesSseAdapter:
    """把 chat.completion.chunk 的 SSE 行流转换为 Responses API 的 SSE 事件流。"""

    def __init__(self, model: str):
        self.model = model
        self.resp_id = f"resp_{uuid.uuid4().hex[:24]}"
        self.msg_item_id = f"{self.resp_id}_msg"
        self.rs_item_id = f"rs_{self.resp_id}"
        self.created = int(time.time())
        self.started = False
        self.reasoning_started = False
        self.reasoning_done = False
        self.text_content = ""
        self.reasoning_content = ""
        self.tool_calls: dict[int, dict] = {}
        self.completed = False
        self.last_usage = None
        from app.think_stripper import StreamThinkStripper
        self._stripper = StreamThinkStripper()

    def _flush_reasoning_if_needed(self) -> list[str]:
        """关闭当前推理块并生成对应的 Responses API SSE 事件。"""
        events = []
        if self.reasoning_content and not self.reasoning_started:
            self.reasoning_started = True
            events.append(("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"id": self.rs_item_id, "type": "reasoning", "summary": []}
            }))
            events.append(("response.reasoning_summary_part.added", {
                "type": "response.reasoning_summary_part.added",
                "item_id": self.rs_item_id,
                "output_index": 0,
                "summary_index": 0,
                "part": {"type": "summary_text", "text": ""}
            }))
            events.append(("response.reasoning_summary_text.delta", {
                "type": "response.reasoning_summary_text.delta",
                "item_id": self.rs_item_id,
                "output_index": 0,
                "summary_index": 0,
                "delta": self.reasoning_content
            }))
        if self.reasoning_started and not self.reasoning_done:
            self.reasoning_done = True
            events.append(("response.reasoning_summary_text.done", {
                "type": "response.reasoning_summary_text.done",
                "item_id": self.rs_item_id,
                "output_index": 0,
                "summary_index": 0,
                "text": self.reasoning_content
            }))
            events.append(("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": self.rs_item_id,
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": self.reasoning_content}]
                }
            }))
        return [f"event: {ename}\ndata: {json.dumps(edata, ensure_ascii=False)}\n\n" for ename, edata in events]

    def _ensure_visible_content_if_empty(self) -> list[str]:
        """如果模型没有产生工具调用，且正文为空/纯空白，但有思维链内容：
        自动将思维链内容作为文本输出，防止客户端因折叠思维链而展现空气泡 '...'。"""
        events = []
        if not self.text_content.strip() and not self.tool_calls and self.reasoning_content.strip():
            fallback_text = self.reasoning_content.strip()
            self.text_content = fallback_text
            msg_idx = 1 if self.reasoning_started else 0
            c_ev = {
                "type": "response.output_text.delta",
                "item_id": self.msg_item_id,
                "output_index": msg_idx,
                "content_index": 0,
                "delta": fallback_text
            }
            events.append(f"event: response.output_text.delta\ndata: {json.dumps(c_ev, ensure_ascii=False)}\n\n")
        return events

    async def stream_from_chat_iterator(self, body_iterator: AsyncIterator[str | bytes]) -> AsyncIterator[str]:
        # 1. 立即发出 response.created 事件，告知客户端流已启动，重置客户端网络重连计时器
        if not self.started:
            self.started = True
            created_ev = {
                "type": "response.created",
                "response": {
                    "id": self.resp_id,
                    "object": "response",
                    "status": "in_progress",
                    "model": self.model,
                    "created_at": self.created,
                    "output": []
                }
            }
            yield f"event: response.created\ndata: {json.dumps(created_ev, ensure_ascii=False)}\n\n"

        async for raw in body_iterator:
            chunk_str = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            lines = chunk_str.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(":"):
                    yield f"{line}\n\n"
                    continue
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload:
                    continue
                if payload == "[DONE]":
                    vis_tail, rsn_tail = self._stripper.finish()
                    if rsn_tail:
                        self.reasoning_content += rsn_tail
                    if vis_tail:
                        self.text_content += vis_tail
                    for rev in self._flush_reasoning_if_needed():
                        yield rev
                    for cev in self._ensure_visible_content_if_empty():
                        yield cev
                    if not self.completed:
                        self.completed = True
                        comp_ev = self._build_completed_event()
                        yield f"event: response.completed\ndata: {json.dumps(comp_ev, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                if not isinstance(data, dict):
                    continue

                if "error" in data:
                    err_ev = {
                        "type": "response.failed",
                        "response": {
                            "id": self.resp_id,
                            "error": data["error"]
                        }
                    }
                    yield f"event: response.failed\ndata: {json.dumps(err_ev, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                if "usage" in data and isinstance(data["usage"], dict):
                    self.last_usage = data["usage"]

                choices = data.get("choices")
                if not choices or not isinstance(choices, list) or not isinstance(choices[0], dict):
                    continue

                c0 = choices[0]
                delta = c0.get("delta") or {}
                finish_reason = c0.get("finish_reason")

                raw_reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                raw_content = delta.get("content") or ""
                has_tool_calls = bool(delta.get("tool_calls"))

                # 通过 StreamThinkStripper 双重拦截剥离 <thinking> 标签
                if raw_content:
                    vis, rsn = self._stripper.feed(raw_content)
                    raw_content = vis
                    if rsn:
                        raw_reasoning = (raw_reasoning + rsn) if raw_reasoning else rsn

                if has_tool_calls or finish_reason:
                    vis_tail, rsn_tail = self._stripper.finish()
                    if vis_tail:
                        raw_content = (raw_content + vis_tail) if raw_content else vis_tail
                    if rsn_tail:
                        raw_reasoning = (raw_reasoning + rsn_tail) if raw_reasoning else rsn_tail

                # 处理思维链增量 (reasoning / reasoning_content)
                if raw_reasoning:
                    if not self.reasoning_started:
                        self.reasoning_started = True
                        add_ev = {
                            "type": "response.output_item.added",
                            "output_index": 0,
                            "item": {
                                "id": self.rs_item_id,
                                "type": "reasoning",
                                "summary": []
                            }
                        }
                        yield f"event: response.output_item.added\ndata: {json.dumps(add_ev, ensure_ascii=False)}\n\n"
                        part_ev = {
                            "type": "response.reasoning_summary_part.added",
                            "item_id": self.rs_item_id,
                            "output_index": 0,
                            "summary_index": 0,
                            "part": {"type": "summary_text", "text": ""}
                        }
                        yield f"event: response.reasoning_summary_part.added\ndata: {json.dumps(part_ev, ensure_ascii=False)}\n\n"

                    self.reasoning_content += raw_reasoning
                    r_ev = {
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": self.rs_item_id,
                        "output_index": 0,
                        "summary_index": 0,
                        "delta": raw_reasoning
                    }
                    yield f"event: response.reasoning_summary_text.delta\ndata: {json.dumps(r_ev, ensure_ascii=False)}\n\n"

                # 当开始输出文本或工具调用时，闭合之前的思维链事件
                if (raw_content or has_tool_calls) and self.reasoning_started and not self.reasoning_done:
                    for rev in self._flush_reasoning_if_needed():
                        yield rev

                # 处理文本增量
                if raw_content:
                    msg_idx = 1 if self.reasoning_started else 0
                    self.text_content += raw_content
                    c_ev = {
                        "type": "response.output_text.delta",
                        "item_id": self.msg_item_id,
                        "output_index": msg_idx,
                        "content_index": 0,
                        "delta": raw_content
                    }
                    yield f"event: response.output_text.delta\ndata: {json.dumps(c_ev, ensure_ascii=False)}\n\n"

                # 处理工具调用
                tool_calls = delta.get("tool_calls")
                if tool_calls and isinstance(tool_calls, list):
                    for tc in tool_calls:
                        idx = tc.get("index", 0)
                        if idx not in self.tool_calls:
                            t_id = tc.get("id") or f"call_{uuid.uuid4().hex[:12]}"
                            fn = tc.get("function") or {}
                            name = fn.get("name") or ""
                            args = fn.get("arguments") or ""
                            item_id = f"item_{uuid.uuid4().hex[:20]}"
                            base_offset = (1 if self.reasoning_started else 0) + (1 if self.text_content else 0)
                            out_idx = base_offset + len(self.tool_calls)
                            self.tool_calls[idx] = {
                                "item_id": item_id,
                                "call_id": t_id,
                                "name": name,
                                "arguments": args,
                                "output_index": out_idx,
                                "done": False,
                            }
                            add_ev = {
                                "type": "response.output_item.added",
                                "output_index": out_idx,
                                "item": {
                                    "id": item_id,
                                    "type": "function_call",
                                    "call_id": t_id,
                                    "name": name,
                                    "arguments": ""
                                }
                            }
                            yield f"event: response.output_item.added\ndata: {json.dumps(add_ev, ensure_ascii=False)}\n\n"
                            if args:
                                arg_ev = {
                                    "type": "response.function_call_arguments.delta",
                                    "item_id": item_id,
                                    "output_index": out_idx,
                                    "call_id": t_id,
                                    "delta": args
                                }
                                yield f"event: response.function_call_arguments.delta\ndata: {json.dumps(arg_ev, ensure_ascii=False)}\n\n"
                        else:
                            tool = self.tool_calls[idx]
                            fn = tc.get("function") or {}
                            args_delta = fn.get("arguments") or ""
                            if args_delta:
                                tool["arguments"] += args_delta
                                arg_ev = {
                                    "type": "response.function_call_arguments.delta",
                                    "item_id": tool["item_id"],
                                    "output_index": tool["output_index"],
                                    "call_id": tool["call_id"],
                                    "delta": args_delta
                                }
                                yield f"event: response.function_call_arguments.delta\ndata: {json.dumps(arg_ev, ensure_ascii=False)}\n\n"

                if finish_reason:
                    # 检查文本中是否嵌入了原始工具调用（如 Qwen <tool_call>... 或 Markdown 格式）
                    if not self.tool_calls and self.text_content:
                        try:
                            from app.tool_call_parser import parse_raw_tool_calls
                        except ImportError:
                            from gateway.app.tool_call_parser import parse_raw_tool_calls
                        clean_text, extracted_tcs = parse_raw_tool_calls(self.text_content)
                        if extracted_tcs:
                            self.text_content = clean_text
                            base_offset = (1 if self.reasoning_started else 0) + (1 if self.text_content else 0)
                            for e_idx, etc in enumerate(extracted_tcs):
                                e_fn = etc.get("function") or {}
                                e_name = e_fn.get("name") or ""
                                e_args = e_fn.get("arguments") or ""
                                e_item_id = f"item_{uuid.uuid4().hex[:20]}"
                                e_out_idx = base_offset + e_idx
                                e_call_id = etc.get("id") or f"call_{uuid.uuid4().hex[:12]}"
                                self.tool_calls[e_idx] = {
                                    "item_id": e_item_id,
                                    "call_id": e_call_id,
                                    "name": e_name,
                                    "arguments": e_args,
                                    "output_index": e_out_idx,
                                    "done": True,
                                }
                                add_ev = {
                                    "type": "response.output_item.added",
                                    "output_index": e_out_idx,
                                    "item": {
                                        "id": e_item_id,
                                        "type": "function_call",
                                        "call_id": e_call_id,
                                        "name": e_name,
                                        "arguments": ""
                                    }
                                }
                                yield f"event: response.output_item.added\ndata: {json.dumps(add_ev, ensure_ascii=False)}\n\n"
                                if e_args:
                                    arg_ev = {
                                        "type": "response.function_call_arguments.delta",
                                        "item_id": e_item_id,
                                        "output_index": e_out_idx,
                                        "call_id": e_call_id,
                                        "delta": e_args
                                    }
                                    yield f"event: response.function_call_arguments.delta\ndata: {json.dumps(arg_ev, ensure_ascii=False)}\n\n"

                    for rev in self._flush_reasoning_if_needed():
                        yield rev

                    for cev in self._ensure_visible_content_if_empty():
                        yield cev

                    for idx in sorted(self.tool_calls.keys()):
                        tool = self.tool_calls[idx]
                        if not tool["done"]:
                            tool["done"] = True
                            done_ev = {
                                "type": "response.output_item.done",
                                "output_index": tool["output_index"],
                                "item": {
                                    "id": tool["item_id"],
                                    "type": "function_call",
                                    "call_id": tool["call_id"],
                                    "name": tool["name"],
                                    "arguments": tool["arguments"]
                                }
                            }
                            yield f"event: response.output_item.done\ndata: {json.dumps(done_ev, ensure_ascii=False)}\n\n"

                    if not self.completed:
                        self.completed = True
                        comp_ev = self._build_completed_event()
                        yield f"event: response.completed\ndata: {json.dumps(comp_ev, ensure_ascii=False)}\n\n"

        vis_tail, rsn_tail = self._stripper.finish()
        if rsn_tail:
            self.reasoning_content += rsn_tail
        if vis_tail:
            self.text_content += vis_tail

        for rev in self._flush_reasoning_if_needed():
            yield rev

        for cev in self._ensure_visible_content_if_empty():
            yield cev

        for idx in sorted(self.tool_calls.keys()):
            tool = self.tool_calls[idx]
            if not tool["done"]:
                tool["done"] = True
                done_ev = {
                    "type": "response.output_item.done",
                    "output_index": tool["output_index"],
                    "item": {
                        "id": tool["item_id"],
                        "type": "function_call",
                        "call_id": tool["call_id"],
                        "name": tool["name"],
                        "arguments": tool["arguments"]
                    }
                }
                yield f"event: response.output_item.done\ndata: {json.dumps(done_ev, ensure_ascii=False)}\n\n"

        if not self.completed:
            self.completed = True
            comp_ev = self._build_completed_event()
            yield f"event: response.completed\ndata: {json.dumps(comp_ev, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    def _build_completed_event(self) -> dict:
        output_items = []
        if self.reasoning_content:
            output_items.append({
                "id": self.rs_item_id,
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": self.reasoning_content}]
            })
        if self.text_content and self.text_content.strip():
            output_items.append({
                "id": self.msg_item_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": self.text_content.strip()}]
            })
        for idx in sorted(self.tool_calls.keys()):
            t = self.tool_calls[idx]
            output_items.append({
                "id": t["item_id"],
                "type": "function_call",
                "call_id": t["call_id"],
                "name": t["name"],
                "arguments": t["arguments"]
            })
        usage_obj = None
        if self.last_usage:
            pt = self.last_usage.get("prompt_tokens", 0)
            ct = self.last_usage.get("completion_tokens", 0)
            usage_obj = {
                "input_tokens": pt,
                "output_tokens": ct,
                "total_tokens": self.last_usage.get("total_tokens", pt + ct),
            }
        return {
            "type": "response.completed",
            "response": {
                "id": self.resp_id,
                "object": "response",
                "created_at": self.created,
                "status": "completed",
                "model": self.model,
                "output": output_items,
                "usage": usage_obj
            }
        }

