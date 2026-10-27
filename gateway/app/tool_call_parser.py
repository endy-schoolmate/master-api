# -*- coding: utf-8 -*-
"""
Agent ReAct 协议强化与工具调用解析器 (Tool Call Parser & Agent ReAct Enforcer)

核心解决：
1. 开源模型 (Qwen, DeepSeek 等) 在接入 Codex Desktop / ChatGPT Desktop 时，
   习惯性在正文中输出意图宣布 ("Scene complete. Rendering to check."、"我这就替换尾部图层并渲染确认。")，
   但未在同一轮中发起 tool_calls。Codex Desktop 收到 finish_reason="stop" 且无 tool_calls 时，
   误以为任务已全部完成，直接停止并呈现空泡泡/未完成状态，导致用户频繁遇到“截断”。
2. 开源模型在正文中以 <tool_call>...</tool_call>、```tool_call...``` 或 Qwen <arg_key> 格式
   直接输出文本工具调用，网关若原样透传为文本，客户端无法将其识别为可执行工具。
3. Responses API 桥接层中丢弃 custom_tool_call 与 custom_tool_call_output 的问题。
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple


# ========== 1. 工具调用文本解析器 ==========

_QWEN_KV_RE = re.compile(
    r"<tool_call>([a-zA-Z0-9_\-\.]+)(<arg_key>.*?</arg_value>)</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_QWEN_ARG_KV_RE = re.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
    re.DOTALL | re.IGNORECASE,
)
_TOOL_CALL_XML_RE = re.compile(
    r"<tool_call>(.*?)</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_MD_TOOL_CALL_RE = re.compile(
    r"```+(?:tool_call|tool-call|function_call)\s*(.*?)\s*```+",
    re.DOTALL | re.IGNORECASE,
)
_FUNCTION_TAG_RE = re.compile(
    r"<(?:call:|function=)([a-zA-Z0-9_\-\.]+)(?:\s+[^>]*)?>(.*?)</(?:call:[a-zA-Z0-9_\-\.]+|function)>",
    re.DOTALL | re.IGNORECASE,
)


def parse_raw_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    非流式一次性解析：从模型输出文本中提取嵌入的工具调用，返回 (clean_text, tool_calls)。
    """
    if not text or not isinstance(text, str):
        return text or "", []

    tool_calls: List[Dict[str, Any]] = []
    clean_text = text

    # 1. 匹配 Qwen 专用 key-value XML 格式: <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>
    for match in _QWEN_KV_RE.finditer(text):
        name = match.group(1).strip()
        body = match.group(2)
        args: Dict[str, Any] = {}
        for kv in _QWEN_ARG_KV_RE.finditer(body):
            k = kv.group(1).strip()
            v = kv.group(2).strip()
            try:
                args[k] = json.loads(v)
            except Exception:
                args[k] = v
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:16]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })
    clean_text = _QWEN_KV_RE.sub("", clean_text)

    # 2. 匹配标准 JSON XML 格式: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    for match in _TOOL_CALL_XML_RE.finditer(clean_text):
        raw_json = match.group(1).strip()
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict) and "name" in parsed:
                args = parsed.get("arguments", {})
                args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, (dict, list)) else str(args)
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:16]}",
                    "type": "function",
                    "function": {
                        "name": parsed["name"],
                        "arguments": args_str,
                    },
                })
        except Exception:
            pass
    clean_text = _TOOL_CALL_XML_RE.sub("", clean_text)

    # 3. 匹配 Markdown 格式: ```tool_call\n{...}\n```
    for match in _MD_TOOL_CALL_RE.finditer(clean_text):
        raw_json = match.group(1).strip()
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict) and "name" in parsed:
                args = parsed.get("arguments", {})
                args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, (dict, list)) else str(args)
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:16]}",
                    "type": "function",
                    "function": {
                        "name": parsed["name"],
                        "arguments": args_str,
                    },
                })
        except Exception:
            pass
    clean_text = _MD_TOOL_CALL_RE.sub("", clean_text)

    # 4. 匹配通用 <function=name>args</function> 格式
    for match in _FUNCTION_TAG_RE.finditer(clean_text):
        name = match.group(1).strip()
        raw_args = match.group(2).strip()
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:16]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": raw_args or "{}",
            },
        })
    clean_text = _FUNCTION_TAG_RE.sub("", clean_text)

    return clean_text.strip(), tool_calls


# ========== 2. 未执行行动意图识别器 (Action Intent Detector) ==========

_ACTION_INTENT_PATTERNS = [
    # 英文行动叙述（常出现在 Qwen/DeepSeek 输出末尾但没有 tool_calls）
    r"(?:rendering|render)\s+(?:to\s+check|now|it\s+now|and\s+inspect)",
    r"(?:replacing|replace)\s+(?:it\s+now|the\s+tail|now)",
    r"(?:let\s+me|i\s+will|i\x27ll|now\s+i\s+will|now\s+i\x27ll)\s+(?:render|check|confirm|write|patch|update|inspect|run|execute|test)",
    r"now\s+(?:writing|patching|adding|testing|running|executing)\b",
    r"(?:writing|patching|creating)\s+(?:the\s+file|it)\s+now",
    r"confirm\s+by\s+rendering",
    r"scene\s+complete\.\s+rendering",
    # 中文行动叙述
    r"(?:我这就|这就|现在|马上|先来|接下来)(?:去|来|做|进行|替换|渲染|运行|执行|修改|写入|检查|测试|补全)",
    r"(?:渲染|运行|执行|替换|修改|补全|测试|检查)(?:确认|自查|自检|看下|看看|一下)",
    r"(?:替换|写入|补全)(?:尾部|图层|文件|代码|内容)并(?:渲染|确认|运行)",
    r"一次性补完，然后渲染",
    r"截图自查构图",
]
_ACTION_INTENT_RE = re.compile("|".join(_ACTION_INTENT_PATTERNS), re.IGNORECASE)

_COMPLETION_MARKERS = [
    "已经为您", "已经完成", "生成完毕", "绘制完成", "创建完成", "请查看", "修改完成", "保存完毕",
    "all done", "task completed", "successfully finished", "here is the final"
]


def is_unexecuted_action_intent(text: str) -> bool:
    """
    检查一段文本是否表达了“准备去执行某项操作”，但却结束了（未发出工具调用）。
    如果文本很长（>350字符）或明确包含完成标记，则不视为未执行行动意图。
    """
    if not text or not isinstance(text, str):
        return False
    stripped = text.strip()
    if len(stripped) > 350:
        return False
    for marker in _COMPLETION_MARKERS:
        if marker.lower() in stripped.lower():
            return False
    return bool(_ACTION_INTENT_RE.search(stripped))


# ========== 3. Agent 协议强制注入器 (ReAct Protocol Injector) ==========

AGENT_REACT_INSTRUCTION = (
    "\n\n[CRITICAL AGENT EXECUTION PROTOCOL]\n"
    "1. DIRECT TOOL CALLING: When you decide, plan, or need to perform an action (e.g. read/write/edit files, execute shell commands, render/view output, inspect paths, search), you MUST invoke the appropriate tool call in the CURRENT turn.\n"
    "2. NO UNEXECUTED NARRATION: NEVER output conversational announcements such as 'I will now do X', 'Replacing it now', 'Rendering to check', 'Now adding Y', 'Now let me render', '我这就去替换并渲染' without actually invoking the corresponding tool call in that same response. If an action is required, do NOT narrate it; invoke the tool immediately.\n"
    "3. TASK COMPLETION INTEGRITY: Continue invoking tools until the user request is completely solved and verified. Do NOT terminate your turn prematurely while there are remaining actions or verifications."
)

TOOL_RESULT_HINT = "\n[System Protocol: Tool execution finished. If the next step requires an action (such as executing a command, inspecting, editing, or rendering), you MUST invoke the tool call directly in this turn. Do not narrate your intent without invoking the tool.]"


def inject_agent_react_instructions(body: dict) -> dict:
    """
    当请求体包含 tools 时（表明下游是 Agent 客户端，如 Codex Desktop / ChatGPT / Cline），
    向系统提示词与最新工具执行结果中注入严格的 ReAct 协议，防止模型只输出叙述文本而漏掉工具调用。
    """
    tools = body.get("tools")
    if not isinstance(tools, list) or not tools:
        return body

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return body

    # 1. 在首个 system/developer 消息后追加协议，或在消息列表首位插入
    found_sys = False
    for msg in messages:
        if isinstance(msg, dict):
            # 将 developer 统一标准化为 system，防止 AMD 等上游抛 "Unexpected message role"
            if msg.get("role") == "developer":
                msg["role"] = "system"
            if msg.get("role") == "system":
                content = msg.get("content")
                if isinstance(content, str):
                    if "[CRITICAL AGENT EXECUTION PROTOCOL]" not in content:
                        msg["content"] = content + AGENT_REACT_INSTRUCTION
                elif isinstance(content, list):
                    has_proto = any(isinstance(p, dict) and "[CRITICAL AGENT EXECUTION PROTOCOL]" in p.get("text", "") for p in content)
                    if not has_proto:
                        content.append({"type": "text", "text": AGENT_REACT_INSTRUCTION})
                found_sys = True

    if not found_sys:
        messages.insert(0, {
            "role": "system",
            "content": AGENT_REACT_INSTRUCTION.strip(),
        })

    # 2. 如果最新一条消息是工具执行结果 (role == "tool")，在尾部注入就近提示，抵消长上下文注意衰减
    last_msg = messages[-1] if messages else None
    if isinstance(last_msg, dict) and last_msg.get("role") == "tool":
        c = last_msg.get("content")
        if isinstance(c, str):
            if "[System Protocol: Tool execution finished" not in c:
                last_msg["content"] = c + TOOL_RESULT_HINT

    return body
