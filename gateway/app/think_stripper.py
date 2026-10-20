"""
思维链标签提取与剥离模块 (Think / Reasoning Stripper)

解决部分模型 (如 openai/gpt-5.6-luna, deepseek-r1, qwen-qwq 等) 将思考过程
以 <thinking>...</thinking>, <think>...</think>, <thought>...</thought> 等标签
直接输出在 message.content / delta.content 中，导致 ChatGPT Desktop, Codex,
CC Switch 等客户端将其作为普通消息外显到对话界面的问题。

本模块提供：
1. StreamThinkStripper: 流式 SSE 逐块增量解析器，具备跨 chunk 缓冲，支持跨 chunk 标签检测
   与未闭合标签安全沉淀 (如工具调用前未闭合的 <thinking> 视为 reasoning 沉淀，彻底杜绝外显)。
2. strip_think_tags: 非流式完整文本的一次性解析器，返回 (clean_content, extracted_reasoning)。
"""

from __future__ import annotations

import re
from typing import Tuple, List

TAG_NAMES = ("thinking", "think", "thought", "thought_process", "reasoning", "reasoning_content")

# 匹配各类思维链标签（忽略大小写，允许属性与空格）
TAG_RE = re.compile(
    r"</?(?:thinking|think|thought|thought_process|reasoning|reasoning_content)(?:\s+[^>]*)?>",
    re.IGNORECASE
)

MAX_TAG_LEN = 48


def _is_open_tag(token: str) -> bool:
    t = token.lower().strip()
    return t.startswith("<") and not t.startswith("</")


def _is_close_tag(token: str) -> bool:
    t = token.lower().strip()
    return t.startswith("</")


def _tag_prefix_len(text: str) -> int:
    """计算 text 末尾可能构成思维链标签前缀的字符数（用于跨 chunk 缓冲）。"""
    lower = text.lower()
    limit = min(len(lower), MAX_TAG_LEN - 1)
    for size in range(limit, 0, -1):
        tail = lower[-size:]
        for name in TAG_NAMES:
            for prefix in (f"<{name}", f"</{name}"):
                if prefix.startswith(tail):
                    return size
    return 0


class StreamThinkStripper:
    """
    流式增量解析器：将输入 chunk 里的思考标签与内容分离，
    返回 (visible_text, reasoning_text)。
    具备状态机与跨 chunk 缓冲，防止标签被切断时漏判。
    """

    def __init__(self):
        self._pending = ""
        self._in_think = False

    def feed(self, chunk: str) -> Tuple[str, str]:
        """
        处理一个内容增量。
        返回 (visible_text, reasoning_text)。
        """
        if not chunk and not self._pending:
            return "", ""

        data = self._pending + chunk
        self._pending = ""
        visible: List[str] = []
        reasoning: List[str] = []
        cursor = 0

        while cursor < len(data):
            if self._in_think:
                # 处于思考块内部：寻找闭合标签
                match = TAG_RE.search(data, cursor)
                if match is None:
                    # 尚未出现闭合标签，检查尾部是否为潜在的闭合标签前缀（如 </thin）
                    hold = _tag_prefix_len(data[cursor:])
                    if hold > 0:
                        reasoning.append(data[cursor : len(data) - hold])
                        self._pending = data[len(data) - hold :]
                    else:
                        reasoning.append(data[cursor:])
                    break

                token = match.group(0)
                if _is_close_tag(token):
                    # 找到闭合标签：退出思考模式
                    reasoning.append(data[cursor : match.start()])
                    cursor = match.end()
                    self._in_think = False
                else:
                    # 思考块内又遇到了开启标签（罕见冗余嵌套）：丢弃标签本身，保持思考模式
                    reasoning.append(data[cursor : match.start()])
                    cursor = match.end()
            else:
                # 处于普通可见文本模式：寻找开启标签
                match = TAG_RE.search(data, cursor)
                if match is None:
                    # 尚未出现开启标签，检查尾部是否为潜在的开启标签前缀（如 <thin）
                    hold = _tag_prefix_len(data[cursor:])
                    if hold > 0:
                        visible.append(data[cursor : len(data) - hold])
                        self._pending = data[len(data) - hold :]
                    else:
                        visible.append(data[cursor:])
                    break

                token = match.group(0)
                if _is_open_tag(token):
                    # 找到开启标签：进入思考模式
                    visible.append(data[cursor : match.start()])
                    cursor = match.end()
                    self._in_think = True
                else:
                    # 游离的闭合标签：直接丢弃该标签，不影响前后文本
                    visible.append(data[cursor : match.start()])
                    cursor = match.end()

        return "".join(visible), "".join(reasoning)

    def finish(self) -> Tuple[str, str]:
        """
        流结束或遇到工具调用时刷新剩余缓冲。
        如果当前处于未闭合思考标签中（常见于模型在发起 tool_calls 前只输出了 <thinking>xxx
        而未输出闭合标签），缓冲内容沉淀为 reasoning，彻底防止外显为用户可见消息。
        """
        tail = self._pending
        self._pending = ""
        if not tail:
            self._in_think = False
            return "", ""

        if self._in_think:
            self._in_think = False
            match = TAG_RE.match(tail)
            if match and _is_close_tag(match.group(0)):
                return "", ""
            return "", tail
        else:
            return tail, ""


def strip_think_tags(text: str) -> Tuple[str, str]:
    """
    非流式一次性解析器。
    返回 (clean_visible_text, extracted_reasoning)。
    """
    if not text:
        return "", ""
    stripper = StreamThinkStripper()
    v, r = stripper.feed(text)
    tv, tr = stripper.finish()
    clean_v = v + tv
    clean_r = r + tr
    return clean_v.strip(), clean_r.strip()
