"""内置供应商接入目录。

每家上游一个内置定义：id / name / baseUrl / API Key 环境变量 / API 类型，
供控制台"添加上游密钥"时一键预填 Base URL、API 格式与 Provider 标识；
原有手动创建流程完全保留。

字段说明：
- api_format 必须是 openai / anthropic / gemini 之一（与控制台"API 格式"及
  provider_probe 的探测分支一致）；
- base_url 按本网关的转发契约对准可跑通的端点：网关对上游统一按 OpenAI 协议
  （{base_url}/chat/completions + Bearer）转发，走 Responses / Conversations /
  Anthropic 原生协议的供应商填其官方 OpenAI 兼容路径（如 Gemini 的 /v1beta/openai）；
- api_key_env 是该家 Key 的环境变量名，作为界面提示，网关本身不读环境变量。
"""

from __future__ import annotations

import copy

PRESET_SOURCE_URL = "https://github.com/earendil-works/pi/tree/main/packages/ai"

# 分组展示顺序
_GROUP_ORDER = ["主流大厂", "国内平台", "海外平台", "聚合与推理云"]

# ChatGPT 账号登录（openai-codex）默认可用的订阅模型；创建时可再手动增删
DEFAULT_CHATGPT_MODELS = [
    "gpt-5.1-codex",
    "gpt-5.1",
    "gpt-5.1-codex-max",
    "gpt-5.1-codex-mini",
]

_PRESETS: list[dict] = [
    # ---- 主流大厂 ----
    {
        "id": "openai",
        "name": "OpenAI",
        "group": "主流大厂",
        "base_url": "https://api.openai.com/v1",
        "api_format": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "remark": "API Key（按量计费）方式；ChatGPT 订阅账号请选「OpenAI (ChatGPT 账号登录)」",
    },
    {
        "id": "openai-responses",
        "name": "OpenAI (Responses API)",
        "group": "主流大厂",
        "base_url": "https://api.openai.com/v1",
        "api_format": "openai-responses",
        "api_key_env": "OPENAI_API_KEY",
        "remark": "API Key 直调 /v1/responses（官方新接口）；第三方聚合站若提供 /responses 端点也可选此格式并手改 Base URL",
    },
    {
        "id": "openai-codex",
        "name": "OpenAI (ChatGPT 账号登录)",
        "group": "主流大厂",
        "base_url": "https://api.openai.com/v1",
        "api_format": "openai",
        "api_key_env": "—（无需 API Key，浏览器登录 ChatGPT 账号）",
        "auth_method": "oauth_login",
        "remark": "⚠ 必须是 ChatGPT Plus/Pro 订阅账号（free 账号无法换取 token，会报 invalid_grant）；使用 Sign in with ChatGPT 登录，网关自动完成 Responses API 协议桥接与 token 刷新。不支持 embeddings",
    },
    {
        "id": "anthropic",
        "name": "Anthropic (Claude)",
        "group": "主流大厂",
        "base_url": "https://api.anthropic.com/v1",
        "api_format": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "remark": "拉取模型列表走 x-api-key；对话经 Anthropic 的 OpenAI 兼容层转发",
    },
    {
        "id": "google",
        "name": "Google Gemini",
        "group": "主流大厂",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_format": "openai",
        "api_key_env": "GEMINI_API_KEY",
        "remark": "pi 原生接法为 Gemini 协议 https://generativelanguage.googleapis.com/v1beta；网关按 OpenAI 协议转发，故默认填兼容路径",
    },

    # ---- 国内平台 ----
    {
        "id": "zai-coding-cn",
        "name": "智谱 GLM (Z.AI Coding CN)",
        "group": "国内平台",
        "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
        "api_format": "openai",
        "api_key_env": "ZAI_CODING_CN_API_KEY",
        "remark": "智谱国内 Coding Plan 端点；通用平台为 https://open.bigmodel.cn/api/paas/v4",
    },
    {
        "id": "moonshotai-cn",
        "name": "Kimi (Moonshot 国内)",
        "group": "国内平台",
        "base_url": "https://api.moonshot.cn/v1",
        "api_format": "openai",
        "api_key_env": "MOONSHOT_API_KEY",
        "remark": "",
    },
    {
        "id": "deepseek",
        "name": "DeepSeek",
        "group": "国内平台",
        "base_url": "https://api.deepseek.com",
        "api_format": "openai",
        "api_key_env": "DEEPSEEK_API_KEY",
        "remark": "",
    },
    {
        "id": "minimax-cn",
        "name": "MiniMax 国内",
        "group": "国内平台",
        "base_url": "https://api.minimaxi.com/v1",
        "api_format": "openai",
        "api_key_env": "MINIMAX_CN_API_KEY",
        "remark": "pi 原生接法走 Anthropic 协议 https://api.minimaxi.com/anthropic；网关按 OpenAI 协议转发，故默认填 /v1",
    },
    {
        "id": "qwen-token-plan-cn",
        "name": "通义千问 Qwen (Token Plan CN)",
        "group": "国内平台",
        "base_url": "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        "api_format": "openai",
        "api_key_env": "QWEN_TOKEN_PLAN_CN_API_KEY",
        "remark": "Qwen Coding Plan（国内）专用端点；按量付费的百炼平台请手动填写",
    },
    {
        "id": "xiaomi",
        "name": "小米 MiMo",
        "group": "国内平台",
        "base_url": "https://api.xiaomimimo.com/v1",
        "api_format": "openai",
        "api_key_env": "XIAOMI_API_KEY",
        "remark": "",
    },
    {
        "id": "ant-ling",
        "name": "蚂蚁百灵 (Ant Ling)",
        "group": "国内平台",
        "base_url": "https://api.ant-ling.com/v1",
        "api_format": "openai",
        "api_key_env": "ANT_LING_API_KEY",
        "remark": "",
    },

    # ---- 海外平台 ----
    {
        "id": "zai",
        "name": "Z.AI (智谱海外)",
        "group": "海外平台",
        "base_url": "https://api.z.ai/api/coding/paas/v4",
        "api_format": "openai",
        "api_key_env": "ZAI_API_KEY",
        "remark": "Z.AI 海外 Coding Plan 端点",
    },
    {
        "id": "moonshotai",
        "name": "Moonshot AI (海外)",
        "group": "海外平台",
        "base_url": "https://api.moonshot.ai/v1",
        "api_format": "openai",
        "api_key_env": "MOONSHOT_API_KEY",
        "remark": "",
    },
    {
        "id": "minimax",
        "name": "MiniMax (海外)",
        "group": "海外平台",
        "base_url": "https://api.minimax.io/v1",
        "api_format": "openai",
        "api_key_env": "MINIMAX_API_KEY",
        "remark": "pi 原生接法走 Anthropic 协议 https://api.minimax.io/anthropic；网关按 OpenAI 协议转发，故默认填 /v1",
    },
    {
        "id": "qwen-token-plan",
        "name": "Qwen Token Plan (海外)",
        "group": "海外平台",
        "base_url": "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        "api_format": "openai",
        "api_key_env": "QWEN_TOKEN_PLAN_API_KEY",
        "remark": "",
    },
    {
        "id": "groq",
        "name": "Groq",
        "group": "海外平台",
        "base_url": "https://api.groq.com/openai/v1",
        "api_format": "openai",
        "api_key_env": "GROQ_API_KEY",
        "remark": "",
    },
    {
        "id": "cerebras",
        "name": "Cerebras",
        "group": "海外平台",
        "base_url": "https://api.cerebras.ai/v1",
        "api_format": "openai",
        "api_key_env": "CEREBRAS_API_KEY",
        "remark": "",
    },
    {
        "id": "xai",
        "name": "xAI (Grok)",
        "group": "海外平台",
        "base_url": "https://api.x.ai/v1",
        "api_format": "openai",
        "api_key_env": "XAI_API_KEY",
        "remark": "",
    },
    {
        "id": "mistral",
        "name": "Mistral AI",
        "group": "海外平台",
        "base_url": "https://api.mistral.ai/v1",
        "api_format": "openai",
        "api_key_env": "MISTRAL_API_KEY",
        "remark": "pi 原生接法走 Mistral Conversations API；网关按 OpenAI 协议转发，故默认填 OpenAI 兼容 /v1",
    },
    {
        "id": "together",
        "name": "Together AI",
        "group": "海外平台",
        "base_url": "https://api.together.ai/v1",
        "api_format": "openai",
        "api_key_env": "TOGETHER_API_KEY",
        "remark": "",
    },
    {
        "id": "fireworks",
        "name": "Fireworks AI",
        "group": "海外平台",
        "base_url": "https://api.fireworks.ai/inference",
        "api_format": "openai",
        "api_key_env": "FIREWORKS_API_KEY",
        "remark": "",
    },

    # ---- 聚合与推理云 ----
    {
        "id": "openrouter",
        "name": "OpenRouter",
        "group": "聚合与推理云",
        "base_url": "https://openrouter.ai/api/v1",
        "api_format": "openai",
        "api_key_env": "OPENROUTER_API_KEY",
        "remark": "",
    },
    {
        "id": "nvidia",
        "name": "NVIDIA NIM",
        "group": "聚合与推理云",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_format": "openai",
        "api_key_env": "NVIDIA_API_KEY",
        "remark": "",
    },
    {
        "id": "huggingface",
        "name": "Hugging Face",
        "group": "聚合与推理云",
        "base_url": "https://router.huggingface.co/v1",
        "api_format": "openai",
        "api_key_env": "HF_TOKEN",
        "remark": "",
    },
    {
        "id": "baseten",
        "name": "Baseten",
        "group": "聚合与推理云",
        "base_url": "https://inference.baseten.co/v1",
        "api_format": "openai",
        "api_key_env": "BASETEN_API_KEY",
        "remark": "",
    },
    {
        "id": "vercel-ai-gateway",
        "name": "Vercel AI Gateway",
        "group": "聚合与推理云",
        "base_url": "https://ai-gateway.vercel.sh/v1",
        "api_format": "openai",
        "api_key_env": "AI_GATEWAY_API_KEY",
        "remark": "pi 原生接法走 Anthropic 协议根路径；网关按 OpenAI 协议转发，故默认填 /v1",
    },
]


def list_provider_presets() -> list[dict]:
    """返回内置供应商接入目录（按分组顺序），返回副本避免调用方改到常量。"""
    by_group: dict[str, list[dict]] = {}
    for preset in _PRESETS:
        by_group.setdefault(preset["group"], []).append(preset)
    ordered: list[dict] = []
    for group in _GROUP_ORDER:
        for preset in by_group.get(group, []):
            item = copy.deepcopy(preset)
            item["group_order"] = _GROUP_ORDER.index(group)
            ordered.append(item)
    for group, items in by_group.items():
        if group not in _GROUP_ORDER:
            for preset in items:
                item = copy.deepcopy(preset)
                item["group_order"] = len(_GROUP_ORDER)
                ordered.append(item)
    return ordered
