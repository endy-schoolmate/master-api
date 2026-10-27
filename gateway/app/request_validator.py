"""
请求防呆校验模块 v10.1
=============
网关层全面校验，在请求转发至上游前拦截用户配置错误。

功能：
1. 模型ID智能映射与纠错（v10.1: 模糊匹配 + 大小写不敏感 + 多格式兼容）
2. API Key 清洗与校验
3. Base URL 路径标准化
4. 请求体格式强制校验
5. 参数容错与默认值
"""
import re
import json
import time
import logging
import difflib
from typing import Optional, Tuple, Dict, List, Any
from urllib.parse import urlparse, urlunparse

from fastapi import HTTPException

logger = logging.getLogger("acu.request_validator")

# ============================================================
# 1. 模型ID智能映射与纠错（v10.1: 全模糊匹配引擎）
# ============================================================

# 所有上游模型的标准ID列表（从上游动态获取，TTL 300秒）
_VERIFIED_MODELS_CACHE: set = set()
_VERIFIED_MODELS_EXPIRES: float = 0
_VERIFIED_MODELS_TTL: float = 300

# 手动别名映射表（优先级最高，精确匹配）
# 基于社会工程学：模拟用户实际输入习惯（大小写变体、缩写、空格、无分隔符、品牌名等）
_MODEL_ALIAS_MAP: Dict[str, str] = {
    # === deepseek-ai ===
    # deepseek-v4-flash
    "deepseek-v4-flash": "deepseek-ai/deepseek-v4-flash",
    "deepseekv4flash": "deepseek-ai/deepseek-v4-flash",
    "deepseek-v4-flash-instruct": "deepseek-ai/deepseek-v4-flash",
    "deepseek-v4flash": "deepseek-ai/deepseek-v4-flash",
    "deepseek v4 flash": "deepseek-ai/deepseek-v4-flash",
    "ds-v4-flash": "deepseek-ai/deepseek-v4-flash",
    "dsv4flash": "deepseek-ai/deepseek-v4-flash",
    "ds-v4flash": "deepseek-ai/deepseek-v4-flash",
    "DeepSeek-V4-Flash": "deepseek-ai/deepseek-v4-flash",
    "DeepSeekV4Flash": "deepseek-ai/deepseek-v4-flash",
    "deepseek-flash": "deepseek-ai/deepseek-v4-flash",
    "deepseek flash": "deepseek-ai/deepseek-v4-flash",
    # deepseek-v4-pro
    "deepseek-v4-pro": "deepseek-ai/deepseek-v4-pro",
    "deepseekv4pro": "deepseek-ai/deepseek-v4-pro",
    "deepseek-v4pro": "deepseek-ai/deepseek-v4-pro",
    "deepseek v4 pro": "deepseek-ai/deepseek-v4-pro",
    "ds-v4-pro": "deepseek-ai/deepseek-v4-pro",
    "dsv4pro": "deepseek-ai/deepseek-v4-pro",
    "ds-v4pro": "deepseek-ai/deepseek-v4-pro",
    "DeepSeek-V4-Pro": "deepseek-ai/deepseek-v4-pro",
    "DeepSeekV4Pro": "deepseek-ai/deepseek-v4-pro",
    "deepseek-v4": "deepseek-ai/deepseek-v4-pro",
    "deepseekv4": "deepseek-ai/deepseek-v4-pro",
    "deepseek pro": "deepseek-ai/deepseek-v4-pro",
    "deepseek-pro": "deepseek-ai/deepseek-v4-pro",
    "deepseek pro v4": "deepseek-ai/deepseek-v4-pro",
    # deepseek-coder
    "deepseek-coder": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "deepseek-coder-6.7b": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "deepseekcoder": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "deepseek-coder-6.7b-instruct": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "deepseek coder": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "DeepSeek-Coder": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "DeepSeekCoder": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "ds-coder": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "dscoder": "deepseek-ai/deepseek-coder-6.7b-instruct",

    # === z-ai / glm ===
    "glm-5.2": "z-ai/glm-5.2",
    "glm5.2": "z-ai/glm-5.2",
    "glm-5": "z-ai/glm-5.2",
    "glm5": "z-ai/glm-5.2",
    "zhipu-glm-5.2": "z-ai/glm-5.2",
    "chatglm-5.2": "z-ai/glm-5.2",
    "zhipu-glm": "z-ai/glm-5.2",
    "chatglm": "z-ai/glm-5.2",
    "chatglm5": "z-ai/glm-5.2",
    "GLM-5.2": "z-ai/glm-5.2",
    "GLM5.2": "z-ai/glm-5.2",
    "GLM-5": "z-ai/glm-5.2",
    "GLM5": "z-ai/glm-5.2",
    "glm 5.2": "z-ai/glm-5.2",
    "glm5.2instruct": "z-ai/glm-5.2",
    "z-ai-glm-5.2": "z-ai/glm-5.2",
    "zai-glm-5.2": "z-ai/glm-5.2",
    "zhipuai-glm-5.2": "z-ai/glm-5.2",
    "zhipu": "z-ai/glm-5.2",
    "zhipuai": "z-ai/glm-5.2",
    "z-ai": "z-ai/glm-5.2",

    # === qwen ===
    "qwen3.5-397b": "qwen/qwen3.5-397b-a17b",
    "qwen3.5-122b": "qwen/qwen3.5-122b-a10b",
    "qwen3-next": "qwen/qwen3-next-80b-a3b-instruct",
    "q35-397b": "qwen/qwen3.5-397b-a17b",
    "q35-122b": "qwen/qwen3.5-122b-a10b",
    "qwen-3.5-397b": "qwen/qwen3.5-397b-a17b",
    "qwen-3.5-122b": "qwen/qwen3.5-122b-a10b",
    "qwen3.5": "qwen/qwen3.5-397b-a17b",
    "tongyi-qwen": "qwen/qwen3.5-397b-a17b",
    "qwen3.5-397b-a17b": "qwen/qwen3.5-397b-a17b",
    "qwen3.5-122b-a10b": "qwen/qwen3.5-122b-a10b",
    "qwen3.5397b": "qwen/qwen3.5-397b-a17b",
    "qwen3.5122b": "qwen/qwen3.5-122b-a10b",
    "Qwen3.5-397B": "qwen/qwen3.5-397b-a17b",
    "Qwen3.5-122B": "qwen/qwen3.5-122b-a10b",
    "Qwen-3.5": "qwen/qwen3.5-397b-a17b",
    "Qwen3.5": "qwen/qwen3.5-397b-a17b",
    "tongyi": "qwen/qwen3.5-397b-a17b",
    "qwen 3.5 397b": "qwen/qwen3.5-397b-a17b",
    "qwen 3.5 122b": "qwen/qwen3.5-122b-a10b",
    "qwen3-next-80b-a3b-instruct": "qwen/qwen3-next-80b-a3b-instruct",
    "qwen3-next-80b": "qwen/qwen3-next-80b-a3b-instruct",
    "qwen3next": "qwen/qwen3-next-80b-a3b-instruct",
    "qwen-3-next": "qwen/qwen3-next-80b-a3b-instruct",
    "Qwen3-Next": "qwen/qwen3-next-80b-a3b-instruct",
    "qwen next": "qwen/qwen3-next-80b-a3b-instruct",
    "qwen-next": "qwen/qwen3-next-80b-a3b-instruct",

    # === minimaxai ===
    "minimax-m3": "minimaxai/minimax-m3",
    "minimax-m2.7": "minimaxai/minimax-m2.7",
    "minimaxm3": "minimaxai/minimax-m3",
    "minimax": "minimaxai/minimax-m3",
    "mm-m3": "minimaxai/minimax-m3",
    "MiniMax-M3": "minimaxai/minimax-m3",
    "MiniMaxM3": "minimaxai/minimax-m3",
    "MiniMax": "minimaxai/minimax-m3",
    "minimax m3": "minimaxai/minimax-m3",
    "minimaxm2.7": "minimaxai/minimax-m2.7",
    "minimax m2.7": "minimaxai/minimax-m2.7",
    "MiniMax-M2.7": "minimaxai/minimax-m2.7",
    "mm-m2.7": "minimaxai/minimax-m2.7",
    "minimax-m2": "minimaxai/minimax-m2.7",

    # === stepfun ===
    "step-3.5-flash": "stepfun-ai/step-3.5-flash",
    "step-3.7-flash": "stepfun-ai/step-3.7-flash",
    "step3.5": "stepfun-ai/step-3.5-flash",
    "step3.7": "stepfun-ai/step-3.7-flash",
    "step-3.5": "stepfun-ai/step-3.5-flash",
    "step-3.7": "stepfun-ai/step-3.7-flash",
    "step3.5flash": "stepfun-ai/step-3.5-flash",
    "step3.7flash": "stepfun-ai/step-3.7-flash",
    "Step-3.5-Flash": "stepfun-ai/step-3.5-flash",
    "Step-3.7-Flash": "stepfun-ai/step-3.7-flash",
    "Step3.5": "stepfun-ai/step-3.5-flash",
    "Step3.7": "stepfun-ai/step-3.7-flash",
    "step 3.5 flash": "stepfun-ai/step-3.5-flash",
    "step 3.7 flash": "stepfun-ai/step-3.7-flash",
    "stepflash": "stepfun-ai/step-3.7-flash",
    "step-3.5-flash-instruct": "stepfun-ai/step-3.5-flash",
    "step-3.7-flash-instruct": "stepfun-ai/step-3.7-flash",

    # === openai ===
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gptoss-120b": "openai/gpt-oss-120b",
    "gptoss-20b": "openai/gpt-oss-20b",
    "gpt-oss": "openai/gpt-oss-120b",
    "gptoss": "openai/gpt-oss-120b",
    "GPT-OSS-120B": "openai/gpt-oss-120b",
    "GPT-OSS-20B": "openai/gpt-oss-20b",
    "GPTOSS": "openai/gpt-oss-120b",
    "gpt oss 120b": "openai/gpt-oss-120b",
    "gpt oss 20b": "openai/gpt-oss-20b",
    "gptoss120b": "openai/gpt-oss-120b",
    "gptoss20b": "openai/gpt-oss-20b",
    "openai-gpt-oss": "openai/gpt-oss-120b",
    "openai-gpt-oss-120b": "openai/gpt-oss-120b",
    "openai-gpt-oss-20b": "openai/gpt-oss-20b",

    # === meta / llama ===
    "llama-3.1-70b": "meta/llama-3.1-70b-instruct",
    "llama-3.1-8b": "meta/llama-3.1-8b-instruct",
    "llama3.1-70b": "meta/llama-3.1-70b-instruct",
    "llama3.1-8b": "meta/llama-3.1-8b-instruct",
    "llama3.1-70b-instruct": "meta/llama-3.1-70b-instruct",
    "llama3.1-8b-instruct": "meta/llama-3.1-8b-instruct",
    "llama-31-70b": "meta/llama-3.1-70b-instruct",
    "llama-31-8b": "meta/llama-3.1-8b-instruct",
    "llama3.2-11b": "meta/llama-3.2-11b-vision-instruct",
    "llama3.2-90b": "meta/llama-3.2-90b-vision-instruct",
    "llama3.3-70b": "meta/llama-3.3-70b-instruct",
    "llama-3.3-70b": "meta/llama-3.3-70b-instruct",
    "llama-guard": "meta/llama-guard-4-12b",
    "llama-4-maverick": "meta/llama-4-maverick-17b-128e-instruct",
    "llama4-maverick": "meta/llama-4-maverick-17b-128e-instruct",
    "llama-3.2-1b": "meta/llama-3.2-1b-instruct",
    "llama-3.2-3b": "meta/llama-3.2-3b-instruct",
    "Llama-3.1-70B": "meta/llama-3.1-70b-instruct",
    "Llama-3.1-8B": "meta/llama-3.1-8b-instruct",
    "Llama3.1-70B": "meta/llama-3.1-70b-instruct",
    "Llama3.1-8B": "meta/llama-3.1-8b-instruct",
    "Llama-3.3-70B": "meta/llama-3.3-70b-instruct",
    "Llama3.3-70B": "meta/llama-3.3-70b-instruct",
    "Llama-4-Maverick": "meta/llama-4-maverick-17b-128e-instruct",
    "Llama4-Maverick": "meta/llama-4-maverick-17b-128e-instruct",
    "Llama-3.2-11B": "meta/llama-3.2-11b-vision-instruct",
    "Llama-3.2-90B": "meta/llama-3.2-90b-vision-instruct",
    "llama3.2-1b": "meta/llama-3.2-1b-instruct",
    "llama3.2-3b": "meta/llama-3.2-3b-instruct",
    "llama-3.2-11b-vision": "meta/llama-3.2-11b-vision-instruct",
    "llama-3.2-90b-vision": "meta/llama-3.2-90b-vision-instruct",
    "llama3.2-11b-vision": "meta/llama-3.2-11b-vision-instruct",
    "llama3.2-90b-vision": "meta/llama-3.2-90b-vision-instruct",
    "llama-3.2-11b-instruct": "meta/llama-3.2-11b-vision-instruct",
    "llama-3.2-90b-instruct": "meta/llama-3.2-90b-vision-instruct",
    "llama3.1-70b-instruct": "meta/llama-3.1-70b-instruct",
    "llama3.1-8b-instruct": "meta/llama-3.1-8b-instruct",
    "llama3.3-70b-instruct": "meta/llama-3.3-70b-instruct",
    "llama4maverick": "meta/llama-4-maverick-17b-128e-instruct",
    "llama-4-maverick-17b": "meta/llama-4-maverick-17b-128e-instruct",
    "llama-4-maverick-17b-128e": "meta/llama-4-maverick-17b-128e-instruct",
    "llama3.22-1b-instruct": "meta/llama-3.2-1b-instruct",
    "llama3.2-3b-instruct": "meta/llama-3.2-3b-instruct",
    "llama 3.1 70b": "meta/llama-3.1-70b-instruct",
    "llama 3.1 8b": "meta/llama-3.1-8b-instruct",
    "llama 3.3 70b": "meta/llama-3.3-70b-instruct",
    "llama 4 maverick": "meta/llama-4-maverick-17b-128e-instruct",
    "llama70b": "meta/llama-3.1-70b-instruct",
    "llama8b": "meta/llama-3.1-8b-instruct",
    "llamaguard": "meta/llama-guard-4-12b",
    "llama-guard-4": "meta/llama-guard-4-12b",
    "llama-guard-4-12b": "meta/llama-guard-4-12b",
    "Llama-Guard-4-12B": "meta/llama-guard-4-12b",
    "llamaguard4": "meta/llama-guard-4-12b",
    "llamaguard4-12b": "meta/llama-guard-4-12b",

    # === mistralai ===
    "mistral-large-675b": "mistralai/mistral-large-3-675b-instruct-2512",
    "mistral-medium-128b": "mistralai/mistral-medium-3.5-128b",
    "mistral-small-119b": "mistralai/mistral-small-4-119b-2603",
    "mistral-large": "mistralai/mistral-large-3-675b-instruct-2512",
    "mistral-medium": "mistralai/mistral-medium-3.5-128b",
    "mistral-small": "mistralai/mistral-small-4-119b-2603",
    "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct-v0.1",
    "mixtral": "mistralai/mixtral-8x7b-instruct-v0.1",
    "mistral-nemo": "nv-mistralai/mistral-nemo-12b-instruct",
    "Mistral-Large-675B": "mistralai/mistral-large-3-675b-instruct-2512",
    "Mistral-Medium-128B": "mistralai/mistral-medium-3.5-128b",
    "Mistral-Small-119B": "mistralai/mistral-small-4-119b-2603",
    "Mistral-Large": "mistralai/mistral-large-3-675b-instruct-2512",
    "Mistral-Medium": "mistralai/mistral-medium-3.5-128b",
    "Mistral-Small": "mistralai/mistral-small-4-119b-2603",
    "Mixtral-8x7B": "mistralai/mixtral-8x7b-instruct-v0.1",
    "Mixtral": "mistralai/mixtral-8x7b-instruct-v0.1",
    "Mistral-Nemo": "nv-mistralai/mistral-nemo-12b-instruct",
    "mistral-large-3": "mistralai/mistral-large-3-675b-instruct-2512",
    "mistral-large-3-675b": "mistralai/mistral-large-3-675b-instruct-2512",
    "mistral-medium-3.5": "mistralai/mistral-medium-3.5-128b",
    "mistral-small-4": "mistralai/mistral-small-4-119b-2603",
    "mistralnemotron": "mistralai/mistral-nemotron",
    "mistral-nemotron": "mistralai/mistral-nemotron",
    "Mistral-Nemotron": "mistralai/mistral-nemotron",
    "mistral large": "mistralai/mistral-large-3-675b-instruct-2512",
    "mistral medium": "mistralai/mistral-medium-3.5-128b",
    "mistral small": "mistralai/mistral-small-4-119b-2603",
    "mixtral8x7b": "mistralai/mixtral-8x7b-instruct-v0.1",
    "mixtral8x7binstruct": "mistralai/mixtral-8x7b-instruct-v0.1",
    "mixtral-8x7b-instruct": "mistralai/mixtral-8x7b-instruct-v0.1",
    "mixtral-8x7b-instruct-v0.1": "mistralai/mixtral-8x7b-instruct-v0.1",
    "Mixtral-8x7B-Instruct": "mistralai/mixtral-8x7b-instruct-v0.1",
    "mistral-nemo-12b": "nv-mistralai/mistral-nemo-12b-instruct",
    "mistral-nemo-12b-instruct": "nv-mistralai/mistral-nemo-12b-instruct",
    "Mistral-Nemo-12B": "nv-mistralai/mistral-nemo-12b-instruct",
    "mistralnemo": "nv-mistralai/mistral-nemo-12b-instruct",
    "nemo": "nv-mistralai/mistral-nemo-12b-instruct",

    # === google / gemma ===
    "gemma-4-31b": "google/gemma-4-31b-it",
    "gemma4-31b": "google/gemma-4-31b-it",
    "gemma-2-2b": "google/gemma-2-2b-it",
    "gemma2-2b": "google/gemma-2-2b-it",
    "gemma-3-12b": "google/gemma-3-12b-it",
    "gemma-3-4b": "google/gemma-3-4b-it",
    "gemma3-12b": "google/gemma-3-12b-it",
    "gemma3-4b": "google/gemma-3-4b-it",
    "Gemma-4-31B": "google/gemma-4-31b-it",
    "Gemma4-31B": "google/gemma-4-31b-it",
    "Gemma-3-12B": "google/gemma-3-12b-it",
    "Gemma-3-4B": "google/gemma-3-4b-it",
    "Gemma-2-2B": "google/gemma-2-2b-it",
    "gemma-4-31b-it": "google/gemma-4-31b-it",
    "gemma-3-12b-it": "google/gemma-3-12b-it",
    "gemma-3-4b-it": "google/gemma-3-4b-it",
    "gemma-2-2b-it": "google/gemma-2-2b-it",
    "gemma431bit": "google/gemma-4-31b-it",
    "gemma312bit": "google/gemma-3-12b-it",
    "gemma34bit": "google/gemma-3-4b-it",
    "gemma-4": "google/gemma-4-31b-it",
    "gemma-3": "google/gemma-3-12b-it",
    "gemma-2": "google/gemma-2-2b-it",
    "gemma4": "google/gemma-4-31b-it",
    "gemma3": "google/gemma-3-12b-it",
    "gemma2": "google/gemma-2-2b-it",
    "gemma 4 31b": "google/gemma-4-31b-it",
    "gemma 3 12b": "google/gemma-3-12b-it",
    "gemma 3 4b": "google/gemma-3-4b-it",
    "diffusiongemma": "google/diffusiongemma-26b-a4b-it",
    "diffusion-gemma": "google/diffusiongemma-26b-a4b-it",
    "DiffusionGemma": "google/diffusiongemma-26b-a4b-it",
    "diffusiongemma-26b": "google/diffusiongemma-26b-a4b-it",
    "diffusiongemma26b": "google/diffusiongemma-26b-a4b-it",
    "gemma-3n-e2b": "google/gemma-3n-e2b-it",
    "gemma-3n-e4b": "google/gemma-3n-e4b-it",
    "gemma3n-e2b": "google/gemma-3n-e2b-it",
    "gemma3n-e4b": "google/gemma-3n-e4b-it",
    "Gemma-3n-E2B": "google/gemma-3n-e2b-it",
    "Gemma-3n-E4B": "google/gemma-3n-e4b-it",
    "gemma-3n-e2b-it": "google/gemma-3n-e2b-it",
    "gemma-3n-e4b-it": "google/gemma-3n-e4b-it",
    "gemma3n": "google/gemma-3n-e4b-it",
    "gemma 3n": "google/gemma-3n-e4b-it",

    # === nvidia / nemotron ===
    "nemotron-3-ultra": "nvidia/nemotron-3-ultra-550b-a55b",
    "nemotron-3-super": "nvidia/nemotron-3-super-120b-a12b",
    "nemotron-3-nano": "nvidia/nemotron-3-nano-30b-a3b",
    "nemotron-super-49b": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "nemotron-nano-12b": "nvidia/nemotron-nano-12b-v2-vl",
    "nemotron-mini": "nvidia/nemotron-mini-4b-instruct",
    "nemotron-nano-9b": "nvidia/nvidia-nemotron-nano-9b-v2",
    "nemotron-nano-3b": "nvidia/nemotron-nano-3-30b-a3b",
    "nemotron-parse": "nvidia/nemotron-parse",
    "nemotron-safety": "nvidia/nemotron-3.5-content-safety",
    "nemoguard": "nvidia/llama-3.1-nemoguard-8b-content-safety",
    "riva-translate": "nvidia/riva-translate-4b-instruct",
    "vila": "nvidia/vila",
    "nvidia-vila": "nvidia/vila",
    "Nemotron-3-Ultra": "nvidia/nemotron-3-ultra-550b-a55b",
    "Nemotron-3-Super": "nvidia/nemotron-3-super-120b-a12b",
    "Nemotron-3-Nano": "nvidia/nemotron-3-nano-30b-a3b",
    "Nemotron-Ultra": "nvidia/nemotron-3-ultra-550b-a55b",
    "Nemotron-Super": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "Nemotron-Nano": "nvidia/nemotron-3-nano-30b-a3b",
    "Nemotron-Mini": "nvidia/nemotron-mini-4b-instruct",
    "Nemotron-Parse": "nvidia/nemotron-parse",
    "Nemotron-Safety": "nvidia/nemotron-3.5-content-safety",
    "NemoGuard": "nvidia/llama-3.1-nemoguard-8b-content-safety",
    "VILA": "nvidia/vila",
    "nemotron": "nvidia/nemotron-3-ultra-550b-a55b",
    "nemotron3ultra": "nvidia/nemotron-3-ultra-550b-a55b",
    "nemotron3super": "nvidia/nemotron-3-super-120b-a12b",
    "nemotron3nano": "nvidia/nemotron-3-nano-30b-a3b",
    "nemotron ultra": "nvidia/nemotron-3-ultra-550b-a55b",
    "nemotron super": "nvidia/nemotron-3-super-120b-a12b",
    "nemotron nano": "nvidia/nemotron-3-nano-30b-a3b",
    "nemotron-3-ultra-550b": "nvidia/nemotron-3-ultra-550b-a55b",
    "nemotron-3-super-120b": "nvidia/nemotron-3-super-120b-a12b",
    "nemotron-3-nano-30b": "nvidia/nemotron-3-nano-30b-a3b",
    "nemotron-3-ultra-550b-a55b": "nvidia/nemotron-3-ultra-550b-a55b",
    "nemotron-3-super-120b-a12b": "nvidia/nemotron-3-super-120b-a12b",
    "nemotron-3-nano-30b-a3b": "nvidia/nemotron-3-nano-30b-a3b",
    "nemotron-mini-4b": "nvidia/nemotron-mini-4b-instruct",
    "nemotron-mini-4b-instruct": "nvidia/nemotron-mini-4b-instruct",
    "nemotron-nano-12b-v2-vl": "nvidia/nemotron-nano-12b-v2-vl",
    "nemotron-nano-12b-v2": "nvidia/nemotron-nano-12b-v2-vl",
    "nvidia-nemotron-nano-9b-v2": "nvidia/nvidia-nemotron-nano-9b-v2",
    "nemotron-nano-9b": "nvidia/nvidia-nemotron-nano-9b-v2",
    "nemotron-nano-9b-v2": "nvidia/nvidia-nemotron-nano-9b-v2",
    "nemotron-nano-3-30b": "nvidia/nemotron-nano-3-30b-a3b",
    "nemotron-nano-3-30b-a3b": "nvidia/nemotron-nano-3-30b-a3b",
    "nemotron-nano-3": "nvidia/nemotron-nano-3-30b-a3b",
    "nemotron-parse": "nvidia/nemotron-parse",
    "nemotron-3.5-content-safety": "nvidia/nemotron-3.5-content-safety",
    "nemotron-3-5-content-safety": "nvidia/nemotron-3.5-content-safety",
    "nemotron-content-safety": "nvidia/nemotron-3.5-content-safety",
    "nemotron-safety-guard": "nvidia/llama-3.1-nemotron-safety-guard-8b-v3",
    "nemotron-safety-guard-8b": "nvidia/llama-3.1-nemotron-safety-guard-8b-v3",
    "nemotron-safety-guard-8b-v3": "nvidia/llama-3.1-nemotron-safety-guard-8b-v3",
    "nemoguard-8b": "nvidia/llama-3.1-nemoguard-8b-content-safety",
    "nemoguard-8b-content-safety": "nvidia/llama-3.1-nemoguard-8b-content-safety",
    "llama-3.1-nemoguard-8b": "nvidia/llama-3.1-nemoguard-8b-content-safety",
    "llama-3.1-nemoguard-8b-content-safety": "nvidia/llama-3.1-nemoguard-8b-content-safety",
    "nemoguard-topic-control": "nvidia/llama-3.1-nemoguard-8b-topic-control",
    "nemoguard-8b-topic-control": "nvidia/llama-3.1-nemoguard-8b-topic-control",
    "llama-3.1-nemoguard-8b-topic-control": "nvidia/llama-3.1-nemoguard-8b-topic-control",
    "nemotron-super-49b-v1.5": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "nemotron-super-49b-v1": "nvidia/llama-3.3-nemotron-super-49b-v1",
    "llama-3.3-nemotron-super-49b": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "nemotron-super-49b-v1-5": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "Nemotron-Super-49B": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "nemotron-nano-vl": "nvidia/nemotron-nano-12b-v2-vl",
    "nemotron-nano-12b-vl": "nvidia/nemotron-nano-12b-v2-vl",
    "llama-3.1-nemotron-nano-vl-8b": "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    "nemotron-nano-vl-8b": "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    "nemotron-nano-vl-8b-v1": "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    "llama-3.1-nemotron-51b": "nvidia/llama-3.1-nemotron-51b-instruct",
    "llama-3.1-nemotron-70b": "nvidia/llama-3.1-nemotron-70b-instruct",
    "llama-3.1-nemotron-ultra-253b": "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "llama-3.1-nemotron-nano-8b": "nvidia/llama-3.1-nemotron-nano-8b-v1",
    "nemotron-51b": "nvidia/llama-3.1-nemotron-51b-instruct",
    "nemotron-70b": "nvidia/llama-3.1-nemotron-70b-instruct",
    "nemotron-ultra-253b": "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "nemotron-nano-8b": "nvidia/llama-3.1-nemotron-nano-8b-v1",
    "nemotron-nano-omni": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "nemotron-nano-omni-30b": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "nemotron-3-nano-omni": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "nemotron-3-nano-omni-30b": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "nemotron-3-nano-omni-30b-a3b-reasoning": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "nemotron-nano-omni-30b-a3b-reasoning": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "riva-translate-4b": "nvidia/riva-translate-4b-instruct",
    "riva": "nvidia/riva-translate-4b-instruct",
    "riva-translate-4b-instruct": "nvidia/riva-translate-4b-instruct",
    "riva-translate-4b-instruct-v1.1": "nvidia/riva-translate-4b-instruct-v1.1",
    "riva-translate-4b-v1.1": "nvidia/riva-translate-4b-instruct-v1.1",
    "nvidia-vila": "nvidia/vila",
    "nvidia-vila": "nvidia/vila",
    "vila": "nvidia/vila",
    "VILA": "nvidia/vila",
    "gliner-pii": "nvidia/gliner-pii",
    "gliner": "nvidia/gliner-pii",
    "ising-calibration": "nvidia/ising-calibration-1-35b-a3b",
    "ising-calibration-1-35b": "nvidia/ising-calibration-1-35b-a3b",
    "ising-calibration-35b": "nvidia/ising-calibration-1-35b-a3b",
    "ising-1-35b": "nvidia/ising-calibration-1-35b-a3b",

    # === moonshotai ===
    "kimi-k2.6": "moonshotai/kimi-k2.6",
    "kimi": "moonshotai/kimi-k2.6",
    "kimi-k2": "moonshotai/kimi-k2.6",
    "moonshot": "moonshotai/kimi-k2.6",
    "Kimi-K2.6": "moonshotai/kimi-k2.6",
    "Kimi": "moonshotai/kimi-k2.6",
    "Kimi-K2": "moonshotai/kimi-k2.6",
    "Moonshot": "moonshotai/kimi-k2.6",
    "kimik2.6": "moonshotai/kimi-k2.6",
    "kimik2": "moonshotai/kimi-k2.6",
    "kimi k2.6": "moonshotai/kimi-k2.6",
    "kimi k2": "moonshotai/kimi-k2.6",
    "moonshot-kimi": "moonshotai/kimi-k2.6",
    "moonshot-kimi-k2.6": "moonshotai/kimi-k2.6",
    "moonshotai-kimi": "moonshotai/kimi-k2.6",
    "moonshotai-kimi-k2.6": "moonshotai/kimi-k2.6",

    # === bytedance ===
    "seed-oss-36b": "bytedance/seed-oss-36b-instruct",
    "seed-oss": "bytedance/seed-oss-36b-instruct",
    "doubao": "bytedance/seed-oss-36b-instruct",
    "seedoss": "bytedance/seed-oss-36b-instruct",
    "seedoss36b": "bytedance/seed-oss-36b-instruct",
    "Seed-OSS-36B": "bytedance/seed-oss-36b-instruct",
    "Seed-OSS": "bytedance/seed-oss-36b-instruct",
    "Doubao": "bytedance/seed-oss-36b-instruct",
    "seed oss 36b": "bytedance/seed-oss-36b-instruct",
    "seed-oss-36b-instruct": "bytedance/seed-oss-36b-instruct",
    "bytedance-seed-oss": "bytedance/seed-oss-36b-instruct",
    "bytedance-seed-oss-36b": "bytedance/seed-oss-36b-instruct",
    "seed": "bytedance/seed-oss-36b-instruct",

    # === abacusai ===
    "dracarys-70b": "abacusai/dracarys-llama-3.1-70b-instruct",
    "dracarys": "abacusai/dracarys-llama-3.1-70b-instruct",
    "Dracarys-70B": "abacusai/dracarys-llama-3.1-70b-instruct",
    "Dracarys": "abacusai/dracarys-llama-3.1-70b-instruct",
    "dracarys-llama-3.1-70b": "abacusai/dracarys-llama-3.1-70b-instruct",
    "dracarys-llama-3.1-70b-instruct": "abacusai/dracarys-llama-3.1-70b-instruct",
    "dracarysllama": "abacusai/dracarys-llama-3.1-70b-instruct",
    "dracarys70b": "abacusai/dracarys-llama-3.1-70b-instruct",
    "abacusai-dracarys": "abacusai/dracarys-llama-3.1-70b-instruct",

    # === upstage ===
    "solar-10.7b": "upstage/solar-10.7b-instruct",
    "solar": "upstage/solar-10.7b-instruct",
    "Solar-10.7B": "upstage/solar-10.7b-instruct",
    "Solar": "upstage/solar-10.7b-instruct",
    "solar10.7b": "upstage/solar-10.7b-instruct",
    "solar-10.7b-instruct": "upstage/solar-10.7b-instruct",
    "solar10.7binstruct": "upstage/solar-10.7b-instruct",
    "upstage-solar": "upstage/solar-10.7b-instruct",
    "upstage-solar-10.7b": "upstage/solar-10.7b-instruct",

    # === poolside ===
    "laguna-xs": "poolside/laguna-xs-2.1",
    "laguna": "poolside/laguna-xs-2.1",
    "Laguna-XS": "poolside/laguna-xs-2.1",
    "Laguna": "poolside/laguna-xs-2.1",
    "lagunaxs": "poolside/laguna-xs-2.1",
    "laguna-xs-2.1": "poolside/laguna-xs-2.1",
    "laguna-xs-2": "poolside/laguna-xs-2.1",
    "laguna2.1": "poolside/laguna-xs-2.1",
    "poolside-laguna": "poolside/laguna-xs-2.1",
    "poolside-laguna-xs": "poolside/laguna-xs-2.1",

    # === sarvamai ===
    "sarvam": "sarvamai/sarvam-m",
    "sarvam-m": "sarvamai/sarvam-m",
    "Sarvam": "sarvamai/sarvam-m",
    "Sarvam-M": "sarvamai/sarvam-m",
    "sarvamm": "sarvamai/sarvam-m",
    "sarvamai-sarvam": "sarvamai/sarvam-m",
    "sarvamai-sarvam-m": "sarvamai/sarvam-m",

    # === thinkingmachines ===
    "inkling": "thinkingmachines/inkling",
    "Inkling": "thinkingmachines/inkling",
    "thinkingmachines-inkling": "thinkingmachines/inkling",
    "thinking-machines": "thinkingmachines/inkling",
    "thinking-machines-inkling": "thinkingmachines/inkling",
}


def refresh_verified_models(models_list: list):
    """从上游获取的真实模型列表更新缓存"""
    global _VERIFIED_MODELS_CACHE, _VERIFIED_MODELS_EXPIRES
    _VERIFIED_MODELS_CACHE = set(m.get("id", "") for m in models_list if isinstance(m, dict))
    _VERIFIED_MODELS_EXPIRES = time.time() + _VERIFIED_MODELS_TTL


def _get_verified_models() -> set:
    """获取已验证的模型ID集合"""
    return _VERIFIED_MODELS_CACHE


def _normalize(s: str) -> str:
    """
    标准化字符串用于模糊匹配：
    - 转小写
    - 去除所有非字母数字字符（- _ . / 空格等全部去掉）
    """
    return re.sub(r'[^a-zA-Z0-9]', '', s).lower()


def _build_search_index() -> List[Tuple[str, str]]:
    """
    构建搜索索引：[(normalized_key, standard_id), ...]
    包含标准ID本身 + 所有别名 + NIM目录中的display_name
    """
    index = []
    # 标准ID自身
    for mid in _VERIFIED_MODELS_CACHE:
        index.append((_normalize(mid), mid))
        # 同时索引去掉 provider 前缀的部分
        if "/" in mid:
            name_part = mid.split("/", 1)[1]
            index.append((_normalize(name_part), mid))
    # 别名
    for alias, standard_id in _MODEL_ALIAS_MAP.items():
        index.append((_normalize(alias), standard_id))
    # 从 NIM_MODEL_CATALOG 补充 display_name 作为搜索索引
    try:
        from .nim_models import NIM_MODEL_CATALOG
        for model_id, info in NIM_MODEL_CATALOG.items():
            if info.display_name:
                index.append((_normalize(info.display_name), model_id))
    except ImportError:
        pass
    return index


def validate_and_correct_model(model_id: str) -> Tuple[str, bool]:
    """
    v10.1: 校验并修正模型ID。
    返回 (corrected_model_id, was_corrected)

    匹配策略（按优先级）：
    1. 精确匹配（标准ID或别名，大小写敏感）
    2. 大小写不敏感精确匹配
    3. 标准化精确匹配（去除所有分隔符后比较）
    4. 模糊匹配（difflib 相似度 ≥ 0.85）
    5. 子串包含匹配（用户输入是标准ID的子串）
    """
    if not model_id:
        return model_id, False

    model_stripped = model_id.strip()
    corrected = None

    # === 策略1: 精确匹配（大小写敏感）===
    # 1a. 标准ID直接命中
    if model_stripped in _VERIFIED_MODELS_CACHE:
        return model_stripped, False

    # 1b. 别名映射精确命中
    if model_stripped in _MODEL_ALIAS_MAP:
        corrected = _MODEL_ALIAS_MAP[model_stripped]
        logger.info(f"模型ID别名映射: '{model_stripped}' -> '{corrected}'")
        return corrected, True

    # === 策略2: 大小写不敏感精确匹配 ===
    model_lower = model_stripped.lower()

    # 2a. 标准ID大小写不敏感
    for mid in _VERIFIED_MODELS_CACHE:
        if mid.lower() == model_lower:
            if mid != model_stripped:
                logger.info(f"模型ID大小写修正: '{model_stripped}' -> '{mid}'")
                return mid, True
            return mid, False

    # 2b. 别名大小写不敏感
    for alias, standard_id in _MODEL_ALIAS_MAP.items():
        if alias.lower() == model_lower:
            logger.info(f"模型ID别名映射(大小写不敏感): '{model_stripped}' -> '{standard_id}'")
            return standard_id, True

    # === 策略3: 标准化精确匹配（去除所有分隔符）===
    model_norm = _normalize(model_stripped)

    if model_norm:
        # 3a. 标准ID的标准化形式
        for mid in _VERIFIED_MODELS_CACHE:
            if _normalize(mid) == model_norm:
                logger.info(f"模型ID标准化修正: '{model_stripped}' -> '{mid}'")
                return mid, True
            # 去掉 provider 前缀后的部分
            if "/" in mid:
                name_part = mid.split("/", 1)[1]
                if _normalize(name_part) == model_norm:
                    logger.info(f"模型ID名称匹配(无provider): '{model_stripped}' -> '{mid}'")
                    return mid, True

        # 3b. 别名的标准化形式
        for alias, standard_id in _MODEL_ALIAS_MAP.items():
            if _normalize(alias) == model_norm:
                logger.info(f"模型ID别名标准化映射: '{model_stripped}' -> '{standard_id}'")
                return standard_id, True

    # === 策略4: 模糊匹配（编辑距离相似度）===
    if model_norm and len(model_norm) >= 3:
        search_index = _build_search_index()

        # 计算相似度，取最佳匹配
        best_match = None
        best_ratio = 0.0
        threshold = 0.85

        for norm_key, standard_id in search_index:
            # 快速过滤：长度差异太大就跳过
            if abs(len(norm_key) - len(model_norm)) > max(len(model_norm), len(norm_key)) * 0.3:
                continue

            ratio = difflib.SequenceMatcher(None, model_norm, norm_key).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = standard_id

        if best_match and best_ratio >= threshold:
            logger.info(f"模型ID模糊匹配(ratio={best_ratio:.2f}): '{model_stripped}' -> '{best_match}'")
            return best_match, True

    # === 策略5: 子串包含匹配 ===
    if model_norm and len(model_norm) >= 4:
        # 用户输入是某个标准模型名称的子串
        candidates = []
        for mid in _VERIFIED_MODELS_CACHE:
            mid_norm = _normalize(mid)
            if "/" in mid:
                name_part = mid.split("/", 1)[1]
                name_norm = _normalize(name_part)
                if model_norm in name_norm and len(model_norm) >= len(name_norm) * 0.5:
                    candidates.append((mid, len(name_norm)))  # 优先选最短匹配

        if len(candidates) == 1:
            corrected = candidates[0][0]
            logger.info(f"模型ID子串匹配: '{model_stripped}' -> '{corrected}'")
            return corrected, True
        elif len(candidates) > 1:
            # 多个候选，选名称最短的（最精确匹配）
            candidates.sort(key=lambda x: x[1])
            corrected = candidates[0][0]
            logger.info(f"模型ID子串匹配(多候选选最短): '{model_stripped}' -> '{corrected}'")
            return corrected, True

    # === 策略6: 智能补全后缀 ===
    corrected = _try_smart_correction(model_stripped)
    if corrected != model_stripped:
        logger.info(f"模型ID智能补全: '{model_stripped}' -> '{corrected}'")
        return corrected, True

    # 无法修正，返回原值（由调用方决定是否放行）
    return model_stripped, False


def _try_smart_correction(model_id: str) -> str:
    """尝试智能补全常见简写缺失的 -instruct / -it 后缀"""
    known_providers = ["meta", "mistralai", "google", "nvidia", "deepseek-ai",
                       "qwen", "z-ai", "minimaxai", "stepfun-ai", "openai",
                       "abacusai", "bytedance", "upstage", "moonshotai",
                       "poolside", "sarvamai", "thinkingmachines", "nv-mistralai"]

    parts = model_id.split("/")
    if len(parts) == 2:
        provider, name = parts
        if provider.lower() in [p.lower() for p in known_providers]:
            need_instruct = ["llama", "mistral", "nemotron", "gemma", "qwen", "seed", "solar"]
            if any(kw in name.lower() for kw in need_instruct):
                if not any(name.lower().endswith(suffix) for suffix in ["-instruct", "-it", "-vl", "-v1", "-v2", "-v3", "-v4"]):
                    corrected_name = name + "-instruct"
                    test_id = f"{provider}/{corrected_name}"
                    verified = _get_verified_models()
                    if test_id in verified:
                        return test_id
                    test_id_it = f"{provider}/{name}-it"
                    if test_id_it in verified:
                        return test_id_it

    return model_id


def build_model_error_suggestion(model_id: str) -> str:
    """构建模型ID错误的友好提示"""
    if not model_id:
        return "模型ID不能为空"

    model_norm = _normalize(model_id)
    suggestions = []

    # 从标准模型列表中找最相似的
    for mid in _VERIFIED_MODELS_CACHE:
        mid_norm = _normalize(mid)
        ratio = difflib.SequenceMatcher(None, model_norm, mid_norm).ratio()
        if ratio >= 0.6:
            suggestions.append((mid, ratio))

    # 从别名中找
    for alias, standard_id in _MODEL_ALIAS_MAP.items():
        alias_norm = _normalize(alias)
        ratio = difflib.SequenceMatcher(None, model_norm, alias_norm).ratio()
        if ratio >= 0.6:
            suggestions.append((standard_id, ratio))

    if suggestions:
        suggestions.sort(key=lambda x: -x[1])
        uniq = list(dict.fromkeys([s[0] for s in suggestions]))[:3]
        return f"模型ID应为 {', '.join(uniq)}，请检查拼写"

    return f"未知模型ID '{model_id}'，请检查拼写或联系管理员"


# ============================================================
# 2. API Key 清洗与校验
# ============================================================

def clean_and_validate_api_key(raw_key: str) -> Tuple[str, Optional[str]]:
    """
    清洗并校验 API Key。
    返回 (cleaned_key, error_message)
    """
    if not raw_key:
        return "", "API密钥不能为空"

    cleaned = raw_key.strip().strip("'\"").strip()
    cleaned = re.sub(r'[\x00-\x1f\x7f]', '', cleaned)

    if cleaned.lower() in ("demo", "test", "example", "sk-demo", "sk-test", "sk-example", "test-key"):
        return cleaned, "API密钥不能使用测试值，请使用真实的上游密钥"

    if cleaned.lower().startswith("sk-demo") or cleaned.lower().startswith("sk-test"):
        return cleaned, "API密钥包含测试前缀，请使用真实的上游密钥"

    if len(cleaned) < 10:
        return cleaned, f"API密钥过短（{len(cleaned)}个字符），请检查是否完整复制"

    if " " in cleaned and len(cleaned.split()) > 1:
        first_part = cleaned.split()[0]
        logger.warning(f"API Key 包含空格，已自动截取: '{cleaned[:20]}...' -> '{first_part[:20]}...'")
        return clean_and_validate_api_key(first_part)

    return cleaned, None


# ============================================================
# 3. Base URL 路径标准化
# ============================================================

def normalize_base_url(raw_url: str) -> Tuple[str, Optional[str]]:
    """标准化 Base URL"""
    if not raw_url:
        return "", "Base URL 不能为空"

    url = raw_url.strip()

    if not url.startswith(("http://", "https://")):
        url = "https://" + url
        logger.info(f"Base URL 缺失协议，已自动补全: '{raw_url[:30]}' -> '{url[:30]}'")

    parsed = urlparse(url)
    path = parsed.path.rstrip("/")

    while "/v1/v1" in path:
        path = path.replace("/v1/v1", "/v1")

    if "/v1/" in path and path != "/v1":
        v1_idx = path.index("/v1/") + 3
        path = path[:v1_idx]
        logger.info(f"Base URL 路径过深，已截断: '{raw_url[:40]}' -> '{urlunparse(parsed._replace(path=path))[:40]}'")

    if not path or path == "":
        path = "/v1"
    elif not path.endswith("/v1"):
        if path.endswith("/v1/"):
            path = path[:-1]
        else:
            if "/v1" in path:
                idx = path.index("/v1")
                path = path[:idx + 3]
            else:
                path = path + "/v1"

    normalized = urlunparse(parsed._replace(path=path))
    warning = None
    if normalized != raw_url.strip():
        warning = f"Base URL 已自动标准化"

    return normalized, warning


# ============================================================
# 4. 请求体格式强制校验
# ============================================================

_MINIMUM_REQUIRED_FIELDS = {"role", "content"}


def validate_request_body(body: dict) -> Optional[str]:
    """校验请求体格式"""
    if not isinstance(body, dict):
        return "请求体必须是JSON对象"

    messages = body.get("messages")
    if messages is None:
        return "请求体缺少 'messages' 字段"
    if not isinstance(messages, list):
        return "'messages' 必须是数组格式，当前类型为 " + type(messages).__name__
    if len(messages) == 0:
        return "'messages' 数组不能为空，请至少包含一条消息"

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return f"messages[{i}] 必须是JSON对象"
        role = msg.get("role")
        if not role:
            return f"messages[{i}] 缺少 'role' 字段"
        if role not in ("system", "developer", "user", "assistant", "tool", "function"):
            return f"messages[{i}] 的 role '{role}' 不合法，应为 system/developer/user/assistant/tool/function"
        content = msg.get("content")
        has_tool_calls = "tool_calls" in msg
        if content is None and not has_tool_calls:
            return f"messages[{i}] 缺少 'content' 字段"
        if content is not None and not isinstance(content, (str, list)):
            return f"messages[{i}] 的 content 类型不合法，应为字符串或数组"
        if isinstance(content, str) and not content.strip() and role in ("system", "user"):
            return f"messages[{i}] 的 content 不能为空"
        if isinstance(content, list):
            for j, part in enumerate(content):
                if not isinstance(part, dict):
                    return f"messages[{i}].content[{j}] 必须是JSON对象"

    return None


# ============================================================
# 5. 参数容错与默认值
# ============================================================

_PARAM_RANGES = {
    "temperature": (0.0, 2.0, float),
    "top_p": (0.0, 1.0, float),
    "top_k": (1, 200, int),
    "max_tokens": (1, 131072, int),
    "frequency_penalty": (-2.0, 2.0, float),
    "presence_penalty": (-2.0, 2.0, float),
    "repetition_penalty": (0.0, 2.0, float),
    "seed": (0, 2**31 - 1, int),
}


def sanitize_parameters(body: dict) -> dict:
    """参数容错：自动截断超范围值，强制类型转换"""
    corrections = []

    for param, (min_val, max_val, expected_type) in _PARAM_RANGES.items():
        if param not in body:
            continue

        val = body[param]

        # 类型转换
        try:
            if expected_type == int:
                converted = int(float(val)) if val is not None else None
            else:
                converted = float(val) if val is not None else None
        except (ValueError, TypeError):
            corrections.append(f"{param}={val} 类型无效，已移除")
            del body[param]
            continue

        if converted is None:
            del body[param]
            continue

        # 范围截断
        original = converted
        if converted < min_val:
            converted = min_val
            corrections.append(f"{param}={original} 低于最小值，已修正为 {min_val}")
        elif converted > max_val:
            converted = max_val
            corrections.append(f"{param}={original} 超过最大值，已修正为 {max_val}")

        body[param] = converted

    # stream 强制转 bool
    if "stream" in body:
        stream_val = body["stream"]
        if isinstance(stream_val, str):
            body["stream"] = stream_val.lower() in ("true", "1", "yes")
        elif not isinstance(stream_val, bool):
            body["stream"] = bool(stream_val)

    # 消息角色标准化：developer -> system（兼容绝大多数第三方 OpenAI 兼容上游）
    if "messages" in body and isinstance(body["messages"], list):
        for msg in body["messages"]:
            if isinstance(msg, dict):
                if msg.get("role") == "developer":
                    msg["role"] = "system"
                # 多模态消息容错：清洗重复拼接的 data URL 前缀（如 data:image/png;base64,data:image/png;base64,...）
                content = msg.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "image_url":
                            img_info = part.get("image_url")
                            if isinstance(img_info, dict):
                                url = img_info.get("url")
                                if isinstance(url, str) and url:
                                    cleaned_url = url
                                    while re.match(r"^data:[^;]+;base64,data:[^;]+;base64,", cleaned_url, re.I):
                                        cleaned_url = re.sub(r"^data:[^;]+;base64,", "", cleaned_url, count=1, flags=re.I)
                                    if cleaned_url != url:
                                        img_info["url"] = cleaned_url
                                        corrections.append("image_url 重复 data: 前缀已规范化")

    # thinking 参数转换或剥离：OpenAI 兼容端点通常不接受顶层 thinking 参数，而接受 reasoning_effort
    if "thinking" in body:
        thinking_val = body.pop("thinking")
        if "reasoning_effort" not in body:
            if isinstance(thinking_val, dict) and thinking_val.get("type") in ("disabled", "none", False):
                body["reasoning_effort"] = "none"
            elif thinking_val is False:
                body["reasoning_effort"] = "none"
            else:
                body["reasoning_effort"] = "high"
        corrections.append("thinking 参数已映射至 reasoning_effort 并从顶层移除")

    # reasoning_effort 标准化容错（如 Codex 发送的 ultra/max 映射为上游 AMD 接受的合法枚举值）
    if "reasoning_effort" in body:
        effort_raw = str(body["reasoning_effort"]).strip().lower()
        effort_map = {
            "ultra": "xhigh",
            "max": "xhigh",
            "highest": "xhigh",
            "medium-high": "high",
        }
        valid_efforts = {"none", "minimal", "low", "medium", "high", "xhigh"}
        if effort_raw in effort_map:
            mapped_effort = effort_map[effort_raw]
            corrections.append(f"reasoning_effort={effort_raw} 映射为 {mapped_effort}")
            body["reasoning_effort"] = mapped_effort
        elif effort_raw not in valid_efforts and effort_raw != "max":
            corrections.append(f"reasoning_effort={effort_raw} 非法，已修正为 xhigh")
            body["reasoning_effort"] = "xhigh"
        else:
            body["reasoning_effort"] = effort_raw

    # max_completion_tokens 与 max_tokens 双向同步
    if "max_completion_tokens" in body and "max_tokens" not in body:
        body["max_tokens"] = body["max_completion_tokens"]
    elif "max_tokens" in body and "max_completion_tokens" not in body:
        body["max_completion_tokens"] = body["max_tokens"]

    # tools 格式容错与过滤（防止 AMD 等上游 Zod 模式校验对非 function/web_search 工具抛 400）
    if "tools" in body and isinstance(body["tools"], list):
        sanitized_tools = []
        for idx, tool in enumerate(body["tools"]):
            if not isinstance(tool, dict):
                continue
            t_type = tool.get("type")
            if t_type == "custom":
                # Responses/Codex 客户端可能发送 type: custom，转换为标准 function
                custom_data = tool.get("custom") if isinstance(tool.get("custom"), dict) else {}
                func_name = custom_data.get("name") or tool.get("name") or f"custom_tool_{idx}"
                func_desc = custom_data.get("description") or tool.get("description", "")
                func_params = custom_data.get("parameters") or tool.get("parameters")
                if func_name == "apply_patch" and (not func_params or not isinstance(func_params, dict) or not func_params.get("properties")):
                    func_params = {
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string",
                                "description": "The complete diff patch text starting with *** Begin Patch and ending with *** End Patch"
                            }
                        },
                        "required": ["input"]
                    }
                elif not func_params or not isinstance(func_params, dict):
                    func_params = {"type": "object", "properties": {}}
                sanitized_tools.append({
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "description": func_desc,
                        "parameters": func_params,
                    }
                })
                corrections.append(f"tools[{idx}] custom工具已转换为function工具")
            elif t_type == "function":
                if "function" in tool and isinstance(tool["function"], dict):
                    fn = dict(tool["function"])
                    if fn.get("name") == "apply_patch" and (not fn.get("parameters") or not isinstance(fn.get("parameters"), dict) or not fn["parameters"].get("properties")):
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
                    sanitized_tools.append({"type": "function", "function": fn})
                elif "name" in tool:
                    # 容错：顶层有 name 但缺少 function 外层的残缺结构
                    name = tool.get("name")
                    fn_params = tool.get("parameters")
                    if name == "apply_patch" and (not fn_params or not isinstance(fn_params, dict) or not fn_params.get("properties")):
                        fn_params = {
                            "type": "object",
                            "properties": {
                                "input": {
                                    "type": "string",
                                    "description": "The complete diff patch text starting with *** Begin Patch and ending with *** End Patch"
                                }
                            },
                            "required": ["input"]
                        }
                    elif not fn_params or not isinstance(fn_params, dict):
                        fn_params = {"type": "object", "properties": {}}
                    sanitized_tools.append({
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": tool.get("description", ""),
                            "parameters": fn_params,
                        }
                    })
            elif t_type == "web_search":
                sanitized_tools.append(tool)
            else:
                corrections.append(f"tools[{idx}] 未知类型 {t_type} 已跳过以防上游报错")
        body["tools"] = sanitized_tools
        if not sanitized_tools:
            del body["tools"]
            if "tool_choice" in body:
                del body["tool_choice"]

    if corrections:
        logger.info(f"参数容错修正: {'; '.join(corrections)}")

    return body


def validate_and_sanitize(body: dict) -> dict:
    """组合校验：请求体格式 + 参数容错 + Agent ReAct 协议注入"""
    error = validate_request_body(body)
    if error:
        raise HTTPException(status_code=400, detail={
            "message": error,
            "type": "invalid_request_error",
            "code": "invalid_body_format",
        })

    body = sanitize_parameters(body)

    # 注入 Agent ReAct 协议：防止模型在有 tools 时仅输出行动叙述而漏发工具调用
    try:
        try:
            from app.tool_call_parser import inject_agent_react_instructions
        except ImportError:
            from gateway.app.tool_call_parser import inject_agent_react_instructions
        if inject_agent_react_instructions:
            body = inject_agent_react_instructions(body)
    except Exception as e:
        logger.warning(f"注入 Agent ReAct 协议异常: {e}")

    return body
