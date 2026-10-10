"""
NVIDIA NIM 模型目录（仅包含实测可用模型）

基于 build.nvidia.com 2026年数据，经过实际API调用验证
模型经过筛选，仅包含实测可用或通过 chat/completions 接口可调用的模型
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class NIMModelInfo:
    """NVIDIA NIM 模型信息"""
    model_id: str  # e.g., "nvidia/nemotron-3-ultra-550b-a55b"
    display_name: str  # e.g., "Nemotron-3-Ultra-550B"
    publisher: str  # e.g., "nvidia", "meta", "deepseek"
    context_length: int  # Max context tokens
    max_output_tokens: int  # Max output tokens
    supports_streaming: bool
    supports_tools: bool  # Function calling
    supports_images: bool  # Vision
    model_family: str  # e.g., "nemotron", "llama", "deepseek"
    description: str = ""
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


# NVIDIA NIM 实测可用模型目录 (经过实际chat/completions调用验证)
NIM_MODEL_CATALOG: Dict[str, NIMModelInfo] = {
    # =========================================================================
    # z-ai
    # =========================================================================
    "z-ai/glm-5.2": NIMModelInfo(
        model_id="z-ai/glm-5.2",
        display_name="GLM-5.2",
        publisher="z-ai",
        context_length=131072,
        max_output_tokens=16384,
        supports_streaming=True,
        supports_tools=True,
        supports_images=False,
        model_family="glm",
        description="GLM-5.2 flagship LLM for agentic workflows, coding, and long-horizon reasoning tasks.",
        tags=["agentic-ai"],
    ),

    # =========================================================================
    # deepseek-ai
    # =========================================================================
    "deepseek-ai/deepseek-v4-flash": NIMModelInfo(
        model_id="deepseek-ai/deepseek-v4-flash",
        display_name="DeepSeek-V4-Flash",
        publisher="deepseek-ai",
        context_length=1048576,
        max_output_tokens=16384,
        supports_streaming=True,
        supports_tools=True,
        supports_images=False,
        model_family="deepseek",
        description="284B MoE with 1M-token context optimized for fast coding and agents.",
        tags=["moe"],
    ),
    "deepseek-ai/deepseek-v4-pro": NIMModelInfo(
        model_id="deepseek-ai/deepseek-v4-pro",
        display_name="DeepSeek-V4-Pro",
        publisher="deepseek-ai",
        context_length=1048576,
        max_output_tokens=32768,
        supports_streaming=True,
        supports_tools=True,
        supports_images=False,
        model_family="deepseek",
        description="DeepSeek V4 scales to 1M-token context with efficient MoE architecture.",
        tags=["moe"],
    ),

    # =========================================================================
    # minimaxai
    # =========================================================================
    "minimaxai/minimax-m3": NIMModelInfo(
        model_id="minimaxai/minimax-m3",
        display_name="MiniMax-M3",
        publisher="minimaxai",
        context_length=1048576,
        max_output_tokens=16384,
        supports_streaming=True,
        supports_tools=True,
        supports_images=True,
        model_family="minimax",
        description="Multimodal MoE vision-language model with strong reasoning, coding, and tool-calling.",
        tags=["coding"],
    ),
    "minimaxai/minimax-m2.7": NIMModelInfo(
        model_id="minimaxai/minimax-m2.7",
        display_name="MiniMax-M2.7",
        publisher="minimaxai",
        context_length=131072,
        max_output_tokens=16384,
        supports_streaming=True,
        supports_tools=False,
        supports_images=False,
        model_family="minimax",
        description="230B-parameter text-to-text AI model excelling in coding, reasoning, and office tasks.",
        tags=["reasoning"],
    ),

    # =========================================================================
    # stepfun-ai
    # =========================================================================
    "stepfun-ai/step-3.7-flash": NIMModelInfo(
        model_id="stepfun-ai/step-3.7-flash",
        display_name="Step-3.7-Flash",
        publisher="stepfun-ai",
        context_length=131072,
        max_output_tokens=16384,
        supports_streaming=True,
        supports_tools=True,
        supports_images=True,
        model_family="step",
        description="Sparse MoE multimodal reasoning model for enterprise, agentic and coding.",
        tags=["coding"],
    ),
    "stepfun-ai/step-3.5-flash": NIMModelInfo(
        model_id="stepfun-ai/step-3.5-flash",
        display_name="Step-3.5-Flash",
        publisher="stepfun-ai",
        context_length=131072,
        max_output_tokens=16384,
        supports_streaming=True,
        supports_tools=True,
        supports_images=False,
        model_family="step",
        description="200B open-source reasoning engine with sparse MoE powering frontier agentic AI.",
        tags=["agentic"],
    ),

    # =========================================================================
    # qwen
    # =========================================================================
    "qwen/qwen3.5-397b-a17b": NIMModelInfo(
        model_id="qwen/qwen3.5-397b-a17b",
        display_name="Qwen3.5-397B",
        publisher="qwen",
        context_length=131072,
        max_output_tokens=16384,
        supports_streaming=True,
        supports_tools=True,
        supports_images=True,
        model_family="qwen",
        description="Next-gen Qwen 3.5 VLM (400B MoE) brings advanced vision, chat, RAG, and agentic capabilities.",
        tags=["moe"],
    ),
    "qwen/qwen3-next-80b-a3b-instruct": NIMModelInfo(
        model_id="qwen/qwen3-next-80b-a3b-instruct",
        display_name="Qwen3-Next-80B",
        publisher="qwen",
        context_length=131072,
        max_output_tokens=16384,
        supports_streaming=True,
        supports_tools=False,
        supports_images=False,
        model_family="qwen",
        description="Qwen3-Next Instruct blends hybrid attention, sparse MoE, and stability boosts for ultra-long context AI.",
        tags=["text-generation"],
    ),

    # v10.0: DeepSeek Coder
    "deepseek-ai/deepseek-coder-6.7b-instruct": NIMModelInfo(
        model_id="deepseek-ai/deepseek-coder-6.7b-instruct",
        display_name="DeepSeek-Coder-6.7B",
        publisher="deepseek-ai",
        context_length=32768,
        max_output_tokens=8192,
        supports_streaming=True,
        supports_tools=False,
        supports_images=False,
        model_family="deepseek",
        description="Code-specialized LLM with 32K context for code generation, completion, and debugging.",
        tags=["code"],
    ),

    # v10.0: Google Gemma 3
    "google/gemma-3-12b-it": NIMModelInfo(
        model_id="google/gemma-3-12b-it",
        display_name="Gemma-3-12B",
        publisher="google",
        context_length=32768,
        max_output_tokens=8192,
        supports_streaming=True,
        supports_tools=False,
        supports_images=False,
        model_family="gemma",
        description="Lightweight instruction-tuned model for text generation.",
        tags=["text-generation"],
    ),
    "google/gemma-3-4b-it": NIMModelInfo(
        model_id="google/gemma-3-4b-it",
        display_name="Gemma-3-4B",
        publisher="google",
        context_length=32768,
        max_output_tokens=8192,
        supports_streaming=True,
        supports_tools=False,
        supports_images=False,
        model_family="gemma",
        description="Compact instruction-tuned model for efficient deployment.",
        tags=["text-generation"],
    ),

    # v10.0: Meta Llama 4
    "meta/llama-4-maverick-17b-128e-instruct": NIMModelInfo(
        model_id="meta/llama-4-maverick-17b-128e-instruct",
        display_name="Llama-4-Maverick-17B",
        publisher="meta",
        context_length=262144,
        max_output_tokens=8192,
        supports_streaming=True,
        supports_tools=True,
        supports_images=False,
        model_family="llama",
        description="MoE instruction model for dialogue use cases, 17B active / 128B total.",
        tags=["moe"],
    ),

    # v10.0: MoonshotAI Kimi
    "moonshotai/kimi-k2.6": NIMModelInfo(
        model_id="moonshotai/kimi-k2.6",
        display_name="Kimi-K2.6",
        publisher="moonshotai",
        context_length=131072,
        max_output_tokens=16384,
        supports_streaming=True,
        supports_tools=True,
        supports_images=False,
        model_family="kimi",
        description="MoE reasoning model optimized for complex problem-solving.",
        tags=["reasoning"],
    ),

    # v10.0: nv-mistralai
    "nv-mistralai/mistral-nemo-12b-instruct": NIMModelInfo(
        model_id="nv-mistralai/mistral-nemo-12b-instruct",
        display_name="Mistral-NeMo-12B",
        publisher="nv-mistralai",
        context_length=128000,
        max_output_tokens=8192,
        supports_streaming=True,
        supports_tools=True,
        supports_images=False,
        model_family="mistral",
        description="NVIDIA-Mistral collaboration model excelling in reasoning, coding, and multilingual tasks.",
        tags=["text-generation"],
    ),

    # =========================================================================
    # openai
    # =========================================================================
    "openai/gpt-oss-120b": NIMModelInfo(
        model_id="openai/gpt-oss-120b",
        display_name="GPT-OSS-120B",
        publisher="openai",
        context_length=131072,
        max_output_tokens=16384,
        supports_streaming=True,
        supports_tools=False,
        supports_images=False,
        model_family="gpt",
        description="MoE reasoning LLM (text-only) designed to fit within 80GB GPU.",
        tags=["reasoning"],
    ),
    "openai/gpt-oss-20b": NIMModelInfo(
        model_id="openai/gpt-oss-20b",
        display_name="GPT-OSS-20B",
        publisher="openai",
        context_length=131072,
        max_output_tokens=16384,
        supports_streaming=True,
        supports_tools=False,
        supports_images=False,
        model_family="gpt",
        description="Smaller MoE text-only LLM for efficient AI reasoning and math.",
        tags=["reasoning"],
    ),

    # =========================================================================
    # meta
    # =========================================================================
    "meta/llama-3.1-70b-instruct": NIMModelInfo(
        model_id="meta/llama-3.1-70b-instruct",
        display_name="Llama-3.1-70B",
        publisher="meta",
        context_length=131072,
        max_output_tokens=16384,
        supports_streaming=True,
        supports_tools=True,
        supports_images=False,
        model_family="llama",
        description="Complex conversations with superior contextual understanding, reasoning and text generation.",
        tags=["chat"],
    ),
    "meta/llama-3.1-8b-instruct": NIMModelInfo(
        model_id="meta/llama-3.1-8b-instruct",
        display_name="Llama-3.1-8B",
        publisher="meta",
        context_length=131072,
        max_output_tokens=8192,
        supports_streaming=True,
        supports_tools=True,
        supports_images=False,
        model_family="llama",
        description="Advanced state-of-the-art model with language understanding, superior reasoning, and text generation.",
        tags=["chat"],
    ),
    "meta/llama-3.2-90b-vision-instruct": NIMModelInfo(
        model_id="meta/llama-3.2-90b-vision-instruct",
        display_name="Llama-3.2-90B-Vision",
        publisher="meta",
        context_length=131072,
        max_output_tokens=16384,
        supports_streaming=True,
        supports_tools=True,
        supports_images=True,
        model_family="llama",
        description="Cutting-edge vision-language model excelling in high-quality reasoning from images.",
        tags=["image-text-retrieval"],
    ),
    "meta/llama-3.2-11b-vision-instruct": NIMModelInfo(
        model_id="meta/llama-3.2-11b-vision-instruct",
        display_name="Llama-3.2-11B-Vision",
        publisher="meta",
        context_length=131072,
        max_output_tokens=8192,
        supports_streaming=True,
        supports_tools=True,
        supports_images=True,
        model_family="llama",
        description="Cutting-edge vision-language model excelling in high-quality reasoning from images.",
        tags=["image-text-retrieval"],
    ),
    "meta/llama-3.2-3b-instruct": NIMModelInfo(
        model_id="meta/llama-3.2-3b-instruct",
        display_name="Llama-3.2-3B",
        publisher="meta",
        context_length=131072,
        max_output_tokens=4096,
        supports_streaming=True,
        supports_tools=False,
        supports_images=False,
        model_family="llama",
        description="Advanced small language model with language understanding, superior reasoning, and text generation.",
        tags=["language-generation"],
    ),
    "meta/llama-3.2-1b-instruct": NIMModelInfo(
        model_id="meta/llama-3.2-1b-instruct",
        display_name="Llama-3.2-1B",
        publisher="meta",
        context_length=131072,
        max_output_tokens=4096,
        supports_streaming=True,
        supports_tools=False,
        supports_images=False,
        model_family="llama",
        description="Advanced small language model with language understanding, superior reasoning, and text generation.",
        tags=["language-generation"],
    ),
    "meta/llama-3.3-70b-instruct": NIMModelInfo(
        model_id="meta/llama-3.3-70b-instruct",
        display_name="Llama-3.3-70B",
        publisher="meta",
        context_length=131072,
        max_output_tokens=16384,
        supports_streaming=True,
        supports_tools=True,
        supports_images=False,
        model_family="llama",
        description="Advanced LLM for reasoning, math, general knowledge, and function calling.",
        tags=["chat", "advanced-reasoning"],
    ),
    "meta/llama-guard-4-12b": NIMModelInfo(
        model_id="meta/llama-guard-4-12b",
        display_name="Llama-Guard-4-12B",
        publisher="meta",
        context_length=131072,
        max_output_tokens=4096,
        supports_streaming=True,
        supports_tools=False,
        supports_images=True,
        model_family="llama-guard",
        description="Multi-modal model to classify safety for input prompts as well output responses.",
        tags=["llm-multimodal-safety"],
    ),

    # =========================================================================
    # google
    # =========================================================================
    "google/gemma-4-31b-it": NIMModelInfo(
        model_id="google/gemma-4-31b-it",
        display_name="Gemma-4-31B",
        publisher="google",
        context_length=131072,
        max_output_tokens=16384,
        supports_streaming=True,
        supports_tools=True,
        supports_images=False,
        model_family="gemma",
        description="Dense 31B model delivering frontier reasoning for coding, agentic workflows.",
        tags=["reasoning"],
    ),
    "google/diffusiongemma-26b-a4b-it": NIMModelInfo(
        model_id="google/diffusiongemma-26b-a4b-it",
        display_name="DiffusionGemma-26B",
        publisher="google",
        context_length=8192,
        max_output_tokens=4096,
        supports_streaming=True,
        supports_tools=False,
        supports_images=False,
        model_family="diffusiongemma",
        description="Diffusion-based 26B parameter LLM enabling parallel token generation.",
        tags=["diffusion-llm"],
    ),
    "google/gemma-3n-e4b-it": NIMModelInfo(
        model_id="google/gemma-3n-e4b-it",
        display_name="Gemma-3n-E4B",
        publisher="google",
        context_length=8192,
        max_output_tokens=4096,
        supports_streaming=True,
        supports_tools=False,
        supports_images=True,
        model_family="gemma",
        description="Edge computing AI model which accepts text, audio and image input.",
        tags=["deprecated"],
    ),
    "google/gemma-3n-e2b-it": NIMModelInfo(
        model_id="google/gemma-3n-e2b-it",
        display_name="Gemma-3n-E2B",
        publisher="google",
        context_length=8192,
        max_output_tokens=4096,
        supports_streaming=True,
        supports_tools=False,
        supports_images=True,
        model_family="gemma",
        description="Edge computing AI model which accepts text, audio and image input.",
        tags=["deprecated"],
    ),
    "google/gemma-2-2b-it": NIMModelInfo(
        model_id="google/gemma-2-2b-it",
        display_name="Gemma-2-2B",
        publisher="google",
        context_length=8192,
        max_output_tokens=4096,
        supports_streaming=True,
        supports_tools=False,
        supports_images=False,
        model_family="gemma",
        description="Gemma 2 2B instruct model for lightweight text generation tasks.",
        tags=["chat", "deprecated"],
    ),

    # =========================================================================
    # mistralai
    # =========================================================================
    "mistralai/mistral-large-3-675b-instruct-2512": NIMModelInfo(
        model_id="mistralai/mistral-large-3-675b-instruct-2512",
        display_name="Mistral-Large-3-675B",
        publisher="mistralai",
        context_length=131072,
        max_output_tokens=16384,
        supports_streaming=True,
        supports_tools=True,
        supports_images=True,
        model_family="mistral",
        description="State-of-the-art general purpose MoE VLM ideal for chat, agentic and instruction based use cases.",
        tags=["language-generation"],
    ),
    "mistralai/mistral-small-4-119b-2603": NIMModelInfo(
        model_id="mistralai/mistral-small-4-119b-2603",
        display_name="Mistral-Small-4-119B",
        publisher="mistralai",
        context_length=256000,
        max_output_tokens=16384,
        supports_streaming=True,
        supports_tools=True,
        supports_images=True,
        model_family="mistral",
        description="Hybrid MoE model unifying instruct, reasoning, and coding with multimodal input and 256k context.",
        tags=["code-generation"],
    ),
    "mistralai/mistral-medium-3.5-128b": NIMModelInfo(
        model_id="mistralai/mistral-medium-3.5-128b",
        display_name="Mistral-Medium-3.5-128B",
        publisher="mistralai",
        context_length=131072,
        max_output_tokens=16384,
        supports_streaming=True,
        supports_tools=True,
        supports_images=False,
        model_family="mistral",
        description="High performing model for text generation, coding and agentic use cases.",
        tags=["coding"],
    ),
    "mistralai/mistral-nemotron": NIMModelInfo(
        model_id="mistralai/mistral-nemotron",
        display_name="Mistral-Nemotron",
        publisher="mistralai",
        context_length=131072,
        max_output_tokens=16384,
        supports_streaming=True,
        supports_tools=True,
        supports_images=False,
        model_family="mistral",
        description="Built for agentic workflows, excels in coding, instruction following, and function calling.",
        tags=["language-generation"],
    ),
    "mistralai/mixtral-8x7b-instruct-v0.1": NIMModelInfo(
        model_id="mistralai/mixtral-8x7b-instruct-v0.1",
        display_name="Mixtral-8x7B",
        publisher="mistralai",
        context_length=32768,
        max_output_tokens=4096,
        supports_streaming=True,
        supports_tools=True,
        supports_images=False,
        model_family="mixtral",
        description="Mixtral 8x7B instruct model with function calling support.",
        tags=["chat", "tool-use"],
    ),

    # =========================================================================
    # nvidia - 对话模型
    # =========================================================================
    "nvidia/nemotron-3-ultra-550b-a55b": NIMModelInfo(
        model_id="nvidia/nemotron-3-ultra-550b-a55b",
        display_name="Nemotron-3-Ultra-550B",
        publisher="nvidia",
        context_length=1048576,
        max_output_tokens=32768,
        supports_streaming=True,
        supports_tools=True,
        supports_images=False,
        model_family="nemotron",
        description="Hybrid Mamba-Transformer MoE with 1M context, agentic reasoning, coding, planning, tool calling.",
        tags=["agent"],
    ),
    "nvidia/nemotron-3-super-120b-a12b": NIMModelInfo(
        model_id="nvidia/nemotron-3-super-120b-a12b",
        display_name="Nemotron-3-Super-120B",
        publisher="nvidia",
        context_length=1048576,
        max_output_tokens=32768,
        supports_streaming=True,
        supports_tools=True,
        supports_images=False,
        model_family="nemotron",
        description="Hybrid Mamba-Transformer MoE with Multi-Token Prediction, 1M context.",
        tags=["moe"],
    ),
    "nvidia/nemotron-3-nano-30b-a3b": NIMModelInfo(
        model_id="nvidia/nemotron-3-nano-30b-a3b",
        display_name="Nemotron-3-Nano-30B",
        publisher="nvidia",
        context_length=1048576,
        max_output_tokens=32768,
        supports_streaming=True,
        supports_tools=True,
        supports_images=False,
        model_family="nemotron",
        description="Open efficient MoE model with 1M context, excelling in coding, reasoning, instruction following, tool calling.",
        tags=["moe"],
    ),
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning": NIMModelInfo(
        model_id="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        display_name="Nemotron-3-Nano-Omni-30B-Reasoning",
        publisher="nvidia",
        context_length=256000,
        max_output_tokens=20480,
        supports_streaming=True,
        supports_tools=True,
        supports_images=True,
        model_family="nemotron",
        description="Omni-modal reasoning model that understands images, video, speech, text.",
        tags=["image-to-text", "reasoning"],
    ),
    "nvidia/llama-3.3-nemotron-super-49b-v1.5": NIMModelInfo(
        model_id="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        display_name="Llama-3.3-Nemotron-Super-49B-v1.5",
        publisher="nvidia",
        context_length=131072,
        max_output_tokens=16384,
        supports_streaming=True,
        supports_tools=True,
        supports_images=False,
        model_family="nemotron",
        description="High efficiency model with leading accuracy for reasoning, tool calling, chat, and instruction following.",
        tags=["advanced-reasoning"],
    ),
    "nvidia/llama-3.3-nemotron-super-49b-v1": NIMModelInfo(
        model_id="nvidia/llama-3.3-nemotron-super-49b-v1",
        display_name="Llama-3.3-Nemotron-Super-49B",
        publisher="nvidia",
        context_length=131072,
        max_output_tokens=16384,
        supports_streaming=True,
        supports_tools=True,
        supports_images=False,
        model_family="nemotron",
        description="Llama 3.3 Nemotron Super 49B model for chat and reasoning.",
        tags=["reasoning"],
    ),
    "nvidia/llama-3.1-nemotron-nano-vl-8b-v1": NIMModelInfo(
        model_id="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
        display_name="Llama-3.1-Nemotron-Nano-VL-8B",
        publisher="nvidia",
        context_length=131072,
        max_output_tokens=8192,
        supports_streaming=True,
        supports_tools=True,
        supports_images=True,
        model_family="nemotron",
        description="Multi-modal vision-language model that understands text/img and creates informative responses.",
        tags=["doc-intelligence"],
    ),
    "nvidia/nemotron-nano-12b-v2-vl": NIMModelInfo(
        model_id="nvidia/nemotron-nano-12b-v2-vl",
        display_name="Nemotron-Nano-12B-v2-VL",
        publisher="nvidia",
        context_length=32768,
        max_output_tokens=8192,
        supports_streaming=True,
        supports_tools=False,
        supports_images=True,
        model_family="nemotron",
        description="Nemotron Nano 12B v2 VL enables multi-image and video understanding, along with visual Q&A and summarization.",
        tags=["language-generation"],
    ),
    "nvidia/nvidia-nemotron-nano-9b-v2": NIMModelInfo(
        model_id="nvidia/nvidia-nemotron-nano-9b-v2",
        display_name="Nemotron-Nano-9B-v2",
        publisher="nvidia",
        context_length=131072,
        max_output_tokens=8192,
        supports_streaming=True,
        supports_tools=True,
        supports_images=False,
        model_family="nemotron",
        description="High-efficiency LLM with hybrid Transformer-Mamba design, excelling in reasoning and agentic tasks.",
        tags=["thinking-budget"],
    ),
    "nvidia/nemotron-mini-4b-instruct": NIMModelInfo(
        model_id="nvidia/nemotron-mini-4b-instruct",
        display_name="Nemotron-Mini-4B",
        publisher="nvidia",
        context_length=4096,
        max_output_tokens=4096,
        supports_streaming=True,
        supports_tools=True,
        supports_images=False,
        model_family="nemotron",
        description="Optimized SLM for on-device inference and fine-tuned for roleplay, RAG and function calling.",
        tags=["chat"],
    ),

    # v10.0: NVIDIA 新增对话模型
    "nvidia/llama-3.1-nemotron-51b-instruct": NIMModelInfo(
        model_id="nvidia/llama-3.1-nemotron-51b-instruct",
        display_name="Llama-3.1-Nemotron-51B",
        publisher="nvidia",
        context_length=131072,
        max_output_tokens=16384,
        supports_streaming=True,
        supports_tools=True,
        supports_images=False,
        model_family="nemotron",
        description="Efficient 51B LLM with FP6 quantization for reasoning and chat.",
        tags=["chat"],
    ),
    "nvidia/llama-3.1-nemotron-70b-instruct": NIMModelInfo(
        model_id="nvidia/llama-3.1-nemotron-70b-instruct",
        display_name="Llama-3.1-Nemotron-70B",
        publisher="nvidia",
        context_length=131072,
        max_output_tokens=16384,
        supports_streaming=True,
        supports_tools=True,
        supports_images=False,
        model_family="nemotron",
        description="Nemotron-tuned Llama 3.1 70B for instruction following and dialogue.",
        tags=["chat"],
    ),
    "nvidia/llama-3.1-nemotron-ultra-253b-v1": NIMModelInfo(
        model_id="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        display_name="Nemotron-Ultra-253B",
        publisher="nvidia",
        context_length=131072,
        max_output_tokens=16384,
        supports_streaming=True,
        supports_tools=True,
        supports_images=False,
        model_family="nemotron",
        description="Flagship Nemotron model with exceptional reasoning and coding capabilities.",
        tags=["chat", "reasoning"],
    ),
    "nvidia/llama-3.1-nemotron-nano-8b-v1": NIMModelInfo(
        model_id="nvidia/llama-3.1-nemotron-nano-8b-v1",
        display_name="Nemotron-Nano-8B-v1",
        publisher="nvidia",
        context_length=131072,
        max_output_tokens=16384,
        supports_streaming=True,
        supports_tools=True,
        supports_images=False,
        model_family="nemotron",
        description="Compact 8B Nemotron model for lightweight inference.",
        tags=["chat"],
    ),
    "nvidia/nemotron-nano-3-30b-a3b": NIMModelInfo(
        model_id="nvidia/nemotron-nano-3-30b-a3b",
        display_name="Nemotron-Nano-3-30B",
        publisher="nvidia",
        context_length=259072,
        max_output_tokens=32768,
        supports_streaming=True,
        supports_tools=True,
        supports_images=False,
        model_family="nemotron",
        description="Third-gen Nemotron Nano, 30B MoE with 259K context.",
        tags=["moe"],
    ),

    # =========================================================================
    # nvidia - 安全/检测模型
    # =========================================================================
    "nvidia/llama-3.1-nemoguard-8b-content-safety": NIMModelInfo(
        model_id="nvidia/llama-3.1-nemoguard-8b-content-safety",
        display_name="Llama-3.1-NemoGuard-8B-Content-Safety",
        publisher="nvidia",
        context_length=131072,
        max_output_tokens=4096,
        supports_streaming=True,
        supports_tools=False,
        supports_images=False,
        model_family="nemoguard",
        description="Content safety guard model based on Llama 3.1 8B for detecting unsafe content.",
        tags=["safety"],
    ),
    "nvidia/llama-3.1-nemoguard-8b-topic-control": NIMModelInfo(
        model_id="nvidia/llama-3.1-nemoguard-8b-topic-control",
        display_name="Llama-3.1-NemoGuard-8B-Topic-Control",
        publisher="nvidia",
        context_length=131072,
        max_output_tokens=4096,
        supports_streaming=True,
        supports_tools=False,
        supports_images=False,
        model_family="nemoguard",
        description="Topic control guard model based on Llama 3.1 8B for steering conversations.",
        tags=["safety"],
    ),
    "nvidia/llama-3.1-nemotron-safety-guard-8b-v3": NIMModelInfo(
        model_id="nvidia/llama-3.1-nemotron-safety-guard-8b-v3",
        display_name="Llama-3.1-Nemotron-Safety-Guard-8B-v3",
        publisher="nvidia",
        context_length=131072,
        max_output_tokens=4096,
        supports_streaming=True,
        supports_tools=False,
        supports_images=False,
        model_family="nemotron",
        description="Safety guard model based on Llama 3.1 Nemotron 8B v3 for content moderation.",
        tags=["safety"],
    ),
    "nvidia/nemotron-3.5-content-safety": NIMModelInfo(
        model_id="nvidia/nemotron-3.5-content-safety",
        display_name="Nemotron-3.5-Content-Safety",
        publisher="nvidia",
        context_length=131072,
        max_output_tokens=4096,
        supports_streaming=True,
        supports_tools=False,
        supports_images=True,
        model_family="nemotron",
        description="Multilingual multimodal model for detecting unsafe and toxic content.",
        tags=["llm-safety"],
    ),
    "nvidia/gliner-pii": NIMModelInfo(
        model_id="nvidia/gliner-pii",
        display_name="GLiNER-PII",
        publisher="nvidia",
        context_length=4096,
        max_output_tokens=1024,
        supports_streaming=True,
        supports_tools=False,
        supports_images=False,
        model_family="gliner",
        description="GLiNER model for personally identifiable information detection.",
        tags=["ner", "pii"],
    ),
    "nvidia/ising-calibration-1-35b-a3b": NIMModelInfo(
        model_id="nvidia/ising-calibration-1-35b-a3b",
        display_name="Ising-Calibration-1-35B",
        publisher="nvidia",
        context_length=32768,
        max_output_tokens=4096,
        supports_streaming=True,
        supports_tools=False,
        supports_images=False,
        model_family="ising",
        description="Ising-based calibration model with 35B parameters (3B active).",
        tags=["calibration"],
    ),

    # =========================================================================
    # nvidia - 翻译
    # =========================================================================
    "nvidia/riva-translate-4b-instruct-v1.1": NIMModelInfo(
        model_id="nvidia/riva-translate-4b-instruct-v1.1",
        display_name="Riva-Translate-4B-v1.1",
        publisher="nvidia",
        context_length=4096,
        max_output_tokens=4096,
        supports_streaming=True,
        supports_tools=False,
        supports_images=False,
        model_family="riva",
        description="Riva Translate 4B instruct v1.1 model for translation tasks.",
        tags=["translation"],
    ),

    # v10.0: Riva Translate 原版
    "nvidia/riva-translate-4b-instruct": NIMModelInfo(
        model_id="nvidia/riva-translate-4b-instruct",
        display_name="Riva-Translate-4B",
        publisher="nvidia",
        context_length=2048,
        max_output_tokens=512,
        supports_streaming=False,
        supports_tools=False,
        supports_images=False,
        model_family="riva",
        description="Riva Translate 4B instruct model for translation tasks.",
        tags=["translation"],
    ),

    # v10.0: NVIDIA VILA
    "nvidia/vila": NIMModelInfo(
        model_id="nvidia/vila",
        display_name="VILA",
        publisher="nvidia",
        context_length=8192,
        max_output_tokens=4096,
        supports_streaming=True,
        supports_tools=False,
        supports_images=True,
        model_family="nvidia",
        description="Vision-language model for visual reasoning and image understanding.",
        tags=["vision"],
    ),

    # v10.0: Nemotron-Parse
    "nvidia/nemotron-parse": NIMModelInfo(
        model_id="nvidia/nemotron-parse",
        display_name="Nemotron-Parse",
        publisher="nvidia",
        context_length=131072,
        max_output_tokens=16384,
        supports_streaming=True,
        supports_tools=False,
        supports_images=True,
        model_family="nvidia",
        description="Multi-modal model for parsing complex documents, tables, and charts.",
        tags=["vision", "ocr"],
    ),

    # =========================================================================
    # 其他厂商
    # =========================================================================
    "abacusai/dracarys-llama-3.1-70b-instruct": NIMModelInfo(
        model_id="abacusai/dracarys-llama-3.1-70b-instruct",
        display_name="Dracarys-Llama-3.1-70B",
        publisher="abacusai",
        context_length=131072,
        max_output_tokens=16384,
        supports_streaming=True,
        supports_tools=True,
        supports_images=False,
        model_family="dracarys",
        description="Fine-tuned Llama 3.1 70B model for code generation, summarization.",
        tags=["deprecated"],
    ),
    "bytedance/seed-oss-36b-instruct": NIMModelInfo(
        model_id="bytedance/seed-oss-36b-instruct",
        display_name="Seed-OSS-36B",
        publisher="bytedance",
        context_length=131072,
        max_output_tokens=16384,
        supports_streaming=True,
        supports_tools=True,
        supports_images=False,
        model_family="seed",
        description="ByteDance open-source LLM with long-context, reasoning, and agentic intelligence.",
        tags=["thinking-budget", "deprecated"],
    ),
    "poolside/laguna-xs-2.1": NIMModelInfo(
        model_id="poolside/laguna-xs-2.1",
        display_name="Laguna-XS-2.1",
        publisher="poolside",
        context_length=32768,
        max_output_tokens=8192,
        supports_streaming=True,
        supports_tools=True,
        supports_images=False,
        model_family="laguna",
        description="Poolside Laguna XS 2.1 code model for software engineering tasks.",
        tags=["coding"],
    ),
    "sarvamai/sarvam-m": NIMModelInfo(
        model_id="sarvamai/sarvam-m",
        display_name="Sarvam-M",
        publisher="sarvamai",
        context_length=131072,
        max_output_tokens=8192,
        supports_streaming=True,
        supports_tools=True,
        supports_images=False,
        model_family="sarvam",
        description="Multilingual, hybrid-reasoning model optimized for Indian language tasks, programming, mathematical reasoning.",
        tags=["coding"],
    ),
    "thinkingmachines/inkling": NIMModelInfo(
        model_id="thinkingmachines/inkling",
        display_name="Inkling",
        publisher="thinkingmachines",
        context_length=32768,
        max_output_tokens=4096,
        supports_streaming=True,
        supports_tools=False,
        supports_images=False,
        model_family="inkling",
        description="Inkling model for specialized text generation tasks.",
        tags=["chat"],
    ),
    "upstage/solar-10.7b-instruct": NIMModelInfo(
        model_id="upstage/solar-10.7b-instruct",
        display_name="Solar-10.7B",
        publisher="upstage",
        context_length=4096,
        max_output_tokens=4096,
        supports_streaming=True,
        supports_tools=False,
        supports_images=False,
        model_family="solar",
        description="Solar 10.7B instruct model for text generation.",
        tags=["chat", "deprecated"],
    ),
}


def get_model_info(model_id: str) -> Optional[NIMModelInfo]:
    """获取模型信息"""
    return NIM_MODEL_CATALOG.get(model_id)


def list_models(publisher: str = None, family: str = None, supports_tools: bool = None) -> List[NIMModelInfo]:
    """列出可用模型"""
    models = list(NIM_MODEL_CATALOG.values())
    if publisher:
        models = [m for m in models if m.publisher == publisher]
    if family:
        models = [m for m in models if m.model_family == family]
    if supports_tools is not None:
        models = [m for m in models if m.supports_tools == supports_tools]
    return models


def get_openai_models_list() -> list:
    """生成 OpenAI /v1/models 兼容格式的模型列表"""
    return [
        {
            "id": model.model_id,
            "object": "model",
            "created": 1700000000,
            "owned_by": model.publisher,
            "permission": [],
        }
        for model in NIM_MODEL_CATALOG.values()
    ]


def get_model_sort_priority() -> dict:
    """模型排序优先级 (用于前端展示排序, 仅包含实测可用的热门模型)"""
    priorities = {}
    for i, model_id in enumerate([
        "deepseek-ai/deepseek-v4-flash",
        "deepseek-ai/deepseek-v4-pro",
        "nvidia/nemotron-3-ultra-550b-a55b",
        "z-ai/glm-5.2",
        "nvidia/nemotron-3-super-120b-a12b",
        "minimaxai/minimax-m3",
        "stepfun-ai/step-3.7-flash",
        "qwen/qwen3.5-397b-a17b",
        "mistralai/mistral-large-3-675b-instruct-2512",
        "openai/gpt-oss-120b",
        "google/gemma-4-31b-it",
        "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        "nvidia/nvidia-nemotron-nano-9b-v2",
        "mistralai/mistral-small-4-119b-2603",
        "minimaxai/minimax-m2.7",
        "meta/llama-3.1-70b-instruct",
        "meta/llama-3.2-90b-vision-instruct",
        "stepfun-ai/step-3.5-flash",
        "openai/gpt-oss-20b",
        "moonshotai/kimi-k2.6",                 # v10.0
        "meta/llama-4-maverick-17b-128e-instruct",  # v10.0
        "deepseek-ai/deepseek-coder-6.7b-instruct", # v10.0
        "google/gemma-3-12b-it",               # v10.0
        "google/gemma-3-4b-it",                # v10.0
        "nv-mistralai/mistral-nemo-12b-instruct",   # v10.0
        "nvidia/llama-3.1-nemotron-51b-instruct",   # v10.0
        "nvidia/llama-3.1-nemotron-70b-instruct",   # v10.0
        "nvidia/llama-3.1-nemotron-ultra-253b-v1",  # v10.0
        "nvidia/llama-3.1-nemotron-nano-8b-v1",     # v10.0
        "nvidia/nemotron-nano-3-30b-a3b",           # v10.0
        "nvidia/riva-translate-4b-instruct",        # v10.0
        "nvidia/vila",                              # v10.0
        "nvidia/nemotron-parse",                    # v10.0
    ]):
        priorities[model_id] = i
    return priorities
