"""
平台适配器初始化 - v9.0

在应用启动时注册所有平台适配器
"""
import logging

from app.platforms.base import PlatformAdapterRegistry
from app.platforms.nvidia import NvidiaAdapter
from app.platforms.openai import OpenAIAdapter

logger = logging.getLogger("acu.platform")


def register_all_adapters():
    """注册所有平台适配器"""
    PlatformAdapterRegistry.register("nvidia", NvidiaAdapter())
    PlatformAdapterRegistry.register("openai", OpenAIAdapter())
    logger.info(f"平台适配器注册完成: {PlatformAdapterRegistry.list_providers()}")
