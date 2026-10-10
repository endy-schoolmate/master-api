"""
多级缓存架构 - v9.0

- L1: 进程内热点缓存（Python dict + LRU）
- L2: 本地过期缓存（TTL）
- L3: 分布式缓存（Redis，可选）

使用场景：
- API Key 认证结果缓存
- 模型列表缓存
- 调度决策缓存
- 速率限制计数
"""
import time
import logging
from collections import OrderedDict
from typing import Any, Optional, Callable, Dict

logger = logging.getLogger("acu.cache")


class LRUCache:
    """
    L1 热点缓存 - LRU淘汰策略

    适用于高频读取、容忍短时间不一致的数据
    如：API Key 认证结果、模型列表
    """

    def __init__(self, maxsize: int = 1024, default_ttl: float = 60.0):
        self._maxsize = maxsize
        self._default_ttl = default_ttl
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        """获取缓存项（访问即刷新LRU位置）"""
        if key not in self._cache:
            return None
        value, expires = self._cache[key]
        if time.time() > expires:
            del self._cache[key]
            return None
        # 移动到末尾（最近使用）
        self._cache.move_to_end(key)
        return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """设置缓存项"""
        expires = time.time() + (ttl if ttl is not None else self._default_ttl)
        self._cache[key] = (value, expires)
        self._cache.move_to_end(key)
        # LRU淘汰
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def delete(self, key: str):
        """删除缓存项"""
        self._cache.pop(key, None)

    def clear(self):
        """清空缓存"""
        self._cache.clear()

    def size(self) -> int:
        """当前缓存项数"""
        return len(self._cache)

    def cleanup_expired(self):
        """清理过期项"""
        now = time.time()
        expired = [k for k, (_, expires) in self._cache.items() if now > expires]
        for k in expired:
            del self._cache[k]
        if expired:
            logger.debug(f"L1缓存清理: 移除{len(expired)}个过期项")


class TTLCache:
    """
    L2 本地过期缓存

    适用于中等频率读取、可接受短时间过期的数据
    如：调度决策、密钥缓存、上游配置
    """

    def __init__(self, default_ttl: float = 300.0):
        self._default_ttl = default_ttl
        self._cache: Dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        if key not in self._cache:
            return None
        value, expires = self._cache[key]
        if time.time() > expires:
            del self._cache[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """设置缓存项"""
        expires = time.time() + (ttl if ttl is not None else self._default_ttl)
        self._cache[key] = (value, expires)

    def get_or_set(self, key: str, factory: Callable[[], Any], ttl: Optional[float] = None) -> Any:
        """获取或设置（缓存穿透保护）"""
        value = self.get(key)
        if value is not None:
            return value
        value = factory()
        self.set(key, value, ttl)
        return value

    def delete(self, key: str):
        """删除缓存项"""
        self._cache.pop(key, None)

    def clear(self):
        """清空缓存"""
        self._cache.clear()

    def cleanup_expired(self):
        """清理过期项"""
        now = time.time()
        expired = [k for k, (_, expires) in self._cache.items() if now > expires]
        for k in expired:
            del self._cache[k]

    def size(self) -> int:
        return len(self._cache)


class MultiLevelCache:
    """
    多级缓存管理器

    自动使用 L1 → L2 两级缓存，L3 (Redis) 可选。
    读取顺序：L1 → L2 → 数据源
    写入顺序：同时写入 L1 + L2
    """

    def __init__(self):
        self.l1 = LRUCache(maxsize=1024, default_ttl=60.0)   # API Key认证、模型列表
        self.l2 = TTLCache(default_ttl=300.0)  # 调度决策、密钥缓存
        self._stats = {"l1_hits": 0, "l2_hits": 0, "misses": 0}

    def get(self, key: str, level: int = 2) -> Optional[Any]:
        """
        获取缓存
        level=1: 仅L1
        level=2: L1 → L2
        """
        # L1
        value = self.l1.get(key)
        if value is not None:
            self._stats["l1_hits"] += 1
            return value

        if level == 1:
            self._stats["misses"] += 1
            return None

        # L2
        value = self.l2.get(key)
        if value is not None:
            self._stats["l2_hits"] += 1
            # 提升到L1
            self.l1.set(key, value)
            return value

        self._stats["misses"] += 1
        return None

    def set(self, key: str, value: Any, l1_ttl: Optional[float] = None, l2_ttl: Optional[float] = None):
        """写入所有缓存层级"""
        self.l1.set(key, value, l1_ttl)
        self.l2.set(key, value, l2_ttl)

    def delete(self, key: str):
        """删除所有缓存层级"""
        self.l1.delete(key)
        self.l2.delete(key)

    def clear(self):
        """清空所有缓存"""
        self.l1.clear()
        self.l2.clear()

    def cleanup(self):
        """清理过期项"""
        self.l1.cleanup_expired()
        self.l2.cleanup_expired()

    def get_stats(self) -> dict:
        """获取缓存统计"""
        total = self._stats["l1_hits"] + self._stats["l2_hits"] + self._stats["misses"]
        hit_rate = ((self._stats["l1_hits"] + self._stats["l2_hits"]) / total * 100) if total > 0 else 0
        return {
            "l1_hits": self._stats["l1_hits"],
            "l2_hits": self._stats["l2_hits"],
            "misses": self._stats["misses"],
            "hit_rate": round(hit_rate, 2),
            "l1_size": self.l1.size(),
            "l2_size": self.l2.size(),
            "total": total,
        }


# 全局实例
_cache: Optional[MultiLevelCache] = None


def get_cache() -> MultiLevelCache:
    """获取全局缓存实例"""
    global _cache
    if _cache is None:
        _cache = MultiLevelCache()
    return _cache
