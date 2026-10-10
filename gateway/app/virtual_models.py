"""
虚拟聚合模型（Failover 链）

一个虚拟模型 ID 背后是一个"有序的真实模型链"。客户端只填这个虚拟 ID，
网关内部按 chain 顺序依次尝试真实模型：某个模型不可用或持续限速时，
自动无缝切换到链里的下一个，直到成功或全部失败。

配置存储：admin_settings 表的 `virtual_models` 键（JSON 数组），复用现有
get_setting/set_setting，无需新建表。带短 TTL 内存缓存降低 DB 压力。

限速容错（模型级软熔断，跨请求生效）：
- 记录每个真实模型最近的失败时间戳（滑动窗口）
- 窗口内失败次数达阈值 → 熔断该模型一段时间，后续请求直接跳过
- 成功一次即清零并解除熔断
这样偶发 429 不会乱切，真正持续限速时才快速跳过。
"""
import json
import time
import threading
import logging
from typing import Optional

from app.database import get_setting, set_setting

logger = logging.getLogger("acu.vmodel")

# ========== 容错参数（默认值，可按需调整） ==========

# 单个真实模型在一次请求内最多尝试的次数（含换 key 重试）。
# 比普通请求的 5 次更小，加快 chain 内切换节奏。
CHAIN_PER_MODEL_ATTEMPTS = 2

# 模型级软熔断：滑动窗口时长（秒）
FAIL_WINDOW_SECONDS = 60.0
# 窗口内失败次数达到此值 → 熔断
FAIL_THRESHOLD = 3
# 熔断时长（秒）：期间该真实模型在 chain 展开时被跳过
MODEL_COOLDOWN_SECONDS = 90.0

# ========== 配置读写（带 TTL 缓存） ==========

_SETTING_KEY = "virtual_models"
_cache = {"data": None, "expires": 0.0}
_cache_ttl = 10.0
_lock = threading.RLock()


def _load_raw() -> list:
    """从 admin_settings 读取原始虚拟模型配置数组。"""
    raw = get_setting(_SETTING_KEY)
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning(f"解析 virtual_models 配置失败: {e}")
        return []


def get_virtual_models(use_cache: bool = True) -> list:
    """返回所有虚拟模型配置。

    每项形如：
        {"id": "my-super", "display_name": "...", "chain": ["m1","m2"], "enabled": true}
    """
    now = time.time()
    with _lock:
        if use_cache and _cache["data"] is not None and _cache["expires"] > now:
            return _cache["data"]
        data = _load_raw()
        _cache["data"] = data
        _cache["expires"] = now + _cache_ttl
        return data


def _invalidate_cache() -> None:
    with _lock:
        _cache["data"] = None
        _cache["expires"] = 0.0


def save_virtual_models(models: list) -> None:
    """整体保存虚拟模型配置并失效缓存。"""
    set_setting(_SETTING_KEY, json.dumps(models, ensure_ascii=False))
    _invalidate_cache()


def get_virtual_model(vid: str) -> Optional[dict]:
    """按 id 获取单个虚拟模型配置（不区分 enabled）。"""
    if not vid:
        return None
    for m in get_virtual_models():
        if m.get("id") == vid:
            return m
    return None


def is_virtual_model(model_id: str) -> bool:
    """判断给定 model_id 是否是已启用的虚拟模型。"""
    if not model_id:
        return False
    for m in get_virtual_models():
        if m.get("id") == model_id and m.get("enabled", True):
            return True
    return False


def get_chain(model_id: str) -> list:
    """返回虚拟模型的真实模型链（有序）。非虚拟模型返回空列表。"""
    m = get_virtual_model(model_id)
    if not m or not m.get("enabled", True):
        return []
    chain = m.get("chain") or []
    return [c for c in chain if isinstance(c, str) and c.strip()]


# ========== 模型级软熔断状态（内存，跨请求） ==========

_fail_history: dict[str, list[float]] = {}   # real_model -> [失败时间戳...]
_cooldown_until: dict[str, float] = {}        # real_model -> 熔断截止时间戳
_state_lock = threading.RLock()


def is_model_cooling(real_model: str) -> bool:
    """该真实模型当前是否在软熔断中（应跳过）。"""
    with _state_lock:
        until = _cooldown_until.get(real_model, 0.0)
        if until <= 0:
            return False
        if time.time() >= until:
            # 熔断到期，清理
            _cooldown_until.pop(real_model, None)
            return False
        return True


def cooldown_remaining(real_model: str) -> float:
    """返回剩余熔断秒数（未熔断为 0）。"""
    with _state_lock:
        until = _cooldown_until.get(real_model, 0.0)
        rem = until - time.time()
        return rem if rem > 0 else 0.0


def record_model_failure(real_model: str) -> bool:
    """记录一次真实模型失败。若达到阈值则触发熔断。

    返回 True 表示本次触发了熔断。
    """
    now = time.time()
    with _state_lock:
        hist = _fail_history.setdefault(real_model, [])
        hist.append(now)
        # 清理窗口外的
        cutoff = now - FAIL_WINDOW_SECONDS
        hist[:] = [t for t in hist if t >= cutoff]
        if len(hist) >= FAIL_THRESHOLD:
            _cooldown_until[real_model] = now + MODEL_COOLDOWN_SECONDS
            hist.clear()
            logger.warning(
                f"虚拟模型链: 真实模型 '{real_model}' {FAIL_WINDOW_SECONDS:.0f}s 内失败达 "
                f"{FAIL_THRESHOLD} 次，软熔断 {MODEL_COOLDOWN_SECONDS:.0f}s"
            )
            return True
    return False


def record_model_success(real_model: str) -> None:
    """记录一次成功：清空失败历史并解除熔断。"""
    with _state_lock:
        _fail_history.pop(real_model, None)
        _cooldown_until.pop(real_model, None)


def get_cooldown_snapshot() -> dict:
    """返回当前所有熔断中的真实模型及剩余秒数（供管理台展示）。"""
    now = time.time()
    with _state_lock:
        out = {}
        for m, until in list(_cooldown_until.items()):
            rem = until - now
            if rem > 0:
                out[m] = round(rem, 1)
            else:
                _cooldown_until.pop(m, None)
        return out
