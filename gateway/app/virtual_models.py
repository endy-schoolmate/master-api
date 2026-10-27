"""
虚拟聚合模型（Failover 链）

一个虚拟模型 ID 背后是一个"有序的真实模型链"。客户端只填这个虚拟 ID，
网关内部按 chain 顺序依次尝试真实模型：某个模型不可用或持续限速时，
自动无缝切换到链里的下一个，直到成功或全部失败。

配置存储：admin_settings 表的 `virtual_models` 键（JSON 数组），复用现有
get_setting/set_setting，无需新建表。带短 TTL 内存缓存降低 DB 压力。

切换速度与容错的平衡（本模块与 public_api._call_upstream_chain 共同实现）：

1. 错误分级重试（chain_attempts_for_kind）：
   - 超时/挂起类错误（timeout）只对同一候选试 1 次就切下一个候选——
     再等一轮完整超时只会成倍放大延迟，换模型才是出路；
   - 429/401/5xx 等瞬时错误仍保留 2 次机会（换 key 重试），容忍偶发抖动；
   - no_key（该模型当前没有可用密钥）直接换候选，且不计入模型失败——
     密钥暂时冷却 ≠ 模型坏了，避免误伤健康模型。

2. 候选超时预算（public_api 侧使用 CHAIN_CANDIDATE_TIMEOUT_SECONDS）：
   链中非末位候选的等待时间封顶（默认 60s），末位候选保留完整模型超时，
   保证"前面快速试错、最后兜底耐心"。

3. 自适应软熔断（升级冷却 + 试探恢复）：
   - 偶发失败（窗口内 < FAIL_THRESHOLD 次）不触发任何熔断，容忍抖动；
   - 触发熔断后冷却时长按连续触发次数指数升级（30s→60s→120s→…封顶），
     持续坏掉的模型被跳过得越来越久；
   - 冷却到期后进入"试探期"：允许请求通过，但试探期一失败立即再次熔断
     （不再累计窗口），好得快就快速回归，还没好就不浪费第二个请求。
"""
import json
import os
import time
import threading
import logging
from typing import Optional

from app.database import get_setting, set_setting

logger = logging.getLogger("acu.vmodel")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


# ========== 容错参数（默认值，可用环境变量覆盖） ==========

# 单个真实模型在一次请求内最多尝试的次数（含换 key 重试）。
# 比普通请求的 5 次更小，加快 chain 内切换节奏。
CHAIN_PER_MODEL_ATTEMPTS = 2

# 链中"非末位"候选的单次尝试预算（秒）：非流式为整个响应的读超时上限，
# 流式为"等到响应头"的预算。末位候选不受此限制，保留完整模型超时。
# 对于大上下文（如 Codex 100k+ tokens），首包预填充可能耗时 60~90s，默认调大至 180s。
CHAIN_CANDIDATE_TIMEOUT_SECONDS = _env_float("MASTER_API_CHAIN_CANDIDATE_TIMEOUT", 180.0)

# 流式候选"提交前探测"窗口（秒）：200 之后先缓冲流的前几个事件再决定提交，
# 覆盖聚合上游"先回200再在SSE里吐error"、"只发角色块后挂起"、"200后长时间
# 不出首块"三类故障。必须等到真实内容块（文本/推理/工具调用）才提交。
# 探测期内没等到即判失败并切换候选（不再让客户端挂到180s空闲超时）。
CHAIN_STREAM_PROBE_SECONDS = _env_float("MASTER_API_CHAIN_STREAM_PROBE", 45.0)

# 整链尝试预算（秒）：链开始后超过该时长，剩余"还有备选"的候选直接跳过，
# 末位兜底候选的等待也被压到剩余预算内（保底 CHAIN_FINAL_MIN_SECONDS）。
CHAIN_TOTAL_BUDGET_SECONDS = _env_float("MASTER_API_CHAIN_TOTAL_BUDGET", 360.0)

# 末位兜底候选的最短等待预算（秒）：即使整链预算耗尽，也保证兜底候选
# 至少有一次像样的尝试机会（而不是只给它 1~2s 直接判死）。
CHAIN_FINAL_MIN_SECONDS = _env_float("MASTER_API_CHAIN_FINAL_MIN", 60.0)

# ---- 错误分级 → 单模型内重试次数 ----
# timeout: 读/响应超时（上游挂起），等待成本高 → 立即切下一个候选；
# no_key:  调度器选不出密钥（密钥级冷却/无密钥），重选结果不变 → 直接换候选；
# 其余（429/401/403/5xx/网络/空响应等）：瞬时或密钥级错误，换 key 有意义 → 保持 2 次。
CHAIN_ATTEMPTS_BY_KIND = {
    "timeout": 1,
    "no_key": 1,
}

# 模型级软熔断：滑动窗口时长（秒，调大至600秒，适配LLM长请求场景）
FAIL_WINDOW_SECONDS = _env_float("MASTER_API_CHAIN_FAIL_WINDOW", 600.0)
# 窗口内失败次数达到此值 → 熔断（偶发失败低于该值不会触发任何切换惩罚）
FAIL_THRESHOLD = int(os.getenv("MASTER_API_CHAIN_FAIL_THRESHOLD", "3"))
# 连续失败熔断阈值（连续失败次数达到此值立即熔断，无需等待滑动窗口）
CONSECUTIVE_FAIL_THRESHOLD = int(os.getenv("MASTER_API_CHAIN_CONSECUTIVE_FAIL_THRESHOLD", "2"))
# 熔断时长：首次触发的基准值，之后按连续触发次数翻倍升级，封顶 COOLDOWN_MAX_SECONDS
COOLDOWN_BASE_SECONDS = _env_float("MASTER_API_CHAIN_COOLDOWN_BASE", 30.0)
COOLDOWN_MAX_SECONDS = _env_float("MASTER_API_CHAIN_COOLDOWN_MAX", 300.0)


def chain_attempts_for_kind(kind: str) -> int:
    """按错误分级返回该真实模型内允许的尝试次数（不超过 CHAIN_PER_MODEL_ATTEMPTS）。"""
    return CHAIN_ATTEMPTS_BY_KIND.get(kind, CHAIN_PER_MODEL_ATTEMPTS)

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
        {"id": "my-super", "display_name": "...", "chain": ["m1","m2"], "routing_strategy": "round-robin", "enabled": true}
    """
    now = time.time()
    with _lock:
        if use_cache and _cache["data"] is not None and _cache["expires"] > now:
            return _cache["data"]
        data = _load_raw()
        # 确保每个虚拟模型都具有 routing_strategy 字段（默认 round-robin 轮询）
        for item in data:
            if isinstance(item, dict) and "routing_strategy" not in item:
                item["routing_strategy"] = "round-robin"
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


def get_ordered_chain(model_id: str) -> list:
    """根据虚拟模型的路由策略（轮询 / 优先级 / 自适应）返回本次请求应尝试的候选链顺序。

    - round-robin: 平滑动态轮询（每次请求依次轮换起始模型，若故障顺延尝试链中剩余健康模型）；
    - priority: 主备优先故障转移（严格按配置顺序从第一个健康模型开始）；
    - adaptive: 健康感知自适应（优先按非熔断、低连续失败、低延迟排序）。
    """
    m = get_virtual_model(model_id)
    if not m or not m.get("enabled", True):
        return []
    raw_chain = [c for c in (m.get("chain") or []) if isinstance(c, str) and c.strip()]
    if not raw_chain:
        return []

    strategy = m.get("routing_strategy", "priority")

    if strategy == "round-robin":
        with _state_lock:
            idx = _rr_counters.get(model_id, 0)
            _rr_counters[model_id] = (idx + 1) % len(raw_chain)
        # 将候选列表以 idx 为起点平移轮询
        return raw_chain[idx:] + raw_chain[:idx]

    elif strategy == "adaptive":
        with _state_lock:
            now = time.time()
            def _model_rank(rm: str):
                is_cooling = 1 if _cooldown_until.get(rm, 0.0) > now else 0
                fails = _consecutive_fails.get(rm, 0)
                return (is_cooling, fails)
            return sorted(raw_chain, key=_model_rank)

    else:
        # 默认 priority: 严格按配置主备顺序
        return list(raw_chain)


# ========== 模型级软熔断与健康状态（内存，跨请求） ==========
#
# 状态机：健康 → (连续失败/严重错误/窗口累计) → 熔断 → (到期) → 试探期 → 成功:健康 / 失败:再熔断
# 冷却时长按连续触发次数升级：base × 2^(n-1)，封顶 MAX；任何一次成功全部清零。

_fail_history: dict[str, list[float]] = {}   # real_model -> [失败时间戳...]
_cooldown_until: dict[str, float] = {}        # real_model -> 熔断截止时间戳
_breach_count: dict[str, int] = {}            # real_model -> 连续熔断触发次数（升级用）
_probation: set[str] = set()                  # 试探期中的模型（冷却到期后首次回归）
_consecutive_fails: dict[str, int] = {}       # real_model -> 连续失败次数
_last_fail_reason: dict[str, str] = {}        # real_model -> 最近一次失败原因
_rr_counters: dict[str, int] = {}             # virtual_model_id -> 轮询指针
_state_lock = threading.RLock()


def _escalated_cooldown(real_model: str, now: float) -> float:
    """按连续触发次数计算本次熔断时长（指数升级，封顶）。调用方需持锁。"""
    breaches = _breach_count.get(real_model, 0)
    seconds = min(COOLDOWN_BASE_SECONDS * (2 ** breaches), COOLDOWN_MAX_SECONDS)
    _breach_count[real_model] = breaches + 1
    return seconds


def is_model_cooling(real_model: str) -> bool:
    """该真实模型当前是否在软熔断中（应跳过）。

    熔断到期的模型转入"试探期"并放行：试探成功即完全恢复，
    试探失败由 record_model_failure 立即再次熔断，不浪费重试窗口。
    """
    with _state_lock:
        until = _cooldown_until.get(real_model, 0.0)
        if until <= 0:
            return False
        if time.time() >= until:
            # 熔断到期：进入试探期，放行但保持监视
            _cooldown_until.pop(real_model, None)
            _probation.add(real_model)
            logger.info(f"虚拟模型链: 真实模型 '{real_model}' 熔断到期，进入试探期")
            return False
        return True


def cooldown_remaining(real_model: str) -> float:
    """返回剩余熔断秒数（未熔断为 0）。"""
    with _state_lock:
        until = _cooldown_until.get(real_model, 0.0)
        rem = until - time.time()
        return rem if rem > 0 else 0.0


def record_model_failure(real_model: str, reason: str = "", is_severe: bool = False) -> bool:
    """记录一次真实模型失败。达到阈值（或试探期失败、严重错误、连续失败达到阈值）则触发熔断。

    返回 True 表示本次触发了熔断。
    """
    now = time.time()
    with _state_lock:
        _consecutive_fails[real_model] = _consecutive_fails.get(real_model, 0) + 1
        if reason:
            _last_fail_reason[real_model] = str(reason)[:200]

        should_trip = False
        # 1. 试探期失败：模型还是坏的，立即再次熔断（时长升级），不走窗口累计
        if real_model in _probation:
            _probation.discard(real_model)
            should_trip = True
        # 2. 严重错误（如 SSLError、网络中断、503无渠道、502 stream error等）：立即熔断
        elif is_severe:
            should_trip = True
        # 3. 连续失败次数达到阈值（默认 2 次）：立即熔断
        elif _consecutive_fails[real_model] >= CONSECUTIVE_FAIL_THRESHOLD:
            should_trip = True
        else:
            # 4. 滑动窗口累计
            hist = _fail_history.setdefault(real_model, [])
            hist.append(now)
            cutoff = now - FAIL_WINDOW_SECONDS
            hist[:] = [t for t in hist if t >= cutoff]
            if len(hist) >= FAIL_THRESHOLD:
                should_trip = True

        if should_trip:
            seconds = _escalated_cooldown(real_model, now)
            _cooldown_until[real_model] = now + seconds
            _fail_history.pop(real_model, None)
            logger.warning(
                f"虚拟模型链: 真实模型 '{real_model}' 触发软熔断 {seconds:.0f}s "
                f"（连续失败 {_consecutive_fails[real_model]} 次, 原因={reason or 'failure'}, "
                f"第 {_breach_count[real_model]} 次触发）"
            )
            return True
    return False


def record_model_success(real_model: str) -> None:
    """记录一次成功：清空失败历史并解除熔断/试探状态。"""
    with _state_lock:
        _fail_history.pop(real_model, None)
        _cooldown_until.pop(real_model, None)
        _breach_count.pop(real_model, None)
        _probation.discard(real_model)
        _consecutive_fails[real_model] = 0
        _last_fail_reason.pop(real_model, None)


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


def get_model_health_stats() -> dict:
    """返回所有真实模型的健康诊断状态（熔断剩余、连续失败、最近错误）。"""
    now = time.time()
    with _state_lock:
        out = {}
        all_models = set(_cooldown_until.keys()) | set(_consecutive_fails.keys()) | set(_last_fail_reason.keys())
        for m in all_models:
            rem = max(0.0, _cooldown_until.get(m, 0.0) - now)
            out[m] = {
                "cooling": rem > 0,
                "cooldown_remaining": round(rem, 1),
                "consecutive_fails": _consecutive_fails.get(m, 0),
                "last_reason": _last_fail_reason.get(m, ""),
                "probation": m in _probation,
            }
        return out
