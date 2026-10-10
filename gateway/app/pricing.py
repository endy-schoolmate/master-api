"""
模型成本定价与计费计算模块
Master API Gateway - Pricing Engine
"""
from __future__ import annotations

import time
import logging
from typing import Optional, Dict, Any, List, Tuple
from app.database import fetch_all, fetch_one, execute, localnow

logger = logging.getLogger("acu.pricing")

# 内存缓存，避免高并发下每次请求重复查询数据库
_pricing_cache: Optional[List[Dict[str, Any]]] = None
_pricing_cache_time: float = 0.0
_CACHE_TTL = 30.0  # 30秒自动过期刷新


def invalidate_pricing_cache() -> None:
    """清除定价缓存"""
    global _pricing_cache, _pricing_cache_time
    _pricing_cache = None
    _pricing_cache_time = 0.0


def get_all_pricings(force_refresh: bool = False) -> List[Dict[str, Any]]:
    """获取所有模型定价配置规则"""
    global _pricing_cache, _pricing_cache_time
    now = time.time()
    if not force_refresh and _pricing_cache is not None and (now - _pricing_cache_time < _CACHE_TTL):
        return _pricing_cache

    try:
        rows = fetch_all(
            "SELECT id, model_id, display_name, provider, currency, "
            "input_cost, output_cost, cache_read_cost, cache_write_cost, "
            "created_at, updated_at FROM model_pricing ORDER BY updated_at DESC"
        )
        _pricing_cache = rows or []
        _pricing_cache_time = now
        return _pricing_cache
    except Exception as e:
        logger.error(f"Failed to fetch model pricing from database: {e}")
        return _pricing_cache or []


def find_pricing_rule(model: str) -> Optional[Dict[str, Any]]:
    """
    根据模型名称查找最匹配的定价规则。
    支持：
    1. 精确匹配：model == rule.model_id
    2. 带 provider 前缀匹配：'hcnsec::longcat-2.5' -> 'longcat-2.5'
    3. 不带 provider 前缀匹配：'longcat-2.5' -> 'hcnsec::longcat-2.5'
    4. 大小写不敏感匹配
    """
    if not model:
        return None

    rules = get_all_pricings()
    if not rules:
        return None

    model_clean = model.strip()
    model_lower = model_clean.lower()
    
    # 拆分 provider 前缀，例如 'hcnsec::longcat-2.5' -> ('hcnsec', 'longcat-2.5')
    if "::" in model_clean:
        prefix, sub_model = model_clean.split("::", 1)
    else:
        prefix, sub_model = "", model_clean

    # 1. 精确匹配
    for r in rules:
        if r["model_id"] == model_clean:
            return r

    # 2. sub_model 精确匹配
    if sub_model and sub_model != model_clean:
        for r in rules:
            if r["model_id"] == sub_model:
                return r

    # 3. 忽略大小写匹配
    for r in rules:
        rid = (r["model_id"] or "").strip().lower()
        if rid == model_lower:
            return r
        if sub_model and rid == sub_model.lower():
            return r
        if "::" in rid:
            _, r_sub = rid.split("::", 1)
            if r_sub == model_lower or (sub_model and r_sub == sub_model.lower()):
                return r

    return None


def calculate_request_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> Dict[str, Any]:
    """
    计算单个请求的费用。
    价格单位为每百万 tokens (USD / CNY)。
    """
    rule = find_pricing_rule(model)
    if not rule:
        return {
            "cost": 0.0,
            "currency": "USD",
            "matched_model": "",
            "input_cost": 0.0,
            "output_cost": 0.0,
            "cache_read_cost": 0.0,
            "cache_write_cost": 0.0,
        }

    input_price = float(rule.get("input_cost") or 0.0)
    output_price = float(rule.get("output_cost") or 0.0)
    cache_read_price = float(rule.get("cache_read_cost") or 0.0)
    cache_write_price = float(rule.get("cache_write_cost") or 0.0)
    currency = (rule.get("currency") or "USD").upper()

    prompt_toks = max(0, int(prompt_tokens or 0))
    comp_toks = max(0, int(completion_tokens or 0))
    cached_toks = max(0, int(cached_tokens or 0))
    cache_write_toks = max(0, int(cache_write_tokens or 0))

    # 计算输入未命中和命中缓存的 token
    if cached_toks > 0 and prompt_toks >= cached_toks:
        normal_prompt_toks = prompt_toks - cached_toks
    else:
        normal_prompt_toks = prompt_toks

    cost_input = normal_prompt_toks * (input_price / 1_000_000.0)
    cost_output = comp_toks * (output_price / 1_000_000.0)
    cost_cache_read = cached_toks * (cache_read_price / 1_000_000.0)
    cost_cache_write = cache_write_toks * (cache_write_price / 1_000_000.0)

    total_cost = cost_input + cost_output + cost_cache_read + cost_cache_write

    return {
        "cost": round(total_cost, 6),
        "currency": currency,
        "matched_model": rule.get("model_id", ""),
        "input_cost": round(cost_input, 6),
        "output_cost": round(cost_output, 6),
        "cache_read_cost": round(cost_cache_read, 6),
        "cache_write_cost": round(cost_cache_write, 6),
    }


def calculate_aggregated_cost(model_group_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    根据按模型分组的 token 聚合结果计算总消费，分别统计 USD 和 CNY。
    """
    total_usd = 0.0
    total_cny = 0.0

    for row in model_group_rows:
        model = row.get("model") or ""
        p_toks = int(row.get("prompt_tokens") or 0)
        c_toks = int(row.get("completion_tokens") or 0)
        cached_toks = int(row.get("cached_tokens") or 0)
        cache_write_toks = int(row.get("cache_write_tokens") or 0)

        cost_info = calculate_request_cost(
            model=model,
            prompt_tokens=p_toks,
            completion_tokens=c_toks,
            cached_tokens=cached_toks,
            cache_write_tokens=cache_write_toks,
        )
        cost_val = cost_info["cost"]
        curr = cost_info["currency"]
        if curr == "CNY":
            total_cny += cost_val
        else:
            total_usd += cost_val

    # 格式化展示字符串
    usd_str = f"${total_usd:.4f}"
    cny_str = f"¥{total_cny:.4f}"

    if total_usd > 0 and total_cny > 0:
        display = f"{usd_str} / {cny_str}"
    elif total_cny > 0:
        display = cny_str
    else:
        display = usd_str

    return {
        "usd": round(total_usd, 4),
        "cny": round(total_cny, 4),
        "usd_formatted": usd_str,
        "cny_formatted": cny_str,
        "display": display,
    }
