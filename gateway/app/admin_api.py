"""
管理后台API路由 - 管理功能（个人自用精简版）

端点分组：
- 管理员认证: /gw/admin/login
- 上游密钥管理: /gw/admin/upstreams
- 下游客户管理: /gw/admin/clients
- 仪表盘: /gw/admin/dashboard
- 请求日志: /gw/admin/request-logs
- 审计日志: /gw/admin/audit-logs
- 全局状态: /gw/admin/global-status
- 接口调试: /gw/admin/debug/test
"""
import json
import os
import bcrypt
import time
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List

from fastapi import APIRouter, Request, HTTPException, Depends, Query
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, ConfigDict
import httpx

from app.database import (
    init_db, fetch_one, fetch_all, execute, get_setting, set_setting,
    insert_audit, utcnow, localnow, today_start_utc, today_start_local, days_ago_utc,
    cleanup_success_logs,
)
from app.security import (
    encrypt_upstream_key, decrypt_upstream_key, generate_platform_key,
    encrypt_secret, decrypt_secret, hash_secret, mask_secret,
    create_admin_token, verify_admin_token,
)
from app.scheduler import get_scheduler, get_threshold_for_model
from app.public_api import _VERIFIED_WORKING_MODELS
from app.middleware import (
    _login_rate_limiter, _admin_rate_limiter, get_client_ip,
)
from app.provider_probe import ProviderProbe, probe_provider, split_provider_model_ref, normalize_pasted_key
from app.provider_presets import list_provider_presets, PRESET_SOURCE_URL, DEFAULT_CHATGPT_MODELS
from app.responses_bridge import OAUTH_PROVIDER_ID
from app.openai_oauth import (
    start_login, complete_with_paste, session_status, pop_completed_session,
    parse_envelope, ensure_fresh_access, redacted_envelope, envelope_for_storage,
    get_completed_access, get_oauth_proxy, OAuthError,
)

logger = logging.getLogger("acu.admin")

router = APIRouter(prefix="/gw/admin")

def _load_admin_password_hash() -> str:
    """Resolve the bcrypt admin password hash from env or .env file.

    Loads `D:/master API/.env` (project root) when the env var is absent or
    truncated; defends against launchers (e.g. ``restart.bat``) that lose the
    leading ``$2b`` bcrypt salt prefix when exporting entries to ``os.environ``.
    """
    value = (os.environ.get("MASTER_ADMIN_PASSWORD_HASH") or "").strip()
    if value and value.startswith("$2"):
        return value
    # Fall back to reading the raw .env line so a truncated/lost env value still
    # produces the original 60-char bcrypt hash.
    try:
        from pathlib import Path
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"
        if env_path.exists():
            for raw in env_path.read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if raw.startswith("#") or "=" not in raw:
                    continue
                key, _, val = raw.partition("=")
                if key.strip() == "MASTER_ADMIN_PASSWORD_HASH":
                    val = val.strip().strip('"').strip("'")
                    if val and val.startswith("$2"):
                        os.environ["MASTER_ADMIN_PASSWORD_HASH"] = val
                        return val
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("无法从 .env 读取管理员密码哈希: %s", exc)
    if value and value.startswith("$2"):
        return value
    return value


ADMIN_PASSWORD_HASH = _load_admin_password_hash()
if not ADMIN_PASSWORD_HASH or not ADMIN_PASSWORD_HASH.startswith("$2"):
    raise RuntimeError(
        "[FATAL] 环境变量 MASTER_ADMIN_PASSWORD_HASH 未设置或被截断！"
        "请检查 .env 中 MASTER_ADMIN_PASSWORD_HASH= 行以 $2 开头且为 60 字符。"
    )


# ========== 请求模型 ==========

class LoginRequest(BaseModel):
    password: str
    model_config = ConfigDict(extra="ignore")

class UpstreamCreateRequest(BaseModel):
    name: str
    api_key: str = ""          # auth_type=oauth（ChatGPT 账号登录）时留空，凭据由 oauth_state_id 带入
    provider: str = "nvidia"
    base_url: str = ""  # 多 provider 改造：允许该 key 自定上游地址；空串表示走适配器默认
    weight: int = 1
    rpm_limit: int = 40
    switch_threshold: int = 38
    # 创建上游时可同时声明它能跑哪些 model；后端会把每个 model 关联到
    # 此 provider，并允许其它上游同时关联同一个 model。
    models: List[str] = []
    # auth_type: api_key（默认，传统密钥）| oauth（ChatGPT 账号登录，凭据走 oauth_state_id）
    auth_type: str = "api_key"
    oauth_state_id: Optional[str] = None

class UpstreamModelsRequest(BaseModel):
    """保存某供应商的模型清单，并同步维护 model_to_provider 映射"""
    models: List[str]

class UpstreamTestRequest(BaseModel):
    """测试供应商连通性的可选覆盖参数（页面保存前可临时填一组临时 base_url/key 试连）"""
    base_url: Optional[str] = None  # 留空则用 DB 里此 key 已存的 base_url
    api_key: Optional[str] = None   # 留空则用 DB 里此 key 已存的明文（解密）
    model: Optional[str] = None     # 指定则测试单个模型（chat/completions），否则 GET /models


class UpstreamTestAdhocRequest(BaseModel):
    """无 id 的临时连通性测试（新建供应商保存前用）：直接用表单里的 base_url + api_key 探测。"""
    base_url: str
    api_key: str = ""
    provider: str = "openai"
    model: Optional[str] = None     # 指定则测试单个模型（chat/completions），否则 GET /models
    oauth_state_id: Optional[str] = None  # OAuth（ChatGPT 账号登录）新建：用已完成登录会话的凭据测试

class UpstreamUpdateRequest(BaseModel):
    name: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None   # 填写则替换现有密钥；留空/省略表示不修改
    provider: Optional[str] = None
    weight: Optional[int] = None
    rpm_limit: Optional[int] = None
    switch_threshold: Optional[int] = None
    status: Optional[str] = None

class ClientCreateRequest(BaseModel):
    name: str
    user_type: str = "old"

class ClientUpdateRequest(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None
    user_type: Optional[str] = None

class DebugChatRequest(BaseModel):
    api_key: str
    model: str
    messages: list
    stream: Optional[bool] = False
    model_config = ConfigDict(extra="allow")


class VirtualModelCreateRequest(BaseModel):
    """新建虚拟聚合模型（failover 链）"""
    id: str                          # 虚拟模型 ID（客户端填这个）
    display_name: str = ""           # 展示名（可选）
    chain: List[str] = []            # 有序真实模型链
    enabled: bool = True


class VirtualModelUpdateRequest(BaseModel):
    """更新虚拟聚合模型（字段均可选，只改传入项；new_id 用于改名）"""
    new_id: Optional[str] = None
    display_name: Optional[str] = None
    chain: Optional[List[str]] = None
    enabled: Optional[bool] = None


class VirtualModelTestRequest(BaseModel):
    """测试聚合链：逐个探测这些真实模型的连通性"""
    chain: List[str] = []


class ModelPricingSaveRequest(BaseModel):
    model_id: str
    display_name: Optional[str] = ""
    provider: Optional[str] = ""
    currency: Optional[str] = "USD"
    input_cost: Optional[float] = 0.0
    output_cost: Optional[float] = 0.0
    cache_read_cost: Optional[float] = 0.0
    cache_write_cost: Optional[float] = 0.0


# ========== 认证依赖 ==========

async def require_admin(request: Request):
    """验证管理员Token"""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
    else:
        token = request.cookies.get("admin_token", "")

    if not token:
        raise HTTPException(status_code=401, detail={
            "message": "未登录", "type": "unauthorized", "code": "no_token"
        })

    # 管理员Token鉴权
    secret = get_setting("gateway_secret")
    if not secret or not verify_admin_token(token, secret):
        raise HTTPException(status_code=401, detail={
            "message": "Token无效或已过期", "type": "unauthorized", "code": "invalid_token"
        })
    request.state.auth_type = "admin"


# ========== 管理员登录 ==========

@router.post("/login", tags=["管理员"])
async def admin_login(req: LoginRequest, request: Request):
    """管理员登录"""
    ip = get_client_ip(request)

    # 登录频率限制
    if not _login_rate_limiter.is_allowed(ip):
        raise HTTPException(status_code=429, detail={
            "message": "请求过于频繁，请稍后再试",
            "type": "rate_limit_error",
            "code": "rate_limited",
        })


    if not bcrypt.checkpw(req.password.encode("utf-8"), ADMIN_PASSWORD_HASH.encode("utf-8")):
        raise HTTPException(status_code=401, detail={
            "message": "密码错误", "type": "unauthorized", "code": "wrong_password"
        })

    secret = get_setting("gateway_secret")
    token = create_admin_token(secret)

    response = JSONResponse(content={
        "token": token,
        "message": "登录成功",
    })
    response.set_cookie(
        key="admin_token", value=token,
        httponly=True, max_age=86400, samesite="lax",
    )
    insert_audit("login", "admin", "", "管理员登录")
    return response


# ========== 仪表盘 ==========

@router.get("/dashboard", tags=["管理员"])
async def dashboard(request: Request):
    """仪表盘汇总数据"""
    await require_admin(request)

    now_ts = time.time()
    # 修复：使用本地时区(CST+8)零点对应的本地时间ISO字符串作为"今日"边界
    today_start = today_start_local()
    
    # 计算本周开始时间（周一 00:00 CST）
    _is_sqlite = (os.environ.get("DATABASE_BACKEND", "sqlite")).lower() == "sqlite"
    if _is_sqlite:
        # SQLite: 用 Python 计算本周一和本月1号的 ISO 字符串（本地时区 +08:00）
        from datetime import datetime, timedelta
        now_local = datetime.now()
        week_start = (now_local - timedelta(days=now_local.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        month_start = now_local.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        week_start_str = week_start.strftime("%Y-%m-%dT%H:%M:%S.000+08:00")
        month_start_str = month_start.strftime("%Y-%m-%dT%H:%M:%S.000+08:00")
        seven_days_ago = (now_local - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%S.000+08:00")
    else:
        # PostgreSQL: 直接用 SQL 计算
        week_start_str = fetch_one(
            "SELECT (date_trunc('week', NOW() AT TIME ZONE 'Asia/Shanghai') AT TIME ZONE 'Asia/Shanghai')::text"
        )[0]
        month_start_str = fetch_one(
            "SELECT (date_trunc('month', NOW() AT TIME ZONE 'Asia/Shanghai') AT TIME ZONE 'Asia/Shanghai')::text"
        )[0]
        seven_days_ago = days_ago_utc(7)

    # 今日统计
    today_stats = fetch_one(
        "SELECT COUNT(*) as total_requests, "
        "SUM(CASE WHEN status_code >= 200 AND status_code < 300 THEN 1 ELSE 0 END) as success_count, "
        "SUM(CASE WHEN status_code = 429 THEN 1 ELSE 0 END) as count_429, "
        "SUM(CASE WHEN status_code >= 500 THEN 1 ELSE 0 END) as count_5xx, "
        "SUM(CASE WHEN status_code = 401 THEN 1 ELSE 0 END) as count_401, "
        "SUM(CASE WHEN status_code = 403 THEN 1 ELSE 0 END) as count_403, "
        "SUM(CASE WHEN status_code = 400 THEN 1 ELSE 0 END) as count_400, "
        "SUM(CASE WHEN status_code = 200 THEN 1 ELSE 0 END) as count_200, "
        "SUM(prompt_tokens) as prompt_tokens, "
        "SUM(completion_tokens) as completion_tokens, "
        "SUM(total_tokens) as total_tokens, "
        "AVG(latency_ms) as avg_latency, "
        "SUM(CASE WHEN is_stream = 1 THEN 1 ELSE 0 END) as stream_count "
        "FROM request_logs WHERE created_at >= %s",
        (today_start,),
    )
    
    # 本周统计
    week_stats = fetch_one(
        "SELECT SUM(total_tokens) as total_tokens "
        "FROM request_logs WHERE created_at >= %s",
        (week_start_str,),
    )
    
    # 本月统计
    month_stats = fetch_one(
        "SELECT SUM(total_tokens) as total_tokens "
        "FROM request_logs WHERE created_at >= %s",
        (month_start_str,),
    )

    # 历史统计（全量）
    total_stats = fetch_one(
        "SELECT COUNT(*) as total_requests, "
        "SUM(CASE WHEN status_code >= 200 AND status_code < 300 THEN 1 ELSE 0 END) as success_count, "
        "SUM(CASE WHEN status_code = 429 THEN 1 ELSE 0 END) as count_429, "
        "SUM(CASE WHEN status_code >= 500 THEN 1 ELSE 0 END) as count_5xx, "
        "SUM(CASE WHEN status_code = 401 THEN 1 ELSE 0 END) as count_401, "
        "SUM(CASE WHEN status_code = 403 THEN 1 ELSE 0 END) as count_403, "
        "SUM(CASE WHEN status_code = 400 THEN 1 ELSE 0 END) as count_400, "
        "SUM(CASE WHEN status_code = 200 THEN 1 ELSE 0 END) as count_200, "
        "SUM(prompt_tokens) as prompt_tokens, "
        "SUM(completion_tokens) as completion_tokens, "
        "SUM(total_tokens) as total_tokens, "
        "AVG(latency_ms) as avg_latency, "
        "SUM(CASE WHEN is_stream = 1 THEN 1 ELSE 0 END) as stream_count "
        "FROM request_logs"
    )

    # 活跃密钥和客户数
    active_keys = fetch_one("SELECT COUNT(*) as cnt FROM upstream_keys WHERE status='active'")["cnt"]
    active_clients = fetch_one("SELECT COUNT(*) as cnt FROM clients WHERE status='active'")["cnt"]

    # 各模型 token 分布（全量，按 token 降序，用于饼图/分布）
    model_token_dist_all = fetch_all(
        "SELECT model, SUM(total_tokens) as total_tokens, SUM(prompt_tokens) as prompt_tokens, "
        "SUM(completion_tokens) as completion_tokens, COUNT(*) as request_count "
        "FROM request_logs "
        "WHERE model IS NOT NULL AND model != '' "
        "GROUP BY model ORDER BY total_tokens DESC",
    )
    # Top4 + 其他
    if len(model_token_dist_all) > 4:
        top4 = model_token_dist_all[:4]
        others_tokens = sum(r["total_tokens"] or 0 for r in model_token_dist_all[4:])
        others_requests = sum(r["request_count"] or 0 for r in model_token_dist_all[4:])
        top4.append({
            "model": "其他模型",
            "total_tokens": others_tokens,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "request_count": others_requests,
        })
        model_token_dist = top4
    else:
        model_token_dist = model_token_dist_all

    # 7天趋势（PostgreSQL：将UTC时间转为CST+8日期分组 / SQLite：按 ISO 时间字符串前10位算日期）
    if _is_sqlite:
        # SQLite 无 timestamptz/AT TIME ZONE；存格式为 ISO 字符串，直接取前 10 字符作 date 够用
        trend = fetch_all(
            "SELECT SUBSTR(created_at, 1, 10) as date, COUNT(*) as requests, "
            "SUM(CASE WHEN status_code >= 200 AND status_code < 300 THEN 1 ELSE 0 END) as success "
            "FROM request_logs WHERE created_at >= ? "
            "GROUP BY SUBSTR(created_at, 1, 10) ORDER BY date",
            (seven_days_ago,),
        )
    else:
        trend = fetch_all(
            "SELECT (created_at::timestamptz AT TIME ZONE 'Asia/Shanghai')::date as date, COUNT(*) as requests, "
            "SUM(CASE WHEN status_code >= 200 AND status_code < 300 THEN 1 ELSE 0 END) as success "
            "FROM request_logs WHERE created_at >= %s "
            "GROUP BY (created_at::timestamptz AT TIME ZONE 'Asia/Shanghai')::date ORDER BY date",
            (seven_days_ago,),
        )

    # 模型分布（今日，按请求数排序，用于原有兼容）
    model_dist = fetch_all(
        "SELECT model, COUNT(*) as count FROM request_logs "
        "WHERE created_at >= %s AND model IS NOT NULL AND model != '' GROUP BY model ORDER BY count DESC LIMIT 10",
        (today_start,),
    )
    # v10.0: 添加 display_name（友好名称）
    try:
        from app.nim_models import NIM_MODEL_CATALOG
        for md in model_dist:
            info = NIM_MODEL_CATALOG.get(md["model"])
            if info:
                md["display_name"] = info.display_name
            else:
                md["display_name"] = md["model"]
        for md in model_token_dist:
            if md["model"] == "其他模型":
                md["display_name"] = "其他模型"
            else:
                info = NIM_MODEL_CATALOG.get(md["model"])
                md["display_name"] = info.display_name if info else md["model"]
    except ImportError:
        for md in model_dist:
            md["display_name"] = md["model"]
        for md in model_token_dist:
            md.setdefault("display_name", md["model"])

    success_count = today_stats["success_count"] or 0
    total_today = today_stats["total_requests"] or 0
    success_rate = (success_count / total_today * 100) if total_today > 0 else 0

    hist_total = total_stats["total_requests"] or 0
    hist_success = total_stats["success_count"] or 0
    hist_success_rate = (hist_success / hist_total * 100) if hist_total > 0 else 0

    # token 汇总（4个时间维度）
    total_tokens_all = int(total_stats["total_tokens"] or 0)
    total_tokens_month = int(month_stats["total_tokens"] or 0)
    total_tokens_week = int(week_stats["total_tokens"] or 0)
    total_tokens_today = int(today_stats["total_tokens"] or 0)

    
    # 消费统计（今日消费与总消费，按模型聚合计算，支持 USD 与 CNY）
    try:
        from app.pricing import calculate_aggregated_cost
        today_groups = fetch_all(
            "SELECT model, SUM(prompt_tokens) as prompt_tokens, SUM(completion_tokens) as completion_tokens, "
            "SUM(cached_tokens) as cached_tokens, SUM(cache_write_tokens) as cache_write_tokens "
            "FROM request_logs WHERE created_at >= %s AND status_code >= 200 AND status_code < 300 "
            "GROUP BY model",
            (today_start,),
        )
        today_cost_summary = calculate_aggregated_cost(today_groups or [])

        total_groups = fetch_all(
            "SELECT model, SUM(prompt_tokens) as prompt_tokens, SUM(completion_tokens) as completion_tokens, "
            "SUM(cached_tokens) as cached_tokens, SUM(cache_write_tokens) as cache_write_tokens "
            "FROM request_logs WHERE status_code >= 200 AND status_code < 300 "
            "GROUP BY model"
        )
        total_cost_summary = calculate_aggregated_cost(total_groups or [])
    except Exception as ce:
        logger.warning(f"Error calculating dashboard cost summary: {ce}")
        today_cost_summary = {"usd": 0.0, "cny": 0.0, "usd_formatted": "$0.0000", "cny_formatted": "¥0.0000", "display": "$0.0000"}
        total_cost_summary = {"usd": 0.0, "cny": 0.0, "usd_formatted": "$0.0000", "cny_formatted": "¥0.0000", "display": "$0.0000"}

    return {
        "today": {
            "total_requests": total_today,
            "success_rate": round(success_rate, 2),
            "total_tokens": today_stats["total_tokens"] or 0,
            "prompt_tokens": today_stats["prompt_tokens"] or 0,
            "completion_tokens": today_stats["completion_tokens"] or 0,
            "avg_latency_ms": round(today_stats["avg_latency"] or 0, 2),
            "count_200": today_stats["count_200"] or 0,
            "count_400": today_stats["count_400"] or 0,
            "count_401": today_stats["count_401"] or 0,
            "count_403": today_stats["count_403"] or 0,
            "count_429": today_stats["count_429"] or 0,
            "count_5xx": today_stats["count_5xx"] or 0,
            "stream_count": today_stats["stream_count"] or 0,
            "stream_ratio": round((today_stats["stream_count"] or 0) / total_today * 100, 2) if total_today > 0 else 0,
        },
        "history": {
            "total_requests": hist_total,
            "success_count": hist_success,
            "success_rate": round(hist_success_rate, 2),
            "total_tokens": int(total_stats["total_tokens"] or 0),
            "prompt_tokens": int(total_stats["prompt_tokens"] or 0),
            "completion_tokens": int(total_stats["completion_tokens"] or 0),
            "avg_latency_ms": int(round(total_stats["avg_latency"] or 0)),
            "count_200": total_stats["count_200"] or 0,
            "count_400": total_stats["count_400"] or 0,
            "count_401": total_stats["count_401"] or 0,
            "count_403": total_stats["count_403"] or 0,
            "count_429": total_stats["count_429"] or 0,
            "count_5xx": total_stats["count_5xx"] or 0,
            "stream_count": total_stats["stream_count"] or 0,
        },
        "token_summary": {
            "total": total_tokens_all,
            "month": total_tokens_month,
            "week": total_tokens_week,
            "today": total_tokens_today,
        },
        "cost_summary": {
            "today": today_cost_summary,
            "total": total_cost_summary,
        },
        "active": {
            "upstream_keys": active_keys,
            "clients": active_clients,
        },
        "trend_7d": trend,
        "model_distribution": model_dist,
        "model_token_distribution": model_token_dist,
    }


# ========== 上游密钥管理 ==========

@router.get("/upstreams", tags=["管理员"])
async def list_upstreams(request: Request):
    """上游密钥列表（含调度器实时状态）"""
    await require_admin(request)
    rows = fetch_all("SELECT * FROM upstream_keys ORDER BY created_at")
    scheduler = get_scheduler()
    all_buckets = scheduler.get_bucket_stats()

    # 按 key_id 聚合桶数据
    buckets_by_key = {}
    for b in all_buckets:
        buckets_by_key.setdefault(b["key_id"], []).append(b)

    # 全局在途请求数按key_id聚合
    inflight_by_key = {}
    try:
        for cid, m in scheduler._client_metrics.items():
            # 客户端级粒度，无法直接按key_id聚合，保留为0
            pass
    except Exception:
        pass

    result = []
    for row in rows:
        r = dict(row)
        key_buckets = buckets_by_key.get(r["id"], [])
        r["bucket_count"] = len(key_buckets)
        r["cooled_buckets"] = sum(1 for b in key_buckets if b["cooldown_remaining"] > 0)
        r["soft_busy_buckets"] = sum(1 for b in key_buckets if b["soft_busy"])
        r["isolated_buckets"] = sum(1 for b in key_buckets if b["isolation_remaining"] > 0)
        r["avg_health"] = round(sum(b["health_score"] for b in key_buckets) / len(key_buckets), 1) if key_buckets else None
        # 当前RPM（该密钥所有桶RPM之和）
        r["current_rpm"] = sum(b.get("rpm", 0) for b in key_buckets)
        # 在途请求数（调度器暂未按key_id统计，保留0）
        r["inflight_requests"] = inflight_by_key.get(r["id"], 0)
        # 成功率（聚合）
        total_req = sum(b.get("total_requests", 0) for b in key_buckets)
        total_success = sum(b.get("total_success", 0) for b in key_buckets)
        r["success_rate"] = round(total_success / total_req * 100, 2) if total_req > 0 else None
        r["total_requests"] = total_req
        r["stats_status"] = "observed" if total_req > 0 else "unknown"
        r["total_429"] = sum(b.get("total_429", 0) for b in key_buckets)
        r["total_5xx"] = sum(b.get("total_5xx", 0) for b in key_buckets)
        result.append(r)
    return result


@router.get("/upstreams/presets", tags=["管理员"])
async def list_upstream_presets(request: Request):
    """内置供应商接入目录：供"添加上游密钥"一键预填。

    只读静态数据，不影响任何已保存的上游；手动创建流程不变。
    """
    await require_admin(request)
    return {"presets": list_provider_presets(), "source": PRESET_SOURCE_URL}


class OAuthPasteRequest(BaseModel):
    state_id: str
    url: str


async def _resolve_probe_credential(key_id: str, plaintext: str) -> str:
    """OAuth 信封换成新鲜 access token（可能触发刷新并回写库）；普通密钥原样返回。"""
    if parse_envelope(plaintext) is None:
        return plaintext
    try:
        return await ensure_fresh_access(key_id, plaintext)
    except OAuthError as exc:
        raise HTTPException(status_code=502, detail=f"ChatGPT 账号凭据刷新失败：{exc}")


@router.post("/upstreams/oauth/start", tags=["管理员"])
async def start_oauth_login(request: Request):
    """发起 Sign in with ChatGPT 登录：返回授权 URL + 会话 ID。"""
    await require_admin(request)
    try:
        info = await start_login()
    except OAuthError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    insert_audit("oauth_login_start", "upstream_key", "", "发起 ChatGPT 账号登录")
    return info


@router.get("/upstreams/oauth/status/{state_id}", tags=["管理员"])
async def oauth_login_status(state_id: str, request: Request):
    """轮询登录状态：pending | completed | failed | expired。"""
    await require_admin(request)
    return session_status(state_id)


@router.post("/upstreams/oauth/paste", tags=["管理员"])
async def paste_oauth_callback(req: OAuthPasteRequest, request: Request):
    """手动粘贴浏览器最终回调 URL 完成登录（1455 回调端口不可用时的兜底）。"""
    await require_admin(request)
    try:
        return await complete_with_paste(req.state_id, req.url)
    except OAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ========== 代理设置（网关出站代理，作用于 OpenAI 直连类流量） ==========

class ProxySaveRequest(BaseModel):
    """保存代理配置：url 为空表示直连。"""
    url: str = ""

class ProxyTestRequest(BaseModel):
    """测试代理连通性：url 留空则测试当前生效配置。"""
    url: str = ""

_PROXY_SCHEMES = ("http://", "https://", "socks5://", "socks5h://")

def _proxy_info() -> dict:
    setting = (get_setting("oauth_proxy") or "").strip()
    env = (os.environ.get("MASTER_API_OAUTH_PROXY") or "").strip()
    effective = setting or env or ""
    return {"setting": setting, "env": env, "effective": effective,
            "source": "setting" if setting else ("env" if env else "none"),
            "mode": "manual" if effective else "none"}

@router.get("/proxy", tags=["管理员"])
async def get_proxy_config(request: Request):
    """当前代理配置（控制台配置优先于环境变量）。"""
    await require_admin(request)
    return _proxy_info()

@router.post("/proxy", tags=["管理员"])
async def save_proxy_config(req: ProxySaveRequest, request: Request):
    """保存代理地址（空 = 直连）。写入 admin_settings，即时生效，无需重启。"""
    await require_admin(request)
    url = (req.url or "").strip()
    if url and not url.lower().startswith(_PROXY_SCHEMES):
        raise HTTPException(status_code=400, detail="代理地址必须以 http://、https:// 或 socks5:// 开头")
    set_setting("oauth_proxy", url)
    insert_audit("update", "proxy_config", "", f"设置网关出站代理: {url or '(直连)'}")
    return {"ok": True, **_proxy_info()}

@router.post("/proxy/test", tags=["管理员"])
async def test_proxy_config(req: ProxyTestRequest, request: Request):
    """通过给定（或当前生效）代理请求 ipify，返回出口 IP 与延迟。"""
    await require_admin(request)
    proxy = (req.url or "").strip() or get_oauth_proxy()
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(12.0, connect=6.0), proxy=proxy or None) as client:
            r = await client.get("https://api.ipify.org?format=json")
        ip = ""
        try:
            ip = (r.json() or {}).get("ip", "")
        except Exception:
            pass
        if not ip:
            return {"ok": False, "proxy": proxy or "DIRECT",
                    "error": f"响应异常（HTTP {r.status_code}）: {r.text[:120]}"}
        return {"ok": True, "proxy": proxy or "DIRECT", "ip": ip,
                "latency_ms": round((time.perf_counter() - started) * 1000)}
    except Exception as exc:
        return {"ok": False, "proxy": proxy or "DIRECT",
                "error": f"{type(exc).__name__}: {str(exc)[:160]}"}


def _record_upstream_health_check(key_id: str, result: dict) -> str:
    """Persist an admin connectivity probe without affecting scheduler metrics."""
    http_status = result.get("http_status")
    if result.get("ok"):
        status = "healthy"
    elif http_status in (401, 403):
        status = "invalid"
    else:
        status = "failed"

    error = str(result.get("error") or result.get("reason") or "")[:500]
    execute(
        "UPDATE upstream_keys SET "
        "last_health_check_status = %s, last_health_check_at = %s, "
        "last_health_check_latency_ms = %s, last_health_check_error = %s "
        "WHERE id = %s",
        (status, utcnow(), result.get("latency_ms"), error, key_id),
    )
    return status


def _check_chat_response(body: dict) -> Optional[str]:
    """校验 chat/completions 响应体，返回错误描述字符串；None 表示正常。
    
    校验规则：
    1. body 里有 error 字段 → 报错
    2. 没有 choices 且没有 content → 报错
    3. choices 存在但为空列表 → 报错
    4. choices[0] 里 finish_reason == 'error' → 报错
    """
    if isinstance(body.get("error"), dict):
        return body["error"].get("message") or str(body["error"])[:120]
    if isinstance(body.get("error"), str) and body["error"]:
        return body["error"]

    choices = body.get("choices")
    content_top = body.get("content")  # anthropic 格式

    if not choices and not content_top:
        return body.get("message") or "响应体无有效内容（模型可能不存在）"
    if choices is not None:
        if len(choices) == 0:
            return "choices 为空列表（模型可能不存在或无权限）"
        first = choices[0]
        if first.get("finish_reason") == "error":
            return "模型返回 finish_reason=error"
    return None


def _load_model_provider_mapping() -> dict[str, list[str]]:
    """读取并规范化模型路由配置。

    规范格式为 ``{model_id: [provider, ...]}``。读取时兼容历史的
    ``{model_id: provider}`` 以及曾短暂写入的 ``{model_id: key_id}``，
    避免旧配置在升级后丢失。
    """
    try:
        raw = json.loads(get_setting("model_to_provider") or "{}")
    except Exception:
        raw = {}
    if not isinstance(raw, dict):
        return {}

    try:
        key_rows = fetch_all("SELECT id, provider FROM upstream_keys")
        provider_by_key_id = {
            str(item["id"]): str(item["provider"])
            for item in key_rows
            if item.get("id") and item.get("provider")
        }
    except Exception:
        provider_by_key_id = {}

    normalized: dict[str, list[str]] = {}
    for model_id, targets in raw.items():
        model_id = str(model_id or "").strip()
        if not model_id:
            continue
        values = targets if isinstance(targets, list) else [targets]
        providers: list[str] = []
        for target in values:
            if not isinstance(target, str):
                continue
            provider = provider_by_key_id.get(target, target).strip()
            if provider and provider not in providers:
                providers.append(provider)
        if providers:
            normalized[model_id] = providers
    return normalized


def _store_model_provider_mapping(mapping: dict[str, list[str]]) -> None:
    """保存规范化后的模型路由配置。"""
    clean = {
        model_id: providers
        for model_id, providers in mapping.items()
        if model_id and providers
    }
    set_setting("model_to_provider", json.dumps(clean, ensure_ascii=False))


def _remove_provider_from_model_mapping(provider: str) -> None:
    """从所有模型路由中移除一个 provider，保留其它 provider 的同名模型。"""
    mapping = _load_model_provider_mapping()
    for model_id in list(mapping):
        remaining = [item for item in mapping[model_id] if item != provider]
        if remaining:
            mapping[model_id] = remaining
        else:
            del mapping[model_id]
    _store_model_provider_mapping(mapping)


def _rename_provider_in_model_mapping(old_provider: str, new_provider: str) -> None:
    """重命名 provider 时同步迁移其所有模型路由。"""
    mapping = _load_model_provider_mapping()
    for model_id, providers in mapping.items():
        updated: list[str] = []
        for provider in providers:
            provider = new_provider if provider == old_provider else provider
            if provider not in updated:
                updated.append(provider)
        mapping[model_id] = updated
    _store_model_provider_mapping(mapping)


def _save_key_models(key_id: str, models: List[str]) -> None:
    """保存一个上游的模型清单，而不覆盖其它上游的同名模型。

    每个模型映射到一个 provider 列表，例如
    ``{"shared-model": ["provider-a", "provider-b"]}``。
    """
    row = fetch_one("SELECT provider FROM upstream_keys WHERE id = %s", (key_id,))
    if not row:
        return
    provider = row["provider"]
    mapping = _load_model_provider_mapping()

    # 当前表单是该上游完整的模型清单：只移除当前 provider，绝不触碰其它 provider。
    for model_id in list(mapping):
        remaining = [item for item in mapping[model_id] if item != provider]
        if remaining:
            mapping[model_id] = remaining
        else:
            del mapping[model_id]

    # 将当前 provider 追加到选中的模型；相同模型可以同时属于多个上游。
    selected_models: list[str] = []
    for model_id in models or []:
        model_id = (model_id or "").strip()
        if model_id and model_id not in selected_models:
            selected_models.append(model_id)
    for model_id in selected_models:
        providers = mapping.setdefault(model_id, [])
        if provider not in providers:
            providers.append(provider)

    _store_model_provider_mapping(mapping)


@router.post("/upstreams/{key_id}/models", tags=["管理员"])
async def save_upstream_models(key_id: str, req: UpstreamModelsRequest, request: Request):
    """保存某上游的模型清单，并维护可共享的模型路由。"""
    await require_admin(request)
    row = fetch_one("SELECT provider FROM upstream_keys WHERE id = %s", (key_id,))
    if not row:
        raise HTTPException(status_code=404, detail="供应商不存在")
    _save_key_models(key_id, req.models or [])
    insert_audit("update", "upstream_models", key_id, f"更新供应商 {row['provider']} 的模型清单: {req.models}")
    return {
        "ok": True,
        "key_id": key_id,
        "provider": row["provider"],
        "models": req.models,
        "mapping": _load_model_provider_mapping(),
    }


@router.get("/upstreams/{key_id}/models", tags=["管理员"])
async def get_upstream_models(key_id: str, request: Request):
    """返回一个上游已选择的模型；同名模型可同时属于多个上游。"""
    await require_admin(request)
    row = fetch_one("SELECT provider FROM upstream_keys WHERE id = %s", (key_id,))
    if not row:
        raise HTTPException(status_code=404, detail="供应商不存在")
    provider = row["provider"]
    mapping = _load_model_provider_mapping()
    models_list = [model_id for model_id, providers in mapping.items() if provider in providers]
    return {"provider": provider, "models": models_list}


@router.get("/available-models", tags=["管理员"])
async def list_available_models(request: Request):
    """列出至少在一个 active 上游中可用的真实模型，供聚合模型配置使用。"""
    await require_admin(request)
    mapping = _load_model_provider_mapping()
    try:
        rows = fetch_all("SELECT provider FROM upstream_keys WHERE status = 'active'")
        active_providers = {row["provider"] for row in rows if row.get("provider")}
    except Exception:
        active_providers = set()

    # 每个「供应商 + 模型」都是独立的链路项。同名模型不能合并成一个选项，
    # 否则聚合链无法指定应调用哪个上游。
    models = []
    for model_id, providers in mapping.items():
        for provider in providers:
            if provider not in active_providers:
                continue
            models.append({
                "id": f"{provider}::{model_id}",
                "model_id": model_id,
                "provider": provider,
                "providers": [provider],
                "active_providers": [provider],
                "has_active_key": True,
            })
    models.sort(key=lambda item: (item["model_id"], item["provider"]))
    return {"models": models}


@router.post("/upstreams/{key_id}/test", tags=["管理员"])
async def test_upstream(key_id: str, req: UpstreamTestRequest, request: Request):
    """使用该上游自身的 provider、base URL 与密钥执行无副作用探测。"""
    await require_admin(request)
    row = fetch_one(
        "SELECT name, provider, base_url, api_key_ciphertext FROM upstream_keys WHERE id = %s",
        (key_id,),
    )
    if not row:
        raise HTTPException(status_code=404, detail="供应商不存在")

    if req.api_key:
        api_key = req.api_key
    else:
        master_key = get_setting("upstream_master_key")
        if not master_key:
            raise HTTPException(status_code=500, detail="主密钥未配置")
        try:
            api_key = decrypt_upstream_key(row["api_key_ciphertext"], master_key)
        except Exception:
            raise HTTPException(status_code=500, detail="密钥解密失败")
        # OAuth 信封自动换出新鲜 access token；普通密钥原样使用
        api_key = await _resolve_probe_credential(key_id, api_key)

    result = await probe_provider(ProviderProbe(
        provider=row["provider"],
        base_url=req.base_url if req.base_url is not None else row.get("base_url", ""),
        api_key=api_key,
        model=req.model,
    ))
    result["status"] = _record_upstream_health_check(key_id, result)
    result["name"] = row["name"]
    return result


@router.post("/upstreams/test", tags=["管理员"])
async def test_upstream_adhoc(req: UpstreamTestAdhocRequest, request: Request):
    """保存前的临时连通性测试；不写数据库或调度器状态。"""
    await require_admin(request)
    base_url = (req.base_url or "").rstrip("/")
    if not base_url:
        raise HTTPException(status_code=400, detail="缺少 base_url")
    if not base_url.startswith(("http://", "https://")):
        return {"ok": False, "http_status": 0, "reason": "invalid_base_url", "error": "Base URL 必须以 http:// 或 https:// 开头"}
    api_key = req.api_key
    if req.oauth_state_id:
        api_key = get_completed_access(req.oauth_state_id)
        if not api_key:
            return {"ok": False, "http_status": 0, "reason": "oauth_not_completed",
                    "error": "登录会话未完成或已过期，请先完成 ChatGPT 授权"}
    return await probe_provider(ProviderProbe(
        provider=req.provider,
        base_url=base_url,
        api_key=api_key,
        model=req.model,
    ))


@router.post("/upstreams", tags=["管理员"])
async def create_upstream(req: UpstreamCreateRequest, request: Request):
    """添加上游密钥（支持同时声明该供应商可跑的模型清单）

    auth_type=oauth（ChatGPT 账号登录）：api_key 留空，凭据从已完成的
    OAuth 登录会话（oauth_state_id）取；其余流程与传统密钥一致。
    """
    await require_admin(request)

    is_oauth = req.auth_type == "oauth"
    if is_oauth and not req.oauth_state_id:
        raise HTTPException(status_code=400, detail="OAuth 添加缺少 oauth_state_id（请先完成 ChatGPT 登录）")

    # provider 唯一性校验
    conflict = fetch_one("SELECT name FROM upstream_keys WHERE provider = %s", (req.provider,))
    if conflict:
        raise HTTPException(status_code=409, detail=f"Provider 标识「{req.provider}」已被「{conflict['name']}」使用，请换一个")

    master_key = get_setting("upstream_master_key")
    if not master_key:
        raise HTTPException(status_code=500, detail="主密钥未配置")

    if is_oauth:
        # 登录会话换凭据，加密信封入库存进 api_key_ciphertext（auth_type=oauth）
        try:
            sess = pop_completed_session(req.oauth_state_id)
        except OAuthError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        ciphertext, key_prefix = envelope_for_storage(sess.cred, master_key)
        audit_note = "创建上游密钥(ChatGPT 账号登录)"
    else:
        # 粘贴的 key 常见误带空白或 "Bearer " 前缀，统一清洗后再入库
        api_key = normalize_pasted_key(req.api_key)
        if not api_key:
            raise HTTPException(status_code=400, detail="API Key 不能为空")
        ciphertext = encrypt_upstream_key(api_key, master_key)
        key_prefix = mask_secret(api_key)
        audit_note = "创建上游密钥"

    key_id = str(uuid.uuid4())
    now = utcnow()

    execute(
        "INSERT INTO upstream_keys "
        "(id, name, provider, api_key_ciphertext, key_prefix, base_url, weight, rpm_limit, switch_threshold, status, created_at, updated_at, auth_type) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'active', %s, %s, %s)",
        (key_id, req.name, req.provider, ciphertext, key_prefix,
         req.base_url, req.weight, req.rpm_limit, req.switch_threshold, now, now,
         "oauth" if is_oauth else "api_key"),
    )

    # 若有 models，立刻保存该上游的模型关联；同名模型不会覆盖其它上游。
    models = req.models or (DEFAULT_CHATGPT_MODELS if is_oauth else [])
    if models:
        _save_key_models(key_id, models)

    # 清除调度器缓存
    scheduler = get_scheduler()
    scheduler.invalidate_key_cache(key_id)
    scheduler.invalidate_active_keys_cache()

    insert_audit("create", "upstream_key", key_id, f"{audit_note}: {req.name} (models={models})")
    return {"id": key_id, "message": "创建成功"}


@router.put("/upstreams/{key_id}", tags=["管理员"])
async def update_upstream(key_id: str, req: UpstreamUpdateRequest, request: Request):
    """编辑上游密钥"""
    await require_admin(request)

    existing = fetch_one("SELECT * FROM upstream_keys WHERE id = %s", (key_id,))
    if not existing:
        raise HTTPException(status_code=404, detail="密钥不存在")

    # provider 唯一性校验
    if req.provider and req.provider != existing["provider"]:
        conflict = fetch_one(
            "SELECT name FROM upstream_keys WHERE provider = %s AND id != %s",
            (req.provider, key_id)
        )
        if conflict:
            raise HTTPException(status_code=409, detail=f"Provider 标识「{req.provider}」已被「{conflict['name']}」使用，请换一个")

    updates = []
    params = []
    for field in ["name", "base_url", "provider", "weight", "rpm_limit", "switch_threshold", "status"]:
        val = getattr(req, field, None)
        if val is not None:
            updates.append(f"{field} = %s")
            params.append(val)

    # 密钥替换：仅当编辑表单明确填写了新 key 时才更新（留空表示不修改）
    if req.api_key is not None and req.api_key != "":
        new_key = normalize_pasted_key(req.api_key)
        if not new_key:
            raise HTTPException(status_code=400, detail="API Key 不能为空")
        master_key = get_setting("upstream_master_key")
        if not master_key:
            raise HTTPException(status_code=500, detail="主密钥未配置")
        updates.extend(["api_key_ciphertext = %s", "key_prefix = %s"])
        params.extend([encrypt_upstream_key(new_key, master_key), mask_secret(new_key)])
        # 密钥变了，旧的健康检测结果作废
        updates.extend([
            "last_health_check_status = 'unknown'",
            "last_health_check_at = ''",
            "last_health_check_latency_ms = NULL",
            "last_health_check_error = ''",
        ])

    if updates:
        # 仅当 base_url / provider 实际发生变化时才作废旧的健康检测结果，
        # 避免编辑名称等无关字段后测试状态被重置为"未检测"。
        config_changed = (
            (req.base_url is not None and (req.base_url or "") != (existing["base_url"] or ""))
            or (req.provider is not None and req.provider != existing["provider"])
        )
        if config_changed:
            updates.extend([
                "last_health_check_status = 'unknown'",
                "last_health_check_at = ''",
                "last_health_check_latency_ms = NULL",
                "last_health_check_error = ''",
            ])
        updates.append("updated_at = %s")
        params.append(utcnow())
        params.append(key_id)
        execute(f"UPDATE upstream_keys SET {', '.join(updates)} WHERE id = %s", tuple(params))

    old_provider = existing["provider"]
    new_provider = req.provider if req.provider is not None else old_provider
    if new_provider != old_provider:
        _rename_provider_in_model_mapping(old_provider, new_provider)

    scheduler = get_scheduler()
    scheduler.invalidate_key_cache(key_id)
    scheduler.invalidate_active_keys_cache()

    insert_audit("update", "upstream_key", key_id, f"更新上游密钥")
    return {"message": "更新成功"}


@router.delete("/upstreams/{key_id}", tags=["管理员"])
async def delete_upstream(key_id: str, request: Request):
    """删除上游密钥，并只移除它自己声明的模型路由。"""
    await require_admin(request)
    existing = fetch_one("SELECT provider FROM upstream_keys WHERE id = %s", (key_id,))
    if not existing:
        raise HTTPException(status_code=404, detail="密钥不存在")

    _remove_provider_from_model_mapping(existing["provider"])
    execute("DELETE FROM upstream_keys WHERE id = %s", (key_id,))

    scheduler = get_scheduler()
    scheduler.invalidate_key_cache(key_id)
    scheduler.invalidate_active_keys_cache()

    insert_audit("delete", "upstream_key", key_id, f"删除上游密钥")
    return {"message": "删除成功"}


@router.post("/upstreams/{key_id}/unfreeze", tags=["管理员"])
async def unfreeze_upstream(key_id: str, request: Request):
    """解冻密钥所有桶"""
    await require_admin(request)

    scheduler = get_scheduler()
    count = scheduler.unfreeze_key_all_buckets(key_id)

    insert_audit("unfreeze", "upstream_key", key_id, f"解冻{count}个桶")
    return {"message": f"已解冻{count}个桶", "count": count}


@router.get("/upstreams/{key_id}/reveal", tags=["管理员"])
async def reveal_upstream_key(key_id: str, request: Request):
    """解密并返回上游密钥明文（仅管理员可调用，会写入审计日志）"""
    await require_admin(request)

    row = fetch_one("SELECT name, api_key_ciphertext, key_prefix FROM upstream_keys WHERE id = %s", (key_id,))
    if not row:
        raise HTTPException(status_code=404, detail="密钥不存在")

    master_key = get_setting("upstream_master_key")
    if not master_key:
        raise HTTPException(status_code=500, detail="主密钥未配置")

    try:
        plaintext = decrypt_upstream_key(row["api_key_ciphertext"], master_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"密钥解密失败: {e}")

    # OAuth 凭据（ChatGPT 账号登录）：只展示概要，refresh_token 打码
    if parse_envelope(plaintext) is not None:
        insert_audit("reveal", "upstream_key", key_id, f"查看上游 OAuth 凭据概要: {row['name']}")
        return {"key": redacted_envelope(plaintext), "prefix": row["key_prefix"],
                "name": row["name"], "auth_type": "oauth"}

    insert_audit("reveal", "upstream_key", key_id, f"查看上游密钥明文: {row['name']}")
    return {"key": plaintext, "prefix": row["key_prefix"], "name": row["name"]}


@router.post("/upstreams/health-check", tags=["管理员"])
async def check_upstream_keys_health(request: Request):
    """逐条按其自身 provider/base URL 探测，绝不修改密钥或调度器状态。"""
    await require_admin(request)
    rows = fetch_all(
        "SELECT id, name, provider, base_url, api_key_ciphertext FROM upstream_keys WHERE status = 'active'"
    )
    master_key = get_setting("upstream_master_key")
    results = []
    for row in rows:
        if not master_key:
            result = {"ok": False, "reason": "no_master_key", "error": "主密钥未配置"}
        else:
            try:
                api_key = decrypt_upstream_key(row["api_key_ciphertext"], master_key)
                # OAuth 信封换成新鲜 access token 再探测（openai-codex 上游走 /responses）
                api_key = await _resolve_probe_credential(row["id"], api_key)
                result = await probe_provider(ProviderProbe(
                    provider=row["provider"],
                    base_url=row.get("base_url", ""),
                    api_key=api_key,
                ))
            except HTTPException as exc:
                result = {"ok": False, "reason": "oauth_refresh_failed", "error": str(exc.detail)}
            except Exception:
                result = {"ok": False, "reason": "decrypt_failed", "error": "密钥解密失败"}
        status = "healthy" if result.get("ok") else (
            "invalid" if result.get("http_status") in (401, 403) else "failed"
        )
        _record_upstream_health_check(row["id"], result)
        results.append({"id": row["id"], "name": row["name"], "status": status, **result})

    return {
        "total": len(results),
        "healthy": sum(1 for item in results if item["status"] == "healthy"),
        "invalid": sum(1 for item in results if item["status"] == "invalid"),
        "errors": sum(1 for item in results if item["status"] == "unknown"),
        "results": results,
    }


# ========== 虚拟聚合模型（Failover 链）管理 ==========


def _validate_virtual_id(vid: str) -> str:
    """校验并规范化虚拟模型 ID，非法抛 HTTPException。"""
    vid = (vid or "").strip()
    if not vid:
        raise HTTPException(status_code=400, detail="虚拟模型 ID 不能为空")
    # 允许字母数字及 - _ . / 冒号，避免奇怪字符影响 agent 传参
    import re as _re
    if not _re.match(r"^[A-Za-z0-9._:/\-]+$", vid):
        raise HTTPException(status_code=400, detail="虚拟模型 ID 只能包含字母、数字和 . _ - : / 字符")
    return vid


def _validate_virtual_payload(vid: str, chain: list) -> tuple[str, list]:
    """校验并规范化虚拟模型入参，返回 (id, chain)。校验失败抛 HTTPException。"""
    vid = _validate_virtual_id(vid)
    clean_chain = []
    for m in (chain or []):
        if isinstance(m, str) and m.strip():
            clean_chain.append(m.strip())
    if not clean_chain:
        raise HTTPException(status_code=400, detail="模型链至少需要一个真实模型")
    return vid, clean_chain


@router.get("/virtual-models", tags=["管理员"])
async def list_virtual_models(request: Request):
    """列出所有虚拟聚合模型（含各真实模型当前软熔断剩余秒数）。"""
    await require_admin(request)
    from app import virtual_models as vm
    models = vm.get_virtual_models(use_cache=False)
    cooldowns = vm.get_cooldown_snapshot()
    # 附带每条 chain 中各真实模型的熔断剩余
    result = []
    for m in models:
        item = dict(m)
        item["cooldowns"] = {rm: cooldowns.get(rm, 0) for rm in (m.get("chain") or [])}
        result.append(item)
    return {"virtual_models": result, "cooldowns": cooldowns}


@router.post("/virtual-models", tags=["管理员"])
async def create_virtual_model(req: VirtualModelCreateRequest, request: Request):
    """新建一个虚拟聚合模型。"""
    await require_admin(request)
    from app import virtual_models as vm
    vid, chain = _validate_virtual_payload(req.id, req.chain)

    models = vm.get_virtual_models(use_cache=False)
    if any(m.get("id") == vid for m in models):
        raise HTTPException(status_code=409, detail=f"虚拟模型 ID '{vid}' 已存在")

    # 与真实模型 ID 冲突检查（避免遮蔽真实模型）
    try:
        if vid in _VERIFIED_WORKING_MODELS:
            raise HTTPException(status_code=409, detail=f"'{vid}' 与已有真实模型同名，请换一个 ID")
    except HTTPException:
        raise
    except Exception:
        pass

    models.append({
        "id": vid,
        "display_name": (req.display_name or vid).strip(),
        "chain": chain,
        "enabled": bool(req.enabled),
    })
    vm.save_virtual_models(models)
    _invalidate_models_cache()
    insert_audit("create", "virtual_model", vid, f"创建聚合模型: chain={chain}")
    return {"ok": True, "id": vid}


@router.put("/virtual-models/{vid}", tags=["管理员"])
async def update_virtual_model(vid: str, req: VirtualModelUpdateRequest, request: Request):
    """更新虚拟聚合模型（改链、改显示名、改ID、启停）。"""
    await require_admin(request)
    from app import virtual_models as vm
    models = vm.get_virtual_models(use_cache=False)
    idx = next((i for i, m in enumerate(models) if m.get("id") == vid), -1)
    if idx < 0:
        raise HTTPException(status_code=404, detail="虚拟模型不存在")

    # 改名（可选）：校验新 ID 合法性、唯一性、不与真实模型冲突
    new_vid = vid
    if req.new_id is not None and req.new_id.strip() and req.new_id.strip() != vid:
        new_vid = _validate_virtual_id(req.new_id)
        if any(m.get("id") == new_vid for m in models):
            raise HTTPException(status_code=409, detail=f"虚拟模型 ID '{new_vid}' 已存在")
        try:
            if new_vid in _VERIFIED_WORKING_MODELS:
                raise HTTPException(status_code=409, detail=f"'{new_vid}' 与已有真实模型同名，请换一个 ID")
        except HTTPException:
            raise
        except Exception:
            pass

    target = dict(models[idx])
    if req.chain is not None:
        _, chain = _validate_virtual_payload(new_vid, req.chain)
        target["chain"] = chain
    if req.display_name is not None:
        target["display_name"] = (req.display_name or new_vid).strip()
    elif target.get("display_name") == vid:
        target["display_name"] = new_vid  # 显示名此前跟随旧 ID，改名时一并同步
    if req.enabled is not None:
        target["enabled"] = bool(req.enabled)
    target["id"] = new_vid
    models[idx] = target
    vm.save_virtual_models(models)

    if new_vid != vid:
        # 让 /v1/models 的 30 秒缓存立即失效，客户端马上能看到新 ID
        _invalidate_models_cache()
        # 同步 model_to_provider 路由映射里的旧 ID（如有）
        try:
            mapping = _load_model_provider_mapping()
            if vid in mapping:
                mapping[new_vid] = mapping.pop(vid)
                _store_model_provider_mapping(mapping)
        except Exception:
            pass
        insert_audit("update", "virtual_model", new_vid, f"聚合模型改名: {vid} → {new_vid}")
    else:
        insert_audit("update", "virtual_model", vid, f"更新聚合模型: {target}")
    return {"ok": True, "id": new_vid}


@router.delete("/virtual-models/{vid}", tags=["管理员"])
async def delete_virtual_model(vid: str, request: Request):
    """删除虚拟聚合模型。"""
    await require_admin(request)
    from app import virtual_models as vm
    models = vm.get_virtual_models(use_cache=False)
    new_models = [m for m in models if m.get("id") != vid]
    if len(new_models) == len(models):
        raise HTTPException(status_code=404, detail="虚拟模型不存在")
    vm.save_virtual_models(new_models)
    _invalidate_models_cache()
    insert_audit("delete", "virtual_model", vid, "删除聚合模型")
    return {"ok": True}


def _invalidate_models_cache() -> None:
    """虚拟模型变更后让 /v1/models 的 30 秒缓存立即失效。"""
    try:
        from app.public_api import _models_cache
        _models_cache["expires"] = 0
    except Exception:
        pass


@router.post("/virtual-models/test", tags=["管理员"])
async def test_virtual_chain(req: VirtualModelTestRequest, request: Request):
    """实际调用每个候选模型的最小 chat 请求，而非仅检查 /models。"""
    await require_admin(request)
    chain = [item.strip() for item in (req.chain or []) if isinstance(item, str) and item.strip()]
    if not chain:
        raise HTTPException(status_code=400, detail="模型链为空")

    mapping = _load_model_provider_mapping()
    master_key = get_setting("upstream_master_key")
    key_rows = fetch_all(
        "SELECT name, provider, base_url, api_key_ciphertext FROM upstream_keys "
        "WHERE status = 'active' ORDER BY created_at"
    )
    keys_by_provider = {row["provider"]: row for row in key_rows if row.get("provider")}

    async def probe_candidate(model_ref: str) -> dict:
        explicit_provider, raw_model = split_provider_model_ref(model_ref)
        providers = [explicit_provider] if explicit_provider else mapping.get(raw_model, [])
        providers = [provider for provider in providers if provider]
        if not providers:
            return {"model": model_ref, "ok": False, "provider": "", "providers": [],
                    "reason": "no_mapping", "message": "该模型未在任何上游供应商里配置", "checks": []}

        checks = []
        for provider in providers:
            key_row = keys_by_provider.get(provider)
            if not key_row:
                checks.append({"provider": provider, "ok": False, "reason": "no_key", "error": "没有可用的上游密钥"})
                continue
            if not master_key:
                checks.append({"provider": provider, "ok": False, "reason": "no_master_key", "error": "主密钥未配置"})
                continue
            try:
                api_key = decrypt_upstream_key(key_row["api_key_ciphertext"], master_key)
            except Exception:
                checks.append({"provider": provider, "ok": False, "reason": "decrypt_failed", "error": "密钥解密失败"})
                continue
            checks.append(await probe_provider(ProviderProbe(
                provider=provider,
                base_url=key_row.get("base_url", ""),
                api_key=api_key,
                model=raw_model,
            )))

        success = next((item for item in checks if item.get("ok")), None)
        primary = success or checks[0]
        return {
            "model": model_ref,
            "ok": success is not None,
            "provider": primary.get("provider", ""),
            "providers": providers,
            "checks": checks,
            "http_status": primary.get("http_status"),
            "reason": primary.get("reason"),
            "message": "至少一个已配置上游可实际调用" if success else (primary.get("error") or "探测失败"),
        }

    import asyncio as _asyncio
    results = await _asyncio.gather(*[probe_candidate(model_ref) for model_ref in chain])
    return {"total": len(results), "ok_count": sum(1 for item in results if item.get("ok")), "results": list(results)}


# ========== 下游客户管理 ==========

@router.get("/clients", tags=["管理员"])
async def list_clients(request: Request):
    """下游客户列表"""
    await require_admin(request)
    rows = fetch_all(
        "SELECT c.*, "
        "(SELECT COUNT(*) FROM client_api_keys WHERE client_id = c.id) as key_count "
        "FROM clients c ORDER BY c.created_at DESC"
    )
    result = []
    for row in rows:
        r = dict(row)
        # display_name 直接使用数据库中的 name（已包含名称和ID，格式如：用户名(ID:123)）
        r["display_name"] = r.get("name", r["id"])
        result.append(r)
    return result


@router.post("/clients", tags=["管理员"])
async def create_client(req: ClientCreateRequest, request: Request):
    """创建下游客户"""
    await require_admin(request)

    client_id = str(uuid.uuid4())
    now = utcnow()

    execute(
        "INSERT INTO clients (id, name, user_type, status, created_at, updated_at) VALUES (%s, %s, %s, 'active', %s, %s)",
        (client_id, req.name, (req.user_type or "old"), now, now),
    )

    insert_audit("create", "client", client_id, f"创建客户: {req.name} (type={req.user_type or 'old'})")
    return {"id": client_id, "message": "创建成功"}


@router.get("/clients/{client_id}", tags=["管理员"])
async def get_client(client_id: str, request: Request):
    """客户详情"""
    await require_admin(request)
    row = fetch_one("SELECT * FROM clients WHERE id = %s", (client_id,))
    if not row:
        raise HTTPException(status_code=404, detail="客户不存在")
    return row


@router.put("/clients/{client_id}", tags=["管理员"])
async def update_client(client_id: str, req: ClientUpdateRequest, request: Request):
    """更新客户"""
    await require_admin(request)

    updates = []
    params = []
    for field in ["name", "status", "user_type"]:
        val = getattr(req, field, None)
        if val is not None:
            updates.append(f"{field} = %s")
            params.append(val)

    if updates:
        updates.append("updated_at = %s")
        params.append(utcnow())
        params.append(client_id)
        execute(f"UPDATE clients SET {', '.join(updates)} WHERE id = %s", tuple(params))

    insert_audit("update", "client", client_id, f"更新客户")
    return {"message": "更新成功"}


@router.delete("/clients/{client_id}", tags=["管理员"])
async def delete_client(client_id: str, request: Request):
    """删除客户"""
    await require_admin(request)
    execute("DELETE FROM client_api_keys WHERE client_id = %s", (client_id,))
    execute("DELETE FROM clients WHERE id = %s", (client_id,))
    insert_audit("delete", "client", client_id, f"删除客户")
    return {"message": "删除成功"}


@router.get("/clients/{client_id}/keys", tags=["管理员"])
async def list_client_keys(client_id: str, request: Request):
    """客户密钥列表"""
    await require_admin(request)
    return fetch_all("SELECT * FROM client_api_keys WHERE client_id = %s ORDER BY created_at", (client_id,))


@router.get("/clients/{client_id}/keys/{key_id}/reveal", tags=["管理员"])
async def reveal_client_key(client_id: str, key_id: str, request: Request):
    """解密并返回客户密钥明文（仅限平台同步使用）"""
    await require_admin(request)
    key_row = fetch_one(
        "SELECT id, key_ciphertext, key_prefix, status FROM client_api_keys WHERE id=%s AND client_id=%s",
        (key_id, client_id),
    )
    if not key_row:
        raise HTTPException(status_code=404, detail="密钥不存在")
    if not key_row["key_ciphertext"]:
        raise HTTPException(status_code=500, detail={
            "message": "密钥加密数据缺失（无法复原），建议删除后重新创建",
            "type": "decryption_error", "code": "missing_ciphertext",
        })
    try:
        master_key = get_setting("upstream_master_key")
        if not master_key:
            raise HTTPException(status_code=500, detail="主密钥未配置")
        plaintext = decrypt_secret(key_row["key_ciphertext"], master_key)
        insert_audit("reveal", "client_key", key_id, f"查看客户密钥明文: {key_row['key_prefix']}")
        return {"key": plaintext, "prefix": key_row["key_prefix"], "status": key_row["status"]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解密失败: {e}")


@router.post("/clients/{client_id}/keys", tags=["管理员"])
async def create_client_key(client_id: str, request: Request):
    """创建客户API密钥"""
    await require_admin(request)

    client = fetch_one("SELECT * FROM clients WHERE id = %s", (client_id,))
    if not client:
        raise HTTPException(status_code=404, detail="客户不存在")

    master_key = get_setting("upstream_master_key")
    if not master_key:
        raise HTTPException(status_code=500, detail="主密钥未配置")

    # 生成平台API密钥
    api_key = generate_platform_key()
    key_hash = hash_secret(api_key)
    ciphertext = encrypt_secret(api_key, master_key)
    key_id = str(uuid.uuid4())
    now = utcnow()

    # 强制命名规范：客户平台ID+平台ID号
    key_name = f"{client['id'][:8]}+{key_id[:8]}"

    execute(
        "INSERT INTO client_api_keys "
        "(id, client_id, key_hash, key_prefix, key_ciphertext, status, created_at, last_used_at) "
        "VALUES (%s, %s, %s, %s, %s, 'active', %s, NULL)",
        (key_id, client_id, key_hash, mask_secret(api_key), ciphertext, now),
    )

    insert_audit("create", "client_key", key_id, f"创建客户密钥: {key_name} (client={client['name']})")

    # 返回完整密钥（仅此一次显示）
    return {
        "id": key_id,
        "key": api_key,
        "key_prefix": mask_secret(api_key),
        "key_name": key_name,
        "message": "密钥已创建，请妥善保存（仅此一次显示完整密钥）",
    }


@router.delete("/clients/{client_id}/keys/{key_id}", tags=["管理员"])
async def delete_client_key(client_id: str, key_id: str, request: Request):
    """删除客户密钥"""
    await require_admin(request)
    execute("DELETE FROM client_api_keys WHERE id = %s AND client_id = %s", (key_id, client_id))
    insert_audit("delete", "client_key", key_id, f"删除客户密钥")
    return {"message": "删除成功"}


@router.put("/clients/{client_id}/keys/{key_id}", tags=["管理员"])
async def update_client_key(client_id: str, key_id: str, request: Request):
    """更新客户密钥（启用/禁用）"""
    await require_admin(request)
    body = await request.json()
    new_status = body.get("status", "")
    if new_status not in ("active", "revoked"):
        raise HTTPException(status_code=400, detail="状态只能是 active 或 revoked")
    key_row = fetch_one(
        "SELECT id, status FROM client_api_keys WHERE id=%s AND client_id=%s",
        (key_id, client_id),
    )
    if not key_row:
        raise HTTPException(status_code=404, detail="密钥不存在")
    execute(
        "UPDATE client_api_keys SET status = %s WHERE id = %s AND client_id = %s",
        (new_status, key_id, client_id),
    )
    from app.scheduler import get_scheduler
    get_scheduler().invalidate_active_keys_cache()
    insert_audit("update", "client_key", key_id, f"密钥状态改为 {new_status}")
    return {"message": f"密钥状态已更新为 {new_status}", "status": new_status}


@router.get("/clients/{client_id}/usage", tags=["管理员"])
async def client_usage(client_id: str, request: Request):
    """客户用量统计"""
    await require_admin(request)
    today = utcnow()[:10]
    stats = fetch_one(
        "SELECT COUNT(*) as total, "
        "SUM(CASE WHEN status_code = 200 THEN 1 ELSE 0 END) as success, "
        "SUM(total_tokens) as tokens "
        "FROM request_logs WHERE client_id = %s AND created_at >= %s",
        (client_id, today + "T00:00:00Z"),
    )
    return stats or {"total": 0, "success": 0, "tokens": 0}




# ========== 模型成本定价 ==========

@router.get("/model-pricing", tags=["管理员"])
async def list_model_pricing(request: Request):
    """获取所有模型成本定价规则"""
    await require_admin(request)
    from app.pricing import get_all_pricings
    return {"data": get_all_pricings(force_refresh=True)}


@router.post("/model-pricing", tags=["管理员"])
async def save_model_pricing(req: ModelPricingSaveRequest, request: Request):
    """保存或更新模型成本定价规则"""
    await require_admin(request)
    model_id = req.model_id.strip()
    if not model_id:
        raise HTTPException(status_code=400, detail={"message": "模型ID不能为空", "type": "bad_request"})

    from app.pricing import invalidate_pricing_cache
    now_str = localnow()
    currency = (req.currency or "USD").upper().strip()
    if currency not in ("USD", "CNY"):
        currency = "USD"

    existing = fetch_one("SELECT id FROM model_pricing WHERE model_id = %s", (model_id,))
    if existing:
        execute(
            """UPDATE model_pricing SET
               display_name = %s, provider = %s, currency = %s,
               input_cost = %s, output_cost = %s,
               cache_read_cost = %s, cache_write_cost = %s,
               updated_at = %s
               WHERE id = %s""",
            (
                req.display_name or "", req.provider or "", currency,
                float(req.input_cost or 0.0), float(req.output_cost or 0.0),
                float(req.cache_read_cost or 0.0), float(req.cache_write_cost or 0.0),
                now_str, existing["id"],
            )
        )
        pricing_id = existing["id"]
        insert_audit("update", "model_pricing", pricing_id, f"更新模型定价: {model_id}")
    else:
        pricing_id = str(uuid.uuid4())
        execute(
            """INSERT INTO model_pricing
               (id, model_id, display_name, provider, currency,
                input_cost, output_cost, cache_read_cost, cache_write_cost,
                created_at, updated_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                pricing_id, model_id, req.display_name or "", req.provider or "", currency,
                float(req.input_cost or 0.0), float(req.output_cost or 0.0),
                float(req.cache_read_cost or 0.0), float(req.cache_write_cost or 0.0),
                now_str, now_str,
            )
        )
        insert_audit("create", "model_pricing", pricing_id, f"创建模型定价: {model_id}")

    invalidate_pricing_cache()
    return {"message": "保存成功", "id": pricing_id}


@router.delete("/model-pricing/{pricing_id}", tags=["管理员"])
async def delete_model_pricing(pricing_id: str, request: Request):
    """删除模型成本定价规则"""
    await require_admin(request)
    row = fetch_one("SELECT * FROM model_pricing WHERE id = %s", (pricing_id,))
    if not row:
        raise HTTPException(status_code=404, detail={"message": "规则不存在", "type": "not_found"})

    from app.pricing import invalidate_pricing_cache
    execute("DELETE FROM model_pricing WHERE id = %s", (pricing_id,))
    invalidate_pricing_cache()
    insert_audit("delete", "model_pricing", pricing_id, f"删除模型定价: {row.get('model_id')}")
    return {"message": "删除成功"}

@router.get("/request-logs", tags=["管理员"])
async def request_logs(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    client_id: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    status_code: Optional[int] = Query(None),
    request_path: Optional[str] = Query(None),
    http_method: Optional[str] = Query(None),
    client_ip: Optional[str] = Query(None),
    log_category: Optional[str] = Query(None),
    error_type: Optional[str] = Query(None),
    business_code: Optional[str] = Query(None),
    start_time: Optional[str] = Query(None, description="起始时间(ISO格式,如2026-01-01T00:00:00)"),
    end_time: Optional[str] = Query(None, description="结束时间(ISO格式)"),
    search: Optional[str] = Query(None, description="全文搜索(error_msg/error_detail/request_body)"),
    sort_by: Optional[str] = Query(None),
    sort_order: Optional[str] = Query("desc"),
):
    """请求日志（分页 + 多维度筛选 + 排序）增强"""
    await require_admin(request)

    where = []
    params = []

    # 默认返回所有请求，包括缺少客户端或模型信息的失败请求。
    if client_id:
        where.append("client_id = %s")
        params.append(client_id)
    if model:
        where.append("model LIKE %s")
        params.append(f"%{model}%")
    if status_code:
        where.append("status_code = %s")
        params.append(status_code)
    if request_path:
        where.append("request_path LIKE %s")
        params.append(f"%{request_path}%")
    if http_method:
        where.append("http_method = %s")
        params.append(http_method)
    if client_ip:
        where.append("client_ip = %s")
        params.append(client_ip)
    if log_category:
        where.append("log_category = %s")
        params.append(log_category)
    if error_type:
        where.append("error_type LIKE %s")
        params.append(f"%{error_type}%")
    if business_code:
        where.append("business_code = %s")
        params.append(business_code)
    if start_time:
        where.append("created_at >= %s")
        params.append(start_time)
    if end_time:
        where.append("created_at <= %s")
        params.append(end_time)
    if search:
        where.append("(error_msg LIKE %s OR error_detail LIKE %s OR request_body LIKE %s)")
        params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])

    where_clause = " WHERE " + " AND ".join(where) if where else ""

    # 排序（白名单字段，防注入）
    allowed_sort = {
        "created_at": "created_at",
        "latency_ms": "latency_ms",
        "latency_us": "latency_us",
        "total_tokens": "total_tokens",
        "prompt_tokens": "prompt_tokens",
        "completion_tokens": "completion_tokens",
        "status_code": "status_code",
        "request_path": "request_path",
        "client_ip": "client_ip",
    }
    order_field = allowed_sort.get(sort_by, "created_at")
    order_dir = "ASC" if (sort_order or "desc").lower() == "asc" else "DESC"
    order_clause = f" ORDER BY {order_field} {order_dir}"

    offset = (page - 1) * page_size

    total = fetch_one(f"SELECT COUNT(*) as cnt FROM request_logs{where_clause}", tuple(params))["cnt"]
    rows = fetch_all(
        f"SELECT r.*, COALESCE(c.name, '') as client_name FROM request_logs r LEFT JOIN clients c ON r.client_id = c.id{where_clause}{order_clause} LIMIT %s OFFSET %s",
        tuple(params) + (page_size, offset),
    )

    # 确保关键字段非空并计算费用与缓存tokens
    from app.pricing import calculate_request_cost
    for row in rows:
        if "client_name" not in row or not row["client_name"]:
            row["client_name"] = ""
        for field in ("request_path", "http_method", "client_ip"):
            if field not in row or row[field] is None:
                row[field] = ""
        cached_toks = int(row.get("cached_tokens") or 0)
        cache_write_toks = int(row.get("cache_write_tokens") or 0)
        prompt_toks = int(row.get("prompt_tokens") or 0)
        comp_toks = int(row.get("completion_tokens") or 0)
        model_name = row.get("model") or ""

        # 尝试从 response_body 兜底读取 cached_tokens（如果是历史日志且表中为0）
        if cached_toks == 0 and row.get("response_body"):
            try:
                resp_json = json.loads(row["response_body"])
                u = resp_json.get("usage") if isinstance(resp_json, dict) else None
                if u:
                    ptd = u.get("prompt_tokens_details") or {}
                    cached_toks = ptd.get("cached_tokens", 0) or ptd.get("cache_read_tokens", 0) or u.get("prompt_cache_hit_tokens", 0) or u.get("cache_read_input_tokens", 0) or 0
                    cache_write_toks = ptd.get("cache_write_tokens", 0) or u.get("cache_creation_input_tokens", 0) or 0
            except Exception:
                pass

        row["cached_tokens"] = cached_toks
        row["cache_write_tokens"] = cache_write_toks

        saved_cost = float(row.get("cost") or 0.0)
        if saved_cost > 0:
            row["cost"] = saved_cost
            row["currency"] = (row.get("currency") or "USD").upper()
        else:
            cost_info = calculate_request_cost(
                model=model_name,
                prompt_tokens=prompt_toks,
                completion_tokens=comp_toks,
                cached_tokens=cached_toks,
                cache_write_tokens=cache_write_toks,
            )
            row["cost"] = cost_info["cost"]
            row["currency"] = cost_info["currency"]

    return {"total": total, "page": page, "page_size": page_size, "data": rows}


@router.get("/request-logs/{log_id}", tags=["管理员"])
async def request_log_detail(log_id: str, request: Request):
    """日志详情"""
    await require_admin(request)
    row = fetch_one(
        "SELECT r.*, COALESCE(c.name, '') as client_name FROM request_logs r LEFT JOIN clients c ON r.client_id = c.id WHERE r.id = %s",
        (log_id,),
    )
    if not row:
        raise HTTPException(status_code=404, detail={"message": "日志未找到", "type": "not_found", "code": "not_found"})

    from app.pricing import calculate_request_cost
    cached_toks = int(row.get("cached_tokens") or 0)
    cache_write_toks = int(row.get("cache_write_tokens") or 0)
    prompt_toks = int(row.get("prompt_tokens") or 0)
    comp_toks = int(row.get("completion_tokens") or 0)
    model_name = row.get("model") or ""

    if cached_toks == 0 and row.get("response_body"):
        try:
            resp_json = json.loads(row["response_body"])
            u = resp_json.get("usage") if isinstance(resp_json, dict) else None
            if u:
                ptd = u.get("prompt_tokens_details") or {}
                cached_toks = ptd.get("cached_tokens", 0) or ptd.get("cache_read_tokens", 0) or u.get("prompt_cache_hit_tokens", 0) or u.get("cache_read_input_tokens", 0) or 0
                cache_write_toks = ptd.get("cache_write_tokens", 0) or u.get("cache_creation_input_tokens", 0) or 0
        except Exception:
            pass

    row["cached_tokens"] = cached_toks
    row["cache_write_tokens"] = cache_write_toks

    cost_info = calculate_request_cost(
        model=model_name,
        prompt_tokens=prompt_toks,
        completion_tokens=comp_toks,
        cached_tokens=cached_toks,
        cache_write_tokens=cache_write_toks,
    )
    saved_cost = float(row.get("cost") or 0.0)
    row["cost"] = saved_cost if saved_cost > 0 else cost_info["cost"]
    row["currency"] = (row.get("currency") or cost_info["currency"]).upper()
    row["cost_details"] = cost_info
    return row


@router.get("/request-logs-stats/summary", tags=["管理员"])
async def request_logs_stats_summary(request: Request):
    """请求日志统计概览  - 实时数据，含错误追踪"""
    await require_admin(request)

    # 今日边界（本地零点 CST+8）
    today_start = today_start_local()

    # 今日统计（与仪表盘统计口径保持完全一致）
    today_stats = fetch_one(
        """SELECT COUNT(*) as total_requests,
                  SUM(CASE WHEN status_code >= 200 AND status_code < 300 THEN 1 ELSE 0 END) as success_count,
                  AVG(latency_ms) as avg_latency
           FROM request_logs WHERE created_at >= %s""",
        (today_start,),
    )
    today_total = today_stats["total_requests"] or 0
    today_success = today_stats["success_count"] or 0
    today_success_rate = round((today_success / today_total * 100), 2) if today_total > 0 else 0

    # 总数统计
    total = fetch_one("SELECT COUNT(*) as cnt FROM request_logs")["cnt"]

    # 按日志分类统计
    category_stats = fetch_all(
        "SELECT log_category, COUNT(*) as cnt FROM request_logs GROUP BY log_category"
    )

    # 按状态码统计
    status_stats = fetch_all(
        "SELECT status_code, COUNT(*) as cnt FROM request_logs GROUP BY status_code ORDER BY cnt DESC"
    )

    # 按HTTP方法统计
    method_stats = fetch_all(
        "SELECT http_method, COUNT(*) as cnt FROM request_logs WHERE http_method != '' GROUP BY http_method"
    )

    # 最近24小时请求量趋势（按小时分组）
    _is_sqlite = (os.environ.get("DATABASE_BACKEND", "sqlite")).lower() == "sqlite"
    if _is_sqlite:
        # SQLite：created_at 是 ISO 字符串，按 SUBSTR 截下日期 + 小时；用 datetime() 计算门槛
        from app.database import utcnow as _utcnow
        cutoff_utc = (datetime.utcnow() - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        hourly_stats = fetch_all(
            "SELECT SUBSTR(created_at, 12, 2) as hour, COUNT(*) as cnt "
            "FROM request_logs WHERE created_at >= ? "
            "GROUP BY hour ORDER BY hour",
            (cutoff_utc,),
        )
    else:
        hourly_stats = fetch_all(
            """SELECT substr(created_at, 12, 2) as hour, COUNT(*) as cnt
               FROM request_logs
               WHERE created_at::timestamptz >= NOW() - INTERVAL '24 hours'
               GROUP BY hour ORDER BY hour"""
        )

    # Top 10 请求路径
    top_paths = fetch_all(
        """SELECT request_path, COUNT(*) as cnt FROM request_logs
           WHERE request_path != '' GROUP BY request_path ORDER BY cnt DESC LIMIT 10"""
    )

    # Top 10 客户端（含client_name）
    top_clients = fetch_all(
        """SELECT COALESCE(MAX(c.name), r.client_id) as client_display, COUNT(*) as cnt,
                  SUM(r.total_tokens) as total_tokens, AVG(r.latency_ms) as avg_latency
           FROM request_logs r
           LEFT JOIN clients c ON r.client_id = c.id
           WHERE r.client_id != '' AND r.client_id IS NOT NULL
           GROUP BY r.client_id ORDER BY cnt DESC LIMIT 10"""
    )

    # Top 10 客户端IP
    top_ips = fetch_all(
        """SELECT client_ip, COUNT(*) as cnt FROM request_logs
           WHERE client_ip != '' GROUP BY client_ip ORDER BY cnt DESC LIMIT 10"""
    )

    # 按日期统计（最近7天每日请求量）
    if _is_sqlite:
        cutoff_utc7 = (datetime.now() - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%S.000+08:00")
        daily_stats = fetch_all(
            """SELECT SUBSTR(created_at, 1, 10) as date,
                      COUNT(*) as cnt,
                      SUM(CASE WHEN status_code >= 200 AND status_code < 300 THEN 1 ELSE 0 END) as success_cnt,
                      SUM(CASE WHEN status_code>=400 AND status_code<500 THEN 1 ELSE 0 END) as client_err_cnt,
                      SUM(CASE WHEN status_code>=500 THEN 1 ELSE 0 END) as server_err_cnt,
                      SUM(total_tokens) as total_tokens,
                      CAST(AVG(latency_ms) AS INTEGER) as avg_latency
               FROM request_logs
               WHERE created_at >= ?
               GROUP BY date ORDER BY date""",
            (cutoff_utc7,),
        )
    else:
        daily_stats = fetch_all(
            """SELECT (created_at::timestamptz AT TIME ZONE 'Asia/Shanghai')::date as date,
                      COUNT(*) as cnt,
                      SUM(CASE WHEN status_code >= 200 AND status_code < 300 THEN 1 ELSE 0 END) as success_cnt,
                      SUM(CASE WHEN status_code>=400 AND status_code<500 THEN 1 ELSE 0 END) as client_err_cnt,
                      SUM(CASE WHEN status_code>=500 THEN 1 ELSE 0 END) as server_err_cnt,
                      SUM(total_tokens) as total_tokens,
                      AVG(latency_ms)::int as avg_latency
               FROM request_logs
               WHERE created_at::timestamptz >= NOW() - INTERVAL '7 days'
               GROUP BY date ORDER BY date"""
        )

    # 按客户端ID统计（最近7天并发分布）
    if _is_sqlite:
        concurrency_stats = fetch_all(
            """SELECT client_id, COUNT(*) as cnt,
                      CAST(AVG(latency_ms) AS INTEGER) as avg_latency,
                      SUM(total_tokens) as total_tokens,
                      COUNT(DISTINCT model) as model_count
               FROM request_logs
               WHERE client_id != '' AND created_at >= ?
               GROUP BY client_id ORDER BY cnt DESC LIMIT 10""",
            (cutoff_utc7,),
        )
    else:
        concurrency_stats = fetch_all(
            """SELECT client_id, COUNT(*) as cnt,
                      AVG(latency_ms)::int as avg_latency,
                      SUM(total_tokens) as total_tokens,
                      COUNT(DISTINCT model) as model_count
               FROM request_logs
               WHERE client_id != '' AND created_at::timestamptz >= NOW() - INTERVAL '7 days'
               GROUP BY client_id ORDER BY cnt DESC LIMIT 10"""
        )

    # Top 10 错误类型（含错误详情） - SQLite: GROUP_CONCAT / PG: string_agg
    if _is_sqlite:
        top_errors = fetch_all(
            """SELECT error_type, business_code, COUNT(*) as cnt,
                      GROUP_CONCAT(DISTINCT error_detail) as details
                FROM request_logs
                WHERE log_category IN ('error', 'auth_fail') AND error_type != ''
                GROUP BY error_type, business_code ORDER BY cnt DESC LIMIT 10"""
        )
    else:
        top_errors = fetch_all(
            """SELECT error_type, business_code, COUNT(*) as cnt,
                      string_agg(DISTINCT error_detail, ', ') as details
                FROM request_logs
                WHERE log_category IN ('error', 'auth_fail') AND error_type != ''
                GROUP BY error_type, business_code ORDER BY cnt DESC LIMIT 10"""
        )

    # Top 10 模型（含token统计）
    top_models = fetch_all(
        """SELECT model, COUNT(*) as cnt,
                  SUM(total_tokens) as total_tokens,
                  AVG(latency_ms) as avg_latency
           FROM request_logs
           WHERE model != '' AND model IS NOT NULL
           GROUP BY model ORDER BY cnt DESC LIMIT 10"""
    )

    # 延迟统计
    latency_stats = fetch_one(
        """SELECT AVG(latency_ms) as avg_latency,
                  MAX(latency_ms) as max_latency,
                  SUM(CASE WHEN status_code >= 200 AND status_code < 300 THEN 1 ELSE 0 END) as success_count,
                  SUM(CASE WHEN status_code>=400 AND status_code<500 THEN 1 ELSE 0 END) as client_error_count,
                  SUM(CASE WHEN status_code>=500 OR status_code=0 THEN 1 ELSE 0 END) as server_error_count,
                  SUM(CASE WHEN status_code=401 THEN 1 ELSE 0 END) as auth_fail_count,
                  SUM(CASE WHEN log_category='error' THEN 1 ELSE 0 END) as error_count,
                  SUM(total_tokens) as total_tokens,
                  SUM(prompt_tokens) as total_prompt_tokens,
                  SUM(completion_tokens) as total_completion_tokens
           FROM request_logs"""
    )

    # P95延迟（PG: percentile_cont / SQLite: 用 ORDER BY + LIMIT 子查询近似）
    if _is_sqlite:
        p95_row = fetch_one(
            """SELECT CAST(AVG(latency_ms) AS INTEGER) as p95_latency
               FROM (SELECT latency_ms FROM request_logs WHERE latency_ms > 0
                     ORDER BY latency_ms LIMIT 1 OFFSET (
                       (SELECT CAST(COUNT(*)*0.95 AS INTEGER) FROM request_logs WHERE latency_ms > 0)
                     ))"""
        )
    else:
        p95_row = fetch_one(
            """SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95_latency
               FROM request_logs WHERE latency_ms > 0"""
        )

    # 最近10条错误日志（实时追踪）
    recent_errors = fetch_all(
        """SELECT r.id, r.status_code, r.error_type, r.business_code, r.error_msg,
                  r.error_detail, r.model, r.request_path,
                  COALESCE(c.name, r.client_id) as client_display,
                  r.created_at
           FROM request_logs r
           LEFT JOIN clients c ON r.client_id = c.id
           WHERE r.log_category IN ('error', 'auth_fail')
           ORDER BY r.created_at DESC LIMIT 10"""
    )

    # 成功率计算
    success_rate = round((latency_stats["success_count"] or 0) / total * 100, 2) if total > 0 else 0

    return {
        "total": total,
        "success_rate": success_rate,
        "today_total": today_total,
        "today_success_rate": today_success_rate,
        "today": {
            "total_requests": today_total,
            "success_count": today_success,
            "success_rate": today_success_rate,
            "avg_latency": round(today_stats["avg_latency"] or 0, 2),
        },
        "history": {
            "total_requests": total,
            "success_count": latency_stats["success_count"] or 0,
            "success_rate": success_rate,
        },
        "category_stats": category_stats,
        "status_stats": status_stats,
        "method_stats": method_stats,
        "hourly_stats": hourly_stats,
        "top_paths": top_paths,
        "top_clients": top_clients,
        "top_ips": top_ips,
        "top_errors": top_errors,
        "top_models": top_models,
        "recent_errors": recent_errors,
        "daily_stats": daily_stats,
        "concurrency_stats": concurrency_stats,
        "latency": {
            "avg_ms": round(latency_stats["avg_latency"] or 0, 2),
            "p95_ms": round((p95_row["p95_latency"] or 0), 2),
            "max_ms": latency_stats["max_latency"] or 0,
        },
        "success_count": latency_stats["success_count"] or 0,
        "error_count": latency_stats["error_count"] or 0,
        "auth_fail_count": latency_stats["auth_fail_count"] or 0,
        "total_tokens": latency_stats["total_tokens"] or 0,
        "total_prompt_tokens": latency_stats["total_prompt_tokens"] or 0,
        "total_completion_tokens": latency_stats["total_completion_tokens"] or 0,
    }


@router.delete("/request-logs/cleanup", tags=["管理员"])
async def cleanup_request_logs(
    request: Request,
    days: int = Query(30, ge=1, description="保留最近N天的日志"),
):
    """清理历史请求日志 """
    await require_admin(request)

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    result = execute("DELETE FROM request_logs WHERE created_at < %s", (cutoff,))
    return {"message": f"已清理 {cutoff} 之前的日志", "cutoff": cutoff}


# ========== 审计日志 ==========

@router.get("/audit-logs", tags=["管理员"])
async def audit_logs(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """审计日志"""
    await require_admin(request)
    offset = (page - 1) * page_size
    total = fetch_one("SELECT COUNT(*) as cnt FROM audit_logs")["cnt"]
    rows = fetch_all(
        "SELECT * FROM audit_logs ORDER BY created_at DESC LIMIT %s OFFSET %s",
        (page_size, offset),
    )
    return {"total": total, "page": page, "page_size": page_size, "data": rows}


# ========== 网关策略配置 ==========

@router.get("/global-status", tags=["管理员"])
async def global_status(request: Request):
    """全局状态总览"""
    await require_admin(request)
    scheduler = get_scheduler()
    status = scheduler.get_global_status()
    status["upstream_keys"] = fetch_one("SELECT COUNT(*) as cnt FROM upstream_keys WHERE status='active'")["cnt"]
    status["clients"] = fetch_one("SELECT COUNT(*) as cnt FROM clients WHERE status='active'")["cnt"]
    # v10.0: 熔断器状态
    try:
        from app.circuit_breaker import get_circuit_breaker
        cb = get_circuit_breaker()
        cb_all = cb.get_all_status()
        status["circuit_breakers"] = {
            "total": len(cb_all),
            "open": sum(1 for v in cb_all.values() if v.get("status") == "open"),
            "details": cb_all,
        }
    except Exception:
        status["circuit_breakers"] = {"total": 0, "open": 0, "details": {}}
    return status


# ========== v10.0 熔断器管理 ==========

@router.get("/circuit-breakers", tags=["管理员"])
async def get_circuit_breakers(request: Request):
    """获取所有熔断器状态"""
    await require_admin(request)
    from app.circuit_breaker import get_circuit_breaker
    cb = get_circuit_breaker()
    return cb.get_all_status()

@router.post("/circuit-breakers/reset", tags=["管理员"])
async def reset_circuit_breaker(request: Request, key: str = None):
    """重置熔断器（可指定key，不指定则重置全部）"""
    await require_admin(request)
    from app.circuit_breaker import get_circuit_breaker
    cb = get_circuit_breaker()
    cb.reset(key)
    return {"message": f"熔断器已重置: {key or 'all'}"}


# ========== 模型ID验证机制 ==========

@router.get("/validate-models", tags=["管理员"])
async def validate_models(request: Request):
    """
    模型ID自动化验证：对比本地模型目录与上游有效模型列表
    
    检查项：
    1. 本地目录中存在但上游不存在的模型（需移除）
    2. 上游存在但本地目录缺失的模型（需添加）
    3. /v1/models API 返回的模型与上游的一致性
    """
    await require_admin(request)
    
    from app.nim_models import NIM_MODEL_CATALOG
    from app.security import decrypt_upstream_key
    
    # 获取上游有效模型列表
    upstream_ids = set()
    upstream_models = {}
    try:
        active_keys = fetch_all(
            "SELECT id, api_key_ciphertext FROM upstream_keys WHERE status = 'active' LIMIT 1"
        )
        if active_keys:
            master_key = get_setting("upstream_master_key")
            if master_key:
                api_key = decrypt_upstream_key(active_keys[0]["api_key_ciphertext"], master_key)
                import httpx
                resp = httpx.get(
                    "https://integrate.api.nvidia.com/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=15.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for m in data.get("data", []):
                        upstream_ids.add(m["id"])
                        upstream_models[m["id"]] = m.get("owned_by", "")
    except Exception as e:
        return {"error": f"获取上游模型列表失败: {str(e)}", "upstream_count": 0}
    
    # 本地目录模型ID
    catalog_ids = set(NIM_MODEL_CATALOG.keys())
    
    # 对比
    in_catalog_not_upstream = sorted(catalog_ids - upstream_ids)
    in_upstream_not_catalog = sorted(upstream_ids - catalog_ids)
    
    # 检查 /v1/models API 返回的模型（需要从public_api缓存获取）
    # 由于API是异步的，这里直接用上游数据
    
    return {
        "upstream_count": len(upstream_ids),
        "catalog_count": len(catalog_ids),
        "missing_from_catalog": [
            {"id": mid, "owned_by": upstream_models.get(mid, "")}
            for mid in in_upstream_not_catalog
        ],
        "invalid_in_catalog": [
            {"id": mid, "display_name": NIM_MODEL_CATALOG[mid].display_name}
            for mid in in_catalog_not_upstream
        ],
        "is_synced": len(in_upstream_not_catalog) == 0 and len(in_catalog_not_upstream) == 0,
        "last_check": time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
    }


@router.post("/debug/test", tags=["管理员"])
async def debug_test(req: DebugChatRequest, request: Request):
    """接口调试（直接使用指定密钥测试）
    支持两种模式：
    1. 直接传 api_key 字符串
    2. 传 "__use_key_id__:<key_id>" 由后端自动解密
    """
    await require_admin(request)

    import httpx
    base_url = get_setting("upstream_base_url") or "https://integrate.api.nvidia.com/v1"
    chat_path = get_setting("chat_path") or "/chat/completions"
    url = f"{base_url}{chat_path}"

    # 处理 api_key：如果是 key_id 标记则自动解密
    api_key = req.api_key
    if api_key.startswith("__use_key_id__:"):
        key_id = api_key.split(":", 1)[1]
        row = fetch_one("SELECT api_key_ciphertext FROM upstream_keys WHERE id = %s", (key_id,))
        if not row:
            return {"status_code": 0, "error": "指定的密钥不存在"}
        master_key = get_setting("upstream_master_key")
        if not master_key:
            return {"status_code": 0, "error": "主密钥未配置"}
        try:
            api_key = decrypt_upstream_key(row["api_key_ciphertext"], master_key)
        except Exception as e:
            return {"status_code": 0, "error": f"密钥解密失败: {e}"}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    body = {"model": req.model, "messages": req.messages, "stream": req.stream}
    # 附加额外参数（temperature/max_tokens等）
    for k, v in req.model_dump().items():
        if k not in ("api_key", "model", "messages", "stream") and v is not None:
            body[k] = v

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=body)
            try:
                data = resp.json()
            except Exception:
                data = {"raw": resp.text}
            return {"status_code": resp.status_code, "data": data}
    except Exception as e:
        return {"status_code": 0, "error": str(e)}


@router.get("/nim/models", tags=["管理员"])
async def get_nim_models(request: Request, publisher: str = None, family: str = None):
    """获取NVIDIA NIM模型目录"""
    await require_admin(request)
    from app.nim_models import list_models
    models = list_models(publisher=publisher, family=family)
    return {
        "models": [
            {
                "model_id": m.model_id,
                "display_name": m.display_name,
                "publisher": m.publisher,
                "context_length": m.context_length,
                "max_output_tokens": m.max_output_tokens,
                "supports_streaming": m.supports_streaming,
                "supports_tools": m.supports_tools,
                "supports_images": m.supports_images,
                "model_family": m.model_family,
                "description": m.description,
                "tags": m.tags,
            }
            for m in models
        ],
        "total": len(models),
    }


@router.post("/logs/cleanup", tags=["管理员"])
async def cleanup_logs(request: Request, keep_days: int = Query(3), keep_error_days: int = Query(90)):
    """清理请求日志：成功日志短期保留，错误日志长期保存"""
    await require_admin(request)
    result = cleanup_success_logs(keep_days=keep_days, keep_error_days=keep_error_days)
    insert_audit("cleanup", "request_logs", "", f"日志清理: 删除{result['success_deleted']}条成功日志(>{keep_days}天), {result['error_deleted']}条错误日志(>{keep_error_days}天)")
    return {"message": "清理完成", **result}


@router.get("/stats/latency-distribution", tags=["统计"])
async def get_latency_distribution(request: Request, model: str = Query(None)):
    """耗时分布统计：P50/P90/P99指标，可按模型分组"""
    await require_admin(request)
    model_filter = "WHERE model = %s" if model else ""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    params = (cutoff, model) if model else (cutoff,)
    
    # 各模型P50/P90/P99
    query = f"""
        SELECT model,
               COUNT(*) as request_count,
               AVG(latency_ms) as avg_latency,
               ROUND(AVG(CASE WHEN row_num <= cnt * 0.50 THEN latency_ms END), 1) as p50,
               ROUND(AVG(CASE WHEN row_num <= cnt * 0.90 THEN latency_ms END), 1) as p90,
               ROUND(AVG(CASE WHEN row_num <= cnt * 0.99 THEN latency_ms END), 1) as p99,
               MAX(latency_ms) as max_latency
        FROM (
            SELECT model, latency_ms,
                   ROW_NUMBER() OVER (PARTITION BY model ORDER BY latency_ms) as row_num,
                   COUNT(*) OVER (PARTITION BY model) as cnt
            FROM request_logs
            WHERE created_at >= %s
            AND latency_ms > 0
            { 'AND model = %s' if model else '' }
        )
        GROUP BY model
        ORDER BY request_count DESC
        LIMIT 20
    """
    try:
        rows = fetch_all(query, params)
        return {
            "data": [dict(r) for r in rows] if rows else [],
            "total_models": len(rows) if rows else 0,
        }
    except Exception as e:
        return {"error": str(e), "data": []}


@router.get("/stats/error-analysis", tags=["统计"])
async def get_error_analysis(request: Request, period: str = Query("day", pattern="^(day|week|month)$")):
    """错误类型分析：按429/403/5xx/timeout分组统计"""
    await require_admin(request)
    days = {"day": 1, "week": 7, "month": 30}
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days.get(period, 1))).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    hour_format = '%Y-%m-%dT%H:00:00'
    
    # 按状态码分类统计
    rows = fetch_all(f"""
        SELECT 
            CASE 
                WHEN status_code = 429 THEN '429限流'
                WHEN status_code = 403 THEN '403权限拒绝'
                WHEN status_code = 502 THEN '502网关错误'
                WHEN status_code = 503 THEN '503服务不可用'
                WHEN status_code >= 500 THEN '5xx服务端错误'
                WHEN status_code >= 400 THEN '4xx客户端错误'
                ELSE CAST(status_code AS TEXT)
            END as error_category,
            COUNT(*) as count,
            COALESCE(SUM(total_tokens), 0) as total_tokens,
            ROUND(AVG(latency_ms), 1) as avg_latency
        FROM request_logs 
        WHERE created_at >= %s
        AND status_code >= 400
        GROUP BY error_category
        ORDER BY count DESC
    """, (cutoff,))
    
    # 时间趋势（按小时）
    trend_rows = fetch_all(f"""
        SELECT 
            to_char(created_at::timestamptz, %s) as time_bucket,
            SUM(CASE WHEN status_code = 429 THEN 1 ELSE 0 END) as count_429,
            SUM(CASE WHEN status_code = 403 THEN 1 ELSE 0 END) as count_403,
            SUM(CASE WHEN status_code >= 500 THEN 1 ELSE 0 END) as count_5xx,
            COUNT(*) as total_errors
        FROM request_logs 
        WHERE created_at >= %s
        AND status_code >= 400
        GROUP BY time_bucket
        ORDER BY time_bucket
        LIMIT 168
    """, (hour_format, cutoff))
    
    total = sum((r["count"] for r in rows), 0) if rows else 0
    return {
        "total_errors": total,
        "data": [dict(r) for r in rows] if rows else [],
        "trend": [dict(r) for r in trend_rows] if trend_rows else [],
        "period": period,
    }


@router.get("/stats/request-trend", tags=["统计"])
async def get_request_trend(request: Request, period: str = Query("day", pattern="^(day|week|month)$")):
    """请求量趋势分析，支持日/周/月视图"""
    await require_admin(request)
    
    group_format = {
        "day": "%Y-%m-%dT%H:00:00",
        "week": "%Y-%m-%d",
        "month": "%Y-%m-%d",
    }
    days_map = {"day": 1, "week": 7, "month": 30}
    fmt = group_format.get(period, "%Y-%m-%d")
    d = days_map.get(period, 1)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=d)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    
    rows = fetch_all(f"""
        SELECT 
            to_char(created_at::timestamptz, %s) as time_bucket,
            COUNT(*) as total_requests,
            SUM(CASE WHEN status_code BETWEEN 200 AND 299 THEN 1 ELSE 0 END) as success_count,
            SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as error_count,
            COALESCE(SUM(total_tokens), 0) as total_tokens,
            COALESCE(SUM(prompt_tokens), 0) as prompt_tokens,
            COALESCE(SUM(completion_tokens), 0) as completion_tokens
        FROM request_logs 
        WHERE created_at >= %s
        GROUP BY time_bucket
        ORDER BY time_bucket
    """, (fmt, cutoff))
    
    return {
        "period": period,
        "data": [dict(r) for r in rows] if rows else [],
        "total": sum((r["total_requests"] for r in rows), 0) if rows else 0,
    }
@router.post("/sync-models", tags=["管理员"])
async def sync_upstream_models(request: Request):
    """
    从 NVIDIA 上游实时同步可用模型列表

    调用上游 /v1/models 接口，获取完整模型列表并更新缓存。
    返回同步结果。
    """
    await require_admin(request)

    from app.security import decrypt_upstream_key
    import httpx

    master_key = get_setting("upstream_master_key")
    if not master_key:
        return {"error": "未配置上游主密钥", "count": 0, "models": []}

    active_key = fetch_one(
        "SELECT id, api_key_ciphertext FROM upstream_keys WHERE status = 'active' LIMIT 1"
    )
    if not active_key:
        return {"error": "无活跃上游密钥", "count": 0, "models": []}

    try:
        api_key = decrypt_upstream_key(active_key["api_key_ciphertext"], master_key)
    except Exception as e:
        return {"error": f"解密密钥失败: {e}", "count": 0, "models": []}

    base_url = get_setting("upstream_base_url") or "https://integrate.api.nvidia.com/v1"
    models_path = get_setting("models_path") or "/models"
    url = f"{base_url}{models_path}"

    try:
        from app.public_api import _models_cache, fetch_upstream_models
        # 清除缓存，强制重新获取
        _models_cache["expires"] = 0
        models = await fetch_upstream_models()
        return {"count": len(models), "models": [m.get("id","") for m in models[:50]], "note": "已过滤不可用/弃用模型"}
    except Exception as e:
        logger.error(f"上游模型同步失败: {e}")
        return {"error": str(e), "count": 0, "models": []}


# ========== 网页对话（管理后台独立板块） ==========

class ChatSessionCreateRequest(BaseModel):
    title: str = "新对话"

class ChatSessionRenameRequest(BaseModel):
    title: str

class ChatMessageCreateRequest(BaseModel):
    role: str                       # user / assistant
    content: str = ""
    reasoning: str = ""
    attachments: list = []          # [{name, mime, size, kind, data}]
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0

class ChatSessionTruncateRequest(BaseModel):
    keep: int                       # 保留开头 keep 条消息，其余删除（消息编辑重发/rewind 用）

class ChatConfigUpdateRequest(BaseModel):
    model: Optional[str] = None
    api_key: Optional[str] = None
    stream: Optional[bool] = None


def _chat_touch_session(session_id: str) -> None:
    """刷新会话 updated_at（消息有变动即视为活跃，驱动侧栏按时间分组排序）。"""
    now = utcnow()
    execute("UPDATE chat_sessions SET updated_at = %s WHERE id = %s", (now, session_id))


@router.get("/chat/sessions", tags=["网页对话"])
async def chat_list_sessions(request: Request):
    """会话列表（全量，前端按 今天/7天内/30天内/更早 分组）"""
    await require_admin(request)
    rows = fetch_all(
        "SELECT s.id, s.title, s.created_at, s.updated_at, "
        "(SELECT COUNT(*) FROM chat_messages m WHERE m.session_id = s.id) AS message_count "
        "FROM chat_sessions s ORDER BY s.updated_at DESC"
    )
    return rows


@router.post("/chat/sessions", tags=["网页对话"])
async def chat_create_session(req: ChatSessionCreateRequest, request: Request):
    """新建会话"""
    await require_admin(request)
    sid = str(uuid.uuid4())
    now = utcnow()
    title = (req.title or "新对话").strip()[:60] or "新对话"
    execute(
        "INSERT INTO chat_sessions (id, title, created_at, updated_at) VALUES (%s, %s, %s, %s)",
        (sid, title, now, now),
    )
    return {"id": sid, "title": title, "created_at": now, "updated_at": now, "message_count": 0}


@router.get("/chat/sessions/{session_id}/messages", tags=["网页对话"])
async def chat_get_messages(session_id: str, request: Request):
    """拉取某会话全部消息（按时间升序）"""
    await require_admin(request)
    sess = fetch_one("SELECT id FROM chat_sessions WHERE id = %s", (session_id,))
    if not sess:
        raise HTTPException(status_code=404, detail={"message": "会话不存在", "type": "not_found", "code": "session_not_found"})
    rows = fetch_all(
        "SELECT id, session_id, role, content, reasoning, attachments, model, "
        "prompt_tokens, completion_tokens, latency_ms, created_at "
        "FROM chat_messages WHERE session_id = %s ORDER BY created_at ASC, rowid ASC",
        (session_id,),
    )
    for r in rows:
        try:
            r["attachments"] = json.loads(r.get("attachments") or "[]")
        except Exception:
            r["attachments"] = []
    return rows


@router.post("/chat/sessions/{session_id}/messages", tags=["网页对话"])
async def chat_append_message(session_id: str, req: ChatMessageCreateRequest, request: Request):
    """追加一条消息；首条用户消息自动生成会话标题"""
    await require_admin(request)
    sess = fetch_one("SELECT id, title FROM chat_sessions WHERE id = %s", (session_id,))
    if not sess:
        raise HTTPException(status_code=404, detail={"message": "会话不存在", "type": "not_found", "code": "session_not_found"})
    if req.role not in ("user", "assistant"):
        raise HTTPException(status_code=400, detail={"message": "role 仅支持 user/assistant", "type": "invalid_request", "code": "invalid_role"})

    mid = str(uuid.uuid4())
    now = utcnow()
    attachments_json = json.dumps(req.attachments or [], ensure_ascii=False)
    # 附件以 dataURL 形式入库，单条消息体过大会拖垮本地库，给个硬上限（约8MB）
    if len(attachments_json) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail={"message": "附件总大小超过 8MB 限制", "type": "payload_too_large", "code": "payload_too_large"})

    execute(
        "INSERT INTO chat_messages (id, session_id, role, content, reasoning, attachments, model, "
        "prompt_tokens, completion_tokens, latency_ms, created_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (mid, session_id, req.role, req.content or "", req.reasoning or "", attachments_json,
         req.model or "", req.prompt_tokens or 0, req.completion_tokens or 0, req.latency_ms or 0, now),
    )

    # 首条用户消息 → 用内容截断作为会话标题（前端显示为历史会话名）
    if req.role == "user" and sess["title"] in ("新对话", ""):
        base = (req.content or "").strip()
        if not base and req.attachments:
            base = "、".join(a.get("name", "") for a in req.attachments[:3] if a.get("name"))
        if base:
            title = base.replace("\n", " ").strip()[:30]
            execute("UPDATE chat_sessions SET title = %s WHERE id = %s", (title, session_id))

    _chat_touch_session(session_id)
    return {"id": mid, "session_id": session_id, "role": req.role, "created_at": now}


@router.put("/chat/sessions/{session_id}", tags=["网页对话"])
async def chat_rename_session(session_id: str, req: ChatSessionRenameRequest, request: Request):
    """重命名会话"""
    await require_admin(request)
    sess = fetch_one("SELECT id FROM chat_sessions WHERE id = %s", (session_id,))
    if not sess:
        raise HTTPException(status_code=404, detail={"message": "会话不存在", "type": "not_found", "code": "session_not_found"})
    title = (req.title or "").strip()[:60]
    if not title:
        raise HTTPException(status_code=400, detail={"message": "标题不能为空", "type": "invalid_request", "code": "empty_title"})
    execute("UPDATE chat_sessions SET title = %s WHERE id = %s", (title, session_id))
    _chat_touch_session(session_id)
    return {"id": session_id, "title": title}


@router.post("/chat/sessions/{session_id}/truncate", tags=["网页对话"])
async def chat_truncate_session(session_id: str, req: ChatSessionTruncateRequest, request: Request):
    """截断会话：保留开头 keep 条消息，删除其余（消息编辑重发/rewind 用）"""
    await require_admin(request)
    sess = fetch_one("SELECT id FROM chat_sessions WHERE id = %s", (session_id,))
    if not sess:
        raise HTTPException(status_code=404, detail={"message": "会话不存在", "type": "not_found", "code": "session_not_found"})
    if req.keep < 0:
        raise HTTPException(status_code=400, detail={"message": "keep 不能为负数", "type": "invalid_request", "code": "invalid_keep"})
    rows = fetch_all(
        "SELECT id FROM chat_messages WHERE session_id = %s ORDER BY created_at ASC, rowid ASC",
        (session_id,),
    )
    drop = rows[req.keep:]
    for r in drop:
        execute("DELETE FROM chat_messages WHERE id = %s", (r["id"],))
    if drop:
        _chat_touch_session(session_id)
    insert_audit("chat_session_truncate", "chat_session", session_id,
                 f"rewind截断: 保留前{min(req.keep, len(rows))}条, 删除{len(drop)}条")
    return {"kept": min(req.keep, len(rows)), "deleted": len(drop)}


@router.delete("/chat/sessions/{session_id}", tags=["网页对话"])
async def chat_delete_session(session_id: str, request: Request):
    """删除会话及其全部消息（外键 ON DELETE CASCADE）"""
    await require_admin(request)
    n = execute("DELETE FROM chat_sessions WHERE id = %s", (session_id,))
    if not n:
        raise HTTPException(status_code=404, detail={"message": "会话不存在", "type": "not_found", "code": "session_not_found"})
    execute("DELETE FROM chat_messages WHERE session_id = %s", (session_id,))
    insert_audit("chat_session_delete", "chat_session", session_id, "删除网页对话会话")
    return {"deleted": True, "id": session_id}


@router.get("/chat/config", tags=["网页对话"])
async def chat_get_config(request: Request):
    """网页对话默认配置（模型 / API Key / 流式），供设置二级页回显"""
    await require_admin(request)
    raw = get_setting("chat_config")
    cfg = {}
    if raw:
        try:
            cfg = json.loads(raw)
        except Exception:
            cfg = {}
    demo_key = ""
    try:
        demo_key = os.environ.get("MASTER_API_LOCAL_DEMO_KEY", "")
    except Exception:
        pass
    return {
        "model": cfg.get("model", ""),
        "api_key": cfg.get("api_key", ""),
        "stream": bool(cfg.get("stream", True)),
        "demo_key": demo_key,
    }


@router.put("/chat/config", tags=["网页对话"])
async def chat_update_config(req: ChatConfigUpdateRequest, request: Request):
    """保存网页对话配置（写入 admin_settings.chat_config，全端生效）"""
    await require_admin(request)
    raw = get_setting("chat_config")
    cfg = {}
    if raw:
        try:
            cfg = json.loads(raw)
        except Exception:
            cfg = {}
    if req.model is not None:
        cfg["model"] = req.model.strip()
    if req.api_key is not None:
        cfg["api_key"] = req.api_key.strip()
    if req.stream is not None:
        cfg["stream"] = bool(req.stream)
    set_setting("chat_config", json.dumps(cfg, ensure_ascii=False))
    return {"saved": True, "config": cfg}

