"""
网关服务主入口 - FastAPI应用（个人自用精简版）

集成：
- 密钥池调度器（SurgeScheduler）
- OpenAI兼容公开API（/v1/chat/completions, /v1/models）
- 管理后台API（/gw/admin/*）
- 多协议转换（Anthropic/Gemini/Ollama）
- 后台周期任务（冷却/健康度/预判/自愈/预热）
"""
# 规避 Windows 下 Python 3.14 的 _wmi 模块在系统 WMI 挂起时卡死
try:
    import platform
    platform._wmi = None
except Exception:
    pass

import asyncio
import os
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger("acu.gateway")

# 从项目根目录加载 .env 文件
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(str(_env_path))
    logger.info(f"[网关] 已加载环境变量: {_env_path}")

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles

from app.database import init_db, seed_defaults
from app.middleware import setup_middleware, start_log_worker, stop_log_worker
from app.scheduler import get_scheduler
from app.public_api import router as public_router
from app.admin_api import router as admin_router
from app.admin_panel import create_admin

STATIC_DIR = Path(__file__).resolve().parent / "static"


# ========== 后台任务 ==========

_background_task = None


async def run_background_tasks():
    """后台周期任务：冷却/健康度/负载预判/异常自愈/冷密钥预热"""
    scheduler = get_scheduler()

    logger.info("[网关] 后台任务已启动（调度器周期算法）")

    while True:
        try:
            await scheduler.run_background_tasks()
        except Exception as e:
            logger.info(f"[网关] 后台任务异常: {e}")

        # 每30秒执行一次（各算法内部有独立的周期控制）
        await asyncio.sleep(30)


# ========== 应用生命周期 ==========

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭生命周期"""
    global _background_task

    # 启动：初始化数据库
    init_db()
    seed_defaults()
    logger.info("[网关] 数据库初始化完成")

    # 初始化调度器HTTP连接池
    scheduler = get_scheduler()
    await scheduler._ensure_pools()
    logger.info("[网关] HTTP连接池已创建")

    # 启动后台任务
    _background_task = asyncio.create_task(run_background_tasks())

    # === v10.0 平台适配器初始化 ===
    try:
        from app.platforms import register_all_adapters
        register_all_adapters()
        logger.info("[网关] v10.0 平台适配器已注册 (nvidia, openai)")
    except Exception as e:
        logger.info(f"[网关] 平台适配器注册失败(可降级运行): {e}")

    # === v10.0 多级缓存初始化 ===
    try:
        from app.cache.multilevel import get_cache
        cache = get_cache()
        logger.info(f"[网关] v10.0 多级缓存已初始化 (L1 maxsize={cache.l1._maxsize}, L2 ttl={cache.l2._default_ttl}s)")
    except Exception as e:
        logger.info(f"[网关] 多级缓存初始化失败(可降级运行): {e}")

    # Initialize async database connection pool
    try:
        from app.database import async_get_pool
        await async_get_pool()
        logger.info("[网关] 异步数据库连接池已创建")
    except Exception as e:
        logger.info(f"[网关] 异步数据库连接池创建失败(可降级运行): {e}")

    # Start batch log worker
    start_log_worker()
    logger.info("[网关] 批量日志写入器已启动")

    yield

    # 关闭：清理资源
    if _background_task:
        _background_task.cancel()

    # Stop batch log worker
    await stop_log_worker()
    logger.info("[网关] 批量日志写入器已停止")

    # Close async database pool
    try:
        from app.db_async_pool import close_async_pool
        await close_async_pool()
        logger.info("[网关] 异步数据库连接池已关闭")
    except Exception:
        pass

    # === v10.0 关闭平台适配器连接池 ===
    try:
        from app.platforms.base import PlatformAdapterRegistry
        await PlatformAdapterRegistry.close_all()
        logger.info("[网关] v10.0 平台适配器连接池已关闭")
    except Exception:
        pass

    if scheduler._http_pool and not scheduler._http_pool.is_closed:
        await scheduler._http_pool.aclose()
    if scheduler._stream_pool and not scheduler._stream_pool.is_closed:
        await scheduler._stream_pool.aclose()
    logger.info("[网关] 服务关闭")


# ========== FastAPI 应用 ==========

app = FastAPI(
    title="Master API Gateway",
    version="10.0.0",
    description="多平台AI API网关 | 17算法互锁调度 | 平台适配器 | 智能路由 | 协议转换 | 运行时健康追踪",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)

# 注册中间件
setup_middleware(app)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/gw/static", StaticFiles(directory=str(STATIC_DIR)), name="gw_static")

# 注册API路由
app.include_router(public_router)
app.include_router(admin_router)

# 挂载 SQLAdmin 数据库管理面板
admin_panel = create_admin(app)


# ========== OpenAI兼容错误处理器 ==========

@app.exception_handler(HTTPException)
async def openai_error_handler(request: Request, exc: HTTPException):
    """将HTTPException格式化为OpenAI标准错误格式"""
    path = request.url.path
    is_openai_route = path.startswith("/v1/") or path.startswith("/api/v1/")

    # v10.0: 使用统一的GatewayError系统
    if is_openai_route:
        error_detail = exc.detail
        if isinstance(error_detail, dict):
            message = error_detail.get("message", str(error_detail))
            error_type = error_detail.get("type", "invalid_request_error")
            code = error_detail.get("code", "unknown")
        else:
            message = str(error_detail)
            error_type = "invalid_request_error"
            code = "unknown"
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": message,
                    "type": error_type,
                    "code": code,
                    "param": None,
                }
            },
        )

    # 管理后台等路由保持原有格式
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc.detail)})

# v10.0: 捕获 GatewayError 并转换为 OpenAI 标准格式
from app.errors import GatewayError

@app.exception_handler(GatewayError)
async def gateway_error_handler(request: Request, exc: GatewayError):
    """将GatewayError转换为OpenAI标准错误格式"""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_openai_error(),
        headers={"X-Error-Code": exc.code, "X-Error-Provider": exc.provider} if exc.provider else None,
    )


# ========== 基础路由 ==========

@app.get("/healthz", tags=["公共"])
async def healthz():
    """健康检查"""
    scheduler = get_scheduler()
    status = scheduler.get_global_status()
    resource = scheduler.get_resource_status()  # New method
    # v10.0: 获取熔断器状态
    try:
        from app.circuit_breaker import get_circuit_breaker
        cb = get_circuit_breaker()
        cb_status = cb.get_all_status()
        open_circuits = {k: v for k, v in cb_status.items() if v.get("status") == "open"}
    except Exception:
        open_circuits = {}
    return {
        "status": "ok",
        "version": "10.0.0",
        "degraded_mode": status["degraded_mode"],
        "healthy_keys": status["healthy_key_count"],
        "resources": resource,
        "circuit_breakers": {
            "total": len(cb_status) if 'cb_status' in dir() else 0,
            "open": len(open_circuits),
            "open_details": open_circuits if open_circuits else {},
        },
    }


@app.get("/robots.txt", tags=["公共"])
async def robots_txt():
    """搜索引擎爬虫规则"""
    content = (
        "User-agent: *\n"
        "Allow: /\n"
        "Disallow: /admin/\n"
        "Disallow: /gw/admin/\n"
    )
    return Response(content=content, media_type="text/plain")


@app.get("/favicon.ico", tags=["公共"])
async def favicon():
    """返回统一favicon图标"""
    ico_path = STATIC_DIR / "favicon.ico"
    if ico_path.exists():
        return FileResponse(str(ico_path), media_type="image/x-icon")
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse, tags=["公共"])
async def index_page():
    """网页前端：不再提供试玩对话页，只提示“配置网关请到管理员后台”。

    网页对话板块已移入管理控制台（/admin/console → 网页对话）。
    """
    html = """<!DOCTYPE html>
<html lang="zh"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Master API Gateway</title>
<link rel="icon" type="image/png" href="/static/brand.png">
<script>
(function(){try{var t=localStorage.getItem('masterapi-theme');if(t!=='light'&&t!=='dark'){t='dark';}document.documentElement.setAttribute('data-theme',t);}catch(e){document.documentElement.setAttribute('data-theme','dark');}})();
</script>
<style>
:root, html[data-theme="dark"] {
  color-scheme: dark;
  --bg: #0a0a0b; --bg-card: #131315; --bg-card-2: #1a1a1d; --bg-inset: #0d0d0f;
  --border: #1f1f23; --border-strong: #2d2d32;
  --text: #f4f4f5; --text-sec: #a1a1aa; --text-dim: #62626b;
  --accent: #f4f4f5; --accent-hover: #dcdcdf; --accent-press: #c7c7cc; --accent-contrast: #0a0a0b;
  --accent-soft: rgba(255,255,255,.07);
  --success: #22c55e; --info: #52525b; --info-soft: rgba(82,82,91,.08);
}
html[data-theme="light"] {
  color-scheme: light;
  --bg: #f6f6f7; --bg-card: #ffffff; --bg-card-2: #f4f4f5; --bg-inset: #f4f4f5;
  --border: #e8e8ea; --border-strong: #d9d9de;
  --text: #131316; --text-sec: #66666f; --text-dim: #9b9ba3;
  --accent: #131316; --accent-hover: #2c2c31; --accent-press: #3a3a41; --accent-contrast: #ffffff;
  --accent-soft: rgba(0,0,0,.055);
  --success: #16a34a; --info: #52525b; --info-soft: rgba(82,82,91,.08);
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { font-family: Inter, "Segoe UI Variable Text", "Segoe UI", "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif; }
body { background: var(--bg); color: var(--text); min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 24px; -webkit-font-smoothing: antialiased; }
.wrap { width: 100%; max-width: 480px; }
.card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 20px; padding: 38px 36px; box-shadow: 0 24px 64px rgba(0,0,0,.18); }
.brand { display: flex; align-items: center; gap: 13px; }
.logo-mark { width: 46px; height: 46px; border-radius: 13px; object-fit: cover; display: block; border: 1px solid var(--border); background: var(--bg-card-2); }
.brand-name { font-size: 19px; font-weight: 700; letter-spacing: .01em; line-height: 1.25; }
.brand-name span { display: block; font-size: 11px; font-weight: 500; color: var(--text-dim); letter-spacing: .12em; text-transform: uppercase; margin-top: 2px; }
.status-row { display: flex; align-items: center; gap: 8px; margin: 24px 0 18px; font-size: 13px; color: var(--text-sec); }
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--success); box-shadow: 0 0 0 3px color-mix(in srgb, var(--success) 22%, transparent); flex-shrink: 0; }
.notice {
  display: flex; gap: 11px; align-items: center;
  background: var(--bg-inset); border: 1px solid var(--border-strong); border-left: 3px solid var(--accent);
  border-radius: 12px; padding: 13px 16px; margin-bottom: 22px;
  font-size: 14px; font-weight: 600; color: var(--text);
}
.notice svg { width: 18px; height: 18px; flex-shrink: 0; color: var(--text); }
.actions { display: flex; gap: 10px; flex-wrap: wrap; }
.btn {
  display: inline-flex; align-items: center; gap: 7px; text-decoration: none;
  border: 1px solid var(--border-strong); background: transparent; color: var(--text);
  border-radius: 11px; padding: 11px 22px; font-size: 13.5px; font-weight: 600; cursor: pointer;
  transition: background .15s, border-color .15s, transform .08s ease;
}
.btn:hover { background: var(--bg-card-2); }
.btn:active { transform: scale(.97); }
.btn-primary { background: var(--accent); border-color: var(--accent); color: var(--accent-contrast); }
.btn-primary:hover { background: var(--accent-hover); border-color: var(--accent-hover); }
.btn svg { width: 15px; height: 15px; }
.theme-toggle { position: fixed; top: 18px; right: 18px; width: 38px; height: 38px; border-radius: 11px; border: 1px solid var(--border-strong); background: var(--bg-card); color: var(--text-sec); cursor: pointer; display: flex; align-items: center; justify-content: center; transition: background .15s, color .15s, transform .08s ease; }
.theme-toggle:hover { background: var(--bg-card-2); color: var(--text); }
.theme-toggle:active { transform: scale(.93); }
.theme-toggle svg { width: 16px; height: 16px; }
</style>
</head><body>
<button class="theme-toggle" id="themeToggle" title="切换深浅模式" aria-label="切换深浅模式"></button>
<div class="wrap">
  <div class="card">
    <div class="brand">
      <img class="logo-mark" src="/static/brand.png" alt="Master API">
      <div class="brand-name">Master API<span>AI Gateway</span></div>
    </div>
    <div class="status-row"><span class="status-dot"></span>网关服务运行中 · OpenAI 兼容端点已就绪</div>
    <div class="notice">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 9v4"/><path d="M12 17h.01"/><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/></svg>
      <span>配置网关请到管理员后台</span>
    </div>
    <div class="actions">
      <a class="btn btn-primary" href="/admin">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
        进入管理员后台
      </a>
    </div>
  </div>
</div>
<script>
(function(){
  var btn = document.getElementById('themeToggle');
  var ICONS = {
    dark: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><line x1="12" y1="2" x2="12" y2="4.5"/><line x1="12" y1="19.5" x2="12" y2="22"/><line x1="2" y1="12" x2="4.5" y2="12"/><line x1="19.5" y1="12" x2="22" y2="12"/><line x1="4.9" y1="4.9" x2="6.7" y2="6.7"/><line x1="17.3" y1="17.3" x2="19.1" y2="19.1"/><line x1="4.9" y1="19.1" x2="6.7" y2="17.3"/><line x1="17.3" y1="6.7" x2="19.1" y2="4.9"/></svg>',
    light: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>'
  };
  function cur(){ return document.documentElement.getAttribute('data-theme') || 'dark'; }
  function paint(){ btn.innerHTML = cur() === 'dark' ? ICONS.light : ICONS.dark; }
  btn.addEventListener('click', function(){
    var n = cur() === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', n);
    try { localStorage.setItem('masterapi-theme', n); } catch(e) {}
    paint();
  });
  paint();
})();
</script>
</body></html>
"""
    return HTMLResponse(content=html)


@app.get("/admin", response_class=HTMLResponse, tags=["管理后台"])
async def admin_login_page():
    """管理员登录页"""
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="zh-CN" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>管理员登录 - Master API</title>
<link rel="icon" type="image/png" href="/static/brand.png">
<script>
(function(){try{var t=localStorage.getItem('masterapi-theme');if(t!=='light'&&t!=='dark'){t='dark';}document.documentElement.setAttribute('data-theme',t);}catch(e){document.documentElement.setAttribute('data-theme','dark');}})();
</script>
<style>
:root, html[data-theme="dark"] {
  color-scheme: dark;
  --bg: #0a0a0b; --bg-card: #131315; --bg-card-2: #1a1a1d; --bg-inset: #0d0d0f;
  --border: #1f1f23; --border-strong: #2d2d32;
  --text: #f4f4f5; --text-sec: #a1a1aa; --text-dim: #62626b;
  --accent: #f4f4f5; --accent-hover: #dcdcdf; --accent-press: #c7c7cc; --accent-contrast: #0a0a0b;
  --accent-soft: rgba(255,255,255,.07);
  --danger: #f04438; --danger-soft: rgba(240,68,56,.13);
  --shadow-pop: 0 24px 64px rgba(0,0,0,.55);
}
html[data-theme="light"] {
  color-scheme: light;
  --bg: #f6f6f7; --bg-card: #ffffff; --bg-card-2: #f4f4f5; --bg-inset: #f4f4f5;
  --border: #e8e8ea; --border-strong: #d9d9de;
  --text: #131316; --text-sec: #66666f; --text-dim: #9b9ba3;
  --accent: #131316; --accent-hover: #2c2c31; --accent-press: #3a3a41; --accent-contrast: #ffffff;
  --accent-soft: rgba(0,0,0,.055);
  --danger: #d92d20; --danger-soft: rgba(217,45,32,.08);
  --shadow-pop: 0 24px 64px rgba(16,16,20,.18);
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; font-family: Inter, "Segoe UI Variable Text", "Segoe UI", "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif; background: var(--bg); color: var(--text); -webkit-font-smoothing: antialiased; }
body { display: flex; align-items: center; justify-content: center; }
.login-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 20px; padding: 40px 36px; width: 100%; max-width: 380px; margin: 20px; box-shadow: var(--shadow-pop); }
.brand { display: flex; align-items: center; gap: 12px; margin-bottom: 30px; }
.logo-mark { width: 40px; height: 40px; border-radius: 12px; object-fit: cover; display: block; border: 1px solid var(--border); background: var(--bg-card-2); }
.brand-name { font-size: 17px; font-weight: 700; letter-spacing: .01em; line-height: 1.2; }
.brand-name span { display: block; font-size: 10.5px; font-weight: 500; color: var(--text-dim); letter-spacing: .1em; text-transform: uppercase; }
.field label { display: block; font-size: 12.5px; font-weight: 500; color: var(--text-sec); margin-bottom: 7px; }
.field input { width: 100%; padding: 11px 13px; border-radius: 11px; border: 1px solid var(--border-strong); background: var(--bg-inset); color: var(--text); font-size: 14px; outline: none; transition: border-color .15s, box-shadow .15s; }
.field input:focus { border-color: var(--text-dim); box-shadow: 0 0 0 3px var(--accent-soft); }
.login-btn { width: 100%; margin-top: 18px; padding: 12px; border-radius: 11px; border: 1px solid var(--accent); background: var(--accent); color: var(--accent-contrast); font-size: 14px; font-weight: 600; cursor: pointer; transition: background .15s, transform .08s ease, box-shadow .08s ease; }
.login-btn:hover { background: var(--accent-hover); border-color: var(--accent-hover); }
.login-btn:active { background: var(--accent-press); border-color: var(--accent-press); transform: scale(.97); box-shadow: inset 0 2px 8px rgba(0,0,0,.3); }
.login-btn:disabled { opacity: .5; cursor: not-allowed; }
.err { color: var(--danger); font-size: 12.5px; margin-top: 12px; min-height: 18px; text-align: center; }
.theme-toggle { position: fixed; top: 18px; right: 18px; width: 38px; height: 38px; border-radius: 11px; border: 1px solid var(--border-strong); background: var(--bg-card); color: var(--text-sec); cursor: pointer; display: flex; align-items: center; justify-content: center; transition: background .15s, color .15s, transform .08s ease, box-shadow .08s ease; }
.theme-toggle:hover { background: var(--bg-card-2); color: var(--text); }
.theme-toggle:active { transform: scale(.93); box-shadow: inset 0 2px 8px rgba(0,0,0,.3); }
.theme-toggle svg { width: 16px; height: 16px; }
</style>
</head>
<body>
<button class="theme-toggle" id="themeToggle" title="切换深浅模式" aria-label="切换深浅模式"></button>
<div class="login-card">
  <div class="brand">
    <img class="logo-mark" src="/static/brand.png" alt="Master API">
    <div class="brand-name">Master API<span>Gateway Console</span></div>
  </div>
  <form id="loginForm">
    <div class="field">
      <label>管理员密码</label>
      <input type="password" id="password" placeholder="请输入管理员密码" autocomplete="current-password">
    </div>
    <button type="submit" class="login-btn" id="loginBtn">登 录</button>
    <p class="err" id="msg"></p>
  </form>
</div>
<script>
(function(){
  var btn = document.getElementById('themeToggle');
  var ICONS = {
    dark: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><line x1="12" y1="2" x2="12" y2="4.5"/><line x1="12" y1="19.5" x2="12" y2="22"/><line x1="2" y1="12" x2="4.5" y2="12"/><line x1="19.5" y1="12" x2="22" y2="12"/><line x1="4.9" y1="4.9" x2="6.7" y2="6.7"/><line x1="17.3" y1="17.3" x2="19.1" y2="19.1"/><line x1="4.9" y1="19.1" x2="6.7" y2="17.3"/><line x1="17.3" y1="6.7" x2="19.1" y2="4.9"/></svg>',
    light: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>'
  };
  function cur(){ return document.documentElement.getAttribute('data-theme') || 'dark'; }
  function paint(){ btn.innerHTML = cur() === 'dark' ? ICONS.light : ICONS.dark; }
  btn.addEventListener('click', function(){
    var n = cur() === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', n);
    try { localStorage.setItem('masterapi-theme', n); } catch(e) {}
    paint();
  });
  paint();
})();
document.getElementById('loginForm').onsubmit = async (e) => {
  e.preventDefault();
  const pwd = document.getElementById('password').value;
  const btn = document.getElementById('loginBtn');
  const msg = document.getElementById('msg');
  if (!pwd) { msg.textContent = '请输入密码'; return; }
  btn.disabled = true;
  msg.textContent = '';
  try {
    const resp = await fetch('/gw/admin/login', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({password: pwd})
    });
    const data = await resp.json();
    if (resp.ok) {
      const token = data.token || data.access_token;
      if (token) localStorage.setItem('t', token);
      window.location.href = '/admin/console';
    } else {
      var errMsg = data.detail && typeof data.detail === 'object' ? data.detail.message || '登录失败' : data.detail || data.message || '登录失败';
      msg.textContent = errMsg;
      btn.disabled = false;
    }
  } catch(err) {
    msg.textContent = '网络错误，请检查连接';
    btn.disabled = false;
  }
};
</script>
</body>
</html>
""")


@app.get("/admin/console", response_class=HTMLResponse, tags=["管理后台"])
async def admin_console_page():
    """管理控制台主页 - v10.0 完整12页面"""
    console_path = STATIC_DIR / "console.html"
    if console_path.exists():
        # no-cache：每次都向服务器校验，避免浏览器长期缓存旧版控制台脚本
        return HTMLResponse(
            content=console_path.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-cache"},
        )
    return HTMLResponse(content="<h1>控制台文件不存在</h1>", status_code=500)


# ========== 启动配置 ==========

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
