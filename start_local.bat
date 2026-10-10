@echo off
rem ============================================================
rem  Master API Windows 本地单机启动脚本（SQLite 模式）
rem  双击即可一键拉起，默认访问 http://127.0.0.1:8000 管理后台
rem ============================================================
cd /d "%~dp0"

rem 默认环境标记
set DATABASE_BACKEND=sqlite
set MASTER_API_SQLITE_PATH=master.db

rem 读取 .env （如果存在则覆盖）
for /f "usebackq tokens=1,* delims==" %%a in ("%~dp0.env") do (
    set "%%a=%%b"
)

rem 防止代理拦截本地回环
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "ALL_PROXY="
set "http_proxy="
set "https_proxy="
set "all_proxy="
set "SSL_CERT_FILE="
set "SSL_CERT_DIR="

echo [start] 正在拉起 Master API @ http://127.0.0.1:8000
echo [start] 管理控制台：http://127.0.0.1:8000/admin/console
echo [start] OpenAI 兼容端点：http://127.0.0.1:8000/v1
echo.

rem 启动网关 Gateway 核心服务（纯 HTTP 模式）
venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --app-dir gateway

if %errorlevel% neq 0 (
    echo.
    echo [error] 网关异常退出，请检查上方错误信息
    pause
)