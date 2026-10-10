@echo off
title Master API 服务
cd /d "%~dp0"

echo [restart] 正在关闭现有 8000 端口服务...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    taskkill /F /T /PID %%a >nul 2>&1
)
ping -n 2 127.0.0.1 >nul

echo [restart] 正在启动网关服务...
set DATABASE_BACKEND=sqlite
set MASTER_API_SQLITE_PATH=master.db
for /f "usebackq tokens=1,* delims==" %%a in ("%~dp0.env") do (
    set "%%a=%%b"
)
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "ALL_PROXY="
set "http_proxy="
set "https_proxy="
set "all_proxy="
set "SSL_CERT_FILE="
set "SSL_CERT_DIR="

echo [restart] 正在启动 Master API @ http://127.0.0.1:8000
echo [restart] 管理控制台：http://127.0.0.1:8000/admin/console
echo [restart] 公开端点：http://127.0.0.1:8000/v1
echo.
venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --app-dir gateway

if %errorlevel% neq 0 (
    echo.
    echo [error] 网关异常退出，请检查上方错误信息
    pause
)