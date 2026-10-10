@echo off
title Master API 停止
cd /d "%~dp0"

echo [stop] 正在关闭 Master API（端口 8000 的所有进程）...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    taskkill /F /T /PID %%a >nul 2>&1
)

echo [stop] 等待端口释放（约 2 秒）...
ping -n 3 127.0.0.1 >nul
echo [stop] 8000 端口服务已关闭。