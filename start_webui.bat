@echo off
chcp 65001 >nul
cd /d "%~dp0"
title CarotKcat Web UI

REM 防止系统代理拦截 Gradio 启动自检（127.0.0.1 不应走代理，否则报 502）
set NO_PROXY=127.0.0.1,localhost,::1
set no_proxy=127.0.0.1,localhost,::1
set HTTP_PROXY=
set HTTPS_PROXY=
set http_proxy=
set https_proxy=
set ALL_PROXY=
set all_proxy=
echo ============================================
echo   CarotKcat 预测服务启动中
echo.
echo   首次启动需加载 AI 模型到显卡，
echo   请耐心等待 1-2 分钟（期间无输出属正常）
echo.
echo   看到 "Running on http://127.0.0.1:7860"
echo   后用浏览器打开该地址即可
echo   关闭本窗口即停止服务
echo ============================================
echo.
"C:\Users\Administrator\.workbuddy\binaries\python\envs\kcat\Scripts\python.exe" -u "C:\Users\Administrator\WorkBuddy\Kcat\app.py"
pause
