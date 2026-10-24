@echo off
chcp 65001 >nul
cd /d "%~dp0"
title CarotKcat Web UI
echo ============================================
echo   CarotKcat 预测服务启动中...
echo   启动后用浏览器打开: http://127.0.0.1:7860
echo   关闭本窗口即停止服务
echo ============================================
echo.
"C:\Users\Administrator\.workbuddy\binaries\python\envs\kcat\Scripts\python.exe" "C:\Users\Administrator\WorkBuddy\Kcat\app.py"
pause
