@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================================
echo   ZenGo 围棋 AI Web 系统 (KOALA 面试展示)
echo   正在启动后端服务与前端界面，请稍候...
echo ========================================================
echo.
echo 本地访问地址: http://127.0.0.1:8000
echo (注意: 必须是 http://，不能是 https://)
echo.
start "" http://127.0.0.1:8000
.venv\Scripts\python.exe start_web.py
pause
