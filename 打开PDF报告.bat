@echo off
chcp 65001 > nul
echo ======================================================
echo   正在打开 ZenGo 围棋 AI 实验报告 PDF...
echo ======================================================
start "" "%~dp0experiment_report.pdf"
exit

