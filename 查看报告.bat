@echo off
chcp 65001 > nul
echo ======================================================
echo   正在打开 ZenGo 围棋 AI 实验报告 (LaTeX 排版)...
echo ======================================================
start "" "%~dp0experiment_report.html"
echo.
echo 提示:
echo 1. 网页已使用 KaTeX 渲染完整数学公式与三线表。
echo 2. 点击右上角 "打印 / 保存为 PDF" 按钮，或在浏览器中按 Ctrl+P，即可一键导出为标准的 A4 实验报告 PDF！
echo 3. 完整的 LaTeX 源码已保存在同目录下的 experiment_report.tex。
echo ======================================================
pause

