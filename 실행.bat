@echo off
chcp 65001 > nul
title 크로마키 배경 제거기
cd /d "%~dp0"
python app.py
if errorlevel 1 (
    echo.
    echo [오류] 실행 중 문제가 발생했습니다.
    pause
)
