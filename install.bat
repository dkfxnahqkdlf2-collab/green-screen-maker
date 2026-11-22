@echo off
chcp 65001 > nul
title 크로마키 배경 제거기 — 설치

echo.
echo ╔══════════════════════════════════════════╗
echo ║      크로마키 배경 제거기 설치 중...      ║
echo ╚══════════════════════════════════════════╝
echo.

:: Python 확인
python --version > nul 2>&1
if errorlevel 1 (
    echo [오류] Python이 설치되어 있지 않습니다.
    echo.
    echo  https://www.python.org/downloads/ 에서
    echo  Python 3.10 이상을 설치한 뒤 다시 실행하세요.
    echo  ※ 설치 시 "Add Python to PATH" 반드시 체크!
    pause
    exit /b 1
)

echo [1/2] 필수 패키지 설치 중... (최초 1회, 수분 소요)
pip install rembg onnxruntime-directml pillow numpy opencv-python imageio-ffmpeg tkinterdnd2 tqdm

echo.
echo [2/2] 설치 완료!
echo.
echo  이제 "실행.bat" 를 더블클릭하면 프로그램이 실행됩니다.
echo  (최초 실행 시 AI 모델 다운로드로 약 1분 소요)
echo.
pause
