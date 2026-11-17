@echo off
chcp 65001 >nul
cd /d "%~dp0"
title H3-Image LoRA - UI (http://127.0.0.1:7860)
call "%~dp0_scripts\env.bat"

if not exist "%AITK%\.venv\Scripts\python.exe" (
    echo [ERREUR] Environnement absent. Lancez d'abord 1_INSTALLER.bat
    pause & exit /b 1
)

REM FFmpeg local (installe par le gestionnaire) visible des entrainements
if exist "%AITK%\.ffmpeg\bin" set "PATH=%AITK%\.ffmpeg\bin;%PATH%"
set "PATH=%AITK%\.venv\Scripts;%PATH%"

echo.
echo  UI : http://127.0.0.1:7860
echo  Fermer cette fenetre arrete l'UI ET l'entrainement en cours
echo  (relancer avec le meme nom de job reprend au dernier checkpoint).
echo.
"%AITK%\.venv\Scripts\python.exe" "%ROOT%\_scripts\h3_ui.py"
pause
