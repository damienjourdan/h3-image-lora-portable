@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"
call "%~dp0_scripts\lang.bat"
title !M_L_TITLE!
call "%~dp0_scripts\env.bat"

if not exist "%AITK%\.venv\Scripts\python.exe" (
    echo !M_E_NOENV!
    pause & exit /b 1
)

REM FFmpeg local (installe par le gestionnaire) visible des entrainements
if exist "%AITK%\.ffmpeg\bin" set "PATH=%AITK%\.ffmpeg\bin;%PATH%"
set "PATH=%AITK%\.venv\Scripts;%PATH%"

echo.
echo  !M_L_1!
echo  !M_L_2!
echo  !M_L_3!
echo  !M_L_4!
echo.
"%AITK%\.venv\Scripts\python.exe" "%ROOT%\_scripts\h3_ui.py"
pause
