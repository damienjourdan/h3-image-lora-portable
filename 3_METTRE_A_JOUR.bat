@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"
call "%~dp0_scripts\lang.bat"
title !M_U_TITLE!
call "%~dp0_scripts\env.bat"

if not exist "%AITK%\.venv\Scripts\python.exe" (
    echo !M_E_NOENV!
    pause & exit /b 1
)
cd /d "%AITK%"
if defined AITK_COMMIT (
    echo !M_I_PINNED! %AITK_COMMIT%
    echo !M_U_PIN2!
    echo !M_U_PIN3!
    goto :sync
)
echo !M_U_PULL!
set GIT_LFS_SKIP_SMUDGE=1
git pull --ff-only --progress
if errorlevel 1 echo !M_U_PULLFAIL!
:sync
echo.
echo !M_U_SYNC!
"%AITK%\.venv\Scripts\python.exe" "%ROOT%\_scripts\install_sans_node.py" sync
pause
