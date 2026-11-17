@echo off
chcp 65001 >nul
cd /d "%~dp0"
title AI Toolkit - Mise a jour
call "%~dp0_scripts\env.bat"

if not exist "%AITK%\.venv\Scripts\python.exe" (
    echo [ERREUR] Environnement absent. Lancez d'abord 1_INSTALLER.bat
    pause & exit /b 1
)
cd /d "%AITK%"
if defined AITK_COMMIT (
    echo Version figee sur le commit %AITK_COMMIT%
    echo Le code n est pas mis a jour. Pour le mettre a jour : bouton Defiger
    echo de l onglet Outils, puis relancez ce script.
    goto :sync
)
echo Mise a jour du code : vos modeles, datasets et sorties ne sont pas touches...
set GIT_LFS_SKIP_SMUDGE=1
git pull --ff-only --progress
if errorlevel 1 echo [i] git pull impossible, modifications locales ou reseau : on garde le code actuel.
:sync
echo.
echo Synchronisation des dependances (sans Node.js / npm)...
"%AITK%\.venv\Scripts\python.exe" "%ROOT%\_scripts\install_sans_node.py" sync
pause
