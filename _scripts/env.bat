@echo off
REM ==========================================================================
REM  Variables d'environnement communes - tout reste dans le dossier portable
REM  Appele par les autres scripts : call "%~dp0_scripts\env.bat"
REM ==========================================================================

REM Racine du dossier portable (ce fichier est dans _scripts\)
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "AITK=%ROOT%\ai-toolkit"

REM Neutralise tout Python / conda / pyenv systeme qui pourrait interferer
set PYTHONPATH=
set PYTHONHOME=
set PYTHONSTARTUP=
set PYTHONUSERBASE=
set VIRTUAL_ENV=
set CONDA_PREFIX=
set CONDA_DEFAULT_ENV=
set PYENV_ROOT=
set PYENV_VERSION=
set PYTHONUTF8=1
set PYTHONNOUSERSITE=1

REM Modeles, caches et telechargements : dans le dossier portable, hors du depot
REM (les mises a jour du depot ne touchent jamais aux modeles telecharges)
set "MODELS_PATH=%ROOT%\models"
set "HF_HOME=%ROOT%\cache\huggingface"
set "HF_HUB_DISABLE_TELEMETRY=1"
set "TORCH_HOME=%ROOT%\cache\torch"
set "TRITON_CACHE_DIR=%ROOT%\cache\triton"
set "UV_CACHE_DIR=%ROOT%\cache\uv"
set "UV_PYTHON_INSTALL_DIR=%AITK%\.uv\python"

REM Python : uniquement celui telecharge par uv dans le dossier, jamais un
REM Python systeme. Ne PAS utiliser UV_PYTHON_PREFERENCE : il est incompatible
REM avec l option --managed-python que le gestionnaire d ai-toolkit passe a uv.
set "UV_MANAGED_PYTHON=1"
set UV_PYTHON_PREFERENCE=

REM Optionnel : figer ai-toolkit sur un commit precis (reproductibilite).
REM Commit audite pour ce paquet : 4a7f28532ed6fe81d9f8c00718878bb3b8739015
REM Laisser vide = derniere version de la branche main.
set "AITK_COMMIT="
REM Le bouton "Figer la version" de l UI ecrit le commit dans ce fichier :
if exist "%ROOT%\_scripts\version_figee.txt" set /p AITK_COMMIT=<"%ROOT%\_scripts\version_figee.txt"

REM Outils portables (uv, MinGit) en tete du PATH
set "PATH=%AITK%\.uv;%AITK%\.mingit\cmd;%ROOT%\_outils\mingit\cmd;%PATH%"

if not exist "%ROOT%\models" mkdir "%ROOT%\models"
if not exist "%ROOT%\cache" mkdir "%ROOT%\cache"
if not exist "%ROOT%\datasets" mkdir "%ROOT%\datasets"
exit /b 0
