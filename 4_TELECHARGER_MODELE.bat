@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title Telechargement MiniMax-H3-Image
call "%~dp0_scripts\env.bat"

REM Optionnel : l'UI telecharge aussi ces fichiers au 1er entrainement
REM (onglet "Modeles" de l'UI = meme chose, avec barre de progression).
REM Reprise : un fichier partiel (.incomplete) est repris la ou il s'est arrete,
REM que la coupure vienne du reseau, d'une fermeture de fenetre ou d'un reboot.

set "HF=%AITK%\.venv\Scripts\hf.exe"
if not exist "%HF%" (
    echo [ERREUR] Environnement absent. Lancez d'abord 1_INSTALLER.bat
    if not "%~1"=="nopause" pause
    exit /b 1
)
set HF_HUB_DISABLE_PROGRESS_BARS=

echo.
echo  Quel transformer telecharger ?
echo   [1] int8  - defaut, chemin sur pour la 3090, avec offload en RAM
echo   [2] w4a8  - plus leger mais EXPERIMENTAL sur 3090, voir LISEZMOI
echo   [3] les deux
echo.
choice /c 123 /m "Choix"
set "SEL=%errorlevel%"

set "FILES=text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors vae/minimax_h3_image_vae_fp16.safetensors"
if "%SEL%"=="1" set "FILES=%FILES% diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors"
if "%SEL%"=="2" set "FILES=%FILES% diffusion_models/minimax_h3_image_pruned_w4a8.safetensors"
if "%SEL%"=="3" set "FILES=%FILES% diffusion_models/minimax_h3_image_pruned_w4a8.safetensors diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors"

echo.
echo Telechargement vers %MODELS_PATH%
echo (une barre de progression par fichier : taille, debit, temps restant)
echo.
set /a TRY=0
:retry_main
set /a TRY+=1
"%HF%" download Comfy-Org/MiniMax-H3-Image %FILES% --local-dir "%MODELS_PATH%"
if errorlevel 1 (
    if !TRY! LSS 5 (
        echo.
        echo [i] Coupure : nouvelle tentative !TRY!/5 dans 10 s, reprise du fichier partiel...
        timeout /t 10 /nobreak >nul
        goto :retry_main
    )
    echo [ERREUR] Telechargement interrompu. Relancez ce script : il reprendra.
    if not "%~1"=="nopause" pause
    exit /b 1
)

echo.
echo Adaptateur d'entrainement (assistant LoRA, requis pour ce modele distille)...
set /a TRY=0
:retry_adapter
set /a TRY+=1
REM ai-toolkit cherche l adaptateur sous MODELS_PATH\loras
"%HF%" download ostris/minimax_h3_training_adapter minimax_h3_image_training_adapter_v1.safetensors --local-dir "%MODELS_PATH%\loras\training_adapters"
if errorlevel 1 if !TRY! LSS 5 (
    timeout /t 10 /nobreak >nul
    goto :retry_adapter
)
echo.
echo Termine.
if not "%~1"=="nopause" pause
exit /b 0
