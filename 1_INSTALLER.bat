@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title Installation AI Toolkit portable - MiniMax-H3-Image LoRA

echo.
echo  ==============================================================
echo    AI Toolkit (ostris) - version portable, SANS Node.js ni npm
echo    Cible : LoRA MiniMax-H3-Image sur RTX 3090 24 Go / 64 Go RAM
echo  ==============================================================
echo    Interruption possible a tout moment : relancez ce fichier,
echo    chaque etape deja faite est sautee et les telechargements
echo    partiels reprennent la ou ils s'etaient arretes.
echo  ==============================================================
echo.

REM ---------------------------------------------------------------- 0. Verifs
set "HERE=%~dp0"
if not "%HERE%"=="%HERE: =%" (
    echo [ERREUR] Le chemin du dossier contient des espaces :
    echo          %HERE%
    echo          Certains outils CUDA/triton cassent dans ce cas.
    echo          Deplacez le dossier, par ex. vers D:\AI\H3-LoRA\
    pause & exit /b 1
)

where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] nvidia-smi introuvable : pilote NVIDIA absent ?
    pause & exit /b 1
)
echo GPU detecte :
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo.
echo  Info : torch 2.13 cu130 demande un pilote recent (serie 580+).
echo         Avec un pilote plus ancien, l'installateur bascule seul en cu126.
echo.

call "%~dp0_scripts\env.bat"
set "DL=%ROOT%\_scripts\dl.bat"

REM Espace disque : ~15 Go d environnement + 35 a 50 Go de modeles
powershell -NoProfile -Command "if ((Get-Item -LiteralPath '%ROOT%').PSDrive.Free -lt 80GB) { exit 1 }"
if errorlevel 1 echo [ATTENTION] Moins de 80 Go libres sur ce disque : risque de manquer de place.
if not exist "%ROOT%\_outils" mkdir "%ROOT%\_outils"

REM ---------------------------------------------------------------- 1. Git
where git >nul 2>&1
if not errorlevel 1 (
    echo [1/4] Git deja present.
    goto :git_ok
)
echo [1/4] MinGit portable (~40 Mo)...
call "%DL%" "https://github.com/git-for-windows/git/releases/download/v2.55.0.windows.3/MinGit-2.55.0.3-64-bit.zip" "%ROOT%\_outils\mingit.zip"
if errorlevel 1 ( pause & exit /b 1 )
REM Controle d integrite (SHA-256 de l archive officielle 2.55.0.3 64-bit)
powershell -NoProfile -Command "if ((Get-FileHash -Algorithm SHA256 -LiteralPath '%ROOT%\_outils\mingit.zip').Hash -ne 'F48E2D2DC74A24454ADC6D8FD0AC25BF9C2386F19CFB06202B9465AAAD4F9F05') { exit 1 }"
if errorlevel 1 (
    echo [ERREUR] Empreinte SHA-256 de MinGit incorrecte : archive supprimee, relancez.
    del "%ROOT%\_outils\mingit.zip" >nul 2>&1
    pause & exit /b 1
)
powershell -NoProfile -Command "Expand-Archive -Force '%ROOT%\_outils\mingit.zip' '%ROOT%\_outils\mingit'"
if not exist "%ROOT%\_outils\mingit\cmd\git.exe" (
    echo [ERREUR] Archive MinGit invalide : elle va etre retelechargee.
    del "%ROOT%\_outils\mingit.zip" >nul 2>&1
    pause & exit /b 1
)
:git_ok

REM ---------------------------------------------------------------- 2. Depot
set GIT_LFS_SKIP_SMUDGE=1
if exist "%AITK%\run.py" (
    echo [2/4] Depot deja present.
    goto :repo_ok
if defined AITK_COMMIT (
    echo       Version figee sur le commit %AITK_COMMIT%
    git -C "%AITK%" checkout --detach %AITK_COMMIT%
    if errorlevel 1 ( echo [ERREUR] Commit introuvable. & pause & exit /b 1 )
)
)
if exist "%AITK%\.git" (
    echo [2/4] Clonage precedent interrompu : reprise...
    pushd "%AITK%"
    git fetch --progress origin
    git checkout -f -B main origin/main
    popd
) else (
    if exist "%AITK%" rd /s /q "%AITK%"
    echo [2/4] Clonage de ostris/ai-toolkit...
    git clone --progress https://github.com/ostris/ai-toolkit.git "%AITK%"
)
if not exist "%AITK%\run.py" (
    echo [ERREUR] Clonage incomplet. Relancez 1_INSTALLER.bat pour reprendre.
    pause & exit /b 1
)
:repo_ok
if defined AITK_COMMIT (
    echo       Version figee sur le commit %AITK_COMMIT%
    git -C "%AITK%" checkout --detach %AITK_COMMIT%
    if errorlevel 1 ( echo [ERREUR] Commit introuvable. & pause & exit /b 1 )
)
REM Le gestionnaire integre cherche son MinGit local dans ai-toolkit\.mingit
if exist "%ROOT%\_outils\mingit\cmd\git.exe" if not exist "%AITK%\.mingit\cmd\git.exe" (
    robocopy "%ROOT%\_outils\mingit" "%AITK%\.mingit" /E /NFL /NDL /NJH /NJS >nul
)

REM ---------------------------------------------------------------- 3. uv + Python
if exist "%AITK%\.uv\uv.exe" (
    echo [3/4] uv deja present.
) else (
    echo [3/4] uv portable, ~20 Mo...
    call "%DL%" "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip" "%ROOT%\_outils\uv.zip"
    if errorlevel 1 ( pause & exit /b 1 )
    powershell -NoProfile -Command "Expand-Archive -Force '%ROOT%\_outils\uv.zip' '%AITK%\.uv'"
    if not exist "%AITK%\.uv\uv.exe" (
        echo [ERREUR] Archive uv invalide : elle va etre retelechargee.
        del "%ROOT%\_outils\uv.zip" >nul 2>&1
        pause & exit /b 1
    )
)
cd /d "%AITK%"
echo       Python 3.12 portable (progression affichee par uv)...
REM --no-bin / --no-registry : rien dans le profil utilisateur ni le registre
uv python install 3.12 --no-bin --no-registry
set "PY="
for /f "delims=" %%P in ('uv python find 3.12 --managed-python') do set "PY=%%P"
if not defined PY (
    echo [ERREUR] Python 3.12 introuvable. Relancez pour reprendre.
    pause & exit /b 1
)

REM ---------------------------------------------------------------- 4. Env complet
echo.
echo [4/4] Environnement Python : torch, dependances, FFmpeg, triton-windows,
echo       flash-attn (~10 Go, 10 a 30 min). Progression affichee par uv.
echo       Les paquets deja telecharges restent dans cache\uv : une relance
echo       reprend sans tout retelecharger.
echo.
set /a TRY=0
:install_retry
set /a TRY+=1
"%PY%" "%ROOT%\_scripts\install_sans_node.py" install
if errorlevel 1 (
    if !TRY! LSS 3 (
        echo.
        echo [i] Echec, nouvelle tentative !TRY!/3 dans 10 s, reprise depuis le cache...
        timeout /t 10 /nobreak >nul
        goto :install_retry
    )
    echo.
    echo [ERREUR] L'installation a echoue. Relancez 1_INSTALLER.bat pour reprendre,
    echo          ou lancez DIAGNOSTIC.bat et gardez la sortie.
    pause & exit /b 1
)

echo.
echo  ==============================================================
echo    Installation terminee (aucun Node.js / npm installe).
echo  ==============================================================
echo.
choice /c ON /m "Pre-telecharger maintenant les poids MiniMax-H3-Image (~30 Go)"
if errorlevel 2 goto :fin
call "%ROOT%\4_TELECHARGER_MODELE.bat" nopause

:fin
echo.
echo  Lancez 2_LANCER_UI.bat pour ouvrir l'interface (http://127.0.0.1:7860).
echo.
pause
