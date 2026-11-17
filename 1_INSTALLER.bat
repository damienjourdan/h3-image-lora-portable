@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"
call "%~dp0_scripts\lang.bat"
title !M_I_TITLE!

echo.
echo  ==============================================================
echo    !M_I_B1!
echo    !M_I_B2!
echo  ==============================================================
echo    !M_I_B3!
echo    !M_I_B4!
echo    !M_I_B5!
echo  ==============================================================
echo.

REM ---------------------------------------------------------------- 0. Verifs
set "HERE=%~dp0"
if not "%HERE%"=="%HERE: =%" (
    echo !M_E_SPACES1!
    echo          %HERE%
    echo          !M_E_SPACES2!
    echo          !M_E_SPACES3!
    pause & exit /b 1
)

where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo !M_E_NOSMI!
    pause & exit /b 1
)
echo !M_I_GPU!
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo.
echo  !M_I_DRV1!
echo  !M_I_DRV2!
echo.

call "%~dp0_scripts\env.bat"
set "DL=%ROOT%\_scripts\dl.bat"

REM Espace disque : ~15 Go d environnement + 35 a 50 Go de modeles
powershell -NoProfile -Command "if ((Get-Item -LiteralPath '%ROOT%').PSDrive.Free -lt 80GB) { exit 1 }"
if errorlevel 1 echo !M_W_DISK!
if not exist "%ROOT%\_outils" mkdir "%ROOT%\_outils"

REM ---------------------------------------------------------------- 1. Git
where git >nul 2>&1
if not errorlevel 1 (
    echo !M_I_GIT_OK!
    goto :git_ok
)
echo !M_I_GIT_DL!
call "%DL%" "https://github.com/git-for-windows/git/releases/download/v2.55.0.windows.3/MinGit-2.55.0.3-64-bit.zip" "%ROOT%\_outils\mingit.zip"
if errorlevel 1 ( pause & exit /b 1 )
REM Controle d integrite (SHA-256 de l archive officielle 2.55.0.3 64-bit)
powershell -NoProfile -Command "if ((Get-FileHash -Algorithm SHA256 -LiteralPath '%ROOT%\_outils\mingit.zip').Hash -ne 'F48E2D2DC74A24454ADC6D8FD0AC25BF9C2386F19CFB06202B9465AAAD4F9F05') { exit 1 }"
if errorlevel 1 (
    echo !M_E_SHA!
    del "%ROOT%\_outils\mingit.zip" >nul 2>&1
    pause & exit /b 1
)
powershell -NoProfile -Command "Expand-Archive -Force '%ROOT%\_outils\mingit.zip' '%ROOT%\_outils\mingit'"
if not exist "%ROOT%\_outils\mingit\cmd\git.exe" (
    echo !M_E_MINGIT!
    del "%ROOT%\_outils\mingit.zip" >nul 2>&1
    pause & exit /b 1
)
:git_ok

REM ---------------------------------------------------------------- 2. Depot
set GIT_LFS_SKIP_SMUDGE=1
if exist "%AITK%\run.py" (
    echo !M_I_REPO_OK!
    goto :repo_ok
)
if exist "%AITK%\.git" (
    echo !M_I_REPO_RESUME!
    pushd "%AITK%"
    git fetch --progress origin
    git checkout -f -B main origin/main
    popd
) else (
    if exist "%AITK%" rd /s /q "%AITK%"
    echo !M_I_REPO_CLONE!
    git clone --progress https://github.com/ostris/ai-toolkit.git "%AITK%"
)
if not exist "%AITK%\run.py" (
    echo !M_E_CLONE!
    pause & exit /b 1
)
:repo_ok
if defined AITK_COMMIT (
    echo       !M_I_PINNED! %AITK_COMMIT%
    git -C "%AITK%" checkout --detach %AITK_COMMIT%
    if errorlevel 1 ( echo !M_E_COMMIT! & pause & exit /b 1 )
)
REM Le gestionnaire integre cherche son MinGit local dans ai-toolkit\.mingit
if exist "%ROOT%\_outils\mingit\cmd\git.exe" if not exist "%AITK%\.mingit\cmd\git.exe" (
    robocopy "%ROOT%\_outils\mingit" "%AITK%\.mingit" /E /NFL /NDL /NJH /NJS >nul
)

REM ---------------------------------------------------------------- 3. uv + Python
if exist "%AITK%\.uv\uv.exe" (
    echo !M_I_UV_OK!
) else (
    echo !M_I_UV_DL!
    call "%DL%" "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip" "%ROOT%\_outils\uv.zip"
    if errorlevel 1 ( pause & exit /b 1 )
    powershell -NoProfile -Command "Expand-Archive -Force '%ROOT%\_outils\uv.zip' '%AITK%\.uv'"
    if not exist "%AITK%\.uv\uv.exe" (
        echo !M_E_UV!
        del "%ROOT%\_outils\uv.zip" >nul 2>&1
        pause & exit /b 1
    )
)
cd /d "%AITK%"
echo       !M_I_PY!
REM --no-bin / --no-registry : rien dans le profil utilisateur ni le registre
uv python install 3.12 --no-bin --no-registry
set "PY="
for /f "delims=" %%P in ('uv python find 3.12 --managed-python') do set "PY=%%P"
if not defined PY (
    echo !M_E_PY!
    pause & exit /b 1
)

REM ---------------------------------------------------------------- 4. Env complet
echo.
echo !M_I_ENV1!
echo       !M_I_ENV2!
echo       !M_I_ENV3!
echo       !M_I_ENV4!
echo.
set /a TRY=0
:install_retry
set /a TRY+=1
"%PY%" "%ROOT%\_scripts\install_sans_node.py" install
if errorlevel 1 (
    if !TRY! LSS 3 (
        echo.
        echo !M_I_RETRY1! !TRY!/3 !M_I_RETRY2!
        timeout /t 10 /nobreak >nul
        goto :install_retry
    )
    echo.
    echo !M_E_INSTALL1!
    echo          !M_E_INSTALL2!
    pause & exit /b 1
)

echo.
echo  ==============================================================
echo    !M_I_DONE!
echo  ==============================================================
echo.
choice /c !M_YN! /m "!M_Q_PREDL!"
if errorlevel 2 goto :fin
call "%ROOT%\4_TELECHARGER_MODELE.bat" nopause

:fin
echo.
echo  !M_I_LAUNCH!
echo.
pause
