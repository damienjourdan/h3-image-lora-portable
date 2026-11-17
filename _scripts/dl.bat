@echo off
REM ==========================================================================
REM  Telechargement avec barre de progression + reprise
REM  Usage : call dl.bat "URL" "C:\chemin\fichier"
REM  - ecrit dans fichier.part, renomme seulement une fois complet
REM  - un .part existant est REPRIS la ou il s'est arrete (curl -C -)
REM  - 5 tentatives, puis repli sur BITS (barre de progression Windows)
REM ==========================================================================
setlocal
set "URL=%~1"
set "OUT=%~2"
set "PART=%~2.part"

if exist "%OUT%" goto :deja
if exist "%PART%" echo   reprise du telechargement partiel : %~nx2

where curl.exe >nul 2>&1
if errorlevel 1 goto :bits

set /a TRY=0
:retry
set /a TRY+=1
curl.exe -L --fail --progress-bar -C - --connect-timeout 20 -o "%PART%" "%URL%"
set "RC=%errorlevel%"
if "%RC%"=="0" goto :ok
REM 33 / 22 avec un .part : reprise refusee par le serveur, on repart de zero
if "%RC%"=="33" del "%PART%" >nul 2>&1
if "%RC%"=="22" del "%PART%" >nul 2>&1
if %TRY% GEQ 5 goto :bits
echo   echec curl code %RC%, nouvelle tentative %TRY%/5 dans 5 s...
timeout /t 5 /nobreak >nul
goto :retry

:bits
echo   repli sur BITS...
del "%PART%" >nul 2>&1
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Start-BitsTransfer -Source '%URL%' -Destination '%PART%' -ErrorAction Stop; exit 0 } catch { Write-Host $_; exit 1 }"
if errorlevel 1 goto :echec

:ok
move /y "%PART%" "%OUT%" >nul
exit /b 0

:deja
echo   deja present : %~nx2
exit /b 0

:echec
echo   [ERREUR] echec du telechargement de %~nx2
echo            relancez le script : il reprendra.
exit /b 1
