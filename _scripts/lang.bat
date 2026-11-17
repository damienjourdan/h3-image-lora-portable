@echo off
REM ==========================================================================
REM  Choix de la langue au debut de chaque script .bat
REM  Usage : call "%~dp0_scripts\lang.bat"
REM  - propose les 7 langues ; sans reponse sous 10 s, garde la derniere choisie
REM  - enregistre le choix dans language.txt (partage avec l'interface web)
REM  - charge les messages M_* de _scripts\lang\<code>.bat
REM ==========================================================================
for %%I in ("%~dp0..") do set "H3_ROOT=%%~fI"
set "H3_LANG=en"
if exist "%H3_ROOT%\language.txt" set /p H3_LANG=<"%H3_ROOT%\language.txt"
set "H3_DEF=1"
if /i "%H3_LANG%"=="fr" set "H3_DEF=2"
if /i "%H3_LANG%"=="de" set "H3_DEF=3"
if /i "%H3_LANG%"=="es" set "H3_DEF=4"
if /i "%H3_LANG%"=="pt" set "H3_DEF=5"
if /i "%H3_LANG%"=="ja" set "H3_DEF=6"
if /i "%H3_LANG%"=="zh" set "H3_DEF=7"
echo.
echo   [1] English   [2] Français   [3] Deutsch   [4] Español
echo   [5] Português   [6] 日本語   [7] 中文
echo.
choice /c 1234567 /t 10 /d %H3_DEF% /n /m "  Language / Langue / Sprache / Idioma / 言語 / 语言 [1-7] (10 s -> %H3_DEF%) : "
set "H3_SEL=%errorlevel%"
set "H3_LANG=en"
if "%H3_SEL%"=="2" set "H3_LANG=fr"
if "%H3_SEL%"=="3" set "H3_LANG=de"
if "%H3_SEL%"=="4" set "H3_LANG=es"
if "%H3_SEL%"=="5" set "H3_LANG=pt"
if "%H3_SEL%"=="6" set "H3_LANG=ja"
if "%H3_SEL%"=="7" set "H3_LANG=zh"
> "%H3_ROOT%\language.txt" echo %H3_LANG%
call "%~dp0lang\%H3_LANG%.bat"
echo.
exit /b 0
