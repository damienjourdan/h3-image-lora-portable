@echo off
chcp 65001 >nul
REM Lance l UI avec les droits administrateur : necessaire uniquement pour
REM la limite de puissance du GPU (onglet Outils). Windows demande confirmation.
REM La langue est demandee par 2_LANCER_UI.bat, dans la fenetre administrateur.
powershell -NoProfile -Command "Start-Process -FilePath '%~dp02_LANCER_UI.bat' -Verb RunAs"
