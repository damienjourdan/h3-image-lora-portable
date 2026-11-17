@echo off
REM Lance l UI avec les droits administrateur : necessaire uniquement pour
REM la limite de puissance du GPU (onglet Outils). Windows demande confirmation.
powershell -NoProfile -Command "Start-Process -FilePath '%~dp02_LANCER_UI.bat' -Verb RunAs"
