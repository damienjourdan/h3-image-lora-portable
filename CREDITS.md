# Crédits

Aucun des projets ci-dessous n'est inclus dans ce dépôt : ils sont téléchargés à l'installation ou
au premier usage, chacun sous sa propre licence.

## ai-toolkit — Ostris (licence MIT)

[ostris/ai-toolkit](https://github.com/ostris/ai-toolkit), Copyright (c) 2024 Ostris, LLC.
C'est le moteur d'entraînement : `1_INSTALLER.bat` clone le dépôt et utilise son gestionnaire
d'installation. Ce projet n'en copie pas le code ; `_scripts/run_job.py` ajuste certains de ses
modules en mémoire, au lancement d'un job, sans modifier ses fichiers. L'adaptateur d'entraînement
MiniMax H3 d'Ostris est téléchargé depuis Hugging Face.

## Fizgig — Peter Neill (licence Apache 2.0)

[shootthesound/Fizgig](https://github.com/shootthesound/Fizgig), Copyright (c) 2026 Peter Neill.
Les éléments suivants ont été réécrits dans ce projet à partir de la lecture de son code et de sa
documentation (version du 16 novembre 2026), sans copie de son code :

- les réglages du procédé « Rapide 512 » : résolution 0,25 MP, rang 32, Automagic v4 plafonné à
  2,5e-4, EMA 0,98, 30 passages sur le dataset ;
- le calendrier de l'optimisation de ressemblance (part des pas au niveau de bruit le plus net) ;
- les réglages de l'échantillonneur LCM (bruit frais de 1,1 à 0,9) ;
- les repères de mémoire et la règle de marge de RAM ;
- la conversion de son adaptateur d'entraînement H3-Image vers le format d'ai-toolkit.

L'adaptateur d'entraînement Fizgig est téléchargé depuis
[Hugging Face](https://huggingface.co/ShootTheSound/Fizgig-MiniMax-H3-Image-Training-Adapter) sous
les conditions de son éditeur.

## Outils téléchargés à l'installation

- [uv](https://github.com/astral-sh/uv) — Astral (licences MIT et Apache 2.0)
- [MinGit](https://github.com/git-for-windows/git) — Git for Windows (GPL v2)
- [Gradio](https://github.com/gradio-app/gradio) (Apache 2.0), installé avec les dépendances d'ai-toolkit
- [OpenCV](https://github.com/opencv/opencv) (Apache 2.0) et le détecteur de texte PP-OCRv3 de
  [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (Apache 2.0), via
  [OpenCV Zoo](https://github.com/opencv/opencv_zoo)
- Qwen3-VL (Apache 2.0), téléchargé par ai-toolkit pour le légendage automatique

## Modèles

MiniMax-H3-Image (MiniMax), reconditionné par [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3-Image),
sous la MiniMax-H3-Image Community License.
