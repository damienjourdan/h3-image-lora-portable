# H3-Image LoRA Portable

A portable Windows installer and a Python graphical interface (Gradio) to train **MiniMax-H3-Image**
LoRAs with [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit), tuned for an
**RTX 3090 24 GB** and **64 GB of RAM**. No Node.js or npm: ai-toolkit's official Next.js interface
is replaced by a Gradio interface.

> **Languages:** the interface, the `.bat` scripts and the manual are available in English (default),
> French, German, Spanish, Portuguese, Japanese and Mandarin. Manuals are in [manuals/](manuals/)
> ([English](manuals/en_README.txt)).

> **Status: experimental.** Validated only on an RTX 3090 under Windows 11, with the "Standard"
> process (int8 base, 35 % of the layers offloaded to RAM). Options marked "experimental" in the
> interface have not been verified on real hardware. Feedback is welcome in the *Issues*.

## Features

- **Portable installation**: MinGit, uv, Python 3.12, torch and FFmpeg are downloaded into the
  folder; nothing is installed system-wide. Downloads show progress and resume after interruptions.
- **Training**: 24 GB presets, a 20-step *quick test* (measures speed and memory, estimates the
  total time), pause with full state saving, resume, job queue, scheduled pause.
- **Choice of training process**: standard ai-toolkit, fast 512 px, or the full Fizgig recipe
  (Fizgig training adapter, likeness optimization, LCM previews).
- **Dataset tools**: automatic captioning (Qwen3-VL) adapted to the LoRA type, caption and image
  size checks, automatic cropping of watermarks and text.
- **Monitoring**: live previews and loss curve, checkpoint comparison, copy to ComfyUI.
- **Seven languages**: a selector next to the title switches the interface between English, French,
  German, Spanish, Portuguese, Japanese and Mandarin; every `.bat` script starts with the same choice.
- **Hardware**: VRAM / RAM / temperature monitor, sleep prevention, thermal alert and pause,
  adjustable GPU power limit and thermostat, GPU electricity cost (spent so far and projected at the
  end of the job, in the currency of your choice), notifications (sound, phone via ntfy.sh).

## Requirements

- Windows 10 or 11, a 24 GB NVIDIA GPU (recent driver: 580 series or later for CUDA 13).
- 64 GB of RAM recommended, 80 GB of disk space (environment + model weights).
- A folder path **without spaces**, for example `D:\AI\H3-LoRA\`.

## Installation

1. Download the repository: **Code → Download ZIP**, or the latest version under **Releases**.
2. Unzip it into a folder whose path contains no spaces.
3. Run `1_INSTALLER.bat` (10 to 30 minutes; it can be restarted at any time and resumes where it stopped).
4. Run `2_LANCER_UI.bat`: the interface opens at http://127.0.0.1:7860.

## Measurements on an RTX 3090

Author's measurements, from the *quick test* and real training runs, at 1024 + 1536 resolution:

| Setting | Speed | Peak VRAM |
|---|---|---|
| int8 base, 35 % in RAM | 4.0 to 6.5 s per step | 19.6 to 20.6 GB |
| w4a8 base | 15.2 s per step | 19.0 GB |

GPU temperature has a large effect on speed: an RTX 3090 throttles above 83 °C. The "fast 512 px"
process has not been measured on this card yet.

## Models and licenses

This repository contains **no model weights**. The MiniMax-H3-Image weights (repackaged by
Comfy-Org) and the training adapters are downloaded from Hugging Face, each under its own license,
which you accept when downloading them. In particular, read the
[MiniMax-H3-Image Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE)
before any use: it may restrict the use of the model and of the images it produces.

## Credits

This project builds on ai-toolkit (Ostris) and reuses settings and methods published by Fizgig.
Details in [CREDITS.md](CREDITS.md).

## License

The code in this repository is released under the [MIT](LICENSE) license. It does not cover
ai-toolkit, the tools downloaded during installation, or the model weights.
