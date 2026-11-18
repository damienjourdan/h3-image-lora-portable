# Credits

None of the projects below is included in this repository: they are downloaded during installation
or on first use, each under its own license.

## ai-toolkit — Ostris (MIT license)

[ostris/ai-toolkit](https://github.com/ostris/ai-toolkit), Copyright (c) 2024 Ostris, LLC.
It is the training engine: `1_INSTALLER.bat` clones the repository and uses its installation
manager. This project does not copy its code; `_scripts/run_job.py` adjusts some of its modules in
memory when a job starts, without modifying its files. Ostris's MiniMax H3 training adapter is
downloaded from Hugging Face.

## Fizgig — Peter Neill (Apache 2.0 license)

[shootthesound/Fizgig](https://github.com/shootthesound/Fizgig), Copyright (c) 2026 Peter Neill.
The following elements were rewritten in this project after reading its code and documentation
(version of 16 November 2026), without copying its code:

- the settings of the "Fast 512" process: 0.25 MP resolution, rank 32, Automagic v4 capped at
  2.5e-4, EMA 0.98, 30 passes over the dataset;
- the schedule of the likeness optimization (share of steps at the sharpest noise level);
- the LCM sampler settings (fresh noise from 1.1 to 0.9);
- the memory benchmarks and the RAM margin rule;
- the conversion of its H3-Image training adapter to the ai-toolkit format.

The Fizgig training adapter is downloaded from
[Hugging Face](https://huggingface.co/ShootTheSound/Fizgig-MiniMax-H3-Image-Training-Adapter) under
its publisher's terms.

## Tools downloaded during installation

- [uv](https://github.com/astral-sh/uv) — Astral (MIT and Apache 2.0 licenses)
- [MinGit](https://github.com/git-for-windows/git) — Git for Windows (GPL v2)
- [Gradio](https://github.com/gradio-app/gradio) (Apache 2.0), installed with ai-toolkit's dependencies
- [OpenCV](https://github.com/opencv/opencv) (Apache 2.0) and the PP-OCRv3 text detector from
  [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (Apache 2.0), through
  [OpenCV Zoo](https://github.com/opencv/opencv_zoo)
- Qwen3-VL (Apache 2.0), downloaded by ai-toolkit for automatic captioning

## Models downloaded on first use for the likeness measurements

- [DINOv2 small](https://huggingface.co/facebook/dinov2-small) — Meta (Apache 2.0): likeness between
  the previews and the dataset
- [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) — OpenAI (MIT): prompt
  fidelity of the previews

## Models

MiniMax-H3-Image (MiniMax), repackaged by [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3-Image),
under the MiniMax-H3-Image Community License.
