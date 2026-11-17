@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"
call "%~dp0_scripts\lang.bat"
call "%~dp0_scripts\env.bat"
cd /d "%AITK%"
echo ===== nvidia-smi =====
nvidia-smi
echo.
echo ===== !M_G_HW! =====
"%AITK%\.venv\Scripts\python.exe" -m manager detect
echo.
echo ===== torch / CUDA / int8 =====
"%AITK%\.venv\Scripts\python.exe" -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda);print('GPU',torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0));a=torch.randint(-8,8,(64,64),dtype=torch.int8,device='cuda');print('int8 matmul OK',torch._int_mm(a,a).shape)"
echo.
echo ===== gradio / triton =====
"%AITK%\.venv\Scripts\python.exe" -c "import gradio,triton;print('gradio',gradio.__version__,'triton',triton.__version__)"
echo.
echo !M_G_REMIND!
pause
