@echo off
REM 一键安装所有依赖脚本
echo ========================================
echo   STGCN项目 - 依赖安装脚本
echo ========================================
echo.

cd /d "%~dp0"
echo 项目目录: %cd%
echo.

echo [1/2] 激活conda环境...
call conda activate kyyc_py311
if %errorlevel% neq 0 (
    echo 错误: 环境激活失败
    echo 请在Anaconda Prompt中运行此脚本
    pause
    exit /b 1
)

echo 当前Python版本:
python --version
echo.

echo [2/2] 安装依赖包...
echo 这可能需要5-10分钟，请耐心等待...
echo.

echo 安装PyTorch (GPU版，支持RTX 50系列)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

echo.
echo 安装其他依赖...
pip install "numpy<2.0"
pip install streamlit pandas scipy matplotlib

echo.
echo ========================================
echo   安装完成！
echo ========================================
echo.
echo 现在可以运行: streamlit run STGCN.py
echo.
pause
