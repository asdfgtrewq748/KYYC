@echo off
chcp 65001 >nul
echo ========================================
echo 🚀 训练增强版模型（改进地质特征融合）
echo ========================================
echo.

cd /d "%~dp0"

echo 激活conda环境...
call conda activate kyyc_py311
if errorlevel 1 (
    echo ❌ conda环境激活失败
    pause
    exit /b 1
)

echo.
echo ✓ 环境激活成功
echo.
echo 开始训练...
echo.

python train_enhanced.py

echo.
echo ========================================
echo ✓ 训练完成
echo ========================================
pause
