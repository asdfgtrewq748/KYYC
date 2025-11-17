@echo off
chcp 65001 >nul
echo.
echo ========================================
echo   矿压预测模型训练系统
echo ========================================
echo.

call conda activate kyyc_py311
if errorlevel 1 (
    echo ❌ 环境激活失败！
    pause
    exit /b 1
)

python train_advanced_model.py

echo.
pause
