@echo off
chcp 65001 >nul
cls
echo.
echo ============================================================
echo 🚀 开始安全训练 (kyyc_py311环境)
echo ============================================================
echo.

call conda activate kyyc_py311

echo ✓ 已激活环境: kyyc_py311
echo.
echo 检查PyTorch...
python -c "import torch; print('✓ PyTorch版本:', torch.__version__); print('✓ CUDA可用:', torch.cuda.is_available())"

echo.
echo ============================================================
echo 🎯 开始训练 - 预计45-60分钟
echo ============================================================
echo.
echo 训练配置:
echo   - 特征数: 17 (固定, 无特征工程)
echo   - 学习率: 0.0001
echo   - 批次大小: 32
echo   - 训练轮数: 50
echo   - 梯度裁剪: 1.0
echo.
echo 按任意键开始训练...
pause >nul

python train_safe.py

echo.
echo ============================================================
echo 训练完成！
echo ============================================================
echo.
echo 查看结果:
echo   - 模型文件: safe_best_model.pth
echo   - 训练历史: training_history.json
echo.
pause
