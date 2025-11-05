@echo off
echo ========================================
echo     启动STGCN矿压预测系统
echo ========================================
echo.

echo [1/3] 激活conda环境...
call conda activate kan
if %errorlevel% neq 0 (
    echo 错误: conda环境激活失败
    pause
    exit /b 1
)

echo [2/3] 检查数据文件...
if not exist "processed_data\sequence_dataset.npz" (
    echo 警告: 未找到预处理数据集
    echo 请先运行 preprocess\prepare_training_data.py
    pause
    exit /b 1
)

echo [3/3] 启动Streamlit应用...
streamlit run STGCN.py

pause
