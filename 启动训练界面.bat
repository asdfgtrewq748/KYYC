@echo off
REM 切换到脚本所在目录（项目根目录）
cd /d "%~dp0"

echo ========================================
echo     启动STGCN矿压预测系统
echo ========================================
echo.
echo 当前目录: %cd%
echo.

echo [1/3] 激活conda环境...
call conda activate kyyc_py311
if %errorlevel% neq 0 (
    echo 错误: conda环境激活失败
    echo 请确保已安装Anaconda/Miniconda并配置好环境变量
    echo 或在Anaconda Prompt中手动运行：
    echo   cd /d %cd%
    echo   conda activate kyyc_py311
    echo   streamlit run STGCN.py
    pause
    exit /b 1
)

echo [2/3] 检查数据文件...
if not exist "processed_data\sequence_dataset.npz" (
    echo 警告: 未找到预处理数据集
    echo 位置: %cd%\processed_data\sequence_dataset.npz
    echo 请先运行 preprocess\prepare_training_data.py
    pause
    exit /b 1
)

echo [3/3] 启动Streamlit应用...
echo 正在启动 STGCN.py ...
streamlit run STGCN.py

pause
