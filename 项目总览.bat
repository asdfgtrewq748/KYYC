@echo off
chcp 65001 >nul
cls
echo.
echo ============================================================
echo 📂 KYYC 项目结构总览
echo ============================================================
echo.
echo ✅ 核心训练系统（活跃）
echo    ├── simple_dataloader.py        数据加载器
echo    ├── stable_transformer.py       Transformer模型  
echo    ├── train_safe.py              主训练脚本
echo    ├── 开始训练.bat                一键启动
echo    └── check_ready.py             环境检查
echo.
echo 📊 数据文件
echo    ├── processed_data/
echo    │   └── sequence_dataset.npz   训练数据（17特征）
echo    ├── 测试钻孔/                   地质数据
echo    ├── kaungya/                   压力数据
echo    └── preprocess/                预处理脚本
echo.
echo 📚 文档
echo    ├── README.md                  项目说明
echo    ├── 安全训练系统说明.md         详细文档
echo    └── requirements.txt           依赖列表
echo.
echo 🛠️ 工具脚本
echo    ├── debug_data.py              数据调试
echo    ├── regenerate_data.py         重生成数据
echo    └── 检查训练就绪.py             就绪检查
echo.
echo 🗂️ 旧系统（已归档）
echo    └── 旧系统存档/
echo        ├── STGCN.py               旧训练系统（4063行）
echo        ├── train_simple.py        旧训练脚本
echo        ├── *.pth                  旧模型文件
echo        └── 诊断修复脚本...
echo.
echo ============================================================
echo 📊 统计信息
echo ============================================================
echo.
echo 活跃文件: ~20个（核心+工具）
echo 归档文件: ~20个（旧系统）
echo Python脚本: 8个（核心训练系统）
echo 文档: 2个MD（README + 说明）
echo.
echo ============================================================
echo 🚀 快速操作
echo ============================================================
echo.
echo [1] 开始训练
echo [2] 检查环境
echo [3] 查看文档
echo [Q] 退出
echo.
set /p choice="请选择操作 (1/2/3/Q): "

if "%choice%"=="1" (
    start 开始训练.bat
)

if "%choice%"=="2" (
    call conda activate kyyc_py311
    python check_ready.py
    pause
)

if "%choice%"=="3" (
    start README.md
    start 安全训练系统说明.md
)

if /i "%choice%"=="Q" (
    exit
)
