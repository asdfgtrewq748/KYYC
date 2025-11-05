@echo off
REM 最简单的启动脚本 - 使用绝对路径

echo ========================================
echo     STGCN训练系统 - 简易启动
echo ========================================
echo.

REM 切换到项目目录
cd /d "d:\xiangmu\KYYC"
echo 项目目录: %cd%
echo.

echo 正在启动Streamlit...
echo 如果失败，请：
echo 1. 打开 Anaconda Prompt
echo 2. 运行: cd /d d:\xiangmu\KYYC
echo 3. 运行: conda activate kyyc_py311
echo 4. 运行: streamlit run STGCN.py
echo.
echo 按任意键继续...
pause >nul

REM 尝试直接启动（如果conda配置正确）
call C:\ProgramData\Anaconda3\Scripts\activate.bat kyyc_py311
streamlit run STGCN.py

pause
