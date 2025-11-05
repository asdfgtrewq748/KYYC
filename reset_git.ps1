# 重置 Git 历史并创建全新初始提交

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  Reset Git History & Fresh Commit" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# 1. 创建备份分支
Write-Host "[1/6] Creating backup branch..." -ForegroundColor Yellow
git branch backup-20251105-final 2>&1 | Out-Null
Write-Host "✓ Backup created: backup-20251105-final" -ForegroundColor Green
Write-Host ""

# 2. 删除 .git 文件夹
Write-Host "[2/6] Removing .git folder..." -ForegroundColor Yellow
Remove-Item -Path .git -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "✓ Git history removed" -ForegroundColor Green
Write-Host ""

# 3. 重新初始化 Git
Write-Host "[3/6] Initializing fresh Git repository..." -ForegroundColor Yellow
git init
Write-Host "✓ Git initialized" -ForegroundColor Green
Write-Host ""

# 4. 添加所有文件
Write-Host "[4/6] Adding all files..." -ForegroundColor Yellow
git add .
Write-Host "✓ All files staged" -ForegroundColor Green
Write-Host ""

# 5. 创建初始提交
Write-Host "[5/6] Creating initial commit..." -ForegroundColor Yellow
git commit -m "Initial commit: STGCN Mining Pressure Prediction System

Features:
- Spatial-Temporal Graph Convolutional Network (STGCN) implementation
- Streamlit web interface for real-time prediction
- GPU acceleration with PyTorch 2.9.0 + CUDA 13.0
- Support for RTX 5070 Ti (Blackwell architecture)
- Complete data preprocessing and visualization
- Model training and evaluation tools

Technical Stack:
- PyTorch 2.9.0 + CUDA 13.0
- Python 3.11
- Streamlit
- NumPy, Pandas, Scipy

Project Structure:
- STGCN.py: Main application
- check_pytorch.py: GPU validation
- test_*.py: Testing utilities
- generate_*.py: Data generation tools
- Documentation in Markdown"

Write-Host "✓ Initial commit created" -ForegroundColor Green
Write-Host ""

# 6. 添加远程仓库
Write-Host "[6/6] Adding remote repository..." -ForegroundColor Yellow
git remote add origin https://github.com/asdfgtrewq748/KYYC.git 2>&1 | Out-Null
Write-Host "✓ Remote added" -ForegroundColor Green
Write-Host ""

# 显示状态
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  Current Status" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Commit:" -ForegroundColor Yellow
git log --oneline -1

Write-Host "`nRemote:" -ForegroundColor Yellow
git remote -v

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "  ✓ Ready to Push!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""

Write-Host "Next step: Run the following command to push:" -ForegroundColor Cyan
Write-Host "  git push -u origin main --force" -ForegroundColor White
Write-Host ""
Write-Host "This will replace all history on GitHub with the fresh commit." -ForegroundColor Yellow
