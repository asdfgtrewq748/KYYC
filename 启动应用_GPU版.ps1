# STGCN Mining Pressure Prediction Application Launcher
# Environment: kyyc_py311 (Python 3.11 + PyTorch 2.9.0 + CUDA 13.0)

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "  STGCN Mining Pressure Prediction - GPU Accelerated" -ForegroundColor Yellow
Write-Host "  Environment: kyyc_py311 (Python 3.11 + PyTorch 2.9.0 + CUDA 13.0)" -ForegroundColor Gray
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# Set environment variable (solve OpenMP conflict)
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Write-Host "[OK] Environment variable set: KMP_DUPLICATE_LIB_OK=TRUE" -ForegroundColor Green

# Check GPU status
Write-Host ""
Write-Host "[INFO] Checking GPU status..." -ForegroundColor Cyan
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n kyyc_py311 python -c "import torch; print('   GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Only'); print('   CUDA:', 'Enabled' if torch.cuda.is_available() else 'Disabled'); print('   PyTorch:', torch.__version__)"

Write-Host ""
Write-Host "[START] Launching Streamlit application..." -ForegroundColor Yellow
Write-Host "Tip: The application will open automatically in your browser" -ForegroundColor Gray
Write-Host ""

# Launch application
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n kyyc_py311 python -m streamlit run STGCN.py

Write-Host ""
Write-Host "[EXIT] Application closed" -ForegroundColor Yellow
