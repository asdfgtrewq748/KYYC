# -*- coding: utf-8 -*-
"""
Quick Check - Verify training system readiness
"""

import numpy as np
from pathlib import Path

print("\n" + "="*70)
print("Check Training System Readiness")
print("="*70)

# 1. Check data file
print("\n1. Check data file...")
data_path = Path('processed_data/sequence_dataset.npz')
if data_path.exists():
    data = np.load(data_path, allow_pickle=True)
    X = data['X']
    y = data['y_final']
    
    print(f"OK Data file exists")
    print(f"OK X shape: {X.shape}")
    print(f"OK y shape: {y.shape}")
    print(f"OK Features: {X.shape[-1]}")
    
    # Check values
    if np.isnan(X).any() or np.isinf(X).any():
        print("ERROR X contains NaN/Inf")
    else:
        print(f"OK X no NaN/Inf, range: [{X.min():.2f}, {X.max():.2f}]")
    
    if np.isnan(y).any() or np.isinf(y).any():
        print("ERROR y contains NaN/Inf")
    else:
        print(f"OK y no NaN/Inf, range: [{y.min():.2f}, {y.max():.2f}]")
else:
    print(f"ERROR Data file not found: {data_path}")

# 2. Check new files
print("\n2. Check training system files...")
files_to_check = [
    'simple_dataloader.py',
    'stable_transformer.py', 
    'train_safe.py'
]

all_files_exist = True
for file in files_to_check:
    if Path(file).exists():
        print(f"OK {file}")
    else:
        print(f"ERROR {file} not found")
        all_files_exist = False

# 3. Check Python environment
print("\n3. Check Python environment...")
try:
    import torch
    print(f"OK PyTorch: {torch.__version__}")
    print(f"OK CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"OK GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("ERROR PyTorch not installed")

try:
    from sklearn.preprocessing import StandardScaler
    print("OK scikit-learn installed")
except ImportError:
    print("ERROR scikit-learn not installed")

# 4. Summary
print("\n" + "="*70)
print("Summary")
print("="*70)

if data_path.exists() and X.shape[-1] == 17 and not np.isnan(X).any():
    print("SUCCESS Data ready (17 features, no NaN)")
else:
    print("WARNING Data needs regeneration")

if all_files_exist:
    print("SUCCESS Training system files complete")
else:
    print("WARNING Some files missing")

try:
    import torch
    print("SUCCESS Ready to train!")
    print("\nRun: python train_safe.py")
except ImportError:
    print("WARNING Need to install PyTorch first")

print("="*70)
