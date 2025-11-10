# -*- coding: utf-8 -*-
"""
快速验证 - 检查新训练系统是否准备就绪
"""

import numpy as np
from pathlib import Path

print("\n" + "="*70)
print("🔍 安全训练系统 - 就绪检查")
print("="*70)

# 1. 检查数据文件
print("\n1️⃣ 检查数据文件...")
data_path = Path('processed_data/sequence_dataset.npz')
if data_path.exists():
    data = np.load(data_path, allow_pickle=True)
    X = data['X']
    y = data['y_final']
    
    print(f"✓ 数据文件存在")
    print(f"✓ X形状: {X.shape}")
    print(f"✓ y形状: {y.shape}")
    print(f"✓ 特征数: {X.shape[-1]}")
    
    # 检查数值
    if np.isnan(X).any() or np.isinf(X).any():
        print("❌ X包含NaN/Inf")
    else:
        print(f"✓ X无NaN/Inf，范围: [{X.min():.2f}, {X.max():.2f}]")
    
    if np.isnan(y).any() or np.isinf(y).any():
        print("❌ y包含NaN/Inf")
    else:
        print(f"✓ y无NaN/Inf，范围: [{y.min():.2f}, {y.max():.2f}]")
else:
    print(f"❌ 数据文件不存在: {data_path}")

# 2. 检查新文件
print("\n2️⃣ 检查新训练系统文件...")
files_to_check = [
    'simple_dataloader.py',
    'stable_transformer.py', 
    'train_safe.py',
    '启动安全训练.bat'
]

for file in files_to_check:
    if Path(file).exists():
        print(f"✓ {file}")
    else:
        print(f"❌ {file} 不存在")

# 3. 检查Python环境
print("\n3️⃣ 检查Python环境...")
try:
    import torch
    print(f"✓ PyTorch已安装: {torch.__version__}")
    print(f"✓ CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("❌ PyTorch未安装")
    print("   请运行: pip install torch")

try:
    from sklearn.preprocessing import StandardScaler
    print("✓ scikit-learn已安装")
except ImportError:
    print("❌ scikit-learn未安装")

# 4. 总结
print("\n" + "="*70)
print("📋 总结")
print("="*70)

if data_path.exists() and X.shape[-1] == 17 and not np.isnan(X).any():
    print("✅ 数据准备完毕（17个特征，无NaN）")
else:
    print("⚠️ 数据需要重新生成")

if all(Path(f).exists() for f in files_to_check):
    print("✅ 训练系统文件完整")
else:
    print("⚠️ 部分文件缺失")

try:
    import torch
    print("✅ 可以开始训练！")
    print("\n运行: python train_safe.py")
except ImportError:
    print("⚠️ 需要先安装PyTorch")
    print("\n运行: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

print("="*70)
