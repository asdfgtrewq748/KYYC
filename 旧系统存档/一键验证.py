# -*- coding: utf-8 -*-
"""一键验证所有修复是否生效"""
import sys
import os

print("=" * 70)
print("🔍 一键验证 - 检查所有NaN修复是否生效")
print("=" * 70)

all_ok = True

# 1. 检查数据集
print("\n1️⃣ 检查数据集...")
try:
    import numpy as np
    data = np.load('processed_data/sequence_dataset.npz', allow_pickle=True)
    X = data['X']
    feature_names = data['feature_names']
    
    num_features = len(feature_names)
    x_mean = X.mean()
    x_std = X.std()
    
    print(f"   特征数量: {num_features}")
    print(f"   X均值: {x_mean:.6f}")
    print(f"   X标准差: {x_std:.6f}")
    
    if num_features != 17:
        print(f"   ❌ 错误：特征数应该是17，当前是{num_features}")
        all_ok = False
    else:
        print(f"   ✅ 特征数正确")
    
    if abs(x_mean) > 0.01 or abs(x_std - 1.0) > 0.1:
        print(f"   ❌ 错误：数据未正确标准化")
        all_ok = False
    else:
        print(f"   ✅ 数据标准化正确")
        
except Exception as e:
    print(f"   ❌ 错误：{e}")
    all_ok = False

# 2. 检查STGCN.py中特征工程是否被禁用
print("\n2️⃣ 检查STGCN.py...")
try:
    with open('STGCN.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否有checkbox
    if 'st.checkbox' in content and 'use_feature_engineering' in content:
        # 检查是否被替换为False
        if 'use_feature_engineering = False' in content:
            print(f"   ✅ 特征工程已强制禁用")
        else:
            print(f"   ⚠️ 警告：特征工程可能未被完全禁用")
            print(f"   建议：运行 python force_disable_feature_engineering.py")
            all_ok = False
    else:
        print(f"   ✅ 特征工程相关代码已移除或禁用")
        
except Exception as e:
    print(f"   ❌ 错误：{e}")
    all_ok = False

# 3. 检查标准化参数文件
print("\n3️⃣ 检查标准化参数...")
try:
    import json
    with open('processed_data/feature_scaler.json', 'r', encoding='utf-8') as f:
        scaler_params = json.load(f)
    
    if len(scaler_params['mean']) == 17:
        print(f"   ✅ 标准化参数文件正确（17个特征）")
    else:
        print(f"   ❌ 错误：标准化参数异常")
        all_ok = False
        
except Exception as e:
    print(f"   ❌ 错误：{e}")
    all_ok = False

# 4. 检查训练配置文件
print("\n4️⃣ 检查安全训练配置...")
if os.path.exists('safe_training_config.py'):
    print(f"   ✅ 安全训练配置文件存在")
else:
    print(f"   ⚠️ 警告：安全训练配置文件不存在（可选）")

# 5. 模拟数据加载
print("\n5️⃣ 模拟训练数据加载...")
try:
    # 加载一小批数据测试
    batch_size = 32
    X_batch = X[:batch_size]
    
    # 检查数值范围
    if np.isnan(X_batch).any():
        print(f"   ❌ 错误：数据中有NaN")
        all_ok = False
    elif np.isinf(X_batch).any():
        print(f"   ❌ 错误：数据中有Inf")
        all_ok = False
    elif np.abs(X_batch).max() > 100:
        print(f"   ⚠️ 警告：数据范围过大 ({np.abs(X_batch).max():.2f})")
        all_ok = False
    else:
        print(f"   ✅ 数据批次正常")
        print(f"      范围: [{X_batch.min():.2f}, {X_batch.max():.2f}]")
        
except Exception as e:
    print(f"   ❌ 错误：{e}")
    all_ok = False

# 最终结论
print("\n" + "=" * 70)
if all_ok:
    print("✅ ✅ ✅ 所有检查通过！可以安全训练 ✅ ✅ ✅")
    print("\n📋 训练前确认清单：")
    print("   □ 重启Streamlit: streamlit run STGCN.py")
    print("   □ 确认特征数显示为17（不是264）")
    print("   □ 学习率设为0.0001")
    print("   □ 批次大小32")
    print("   □ 启用梯度裁剪(1.0)")
    print("   □ 选择Transformer模型")
    print("\n🚀 配置完成后点击\"开始训练\"！")
else:
    print("❌ 检查未完全通过")
    print("\n🔧 修复建议：")
    print("   1. python force_disable_feature_engineering.py")
    print("   2. python preprocess/prepare_training_data.py")
    print("   3. 重新运行本脚本验证")

print("=" * 70)
