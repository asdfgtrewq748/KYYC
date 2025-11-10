"""诊断NaN问题：检查数据集的数值范围和异常值"""
import numpy as np
import pandas as pd

print("=" * 60)
print("数据集诊断 - 查找NaN问题根源")
print("=" * 60)

# 加载数据集
data = np.load('processed_data/sequence_dataset.npz', allow_pickle=True)
X = data['X']
y_final = data['y_final']
feature_names = data['feature_names']

print(f"\n数据集形状:")
print(f"  X: {X.shape}")
print(f"  y: {y_final.shape}")
print(f"  特征数量: {len(feature_names)}")

# 检查NaN和Inf
print("\n" + "=" * 60)
print("步骤1: 检查原始数据中的NaN/Inf")
print("=" * 60)

nan_in_X = np.isnan(X).any()
inf_in_X = np.isinf(X).any()
nan_in_y = np.isnan(y_final).any()
inf_in_y = np.isinf(y_final).any()

print(f"X中是否有NaN: {nan_in_X}")
print(f"X中是否有Inf: {inf_in_X}")
print(f"y中是否有NaN: {nan_in_y}")
print(f"y中是否有Inf: {inf_in_y}")

if nan_in_X:
    nan_count = np.isnan(X).sum()
    print(f"  ⚠️ X中有 {nan_count} 个NaN值！")
if inf_in_X:
    inf_count = np.isinf(X).sum()
    print(f"  ⚠️ X中有 {inf_count} 个Inf值！")

# 检查每个特征的统计信息
print("\n" + "=" * 60)
print("步骤2: 各特征的数值范围")
print("=" * 60)

print(f"\n{'特征名':<40} {'最小值':>12} {'最大值':>12} {'均值':>12} {'标准差':>12}")
print("-" * 88)

problematic_features = []

for i, fname in enumerate(feature_names):
    feat_data = X[:, :, i].flatten()
    
    feat_min = feat_data.min()
    feat_max = feat_data.max()
    feat_mean = feat_data.mean()
    feat_std = feat_data.std()
    
    # 标记问题特征
    is_problem = False
    if abs(feat_max) > 1e6 or abs(feat_min) > 1e6:
        is_problem = True
        problematic_features.append((fname, feat_min, feat_max, feat_mean, feat_std))
    
    marker = "⚠️" if is_problem else "  "
    print(f"{marker}{fname:<40} {feat_min:>12.2f} {feat_max:>12.2f} {feat_mean:>12.2f} {feat_std:>12.2f}")

# 报告问题特征
if problematic_features:
    print("\n" + "=" * 60)
    print("⚠️ 发现问题特征（数值范围过大）：")
    print("=" * 60)
    for fname, fmin, fmax, fmean, fstd in problematic_features:
        print(f"  {fname}:")
        print(f"    范围: [{fmin:.2e}, {fmax:.2e}]")
        print(f"    需要归一化！")

# 检查地质特征
print("\n" + "=" * 60)
print("步骤3: 地质特征专项检查")
print("=" * 60)

geo_features = [i for i, name in enumerate(feature_names) if name.startswith('geo_')]
print(f"\n地质特征数量: {len(geo_features)}")

if geo_features:
    print(f"\n地质特征详细信息:")
    for idx in geo_features:
        fname = feature_names[idx]
        feat_data = X[:, :, idx].flatten()
        print(f"  {fname}:")
        print(f"    范围: [{feat_data.min():.2f}, {feat_data.max():.2f}]")
        print(f"    唯一值数量: {np.unique(feat_data).shape[0]}")
        print(f"    标准差: {feat_data.std():.2f}")

# 检查目标变量
print("\n" + "=" * 60)
print("步骤4: 目标变量检查")
print("=" * 60)

print(f"末阻力值（目标）:")
print(f"  范围: [{y_final.min():.2f}, {y_final.max():.2f}]")
print(f"  均值: {y_final.mean():.2f}")
print(f"  标准差: {y_final.std():.2f}")

# 建议
print("\n" + "=" * 60)
print("🔧 修复建议")
print("=" * 60)

if problematic_features or not geo_features:
    print("\n⚠️ 发现问题：")
    if problematic_features:
        print("  1. 有特征数值范围过大，未正确归一化")
    if not geo_features:
        print("  2. 未检测到地质特征（应该有9个geo_开头的特征）")
    
    print("\n💡 解决方案：")
    print("  1. 重新运行数据预处理，确保所有特征都经过归一化")
    print("  2. 检查 prepare_training_data.py 中的特征提取逻辑")
    print("  3. 在训练前使用 StandardScaler 归一化所有特征")
else:
    print("\n✅ 数据集看起来正常")
    print("\n但训练仍然出现NaN，可能的原因：")
    print("  1. 学习率过大（尝试降低到 0.0001）")
    print("  2. 批次大小过大（尝试32或16）")
    print("  3. 模型架构问题（检查是否有除法操作）")
    print("  4. 梯度裁剪未启用（添加 gradient clipping）")

print("\n" + "=" * 60)
