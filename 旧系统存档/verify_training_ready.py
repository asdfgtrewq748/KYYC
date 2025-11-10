"""最终验证：确认数据集可以安全训练"""
import numpy as np
import json

print("=" * 60)
print("🔍 最终训练前验证")
print("=" * 60)

# 1. 加载数据集
print("\n1️⃣ 加载数据集...")
data = np.load('processed_data/sequence_dataset.npz', allow_pickle=True)
X = data['X']
y_final = data['y_final']
feature_names = data['feature_names']

print(f"   ✓ X形状: {X.shape}")
print(f"   ✓ y形状: {y_final.shape}")
print(f"   ✓ 特征数量: {len(feature_names)}")

# 2. 检查标准化
print("\n2️⃣ 验证标准化效果...")
print(f"   X的均值: {X.mean():.6f} (目标: ≈0)")
print(f"   X的标准差: {X.std():.6f} (目标: ≈1)")
print(f"   X的范围: [{X.min():.2f}, {X.max():.2f}]")

all_good = True

if abs(X.mean()) > 0.01:
    print(f"   ⚠️ 警告：均值偏离0较多")
    all_good = False
else:
    print(f"   ✓ 均值正常")

if abs(X.std() - 1.0) > 0.1:
    print(f"   ⚠️ 警告：标准差偏离1较多")
    all_good = False
else:
    print(f"   ✓ 标准差正常")

# 3. 检查NaN/Inf
print("\n3️⃣ 检查数值问题...")
has_nan = np.isnan(X).any()
has_inf = np.isinf(X).any()

print(f"   是否有NaN: {'❌ 有' if has_nan else '✓ 无'}")
print(f"   是否有Inf: {'❌ 有' if has_inf else '✓ 无'}")

if has_nan or has_inf:
    all_good = False

# 4. 检查标准化参数
print("\n4️⃣ 检查标准化参数...")
try:
    with open('processed_data/feature_scaler.json', 'r', encoding='utf-8') as f:
        scaler_params = json.load(f)
    print(f"   ✓ 标准化参数文件存在")
    print(f"   ✓ 包含 {len(scaler_params['mean'])} 个特征的参数")
except:
    print(f"   ⚠️ 标准化参数文件缺失或损坏")
    all_good = False

# 5. 检查地质特征
print("\n5️⃣ 检查地质特征...")
geo_feature_count = sum(1 for name in feature_names if 'geo_' in name)
print(f"   地质特征数量: {geo_feature_count}")

if geo_feature_count == 9:
    print(f"   ✓ 地质特征完整")
else:
    print(f"   ⚠️ 地质特征数量异常（应该是9个）")
    all_good = False

# 检查地质特征的变化性
geo_indices = [i for i, name in enumerate(feature_names) if 'geo_' in name]
if geo_indices:
    first_geo = X[:, :, geo_indices[0]].flatten()
    unique_count = np.unique(first_geo).shape[0]
    print(f"   第一个地质特征的唯一值数量: {unique_count}")
    if unique_count > 1:
        print(f"   ✓ 地质特征有真实变化")
    else:
        print(f"   ⚠️ 地质特征可能全部相同")
        all_good = False

# 6. 目标变量检查
print("\n6️⃣ 检查目标变量...")
print(f"   y的范围: [{y_final.min():.2f}, {y_final.max():.2f}]")
print(f"   y的均值: {y_final.mean():.2f}")

if np.isnan(y_final).any() or np.isinf(y_final).any():
    print(f"   ⚠️ 目标变量有NaN或Inf")
    all_good = False
else:
    print(f"   ✓ 目标变量正常")

# 7. 模拟一个小批次计算
print("\n7️⃣ 模拟批次计算...")
try:
    batch_size = 32
    sample_batch = X[:batch_size]
    sample_y = y_final[:batch_size]
    
    # 简单的线性计算
    mean_input = sample_batch.mean()
    mean_output = sample_y.mean()
    
    print(f"   批次大小: {batch_size}")
    print(f"   批次X均值: {mean_input:.6f}")
    print(f"   批次y均值: {mean_output:.6f}")
    print(f"   ✓ 批次计算正常")
except Exception as e:
    print(f"   ⚠️ 批次计算失败: {e}")
    all_good = False

# 最终结论
print("\n" + "=" * 60)
if all_good:
    print("✅ 验证通过！数据集可以安全训练")
    print("\n📋 推荐配置:")
    print("   模型: Transformer")
    print("   学习率: 0.0001")
    print("   批次大小: 32")
    print("   特征工程: ❌ 关闭")
    print("   地质特征: ✅ 启用")
    print("\n🚀 可以开始训练了！")
else:
    print("⚠️ 验证未完全通过，请检查上述问题")

print("=" * 60)
