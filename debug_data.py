"""
调试脚本：检查数据文件中的特征数量
"""
import numpy as np
import os

# 加载数据文件
npz_path = os.path.join('processed_data', 'sequence_dataset.npz')

if not os.path.exists(npz_path):
    print(f"❌ 数据文件不存在: {npz_path}")
    exit(1)

print(f"📂 加载数据文件: {npz_path}")
data = np.load(npz_path, allow_pickle=True)

print("\n" + "="*60)
print("📊 数据文件内容:")
print("="*60)

for key in data.keys():
    print(f"\n🔑 {key}:")
    value = data[key]
    if isinstance(value, np.ndarray):
        print(f"   类型: {value.dtype}")
        print(f"   形状: {value.shape}")
        if value.ndim <= 2 and value.size < 20:
            print(f"   内容: {value}")
    else:
        print(f"   类型: {type(value)}")
        print(f"   内容: {value}")

# 重点检查X的特征维度
X = data['X']
print("\n" + "="*60)
print("⭐ 关键信息：X数据")
print("="*60)
print(f"样本数: {X.shape[0]:,}")
print(f"序列长度: {X.shape[1]}")
print(f"🔴 特征数量: {X.shape[2]} 🔴")

# 检查特征名称
if 'feature_names' in data:
    feature_names = data['feature_names'].tolist()
    print(f"\n📝 特征名称列表 (共{len(feature_names)}个):")
    print("="*60)
    
    # 分类显示
    mining_pressure_features = [f for f in feature_names if any(x in f for x in ['矿压', '立柱', '初撑力', '末阻力', '工作阻力', '安全阀', '泵站压力'])]
    geo_features = [f for f in feature_names if any(x in f for x in ['煤厚', '倾角', '断层', '褶皱', '顶板', '底板', '岩性', '强度'])]
    engineered_features = [f for f in feature_names if any(x in f for x in ['_mean', '_std', '_max', '_min', '_range', '_diff', '_roll', 'time_index', 'pos_enc'])]
    other_features = [f for f in feature_names if f not in mining_pressure_features + geo_features + engineered_features]
    
    print(f"\n✅ 矿压特征 ({len(mining_pressure_features)}个):")
    for f in mining_pressure_features[:20]:  # 最多显示20个
        print(f"   - {f}")
    if len(mining_pressure_features) > 20:
        print(f"   ... 还有 {len(mining_pressure_features)-20} 个")
    
    print(f"\n✅ 地质特征 ({len(geo_features)}个):")
    for f in geo_features:
        print(f"   - {f}")
    
    if engineered_features:
        print(f"\n⚠️ 工程特征 ({len(engineered_features)}个):")
        for f in engineered_features[:30]:  # 最多显示30个
            print(f"   - {f}")
        if len(engineered_features) > 30:
            print(f"   ... 还有 {len(engineered_features)-30} 个")
    
    if other_features:
        print(f"\n❓ 其他特征 ({len(other_features)}个):")
        for f in other_features[:10]:
            print(f"   - {f}")
        if len(other_features) > 10:
            print(f"   ... 还有 {len(other_features)-10} 个")

print("\n" + "="*60)
print("🔍 诊断结论:")
print("="*60)

if X.shape[2] == 17:
    print("✅ 正常：17个矿压特征（未融合地质特征）")
elif X.shape[2] == 25:
    print("✅ 正常：25个特征（17矿压 + 8地质）")
elif X.shape[2] > 30:
    print(f"❌ 异常：{X.shape[2]}个特征 - 数据文件已包含工程特征！")
    print("\n💡 解决方案：")
    print("   1. 重新生成数据文件（不要在生成时添加工程特征）")
    print("   2. 或者修改代码，从数据中移除工程特征")
    print("   3. 检查数据预处理脚本，确保没有自动添加特征")
else:
    print(f"⚠️ 特征数量为{X.shape[2]}，需要检查是否符合预期")

# 检查数据值范围
print("\n" + "="*60)
print("📈 数据值范围:")
print("="*60)
print(f"最小值: {X.min():.4f}")
print(f"最大值: {X.max():.4f}")
print(f"均值: {X.mean():.4f}")
print(f"标准差: {X.std():.4f}")
print(f"是否含NaN: {np.isnan(X).any()}")
print(f"是否含Inf: {np.isinf(X).any()}")

if X.max() > 1000:
    print("\n⚠️ 警告：数据未归一化！最大值超过1000")
    print("   建议：在训练前进行归一化处理")

print("\n✅ 检查完成！")
