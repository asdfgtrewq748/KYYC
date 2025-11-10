"""检查地质特征的变化性"""
import pandas as pd
import numpy as np

# 读取处理后的数据
df = pd.read_csv('processed_data/merged_pressure_data.csv')

print("=" * 60)
print("地质特征变化性检查")
print("=" * 60)

# 检查地质特征列
geo_cols = [col for col in df.columns if col.startswith('geo_')]
print(f"\n找到 {len(geo_cols)} 个地质特征列")

print("\n每个地质特征的唯一值数量:")
for col in geo_cols:
    unique_count = df[col].nunique()
    unique_vals = df[col].unique()
    print(f"  {col:30s}: {unique_count:3d} 个唯一值")
    if unique_count <= 5:
        print(f"    值: {unique_vals}")

# 按支架号分组检查
print("\n" + "=" * 60)
print("按支架号检查地质特征（每个支架应该有一个固定的地质特征）")
print("=" * 60)

support_geo = df.groupby('支架号')[geo_cols[0]].first()
print(f"\n不同支架的 {geo_cols[0]} 值:")
print(f"  唯一值数量: {support_geo.nunique()}")
print(f"  前10个支架:")
print(support_geo.head(10))
