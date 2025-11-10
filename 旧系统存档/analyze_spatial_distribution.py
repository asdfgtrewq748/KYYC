"""分析钻孔和支架的空间分布"""
import pandas as pd
import numpy as np

# 读取钻孔数据
geo_df = pd.read_csv('geology_features_extracted.csv')
print("=" * 60)
print("钻孔空间分布")
print("=" * 60)
print(f"\n钻孔数量: {len(geo_df)}")
print(f"X范围: {geo_df['x'].min():.2f} - {geo_df['x'].max():.2f} (跨度: {geo_df['x'].max() - geo_df['x'].min():.2f} 米)")
print(f"Y范围: {geo_df['y'].min():.2f} - {geo_df['y'].max():.2f} (跨度: {geo_df['y'].max() - geo_df['y'].min():.2f} 米)")

print("\n钻孔坐标列表:")
for idx, row in geo_df.iterrows():
    print(f"  {row['borehole']:6s}: X={row['x']:10.2f}, Y={row['y']:10.2f}")

# 读取支架坐标
coord_df = pd.read_csv('processed_data/support_coordinates.csv')
print("\n" + "=" * 60)
print("支架空间分布")
print("=" * 60)
print(f"\n支架数量: {len(coord_df)}")
print(f"X范围: {coord_df['x'].min():.2f} - {coord_df['x'].max():.2f} (跨度: {coord_df['x'].max() - coord_df['x'].min():.2f} 米)")
print(f"Y范围: {coord_df['y'].min():.2f} - {coord_df['y'].max():.2f} (跨度: {coord_df['y'].max() - coord_df['y'].min():.2f} 米)")

print("\n前10个支架坐标:")
print(coord_df.head(10))

# 计算每个支架到最近钻孔的距离
from scipy.spatial import KDTree
tree = KDTree(geo_df[['x', 'y']].values)
distances, indices = tree.query(coord_df[['x', 'y']].values)

print("\n" + "=" * 60)
print("支架-钻孔匹配分析")
print("=" * 60)
print(f"\n平均距离: {np.mean(distances):.2f} 米")
print(f"最小距离: {np.min(distances):.2f} 米")
print(f"最大距离: {np.max(distances):.2f} 米")

# 统计每个钻孔匹配到多少个支架
borehole_counts = {}
for idx in indices:
    borehole = geo_df.iloc[idx]['borehole']
    borehole_counts[borehole] = borehole_counts.get(borehole, 0) + 1

print("\n每个钻孔匹配到的支架数量:")
for borehole, count in sorted(borehole_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {borehole:6s}: {count:3d} 个支架")
