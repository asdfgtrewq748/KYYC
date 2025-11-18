"""
可视化地质特征的影响机制
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False

# 加载数据
data = np.load('processed_data/sequence_dataset.npz')
X = data['X']
y_final = data['y_final']

# 提取地质特征
geo_features = X[:, -1, 6:15]
geo_names = [
    '总厚度(m)',
    '煤层厚度(m)',
    '煤层数量',
    '顶板深度(m)',
    '弹性模量(GPa)',
    '密度(kN/m³)',
    '抗拉强度(MPa)',
    '砂岩占比',
    '泥岩占比'
]

# 创建综合可视化
fig = plt.figure(figsize=(16, 10))

# 1. 地质特征分布
ax1 = plt.subplot(2, 3, 1)
unique_counts = [len(np.unique(geo_features[:, i])) for i in range(9)]
colors = plt.cm.viridis(np.linspace(0, 1, 9))
bars = ax1.bar(range(9), unique_counts, color=colors, alpha=0.7, edgecolor='black')
ax1.set_xticks(range(9))
ax1.set_xticklabels([f'特征{i+1}' for i in range(9)], rotation=45)
ax1.set_ylabel('唯一值数量', fontsize=12)
ax1.set_title('地质特征的离散度分布', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
for i, (bar, count) in enumerate(zip(bars, unique_counts)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{count}', ha='center', va='bottom', fontsize=10)

# 2. 地质特征的值域范围
ax2 = plt.subplot(2, 3, 2)
means = [geo_features[:, i].mean() for i in range(9)]
stds = [geo_features[:, i].std() for i in range(9)]
x_pos = np.arange(9)
ax2.barh(x_pos, means, xerr=stds, color=colors, alpha=0.7, 
         edgecolor='black', error_kw={'elinewidth': 2, 'capsize': 5})
ax2.set_yticks(x_pos)
ax2.set_yticklabels([f'特征{i+1}' for i in range(9)])
ax2.set_xlabel('标准化后的值 (均值±标准差)', fontsize=12)
ax2.set_title('地质特征的数值分布', fontsize=14, fontweight='bold')
ax2.axvline(0, color='red', linestyle='--', alpha=0.5)
ax2.grid(axis='x', alpha=0.3)

# 3. 地质特征与末阻力的相关性
ax3 = plt.subplot(2, 3, 3)
correlations = []
for i in range(9):
    corr = np.corrcoef(geo_features[:, i], y_final.flatten())[0, 1]
    correlations.append(corr)
colors_corr = ['red' if c < 0 else 'green' for c in correlations]
bars = ax3.barh(range(9), correlations, color=colors_corr, alpha=0.7, edgecolor='black')
ax3.set_yticks(range(9))
ax3.set_yticklabels([f'特征{i+1}' for i in range(9)])
ax3.set_xlabel('与末阻力的相关系数', fontsize=12)
ax3.set_title('地质特征的预测重要性', fontsize=14, fontweight='bold')
ax3.axvline(0, color='black', linestyle='-', linewidth=0.5)
ax3.grid(axis='x', alpha=0.3)
for i, (bar, corr) in enumerate(zip(bars, correlations)):
    width = bar.get_width()
    ax3.text(width + (0.01 if width > 0 else -0.01), bar.get_y() + bar.get_height()/2.,
             f'{corr:.3f}', ha='left' if width > 0 else 'right', va='center', fontsize=9)

# 4. 数据流程图
ax4 = plt.subplot(2, 3, 4)
ax4.axis('off')
flow_text = """
【数据处理流程】

1. 空间匹配
   钻孔(19个) → KDTree → 支架
   ├─ 最近邻距离: 平均XX米
   └─ 匹配策略: 1对1复制

2. 时序处理
   时间步1-5 → 地质特征【不变】
   ├─ 物理意义: 地质短期稳定
   └─ 优化: 只取最后时间步

3. 特征编码
   9维原始 → MLP → 128维嵌入
   ├─ Linear(9→128)
   ├─ BatchNorm + ReLU
   └─ Dropout(0.3)
"""
ax4.text(0.1, 0.9, flow_text, fontsize=11, family='monospace',
         verticalalignment='top', bbox=dict(boxstyle='round', 
         facecolor='wheat', alpha=0.3))

# 5. 模型融合架构
ax5 = plt.subplot(2, 3, 5)
ax5.axis('off')
model_text = """
【模型融合逻辑】

输入特征:
├─ 压力序列(6×5) → LSTM → 256维
├─ 地质特征(9×1) → MLP  → 128维
└─ 时间特征(2×1) → MLP  → 64维

融合方式:
   Concat([256, 128, 64]) → 448维
   ├─ 特点: 简单拼接
   └─ 缺点: 无交互建模

预测层:
   MLP(448→256→128→64→1)
   └─ 输出: 末阻力预测值
"""
ax5.text(0.1, 0.9, model_text, fontsize=11, family='monospace',
         verticalalignment='top', bbox=dict(boxstyle='round',
         facecolor='lightblue', alpha=0.3))

# 6. 改进方向
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
improve_text = """
【改进方向】

❌ 当前问题:
├─ 简单拼接融合
├─ 最近邻插值
└─ 单层编码器

✅ 增强方案:
├─ 多头注意力
│  └─ 学习特征间关系
│
├─ 双线性交互
│  └─ 建模 压力×地质
│
└─ 门控机制
   └─ 动态调节权重

预期提升: R² +5~15%
"""
ax6.text(0.1, 0.9, improve_text, fontsize=11, family='monospace',
         verticalalignment='top', bbox=dict(boxstyle='round',
         facecolor='lightgreen', alpha=0.3))

plt.suptitle('地质因素影响逻辑完整分析', fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('geology_influence_analysis.png', dpi=300, bbox_inches='tight')
print("✓ 已保存可视化图表: geology_influence_analysis.png")

# 创建详细的特征对比表
fig2, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

for i in range(9):
    ax = axes[i]
    feature_values = geo_features[:, i]
    target_values = y_final.flatten()
    
    # 散点图
    ax.scatter(feature_values, target_values, alpha=0.1, s=1, c='blue')
    
    # 拟合线
    z = np.polyfit(feature_values, target_values, 1)
    p = np.poly1d(z)
    x_line = np.linspace(feature_values.min(), feature_values.max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8)
    
    # 相关系数
    corr = correlations[i]
    ax.set_title(f'{geo_names[i]}\n相关系数: {corr:.3f}', 
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('特征值(标准化)', fontsize=9)
    ax.set_ylabel('末阻力(标准化)', fontsize=9)
    ax.grid(alpha=0.3)
    
    # 添加统计信息
    unique = len(np.unique(feature_values))
    ax.text(0.05, 0.95, f'唯一值: {unique}', 
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('9个地质特征与末阻力的关系', fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('geology_features_correlation.png', dpi=300, bbox_inches='tight')
print("✓ 已保存特征相关性图表: geology_features_correlation.png")

# 打印详细统计
print("\n" + "="*70)
print("📊 地质特征详细统计")
print("="*70)
print(f"\n{'特征名':<20} {'唯一值':<8} {'相关系数':<10} {'影响等级'}")
print("-"*70)
for i, name in enumerate(geo_names):
    unique = len(np.unique(geo_features[:, i]))
    corr = correlations[i]
    if abs(corr) > 0.3:
        level = '🔴 强'
    elif abs(corr) > 0.1:
        level = '🟡 中'
    else:
        level = '🟢 弱'
    print(f"{name:<20} {unique:<8} {corr:>9.4f}  {level}")

print("\n" + "="*70)
print("✅ 分析完成！生成了2张可视化图表")
print("="*70)
