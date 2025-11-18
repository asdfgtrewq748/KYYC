"""
地质条件综合影响指数计算器
基于煤矿工程理论，将9个地质特征融合为1个影响指数
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class GeologyInfluenceIndex:
    """
    地质条件综合影响指数
    
    核心思路：
    1. 基于物理公式计算多个分指数
    2. 加权组合得到综合影响指数
    3. 替代原始9维特征，降维到1维
    """
    
    def __init__(self, method='comprehensive'):
        """
        参数:
            method: 计算方法
                - 'comprehensive': 综合指数（推荐）
                - 'stability': 顶板稳定性指数
                - 'stress': 应力集中系数
                - 'pca': 主成分分析
        """
        self.method = method
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, geo_features_df):
        """
        拟合标准化器
        
        参数:
            geo_features_df: DataFrame, 包含9个地质特征的数据
        """
        self.feature_names = geo_features_df.columns.tolist()
        self.scaler.fit(geo_features_df)
        self.is_fitted = True
        return self
    
    def calculate_stability_index(self, geo_df):
        """
        方法1: 顶板稳定性指数
        
        公式: 稳定性 = (弹性模量 × 抗拉强度) / (密度 × 总厚度)
        
        物理意义:
        - 分子：岩体强度（刚性 × 抗破坏能力）
        - 分母：重力载荷（质量 × 体积）
        - 高值 → 稳定顶板 → 压力低但集中
        - 低值 → 易垮顶板 → 压力波动大
        """
        elastic = geo_df['avg_elastic_modulus_GPa']
        tensile = geo_df['max_tensile_MPa']
        density = geo_df['avg_density_kN_m3']
        thickness = geo_df['total_thickness_m']
        
        # 避免除零
        denominator = density * thickness
        denominator = np.where(denominator == 0, 1e-6, denominator)
        
        stability = (elastic * tensile) / denominator
        
        return stability
    
    def calculate_stress_concentration(self, geo_df):
        """
        方法2: 应力集中系数
        
        公式: 应力集中 = (硬岩占比 × 弹性模量) / (软岩占比 + 0.1)
        
        物理意义:
        - 硬岩层 → 应力累积 → 突然释放
        - 软岩层 → 应力分散 → 渐进变形
        - 高值 → 应力集中 → 压力峰值高
        - 低值 → 应力均匀 → 压力平稳
        """
        hard_ratio = geo_df['prop_sandstone']
        soft_ratio = geo_df['prop_mudstone']
        elastic = geo_df['avg_elastic_modulus_GPa']
        
        # 避免除零
        denominator = soft_ratio + 0.1
        
        stress_coef = (hard_ratio * elastic) / denominator
        
        return stress_coef
    
    def calculate_lithology_index(self, geo_df):
        """
        方法3: 岩性组合指数
        
        公式: 岩性指数 = 砂岩占比² / (泥岩占比 + 0.1) × 煤层数量
        
        物理意义:
        - 多煤层 + 硬顶板 → 复杂应力分布
        - 泥岩占比高 → 塑性变形大
        """
        sandstone = geo_df['prop_sandstone']
        mudstone = geo_df['prop_mudstone']
        coal_seam = geo_df['coal_seam_count']
        
        lithology = (sandstone ** 2) / (mudstone + 0.1) * coal_seam
        
        return lithology
    
    def calculate_depth_effect(self, geo_df):
        """
        方法4: 埋深效应系数
        
        公式: 埋深效应 = 顶板深度 / 总厚度 × 密度
        
        物理意义:
        - 埋深大 → 地应力高
        - 考虑岩层密度的重力效应
        """
        depth = geo_df['depth_to_top_coal_m']
        thickness = geo_df['total_thickness_m']
        density = geo_df['avg_density_kN_m3']
        
        # 避免除零
        thickness = np.where(thickness == 0, 1e-6, thickness)
        
        depth_effect = (depth / thickness) * density
        
        return depth_effect
    
    def calculate_comprehensive_index(self, geo_df):
        """
        方法5: 综合影响指数（推荐）
        
        公式: 综合指数 = w1×稳定性 + w2×应力集中 + w3×岩性 + w4×埋深
        
        权重设计:
        - w1 = 0.35 (顶板稳定性最重要)
        - w2 = 0.30 (应力集中次之)
        - w3 = 0.20 (岩性组合)
        - w4 = 0.15 (埋深效应)
        """
        # 计算各分指数
        stability = self.calculate_stability_index(geo_df)
        stress = self.calculate_stress_concentration(geo_df)
        lithology = self.calculate_lithology_index(geo_df)
        depth = self.calculate_depth_effect(geo_df)
        
        # 标准化各分指数（使其在相同尺度）
        stability_norm = (stability - stability.mean()) / (stability.std() + 1e-6)
        stress_norm = (stress - stress.mean()) / (stress.std() + 1e-6)
        lithology_norm = (lithology - lithology.mean()) / (lithology.std() + 1e-6)
        depth_norm = (depth - depth.mean()) / (depth.std() + 1e-6)
        
        # 加权组合
        weights = {
            'stability': 0.35,
            'stress': 0.30,
            'lithology': 0.20,
            'depth': 0.15
        }
        
        comprehensive = (
            weights['stability'] * stability_norm +
            weights['stress'] * stress_norm +
            weights['lithology'] * lithology_norm +
            weights['depth'] * depth_norm
        )
        
        return comprehensive
    
    def transform(self, geo_features_df):
        """
        将9维地质特征转换为1维影响指数
        
        参数:
            geo_features_df: DataFrame, 包含9个地质特征
        
        返回:
            index: ndarray, 形状(n_samples,) 综合影响指数
        """
        if not self.is_fitted:
            raise ValueError("必须先调用fit()方法")
        
        if self.method == 'stability':
            index = self.calculate_stability_index(geo_features_df)
        elif self.method == 'stress':
            index = self.calculate_stress_concentration(geo_features_df)
        elif self.method == 'comprehensive':
            index = self.calculate_comprehensive_index(geo_features_df)
        elif self.method == 'pca':
            # 主成分分析降维
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            index = pca.fit_transform(geo_features_df).flatten()
        else:
            raise ValueError(f"未知方法: {self.method}")
        
        return index.values if hasattr(index, 'values') else index
    
    def fit_transform(self, geo_features_df):
        """拟合并转换"""
        self.fit(geo_features_df)
        return self.transform(geo_features_df)
    
    def get_feature_importance(self, geo_features_df):
        """
        分析各个原始特征对综合指数的贡献度
        """
        if self.method != 'comprehensive':
            raise ValueError("仅支持comprehensive方法")
        
        # 计算各分指数
        stability = self.calculate_stability_index(geo_features_df)
        stress = self.calculate_stress_concentration(geo_features_df)
        lithology = self.calculate_lithology_index(geo_features_df)
        depth = self.calculate_depth_effect(geo_features_df)
        
        importance = {
            '顶板稳定性': {
                'weight': 0.35,
                'mean': stability.mean(),
                'std': stability.std(),
                'range': [stability.min(), stability.max()]
            },
            '应力集中': {
                'weight': 0.30,
                'mean': stress.mean(),
                'std': stress.std(),
                'range': [stress.min(), stress.max()]
            },
            '岩性组合': {
                'weight': 0.20,
                'mean': lithology.mean(),
                'std': lithology.std(),
                'range': [lithology.min(), lithology.max()]
            },
            '埋深效应': {
                'weight': 0.15,
                'mean': depth.mean(),
                'std': depth.std(),
                'range': [depth.min(), depth.max()]
            }
        }
        
        return importance


def test_geology_index():
    """测试地质影响指数计算"""
    print("=" * 70)
    print("🧪 地质影响指数计算器测试")
    print("=" * 70)
    
    # 加载地质特征数据
    geo_df = pd.read_csv('geology_features_extracted.csv', encoding='utf-8-sig')
    
    print(f"\n📊 原始数据:")
    print(f"钻孔数量: {len(geo_df)}")
    print(f"特征维度: {len(geo_df.columns) - 3}")  # 减去borehole, x, y
    
    # 提取特征列
    feature_cols = [col for col in geo_df.columns if col not in ['borehole', 'x', 'y']]
    geo_features = geo_df[feature_cols]
    
    print(f"\n地质特征: {feature_cols}")
    
    # 方法1: 综合指数
    print("\n" + "=" * 70)
    print("方法1: 综合影响指数（推荐）")
    print("=" * 70)
    
    calculator = GeologyInfluenceIndex(method='comprehensive')
    comprehensive_index = calculator.fit_transform(geo_features)
    
    print(f"✓ 综合指数统计:")
    print(f"  均值: {comprehensive_index.mean():.4f}")
    print(f"  标准差: {comprehensive_index.std():.4f}")
    print(f"  范围: [{comprehensive_index.min():.4f}, {comprehensive_index.max():.4f}]")
    print(f"  唯一值数量: {len(np.unique(comprehensive_index))}")
    
    # 特征重要性
    print(f"\n📊 分指数贡献度:")
    importance = calculator.get_feature_importance(geo_features)
    for name, info in importance.items():
        print(f"\n  {name}:")
        print(f"    权重: {info['weight']:.2f}")
        print(f"    均值: {info['mean']:.4f}")
        print(f"    标准差: {info['std']:.4f}")
        print(f"    范围: [{info['range'][0]:.4f}, {info['range'][1]:.4f}]")
    
    # 方法2: 顶板稳定性
    print("\n" + "=" * 70)
    print("方法2: 顶板稳定性指数")
    print("=" * 70)
    
    calc_stability = GeologyInfluenceIndex(method='stability')
    stability_index = calc_stability.fit_transform(geo_features)
    
    print(f"✓ 稳定性指数统计:")
    print(f"  均值: {stability_index.mean():.4f}")
    print(f"  标准差: {stability_index.std():.4f}")
    print(f"  范围: [{stability_index.min():.4f}, {stability_index.max():.4f}]")
    
    # 方法3: 应力集中
    print("\n" + "=" * 70)
    print("方法3: 应力集中系数")
    print("=" * 70)
    
    calc_stress = GeologyInfluenceIndex(method='stress')
    stress_index = calc_stress.fit_transform(geo_features)
    
    print(f"✓ 应力集中系数统计:")
    print(f"  均值: {stress_index.mean():.4f}")
    print(f"  标准差: {stress_index.std():.4f}")
    print(f"  范围: [{stress_index.min():.4f}, {stress_index.max():.4f}]")
    
    # 保存结果
    result_df = geo_df[['borehole']].copy()
    result_df['comprehensive_index'] = comprehensive_index
    result_df['stability_index'] = stability_index
    result_df['stress_index'] = stress_index
    
    result_df.to_csv('geology_influence_indices.csv', index=False, encoding='utf-8-sig')
    print(f"\n✓ 已保存影响指数到: geology_influence_indices.csv")
    
    # 对比分析
    print("\n" + "=" * 70)
    print("📈 维度对比")
    print("=" * 70)
    print(f"原始方案: 9个地质特征 → 模型处理")
    print(f"新方案:   1个综合指数 → 模型处理")
    print(f"降维幅度: {(1 - 1/9)*100:.1f}%")
    print(f"\n优势:")
    print(f"  ✓ 物理意义明确")
    print(f"  ✓ 参数量减少89%")
    print(f"  ✓ 可解释性强")
    print(f"  ✓ 过拟合风险低")
    print(f"  ✓ 工程应用方便")
    
    print("\n" + "=" * 70)
    print("🎯 使用建议")
    print("=" * 70)
    print(f"1. 推荐使用'comprehensive'综合指数")
    print(f"2. 如果关注顶板稳定性，用'stability'")
    print(f"3. 如果关注应力分布，用'stress'")
    print(f"4. 可以同时使用多个指数进行对比")
    
    return comprehensive_index, stability_index, stress_index


if __name__ == "__main__":
    comprehensive, stability, stress = test_geology_index()
    
    print("\n" + "=" * 70)
    print("✅ 测试完成！")
    print("=" * 70)
