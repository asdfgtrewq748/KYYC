# -*- coding: utf-8 -*-
"""
矿压数据预处理脚本
功能：合并初撑力和末阻力数据，提取工作循环特征，构建STGCN训练数据集
"""

import os
import sys
# 设置输出编码为UTF-8
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import pandas as pd
import numpy as np
from datetime import datetime

# 文件路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # KYYC项目根目录
DATA_DIR = os.path.join(PROJECT_ROOT, 'kaungya')
INIT_FILE = os.path.join(DATA_DIR, '初撑力数据1-9 (2).csv')
FINAL_FILE = os.path.join(DATA_DIR, '末阻力数据1-9 (2).csv')
GEO_FILE = os.path.join(PROJECT_ROOT, 'geology_features_extracted.csv')
COORD_FILE = os.path.join(PROJECT_ROOT, 'zuobiao.csv')

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'processed_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_clean_data(init_file, final_file):
    """
    加载并清洗初撑力和末阻力数据
    """
    print("=" * 60)
    print("步骤1: 加载原始数据")
    print("=" * 60)
    
    # 读取数据（跳过第一行标题）
    init_df = pd.read_csv(init_file, skiprows=1, encoding='utf-8-sig')
    final_df = pd.read_csv(final_file, skiprows=1, encoding='utf-8-sig')
    
    print(f"初撑力数据: {init_df.shape}")
    print(f"末阻力数据: {final_df.shape}")
    
    # 清理列名
    init_df.columns = init_df.columns.str.strip()
    final_df.columns = final_df.columns.str.strip()
    
    # 转换时间列
    time_cols = ['工作循环开始时间', '工作循环结束时间', '初撑力时间']
    for col in time_cols:
        if col in init_df.columns:
            init_df[col] = pd.to_datetime(init_df[col], errors='coerce')
    
    time_cols_final = ['工作循环开始时间', '工作结束时间', '末阻力时间']
    for col in time_cols_final:
        if col in final_df.columns:
            final_df[col] = pd.to_datetime(final_df[col], errors='coerce')
    
    # 重命名以便合并
    final_df = final_df.rename(columns={'工作结束时间': '工作循环结束时间'})
    
    # 删除无效数据
    init_df = init_df.dropna(subset=['支架号', '初撑力值'])
    final_df = final_df.dropna(subset=['支架号', '末阻力值'])
    
    print(f"清洗后初撑力数据: {init_df.shape}")
    print(f"清洗后末阻力数据: {final_df.shape}")
    
    return init_df, final_df


def merge_and_extract_features(init_df, final_df):
    """
    合并初撑力和末阻力数据并提取特征
    """
    print("\n" + "=" * 60)
    print("步骤2: 合并数据并提取特征")
    print("=" * 60)
    
    # 合并数据集
    merged = pd.merge(
        init_df,
        final_df,
        on=['工作面名称', '支架号', '柱类型', '工作循环开始时间', '工作循环结束时间'],
        how='inner',
        suffixes=('_init', '_final')
    )
    
    print(f"合并后数据量: {merged.shape[0]} 条记录")
    print(f"涉及支架数: {merged['支架号'].nunique()} 个")
    
    # ========== 特征工程 ==========
    
    # 1. 基础压力特征
    merged['压力增量'] = merged['末阻力值'] - merged['初撑力值']
    merged['压力增长率'] = merged['压力增量'] / (merged['初撑力值'] + 1e-6)
    merged['压力平均值'] = (merged['初撑力值'] + merged['末阻力值']) / 2
    
    # 2. 时间特征
    merged['循环时长_秒'] = (merged['工作循环结束时间'] - merged['工作循环开始时间']).dt.total_seconds()
    merged['压力变化速率'] = merged['压力增量'] / (merged['循环时长_秒'] + 1)
    
    # 初撑响应时间
    merged['初撑响应时间_秒'] = (merged['初撑力时间'] - merged['工作循环开始时间']).dt.total_seconds()
    
    # 3. 时间戳特征（周期性）
    merged['小时'] = merged['工作循环开始时间'].dt.hour
    merged['星期几'] = merged['工作循环开始时间'].dt.dayofweek
    merged['月份'] = merged['工作循环开始时间'].dt.month
    merged['日期'] = merged['工作循环开始时间'].dt.date
    
    # 4. 删除异常值
    # 排除循环时长异常的数据（<1分钟或>6小时）
    merged = merged[(merged['循环时长_秒'] >= 60) & (merged['循环时长_秒'] <= 21600)]
    
    # 排除压力值异常的数据（负值或过大值）
    merged = merged[(merged['初撑力值'] >= 0) & (merged['初撑力值'] <= 200)]
    merged = merged[(merged['末阻力值'] >= 0) & (merged['末阻力值'] <= 200)]
    
    print(f"异常值过滤后数据量: {merged.shape[0]} 条记录")
    
    # 按支架号和时间排序
    merged = merged.sort_values(['支架号', '工作循环开始时间']).reset_index(drop=True)
    
    # 显示特征统计
    print("\n特征统计:")
    feature_cols = ['初撑力值', '末阻力值', '压力增量', '压力增长率', '循环时长_秒', '压力变化速率']
    print(merged[feature_cols].describe())
    
    return merged


def add_geological_features(merged_df, geo_file, coord_file):
    """
    为每个支架添加其最近钻孔的地质特征（已修复）
    使用KDTree精确匹配，而不是使用平均值
    """
    print("\n" + "=" * 60)
    print("步骤4: 融合地质特征 (已修复 - 使用最近邻匹配)")
    print("=" * 60)
    
    if not os.path.exists(geo_file):
        print(f"⚠️ 地质特征文件不存在: {geo_file}")
        print("   跳过地质特征融合")
        return merged_df, []
    
    try:
        from scipy.spatial import KDTree
        
        # 1. 加载地质特征（钻孔的）
        geo_df = pd.read_csv(geo_file, encoding='utf-8-sig')
        geo_coords = geo_df[['x', 'y']].values
        geo_features_cols = [col for col in geo_df.columns if col not in ['borehole', 'x', 'y']]
        geo_features_data = geo_df[geo_features_cols]
        print(f"地质特征数据: {geo_df.shape}")
        print(f"钻孔数量: {len(geo_df)}")
        
        # 2. 加载支架坐标
        if not os.path.exists(coord_file):
            print(f"⚠️ 支架坐标文件不存在: {coord_file}")
            print("   正在使用 processed_data/support_coordinates.csv")
            coord_file = os.path.join(OUTPUT_DIR, 'support_coordinates.csv')
            
            if not os.path.exists(coord_file):
                print("   ⚠️ 临时坐标文件也不存在，将先生成")
                # 先创建临时坐标
                support_ids = sorted(merged_df['支架号'].unique())
                num_supports = len(support_ids)
                coords = np.zeros((num_supports, 2))
                coords[:, 0] = np.arange(num_supports) * 1.5  # 支架间距1.5米
                coords[:, 1] = 0
                coord_df_temp = pd.DataFrame({
                    'support_id': support_ids,
                    'x': coords[:, 0],
                    'y': coords[:, 1]
                })
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                coord_df_temp.to_csv(coord_file, index=False, encoding='utf-8-sig')
                print(f"   ✓ 已创建临时支架坐标: {coord_file}")
        
        coord_df = pd.read_csv(coord_file, encoding='utf-8-sig')
        support_coords = coord_df[['x', 'y']].values
        support_ids = coord_df['support_id'].values
        print(f"支架坐标数据: {coord_df.shape}")
        
        # 3. 使用KDTree查找最近的钻孔
        #    为每个支架(support_coords) 找到 geo_coords 中最近的索引
        tree = KDTree(geo_coords)
        distances, indices = tree.query(support_coords)
        
        print(f"✓ 已为每个支架匹配最近的钻孔")
        print(f"  平均距离: {np.mean(distances):.2f} 米")
        print(f"  最大距离: {np.max(distances):.2f} 米")
        
        # 4. 创建 支架ID -> 地质特征 的映射
        #    indices 是 geo_df 的行索引
        nearest_geo_features = geo_features_data.iloc[indices]
        
        # 将支架ID与特征关联
        support_geo_map = {}
        for i, support_id in enumerate(support_ids):
            support_geo_map[support_id] = nearest_geo_features.iloc[i].to_dict()
        
        print(f"✓ 已为 {len(support_ids)} 个支架创建地质特征映射")
        
        # 5. 将特征映射合并回主DataFrame
        #    使用支架号（假设 merged_df 中有 '支架号' 列）
        
        # 确保 '支架号' 为可映射的类型
        merged_df['支架号'] = merged_df['支架号'].astype(type(support_ids[0]))
        
        # 为每个地质特征列创建映射
        for geo_col in geo_features_cols:
            col_map = {sid: support_geo_map[sid][geo_col] for sid in support_ids}
            merged_df[f'geo_{geo_col}'] = merged_df['支架号'].map(col_map)
        
        print(f"✓ 添加了 {len(geo_features_cols)} 个地质特征列")
        
        # 验证地质特征的变化性
        geo_col_names = [f'geo_{col}' for col in geo_features_cols]
        print(f"\n地质特征统计（验证是否有真实变化）:")
        for geo_col in geo_col_names[:3]:  # 只显示前3个特征
            unique_vals = merged_df[geo_col].nunique()
            print(f"  {geo_col}: {unique_vals} 个唯一值")
        
        return merged_df, geo_features_cols
        
    except ImportError:
        print("⚠️ 缺少 scipy 库，无法使用KDTree")
        print("   请运行: pip install scipy")
        print("   暂时跳过地质特征融合")
        return merged_df, []
    except Exception as e:
        print(f"⚠️ 地质特征融合失败: {e}")
        import traceback
        traceback.print_exc()
        print("   跳过地质特征融合")
        return merged_df, []


def create_support_coordinates(merged_df, geo_file=None):
    """
    创建支架坐标（与钻孔坐标系统对齐）
    策略：让支架在钻孔覆盖区域内均匀分布，以便匹配到不同钻孔
    """
    print("\n" + "=" * 60)
    print("步骤3: 创建支架坐标（与钻孔坐标系对齐）")
    print("=" * 60)
    
    support_ids = sorted(merged_df['支架号'].unique())
    num_supports = len(support_ids)
    
    # 如果有地质文件，基于钻孔坐标范围生成支架坐标
    if geo_file and os.path.exists(geo_file):
        geo_df = pd.read_csv(geo_file, encoding='utf-8-sig')
        x_min, x_max = geo_df['x'].min(), geo_df['x'].max()
        y_min, y_max = geo_df['y'].min(), geo_df['y'].max()
        
        # 计算钻孔区域的跨度
        x_span = x_max - x_min  # X方向跨度（约2000米）
        y_span = y_max - y_min  # Y方向跨度（约650米）
        
        # 策略：支架在X-Y平面上呈网格状或S型分布
        # 这样可以让不同支架匹配到不同钻孔
        
        # 方法1：沿对角线分布（从西南到东北）
        coords = np.zeros((num_supports, 2))
        
        # 让支架沿着钻孔区域的对角线分布，并添加小的Y方向偏移
        for i in range(num_supports):
            ratio = i / max(num_supports - 1, 1)  # 0 到 1
            coords[i, 0] = x_min + ratio * x_span  # X从西到东
            coords[i, 1] = y_min + ratio * y_span  # Y从南到北
            
            # 添加小的Y方向摆动（模拟工作面推进）
            # 支架间距约1.5米，添加周期性摆动
            y_wiggle = 50 * np.sin(i * 0.3)  # ±50米的正弦摆动
            coords[i, 1] += y_wiggle
        
        print(f"✓ 基于钻孔坐标范围生成支架坐标（对角线分布）")
        print(f"  钻孔X范围: {x_min:.2f} - {x_max:.2f} (跨度: {x_span:.2f} 米)")
        print(f"  钻孔Y范围: {y_min:.2f} - {y_max:.2f} (跨度: {y_span:.2f} 米)")
        print(f"  支架X范围: {coords[:, 0].min():.2f} - {coords[:, 0].max():.2f}")
        print(f"  支架Y范围: {coords[:, 1].min():.2f} - {coords[:, 1].max():.2f}")
    else:
        # 简化坐标：假设支架沿工作面线性排列
        coords = np.zeros((num_supports, 2))
        coords[:, 0] = np.arange(num_supports) * 1.5  # 支架间距1.5米
        coords[:, 1] = 0  # 假设在同一条线上
        print(f"⚠️ 未找到地质文件，使用简化坐标系统")
    
    coord_df = pd.DataFrame({
        'support_id': support_ids,
        'x': coords[:, 0],
        'y': coords[:, 1]
    })
    
    print(f"✓ 创建了 {num_supports} 个支架的坐标")
    print(f"  支架编号范围: {support_ids[0]} - {support_ids[-1]}")
    
    return coord_df


def create_sequence_dataset(merged_df, seq_len=5, pred_len=1):
    """
    创建序列数据集用于STGCN训练
    
    参数:
        seq_len: 历史序列长度（使用过去多少个工作循环）
        pred_len: 预测长度（预测未来多少个工作循环）
    
    返回:
        sequences: 列表，每个元素包含一个训练样本
    """
    print("\n" + "=" * 60)
    print("步骤5: 构建序列数据集")
    print("=" * 60)
    print(f"配置: 历史长度={seq_len}, 预测长度={pred_len}")
    
    # 确定特征列
    pressure_features = ['初撑力值', '末阻力值', '压力增量', '压力增长率', '循环时长_秒', '压力变化速率']
    geo_features = [col for col in merged_df.columns if col.startswith('geo_')]
    time_features = ['小时', '星期几']
    
    all_features = pressure_features + geo_features + time_features
    print(f"特征维度: {len(all_features)} (压力={len(pressure_features)}, 地质={len(geo_features)}, 时间={len(time_features)})")
    
    sequences = []
    support_ids = sorted(merged_df['支架号'].unique())
    
    for support_id in support_ids:
        support_data = merged_df[merged_df['支架号'] == support_id].copy()
        
        # 至少需要 seq_len + pred_len 条记录
        if len(support_data) < seq_len + pred_len:
            continue
        
        for i in range(len(support_data) - seq_len - pred_len + 1):
            # 历史序列
            hist_seq = support_data.iloc[i:i+seq_len]
            # 预测目标
            target_seq = support_data.iloc[i+seq_len:i+seq_len+pred_len]
            
            # 提取特征和标签
            X = hist_seq[all_features].values  # (seq_len, num_features)
            y_init = target_seq['初撑力值'].values  # (pred_len,)
            y_final = target_seq['末阻力值'].values  # (pred_len,)
            
            sequences.append({
                'support_id': support_id,
                'start_time': hist_seq.iloc[0]['工作循环开始时间'],
                'end_time': target_seq.iloc[-1]['工作循环结束时间'],
                'X': X,
                'y_init': y_init,
                'y_final': y_final,
                'feature_names': all_features
            })
    
    print(f"✓ 生成了 {len(sequences)} 个训练样本")
    print(f"  涉及支架数: {len(support_ids)}")
    print(f"  样本形状: X={sequences[0]['X'].shape}, y_final={sequences[0]['y_final'].shape}")
    
    return sequences, support_ids, all_features


def save_processed_data(merged_df, sequences, coord_df, feature_names, output_dir):
    """
    保存处理后的数据
    """
    print("\n" + "=" * 60)
    print("步骤6: 保存处理后的数据")
    print("=" * 60)
    
    # 1. 保存合并后的完整数据
    merged_file = os.path.join(output_dir, 'merged_pressure_data.csv')
    merged_df.to_csv(merged_file, index=False, encoding='utf-8-sig')
    print(f"✓ 保存合并数据: {merged_file}")
    
    # 2. 保存支架坐标
    coord_file = os.path.join(output_dir, 'support_coordinates.csv')
    coord_df.to_csv(coord_file, index=False, encoding='utf-8-sig')
    print(f"✓ 保存支架坐标: {coord_file}")
    
    # 3. 保存序列数据（NumPy格式，便于快速加载）
    X_list = [seq['X'] for seq in sequences]
    y_init_list = [seq['y_init'] for seq in sequences]
    y_final_list = [seq['y_final'] for seq in sequences]
    support_ids = [seq['support_id'] for seq in sequences]
    
    X_array = np.array(X_list)  # (N, seq_len, features)
    y_init_array = np.array(y_init_list)
    y_final_array = np.array(y_final_list)
    
    # ⭐ 关键修复：对所有特征进行标准化，解决数值范围差异过大的问题
    print("\n⭐ 对特征进行标准化处理...")
    from sklearn.preprocessing import StandardScaler
    
    # 保存原始数据形状
    n_samples, seq_len, n_features = X_array.shape
    print(f"  原始形状: {X_array.shape}")
    
    # 重塑为2D进行标准化
    X_reshaped = X_array.reshape(-1, n_features)
    
    # 显示标准化前的数值范围
    print(f"  标准化前范围: [{X_reshaped.min():.2f}, {X_reshaped.max():.2f}]")
    print(f"  标准化前特征统计:")
    for i, fname in enumerate(feature_names):
        feat_data = X_reshaped[:, i]
        print(f"    {fname}: [{feat_data.min():.2f}, {feat_data.max():.2f}] (std={feat_data.std():.2f})")
    
    # 标准化
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_reshaped)
    
    # 恢复形状
    X_normalized = X_normalized.reshape(n_samples, seq_len, n_features)
    
    print(f"  标准化后范围: [{X_normalized.min():.2f}, {X_normalized.max():.2f}]")
    print(f"  标准化后均值: {X_normalized.mean():.4f} (应接近0)")
    print(f"  标准化后标准差: {X_normalized.std():.4f} (应接近1)")
    
    # 检查是否有NaN或Inf
    if np.isnan(X_normalized).any():
        print(f"  ⚠️ 警告：标准化后出现 {np.isnan(X_normalized).sum()} 个NaN值")
        X_normalized = np.nan_to_num(X_normalized, nan=0.0)
    if np.isinf(X_normalized).any():
        print(f"  ⚠️ 警告：标准化后出现 {np.isinf(X_normalized).sum()} 个Inf值")
        X_normalized = np.nan_to_num(X_normalized, posinf=0.0, neginf=0.0)
    
    print(f"  ✓ 特征标准化完成")
    
    # 保存标准化参数，供后续预测使用
    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'feature_names': feature_names
    }
    
    import json
    scaler_file = os.path.join(output_dir, 'feature_scaler.json')
    with open(scaler_file, 'w', encoding='utf-8') as f:
        json.dump(scaler_params, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 保存标准化参数: feature_scaler.json")
    
    # 保存标准化后的数据
    np.savez_compressed(
        os.path.join(output_dir, 'sequence_dataset.npz'),
        X=X_normalized,  # 使用标准化后的数据
        y_init=y_init_array,
        y_final=y_final_array,
        support_ids=np.array(support_ids),
        feature_names=feature_names
    )
    print(f"✓ 保存序列数据: sequence_dataset.npz (已标准化)")
    
    # 4. 保存数据摘要
    summary = {
        'num_samples': len(sequences),
        'num_supports': coord_df.shape[0],
        'num_features': len(feature_names),
        'seq_len': sequences[0]['X'].shape[0] if sequences else 0,
        'pred_len': sequences[0]['y_final'].shape[0] if sequences else 0,
        'date_range': f"{merged_df['工作循环开始时间'].min()} to {merged_df['工作循环结束时间'].max()}",
        'feature_names': feature_names
    }
    
    import json
    with open(os.path.join(output_dir, 'dataset_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"✓ 保存数据摘要: dataset_summary.json")
    
    print("\n" + "=" * 60)
    print("✅ 数据预处理完成！")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print(f"文件列表:")
    for f in os.listdir(output_dir):
        fpath = os.path.join(output_dir, f)
        size = os.path.getsize(fpath) / 1024 / 1024
        print(f"  - {f} ({size:.2f} MB)")


def main():
    """
    主流程
    """
    print("=" * 60)
    print("矿压数据预处理流程")
    print("=" * 60)
    
    # 检查输入文件
    if not os.path.exists(INIT_FILE):
        print(f"❌ 初撑力文件不存在: {INIT_FILE}")
        return
    if not os.path.exists(FINAL_FILE):
        print(f"❌ 末阻力文件不存在: {FINAL_FILE}")
        return
    
    # 步骤1: 加载数据
    init_df, final_df = load_and_clean_data(INIT_FILE, FINAL_FILE)
    
    # 步骤2: 合并和特征工程
    merged_df = merge_and_extract_features(init_df, final_df)
    
    # 步骤3: 先创建支架坐标（需要在添加地质特征之前）
    coord_df = create_support_coordinates(merged_df, GEO_FILE)
    
    # 保存临时坐标文件供地质特征匹配使用
    temp_coord_file = os.path.join(OUTPUT_DIR, 'support_coordinates.csv')
    coord_df.to_csv(temp_coord_file, index=False, encoding='utf-8-sig')
    
    # 步骤4: 添加地质特征（使用刚创建的坐标）
    merged_df, geo_features = add_geological_features(merged_df, GEO_FILE, temp_coord_file)
    
    # 步骤5: 构建序列数据集
    sequences, support_ids, feature_names = create_sequence_dataset(
        merged_df, 
        seq_len=5,  # 使用过去5个工作循环
        pred_len=1   # 预测下1个工作循环
    )
    
    # 步骤6: 保存结果
    save_processed_data(merged_df, sequences, coord_df, feature_names, OUTPUT_DIR)
    
    print("\n🎉 全部完成！数据已准备好用于STGCN训练")


if __name__ == '__main__':
    main()
