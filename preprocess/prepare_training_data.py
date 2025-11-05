"""
矿压数据预处理脚本
功能：合并初撑力和末阻力数据，提取工作循环特征，构建STGCN训练数据集
"""

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
    为每个支架添加地质特征
    """
    print("\n" + "=" * 60)
    print("步骤3: 融合地质特征")
    print("=" * 60)
    
    if not os.path.exists(geo_file):
        print(f"⚠️ 地质特征文件不存在: {geo_file}")
        print("   跳过地质特征融合")
        return merged_df, []
    
    # 读取地质特征
    geo_df = pd.read_csv(geo_file, encoding='utf-8-sig')
    print(f"地质特征数据: {geo_df.shape}")
    
    # 读取坐标映射（如果存在）
    # 这里假设支架号可以映射到某个钻孔
    # 实际应用中需要根据支架坐标找最近的钻孔
    
    # 简化处理：使用平均地质特征（实际应根据支架位置映射）
    geo_features_cols = [col for col in geo_df.columns if col not in ['borehole', 'x', 'y']]
    geo_mean = geo_df[geo_features_cols].mean()
    
    # 为每条记录添加地质特征
    for col in geo_features_cols:
        merged_df[f'geo_{col}'] = geo_mean[col]
    
    print(f"✓ 添加了 {len(geo_features_cols)} 个地质特征")
    print(f"  特征列表: {geo_features_cols[:5]}...")
    
    return merged_df, geo_features_cols


def create_support_coordinates(merged_df):
    """
    创建支架坐标（简化版：假设支架线性排列）
    实际应用中应使用真实的支架坐标文件
    """
    print("\n" + "=" * 60)
    print("步骤4: 创建支架坐标")
    print("=" * 60)
    
    support_ids = sorted(merged_df['支架号'].unique())
    num_supports = len(support_ids)
    
    # 简化坐标：假设支架沿工作面线性排列
    coords = np.zeros((num_supports, 2))
    coords[:, 0] = np.arange(num_supports) * 1.5  # 支架间距1.5米
    coords[:, 1] = 0  # 假设在同一条线上
    
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
    
    np.savez_compressed(
        os.path.join(output_dir, 'sequence_dataset.npz'),
        X=np.array(X_list),
        y_init=np.array(y_init_list),
        y_final=np.array(y_final_list),
        support_ids=np.array(support_ids),
        feature_names=feature_names
    )
    print(f"✓ 保存序列数据: sequence_dataset.npz")
    
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
    
    # 步骤3: 添加地质特征
    merged_df, geo_features = add_geological_features(merged_df, GEO_FILE, COORD_FILE)
    
    # 步骤4: 创建支架坐标
    coord_df = create_support_coordinates(merged_df)
    
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
