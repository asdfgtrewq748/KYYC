"""
生成示例矿压数据 CSV 文件
用于测试 STGCN 模型
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_sample_mine_pressure_data(
    num_supports=50,      # 支架数量
    num_timesteps=1000,   # 时间步数
    with_time=True,       # 是否包含时间列
    output_file='sample_mine_pressure.csv'
):
    """
    生成模拟的矿压数据
    
    参数:
    - num_supports: 支架/监测点数量
    - num_timesteps: 时间步数
    - with_time: 是否包含时间列
    - output_file: 输出文件名
    """
    
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    
    # 生成基础压力值 (80-120 MPa)
    base_pressure = np.random.uniform(80, 120, num_supports)
    
    # 生成时间序列数据
    data = []
    for t in range(num_timesteps):
        # 添加周期性波动 (模拟采矿推进)
        cycle = np.sin(2 * np.pi * t / 100) * 10
        
        # 添加趋势 (压力逐渐增加)
        trend = t * 0.01
        
        # 添加随机噪声
        noise = np.random.normal(0, 2, num_supports)
        
        # 添加空间相关性 (相邻支架压力相似)
        spatial_correlation = np.zeros(num_supports)
        for i in range(num_supports):
            if i > 0:
                spatial_correlation[i] += np.random.normal(0, 0.5)
            if i < num_supports - 1:
                spatial_correlation[i] += np.random.normal(0, 0.5)
        
        # 计算最终压力值
        pressure = base_pressure + cycle + trend + noise + spatial_correlation
        
        # 确保压力值在合理范围内
        pressure = np.clip(pressure, 50, 200)
        
        data.append(pressure)
    
    # 创建 DataFrame
    columns = [f'支架{i+1}' for i in range(num_supports)]
    df = pd.DataFrame(data, columns=columns)
    
    # 如果需要,添加时间列
    if with_time:
        start_time = datetime(2023, 1, 1, 0, 0, 0)
        time_index = [start_time + timedelta(hours=i) for i in range(num_timesteps)]
        df.insert(0, '时间', time_index)
    
    # 保存为 CSV
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"✓ 已生成示例数据文件: {output_file}")
    print(f"  - 形状: {df.shape}")
    print(f"  - 时间步数: {num_timesteps}")
    print(f"  - 支架数量: {num_supports}")
    print(f"  - 包含时间列: {with_time}")
    print(f"\n数据预览:")
    print(df.head())
    print(f"\n统计信息:")
    print(df.describe())
    
    return df

def generate_adjacency_matrix_file(
    num_supports=50,
    method='chain',
    output_file='sample_adjacency_matrix.npy'
):
    """
    生成邻接矩阵文件
    
    参数:
    - num_supports: 支架数量
    - method: 生成方法 ('chain', 'grid', 'full')
    - output_file: 输出文件名
    """
    adj_mx = np.zeros((num_supports, num_supports))
    
    if method == 'chain':
        # 链式结构
        for i in range(num_supports - 1):
            adj_mx[i, i + 1] = 1
            adj_mx[i + 1, i] = 1
    elif method == 'grid':
        # 网格结构
        rows = int(np.sqrt(num_supports))
        cols = num_supports // rows
        for i in range(num_supports):
            row, col = i // cols, i % cols
            neighbors = []
            if row > 0: neighbors.append((row - 1) * cols + col)
            if row < rows - 1: neighbors.append((row + 1) * cols + col)
            if col > 0: neighbors.append(row * cols + col - 1)
            if col < cols - 1: neighbors.append(row * cols + col + 1)
            for j in neighbors:
                adj_mx[i, j] = 1
    elif method == 'full':
        # 全连接
        adj_mx = np.ones((num_supports, num_supports))
        np.fill_diagonal(adj_mx, 0)
    
    np.save(output_file, adj_mx)
    print(f"\n✓ 已生成邻接矩阵文件: {output_file}")
    print(f"  - 形状: {adj_mx.shape}")
    print(f"  - 方法: {method}")
    print(f"  - 边数: {int(np.sum(adj_mx) / 2)}")

if __name__ == "__main__":
    print("=" * 60)
    print("生成 STGCN 示例数据")
    print("=" * 60)
    
    # 生成包含时间列的 CSV 文件
    print("\n[1] 生成带时间列的 CSV 数据...")
    generate_sample_mine_pressure_data(
        num_supports=50,
        num_timesteps=1000,
        with_time=True,
        output_file='sample_mine_pressure_with_time.csv'
    )
    
    # 生成不含时间列的 CSV 文件
    print("\n" + "=" * 60)
    print("[2] 生成不带时间列的 CSV 数据...")
    generate_sample_mine_pressure_data(
        num_supports=50,
        num_timesteps=1000,
        with_time=False,
        output_file='sample_mine_pressure_no_time.csv'
    )
    
    # 生成邻接矩阵(可选)
    print("\n" + "=" * 60)
    print("[3] 生成邻接矩阵 (可选)...")
    generate_adjacency_matrix_file(
        num_supports=50,
        method='chain',
        output_file='sample_adjacency_matrix.npy'
    )
    
    print("\n" + "=" * 60)
    print("✓ 所有示例文件生成完成!")
    print("=" * 60)
    print("\n使用说明:")
    print("1. 上传 'sample_mine_pressure_with_time.csv' 或")
    print("   'sample_mine_pressure_no_time.csv' 到 Streamlit 应用")
    print("2. 选择邻接矩阵生成方式(推荐使用'链式结构')")
    print("   或上传 'sample_adjacency_matrix.npy'")
    print("3. 开始训练模型")
