"""
生成示例坐标文件和地质特征文件
用于演示坐标对齐功能
"""
import numpy as np
import pandas as pd

def generate_support_coordinates(num_supports=50, layout='linear', output_file='support_coordinates.csv'):
    """
    生成支架坐标文件
    
    参数:
    - num_supports: 支架数量
    - layout: 布局方式 ('linear'=线性, 'grid'=网格)
    - output_file: 输出文件名
    """
    np.random.seed(42)
    
    support_ids = [f'支架{i+1}' for i in range(num_supports)]
    
    if layout == 'linear':
        # 线性排列,沿X轴
        x_coords = np.linspace(1000, 1000 + num_supports * 1.5, num_supports)
        y_coords = np.ones(num_supports) * 2000 + np.random.normal(0, 0.1, num_supports)
        z_coords = np.ones(num_supports) * (-500) + np.random.normal(0, 0.5, num_supports)
        
    elif layout == 'grid':
        # 网格排列
        rows = int(np.sqrt(num_supports))
        cols = (num_supports + rows - 1) // rows
        
        x_coords = []
        y_coords = []
        z_coords = []
        
        for i in range(num_supports):
            row = i // cols
            col = i % cols
            x_coords.append(1000 + col * 1.5 + np.random.normal(0, 0.05))
            y_coords.append(2000 + row * 1.5 + np.random.normal(0, 0.05))
            z_coords.append(-500 + np.random.normal(0, 0.5))
        
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        z_coords = np.array(z_coords)
    
    # 创建 DataFrame
    df = pd.DataFrame({
        '支架ID': support_ids,
        'X坐标': x_coords,
        'Y坐标': y_coords,
        'Z坐标': z_coords,
        '安装日期': pd.date_range('2023-01-01', periods=num_supports, freq='D'),
        '备注': ['正常' for _ in range(num_supports)]
    })
    
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"✓ 已生成支架坐标文件: {output_file}")
    print(f"  - 支架数量: {num_supports}")
    print(f"  - 布局方式: {layout}")
    print(f"\n坐标范围:")
    print(f"  - X: {x_coords.min():.2f} ~ {x_coords.max():.2f}")
    print(f"  - Y: {y_coords.min():.2f} ~ {y_coords.max():.2f}")
    print(f"  - Z: {z_coords.min():.2f} ~ {z_coords.max():.2f}")
    print(f"\n数据预览:")
    print(df.head())
    
    return df

def generate_geological_features(
    x_range=(1000, 1100),
    y_range=(2000, 2020),
    grid_size=20,
    output_file='geological_features.csv'
):
    """
    生成地质特征数据
    
    参数:
    - x_range: X坐标范围
    - y_range: Y坐标范围
    - grid_size: 网格密度
    - output_file: 输出文件名
    """
    np.random.seed(42)
    
    # 生成网格坐标
    x_points = np.linspace(x_range[0], x_range[1], grid_size)
    y_points = np.linspace(y_range[0], y_range[1], grid_size)
    
    X, Y = np.meshgrid(x_points, y_points)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    # 生成模拟的地质特征
    
    # 1. 煤层厚度 (2.5-4.5米)
    coal_thickness = 3.5 + 0.5 * np.sin(X_flat / 20) + 0.3 * np.cos(Y_flat / 10) + np.random.normal(0, 0.2, len(X_flat))
    coal_thickness = np.clip(coal_thickness, 2.5, 4.5)
    
    # 2. 断层距离 (0-100米)
    # 模拟一条从左上到右下的断层
    fault_line_x = 1050
    fault_distance = np.abs(X_flat - fault_line_x) + np.random.normal(0, 2, len(X_flat))
    fault_distance = np.clip(fault_distance, 0, 100)
    
    # 3. 顶板强度 (60-100 MPa)
    roof_strength = 80 + 10 * np.sin(X_flat / 30) + np.random.normal(0, 3, len(X_flat))
    roof_strength = np.clip(roof_strength, 60, 100)
    
    # 4. 地质构造复杂度 (0-1)
    complexity = 0.5 + 0.2 * np.sin(X_flat / 15) * np.cos(Y_flat / 15) + np.random.normal(0, 0.1, len(X_flat))
    complexity = np.clip(complexity, 0, 1)
    
    # 5. 瓦斯含量 (m³/t)
    gas_content = 8 + 3 * np.sin(X_flat / 25) + np.random.normal(0, 0.5, len(X_flat))
    gas_content = np.clip(gas_content, 5, 15)
    
    # 创建 DataFrame
    df = pd.DataFrame({
        'X坐标': X_flat,
        'Y坐标': Y_flat,
        '煤层厚度': coal_thickness,
        '断层距离': fault_distance,
        '顶板强度': roof_strength,
        '构造复杂度': complexity,
        '瓦斯含量': gas_content
    })
    
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n✓ 已生成地质特征文件: {output_file}")
    print(f"  - 数据点数: {len(df)}")
    print(f"  - 特征数量: {len(df.columns) - 2}")  # 减去坐标列
    print(f"\n特征统计:")
    print(df.describe())
    
    return df

def generate_aligned_mine_pressure_data(
    coord_file='support_coordinates.csv',
    num_timesteps=1000,
    output_file='aligned_mine_pressure.csv'
):
    """
    生成与坐标文件对齐的矿压数据
    
    参数:
    - coord_file: 坐标文件路径
    - num_timesteps: 时间步数
    - output_file: 输出文件名
    """
    np.random.seed(42)
    
    # 读取坐标文件
    coord_df = pd.read_csv(coord_file)
    support_ids = coord_df['支架ID'].tolist()
    num_supports = len(support_ids)
    
    # 生成基础压力值
    base_pressure = np.random.uniform(80, 120, num_supports)
    
    # 生成时间序列数据
    data = []
    time_stamps = []
    
    from datetime import datetime, timedelta
    start_time = datetime(2023, 1, 1, 0, 0, 0)
    
    for t in range(num_timesteps):
        time_stamps.append(start_time + timedelta(hours=t))
        
        # 周期性波动
        cycle = np.sin(2 * np.pi * t / 100) * 10
        
        # 趋势
        trend = t * 0.01
        
        # 随机噪声
        noise = np.random.normal(0, 2, num_supports)
        
        # 空间相关性
        spatial_correlation = np.zeros(num_supports)
        for i in range(num_supports):
            if i > 0:
                spatial_correlation[i] += np.random.normal(0, 0.5)
            if i < num_supports - 1:
                spatial_correlation[i] += np.random.normal(0, 0.5)
        
        # 计算压力
        pressure = base_pressure + cycle + trend + noise + spatial_correlation
        pressure = np.clip(pressure, 50, 200)
        
        data.append(pressure)
    
    # 创建 DataFrame
    df = pd.DataFrame(data, columns=support_ids)
    df.insert(0, '时间', time_stamps)
    
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n✓ 已生成对齐的矿压数据文件: {output_file}")
    print(f"  - 时间步数: {num_timesteps}")
    print(f"  - 支架数量: {num_supports}")
    print(f"  - 支架ID列表: {support_ids[:5]}...")
    print(f"\n数据预览:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    print("=" * 70)
    print("生成坐标对齐示例数据")
    print("=" * 70)
    
    # 1. 生成支架坐标(线性布局)
    print("\n[1] 生成支架坐标文件 (线性布局)...")
    coord_df = generate_support_coordinates(
        num_supports=50,
        layout='linear',
        output_file='support_coordinates_linear.csv'
    )
    
    # 2. 生成支架坐标(网格布局)
    print("\n" + "=" * 70)
    print("[2] 生成支架坐标文件 (网格布局)...")
    coord_df_grid = generate_support_coordinates(
        num_supports=49,  # 7x7 网格
        layout='grid',
        output_file='support_coordinates_grid.csv'
    )
    
    # 3. 生成地质特征数据
    print("\n" + "=" * 70)
    print("[3] 生成地质特征文件...")
    geo_df = generate_geological_features(
        x_range=(999, 1080),
        y_range=(1999, 2005),
        grid_size=30,
        output_file='geological_features.csv'
    )
    
    # 4. 生成与坐标对齐的矿压数据
    print("\n" + "=" * 70)
    print("[4] 生成对齐的矿压数据...")
    mine_pressure_df = generate_aligned_mine_pressure_data(
        coord_file='support_coordinates_linear.csv',
        num_timesteps=1000,
        output_file='aligned_mine_pressure.csv'
    )
    
    print("\n" + "=" * 70)
    print("✓ 所有文件生成完成!")
    print("=" * 70)
    print("\n生成的文件:")
    print("1. support_coordinates_linear.csv  - 线性布局支架坐标")
    print("2. support_coordinates_grid.csv    - 网格布局支架坐标")
    print("3. geological_features.csv         - 地质特征数据")
    print("4. aligned_mine_pressure.csv       - 对齐的矿压数据")
    print("\n使用步骤:")
    print("1. 上传 aligned_mine_pressure.csv 作为矿压数据")
    print("2. 上传 support_coordinates_linear.csv 作为坐标文件")
    print("3. (可选) 勾选'融合地质特征',上传 geological_features.csv")
    print("4. 系统将自动完成坐标对齐和特征融合")
