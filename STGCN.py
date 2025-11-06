import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import math
import time
import warnings
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

# 忽略 CUDA 兼容性警告
warnings.filterwarnings('ignore', category=UserWarning, message='.*CUDA capability.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*NVIDIA.*')

# 设置 CUDA 环境
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# 解决 OpenMP 冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ----------------------------------------------------------------------
# 1. 帮助函数 (Utils)
# ----------------------------------------------------------------------

def load_csv_data(csv_file, time_col=None):
    """
    从 CSV 文件加载矿压数据
    :param csv_file: CSV 文件对象
    :param time_col: 时间列名称(如果有)
    :return: numpy array (num_samples, num_nodes, num_features), column_names
    """
    df = pd.read_csv(csv_file)
    
    # 如果有时间列,删除它
    if time_col and time_col in df.columns:
        df = df.drop(columns=[time_col])
    
    # 尝试自动检测时间列(常见名称)
    time_cols = ['时间', 'time', 'Time', 'DATE', 'date', 'datetime', 'Datetime', '日期']
    for col in time_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
            break
    
    # 保存列名(支架名称)
    column_names = df.columns.tolist()
    
    # 转换为 numpy 数组
    data = df.values  # (num_samples, num_nodes)
    
    # 添加特征维度 (假设只有一个特征:压力值)
    data = np.expand_dims(data, axis=-1)  # (num_samples, num_nodes, 1)
    
    return data, column_names

def load_processed_sequence_data(npz_file):
    """
    加载预处理好的序列数据集
    :param npz_file: .npz 文件路径或文件对象
    :return: X, y_final, support_ids, feature_names
    """
    if isinstance(npz_file, str):
        data = np.load(npz_file, allow_pickle=True)
    else:
        data = np.load(npz_file, allow_pickle=True)
    
    X = data['X']  # (num_samples, seq_len, num_features)
    y_final = data['y_final']  # (num_samples, pred_len)
    support_ids = data['support_ids']  # (num_samples,)
    feature_names = data['feature_names'].tolist() if 'feature_names' in data else []
    
    return X, y_final, support_ids, feature_names

def load_coordinate_file(coord_file):
    """
    加载支架坐标文件
    :param coord_file: 坐标文件对象 (CSV 或 Excel)
    :return: DataFrame with columns [支架ID/名称, X坐标, Y坐标, (可选)Z坐标]
    """
    if coord_file.name.endswith('.csv'):
        df = pd.read_csv(coord_file)
    elif coord_file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(coord_file)
    else:
        raise ValueError("坐标文件格式不支持,请使用 CSV 或 Excel 格式")
    
    return df

def align_data_with_coordinates(column_names, coord_df):
    """
    对齐矿压数据和坐标数据
    :param column_names: 矿压数据的列名(支架名称)
    :param coord_df: 坐标数据DataFrame
    :return: 对齐后的坐标数组 (num_nodes, 2 或 3), 对齐信息
    """
    # 尝试找到坐标DataFrame中的支架ID列
    possible_id_cols = ['支架ID', '支架编号', '支架名称', 'ID', 'id', 'Name', 'name', '编号']
    id_col = None
    for col in possible_id_cols:
        if col in coord_df.columns:
            id_col = col
            break
    
    if id_col is None:
        # 如果没有找到,使用第一列作为ID列
        id_col = coord_df.columns[0]
    
    # 尝试找到X,Y坐标列
    possible_x_cols = ['X', 'x', 'X坐标', 'x坐标', 'lon', 'longitude', '经度']
    possible_y_cols = ['Y', 'y', 'Y坐标', 'y坐标', 'lat', 'latitude', '纬度']
    possible_z_cols = ['Z', 'z', 'Z坐标', 'z坐标', 'elevation', '高程']
    
    x_col = next((col for col in possible_x_cols if col in coord_df.columns), None)
    y_col = next((col for col in possible_y_cols if col in coord_df.columns), None)
    z_col = next((col for col in possible_z_cols if col in coord_df.columns), None)
    
    if x_col is None or y_col is None:
        raise ValueError("无法识别坐标列,请确保坐标文件包含 X 和 Y 列")
    
    # 创建坐标字典
    coord_dict = {}
    for _, row in coord_df.iterrows():
        support_id = str(row[id_col]).strip()
        if z_col:
            coord_dict[support_id] = [row[x_col], row[y_col], row[z_col]]
        else:
            coord_dict[support_id] = [row[x_col], row[y_col]]
    
    # 对齐坐标
    aligned_coords = []
    missing_coords = []
    
    for col_name in column_names:
        col_name_str = str(col_name).strip()
        if col_name_str in coord_dict:
            aligned_coords.append(coord_dict[col_name_str])
        else:
            # 尝试模糊匹配
            matched = False
            for key in coord_dict.keys():
                if col_name_str in key or key in col_name_str:
                    aligned_coords.append(coord_dict[key])
                    matched = True
                    break
            if not matched:
                missing_coords.append(col_name_str)
                # 使用默认坐标或跳过
                if len(aligned_coords) > 0:
                    aligned_coords.append(aligned_coords[-1])  # 使用上一个坐标
                else:
                    aligned_coords.append([0, 0] if z_col is None else [0, 0, 0])
    
    coords_array = np.array(aligned_coords)
    
    alignment_info = {
        'total_supports': len(column_names),
        'matched': len(column_names) - len(missing_coords),
        'missing': missing_coords,
        'has_z': z_col is not None
    }
    
    return coords_array, alignment_info

def load_geological_features(geo_file, coords_array, column_names=None):
    """
    加载地质特征数据并映射到支架位置
    :param geo_file: 地质特征文件 (CSV, Excel, 或路径字符串)
    :param coords_array: 支架坐标数组 (num_nodes, 2 或 3)
    :param column_names: 支架名称列表(可选,用于直接匹配钻孔名称)
    :return: 地质特征矩阵 (num_nodes, num_geo_features), 特征列名
    """
    # 支持文件对象或路径字符串
    if isinstance(geo_file, str):
        if geo_file.endswith('.csv'):
            geo_df = pd.read_csv(geo_file, encoding='utf-8-sig')
        elif geo_file.endswith(('.xls', '.xlsx')):
            geo_df = pd.read_excel(geo_file)
        else:
            raise ValueError("地质文件格式不支持")
    else:
        if geo_file.name.endswith('.csv'):
            geo_df = pd.read_csv(geo_file, encoding='utf-8-sig')
        elif geo_file.name.endswith(('.xls', '.xlsx')):
            geo_df = pd.read_excel(geo_file)
        else:
            raise ValueError("地质文件格式不支持")
    
    from scipy.spatial import KDTree
    
    # 提取地质点坐标
    x_col = next((col for col in ['x', 'X', 'X坐标', '坐标x'] if col in geo_df.columns), None)
    y_col = next((col for col in ['y', 'Y', 'Y坐标', '坐标y'] if col in geo_df.columns), None)
    
    if x_col is None or y_col is None:
        raise ValueError("地质文件必须包含 X 和 Y 坐标列")
    
    geo_coords = geo_df[[x_col, y_col]].values
    
    # 提取地质特征列(排除坐标、钻孔名等ID列)
    exclude_cols = [x_col, y_col, 'borehole', '钻孔名', 'id', 'ID', 'name']
    feature_cols = [col for col in geo_df.columns if col not in exclude_cols]
    
    # 如果没有数值特征,返回空数组
    if len(feature_cols) == 0:
        return np.zeros((len(coords_array), 1)), ['dummy_feature']
    
    geo_features = geo_df[feature_cols].values
    
    # 填充缺失值
    geo_features = np.nan_to_num(geo_features, nan=0.0)
    
    # 使用 KNN 插值映射到支架位置
    tree = KDTree(geo_coords)
    distances, indices = tree.query(coords_array[:, :2], k=1)
    
    # 为每个支架分配最近钻孔的地质特征
    support_geo_features = geo_features[indices.flatten()]
    
    return support_geo_features, feature_cols

def generate_adjacency_matrix(num_nodes, method='chain', **kwargs):
    """
    生成邻接矩阵
    :param num_nodes: 节点数量
    :param method: 生成方法
        - 'chain': 链式结构(相邻支架连接)
        - 'grid': 网格结构(如果支架排列成网格)
        - 'distance': 基于距离(需要提供坐标)
        - 'full': 全连接
        - 'knn': K近邻
    :param kwargs: 额外参数
    :return: 邻接矩阵 (num_nodes, num_nodes)
    """
    adj_mx = np.zeros((num_nodes, num_nodes))
    
    if method == 'chain':
        # 链式结构:每个节点与相邻节点连接
        for i in range(num_nodes - 1):
            adj_mx[i, i + 1] = 1
            adj_mx[i + 1, i] = 1
    
    elif method == 'grid':
        # 网格结构(需要指定行列数)
        rows = kwargs.get('rows', int(np.sqrt(num_nodes)))
        cols = num_nodes // rows
        for i in range(num_nodes):
            row, col = i // cols, i % cols
            # 连接上下左右邻居
            neighbors = []
            if row > 0: neighbors.append((row - 1) * cols + col)
            if row < rows - 1: neighbors.append((row + 1) * cols + col)
            if col > 0: neighbors.append(row * cols + col - 1)
            if col < cols - 1: neighbors.append(row * cols + col + 1)
            for j in neighbors:
                adj_mx[i, j] = 1
    
    elif method == 'distance':
        # 基于距离(需要提供坐标)
        coords = kwargs.get('coords')
        threshold = kwargs.get('threshold', 1.0)
        if coords is not None:
            distances = squareform(pdist(coords))
            adj_mx = (distances <= threshold).astype(float)
            np.fill_diagonal(adj_mx, 0)
    
    elif method == 'full':
        # 全连接
        adj_mx = np.ones((num_nodes, num_nodes))
        np.fill_diagonal(adj_mx, 0)
    
    elif method == 'knn':
        # K近邻(需要提供坐标)
        coords = kwargs.get('coords')
        k = kwargs.get('k', 3)
        if coords is not None:
            distances = squareform(pdist(coords))
            for i in range(num_nodes):
                # 找到k个最近邻居
                neighbors = np.argsort(distances[i])[1:k+1]
                adj_mx[i, neighbors] = 1
                adj_mx[neighbors, i] = 1
    
    return adj_mx

def calculate_normalized_laplacian(adj_mx):
    """
    计算标准化的拉普拉斯矩阵 A_hat = D^{-1/2} * (A + I) * D^{-1/2}
    
    :param adj_mx: 邻接矩阵 (N, N)
    :return: A_hat (N, N)
    """
    adj_mx = adj_mx + np.eye(adj_mx.shape[0])
    d = np.array(adj_mx.sum(1)).flatten()
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    normalized_laplacian = d_mat_inv_sqrt.dot(adj_mx).dot(d_mat_inv_sqrt)
    return normalized_laplacian.astype(np.float32)

def generate_dataloader(data, batch_size, seq_len=12, pre_len=1, train_ratio=0.7, val_ratio=0.1):
    """
    生成 PyTorch DataLoaders
    :param data: (num_samples, num_nodes, num_features)
    :param batch_size: 批量大小
    :param seq_len: 历史时间步
    :param pre_len: 预测时间步
    :return: train_loader, val_loader, test_loader, scaler
    """
    # 归一化 (这里使用简单的 Z-Score)
    mean = data.mean()
    std = data.std()
    scaler = {'mean': mean, 'std': std}
    data = (data - mean) / std

    x, y = [], []
    for i in range(len(data) - seq_len - pre_len + 1):
        x.append(data[i : i + seq_len])
        y.append(data[i + seq_len : i + seq_len + pre_len, :, 0:1]) # 假设只预测第一个特征

    x = np.array(x) # (N_samples, seq_len, num_nodes, num_features)
    y = np.array(y) # (N_samples, pre_len, num_nodes, 1)

    # 调整维度以匹配模型 (Batch, Features, Nodes, Time)
    x = np.transpose(x, (0, 3, 2, 1)) 
    # (Batch, Time_out, Nodes, Features_out) -> (Batch, Features_out, Nodes, Time_out)
    y = np.transpose(y, (0, 3, 2, 1))

    # 划分数据集
    num_samples = x.shape[0]
    train_end = int(num_samples * train_ratio)
    val_end = int(num_samples * (train_ratio + val_ratio))

    train_x, train_y = x[:train_end], y[:train_end]
    val_x, val_y = x[train_end:val_end], y[val_end:]
    test_x, test_y = x[val_end:], y[val_end:]

    # 创建 TensorDataset 和 DataLoader
    def create_loader(x_data, y_data):
        tensor_x = torch.tensor(x_data, dtype=torch.float32)
        tensor_y = torch.tensor(y_data, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_loader = create_loader(train_x, train_y)
    val_loader = create_loader(val_x, val_y)
    test_loader = create_loader(test_x, test_y)

    return train_loader, val_loader, test_loader, scaler

# ----------------------------------------------------------------------
# 2. STGCN 模型定义 (PyTorch)
# ----------------------------------------------------------------------

class SimpleLSTM(nn.Module):
    """
    简单的 LSTM 模型（不使用图结构，直接序列预测）
    适用于稀疏图数据
    """
    def __init__(self, num_features, hidden_dim=128, num_layers=2):
        super(SimpleLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM 层
        self.lstm = nn.LSTM(
            num_features, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, X):
        """
        X: (Batch, seq_len, num_features)
        输出: (Batch, 1) - 预测值
        """
        # LSTM
        lstm_out, _ = self.lstm(X)  # (B, T, hidden)
        
        # 取最后一个时间步
        last_hidden = lstm_out[:, -1, :]  # (B, hidden)
        
        # 全连接层
        x = self.relu(self.fc1(last_hidden))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        out = self.fc3(x)  # (B, 1)
        
        return out

class TimeBlock(nn.Module):
    """
    时序卷积块 (TCN)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, (kernel_size - 1) // 2))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, (kernel_size - 1) // 2))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, (kernel_size - 1) // 2))

    def forward(self, X):
        # 输入 X: (Batch, Channels, Nodes, Time_steps)
        # GLU (Gated Linear Unit)
        # X shape: (B, C_in, N, T)
        return (self.conv1(X) + self.conv3(X)) * torch.sigmoid(self.conv2(X))

class STGCNBlock(nn.Module):
    """
    时空卷积块 (Spatio-Temporal GCN Block)
    """
    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes, Kt):
        super(STGCNBlock, self).__init__()
        # 时序卷积 (TCN)
        self.tcn = TimeBlock(in_channels, spatial_channels, Kt)
        # 空间卷积 (GCN)
        self.gcn = nn.Conv2d(spatial_channels, out_channels, (1, 1))
        # 批量归一化 - 应该对 out_channels 进行归一化
        self.bn = nn.BatchNorm2d(out_channels)
        # 残差连接的投影层(如果维度不匹配)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, (1, 1)) if in_channels != out_channels else None
        
    def forward(self, X, A_hat):
        # X: (B, C_in, N, T)
        # A_hat: (N, N)

        # 1. TCN
        X_tcn = self.tcn(X) # (B, spatial_channels, N, T_out)
        
        # 2. GCN
        # (B, spatial_channels, N, T_out) -> (B, T_out, N, spatial_channels)
        X_gcn_input = X_tcn.permute(0, 3, 2, 1) 
        
        # (B, T_out, N, spatial_channels) x (N, N) -> (B, T_out, N, spatial_channels)
        X_gcn = torch.einsum('btni,nm->btmi', X_gcn_input, A_hat)
        
        # (B, T_out, N, spatial_channels) -> (B, spatial_channels, N, T_out)
        X_gcn = X_gcn.permute(0, 3, 2, 1)

        # 3. 1x1 卷积
        X_gcn = self.gcn(X_gcn) # (B, out_channels, N, T_out)
        
        # 4. 批归一化
        X_gcn = self.bn(X_gcn)
        
        # 5. 残差连接(需要处理时间维度和通道维度的变化)
        if X.shape[-1] > X_gcn.shape[-1]:  # 时间维度减少了
            # 截取 X 的最后几个时间步以匹配
            X_res = X[:, :, :, -(X_gcn.shape[-1]):]
        else:
            X_res = X
            
        # 如果通道数不匹配,使用 1x1 卷积投影
        if self.residual_conv is not None:
            X_res = self.residual_conv(X_res)
        
        # 6. 激活
        X_out = F.relu(X_gcn + X_res)
        
        return X_out


class STGCN(nn.Module):
    """
    STGCN 完整模型
    输入维度: (Batch, Features_in, Num_Nodes, Seq_Len)
    输出维度: (Batch, Features_out, Num_Nodes, Pred_Len)
    """
    def __init__(self, num_nodes, num_features, seq_len, pred_len, Kt=3):
        super(STGCN, self).__init__()
        self.num_nodes = num_nodes
        self.pred_len = pred_len
        
        # STGCN Block 1
        self.st_block1 = STGCNBlock(num_features, 64, 64, num_nodes, Kt)
        
        # STGCN Block 2
        self.st_block2 = STGCNBlock(64, 64, 64, num_nodes, Kt)
        
        # 最后一个时序卷积
        self.last_tcn = TimeBlock(64, 128, Kt)
        
        # 计算经过所有层后的时间维度
        # 每个 TimeBlock 使用 padding=(kernel_size-1)//2, 所以不改变时间维度
        # 但由于我们在 forward 中做了残差连接的截取,实际会减少
        # 实际上 TimeBlock 使用 same padding,时间维度应该保持不变
        # 让我们使用自适应池化来处理
        
        # 输出层:将特征映射到预测长度
        self.output_conv = nn.Conv2d(128, 128, (1, 1))
        self.temporal_conv = nn.Conv2d(128, pred_len, (1, 1))
        
    def forward(self, X, A_hat):
        # X: (B, C_in, N, T_in)
        
        # Block 1
        X = self.st_block1(X, A_hat) # (B, 64, N, T)
        
        # Block 2
        X = self.st_block2(X, A_hat) # (B, 64, N, T)
        
        # Last TCN
        X = self.last_tcn(X) # (B, 128, N, T)
        
        # Output layers
        X = F.relu(self.output_conv(X)) # (B, 128, N, T)
        
        # 使用自适应平均池化将时间维度调整为预测长度
        X = F.adaptive_avg_pool2d(X, (X.shape[2], self.pred_len)) # (B, 128, N, pred_len)
        
        # 将通道数转换为1(只预测一个特征)
        # 使用1x1卷积将128个通道压缩为1个通道
        final_conv = nn.Conv2d(128, 1, (1, 1)).to(X.device)
        X = final_conv(X) # (B, 1, N, pred_len)
        
        return X # (B, 1, N, pred_len)


# ----------------------------------------------------------------------
# 3. 训练和评估函数
# ----------------------------------------------------------------------

def train_epoch(model, train_loader, optimizer, loss_fn, device, A_hat):
    model.train()
    total_loss = 0
    A_hat_tensor = torch.tensor(A_hat, dtype=torch.float32).to(device)
    
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        
        # x_batch: (B, F_in, N, T_in)
        # y_batch: (B, F_out, N, T_out)
        y_pred = model(x_batch, A_hat_tensor) # (B, 1, N, pred_len)
        
        # 确保 y_pred 和 y_batch 维度匹配
        # y_batch: (B, F_out, N, T_out)
        # y_pred: (B, 1, N, pred_len)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def evaluate(model, val_loader, loss_fn, device, A_hat):
    model.eval()
    total_loss = 0
    A_hat_tensor = torch.tensor(A_hat, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # x_batch: (B, F_in, N, T_in)
            # y_batch: (B, F_out, N, T_out)
            y_pred = model(x_batch, A_hat_tensor) # (B, 1, N, pred_len)
            
            loss = loss_fn(y_pred, y_batch)
            total_loss += loss.item()
            
    return total_loss / len(val_loader)

# ----------------------------------------------------------------------
# 4. Streamlit 交互式 GUI 界面
# ----------------------------------------------------------------------

st.title("STGCN 矿压预测模型 - 训练界面")

# --- 侧边栏:参数设置 ---
st.sidebar.header("模型参数设置")
SEQ_LEN = st.sidebar.slider("历史时间步 (Seq_Len)", 5, 24, 12)
PRED_LEN = st.sidebar.slider("预测时间步 (Pred_Len)", 1, 12, 1)
BATCH_SIZE = st.sidebar.slider("批量大小 (Batch_Size)", 8, 128, 32)
EPOCHS = st.sidebar.slider("训练轮数 (Epochs)", 10, 200, 50)
LR = st.sidebar.number_input("学习率 (Learning_Rate)", 0.0001, 0.1, 0.001, format="%.4f")

# GPU 检测和设备选择
gpu_available = torch.cuda.is_available()
gpu_usable = False

if gpu_available:
    # 测试 GPU 是否真的可用
    try:
        test_tensor = torch.rand(10, 10).cuda()
        _ = test_tensor * 2
        torch.cuda.synchronize()
        gpu_usable = True
        del test_tensor
        torch.cuda.empty_cache()
    except Exception as e:
        gpu_usable = False
        st.sidebar.warning(f"⚠️ GPU 检测到但不可用: {str(e)[:50]}...")

if gpu_usable:
    DEVICE = st.sidebar.selectbox("设备", ["cuda", "cpu"], index=0)
    st.sidebar.success("✅ GPU 可用且正常工作")
else:
    DEVICE = "cpu"
    st.sidebar.info("ℹ️ 使用 CPU 进行训练")
    if gpu_available:
        st.sidebar.warning("""
        ⚠️ **GPU 兼容性问题**
        
        您的 RTX 5070 Ti (Blackwell 架构) 需要更新的 PyTorch 版本。
        
        当前 PyTorch 2.2.2 不支持该 GPU。
        
        建议:
        1. 使用 CPU 训练(当前选项)
        2. 等待 PyTorch 2.6+ 版本
        3. 或尝试 PyTorch nightly 版本
        """)

st.sidebar.header("数据上传")

# 数据源选择
data_source = st.sidebar.radio(
    "选择数据来源",
    ["上传CSV文件", "使用预处理数据集"],
    help="CSV: 原始矿压时间序列 | 预处理: 已提取特征的训练数据"
)

# 根据数据源显示不同的上传选项
if data_source == "使用预处理数据集":
    st.sidebar.subheader("📦 加载预处理数据")
    
    # 检查默认数据集
    default_npz_path = os.path.join(os.path.dirname(__file__), 'processed_data', 'sequence_dataset.npz')
    use_default_dataset = False
    
    if os.path.exists(default_npz_path):
        use_default_dataset = st.sidebar.checkbox(
            "使用已生成的数据集",
            value=True,
            help=f"路径: {default_npz_path}"
        )
    
    if use_default_dataset:
        npz_file = default_npz_path
        st.sidebar.success("✓ 使用预处理数据集")
        st.sidebar.info(
            """
            **数据集信息:**
            - 195,836 个训练样本
            - 125 个支架
            - 17 维特征
            - 序列长度: 5 → 1
            """
        )
    else:
        npz_file = st.sidebar.file_uploader(
            "上传预处理数据文件 (.npz)",
            type=["npz"],
            help="使用 preprocess/prepare_training_data.py 生成的数据"
        )
    
    # 加载坐标文件
    coord_file_path = os.path.join(os.path.dirname(__file__), 'processed_data', 'support_coordinates.csv')
    if os.path.exists(coord_file_path):
        coord_file = coord_file_path
    else:
        coord_file = None
    
else:  # CSV 文件上传模式
    npz_file = None
    
    # 步骤1: 上传矿压数据
    st.sidebar.subheader("步骤1: 矿压数据")
    data_file = st.sidebar.file_uploader("上传矿压数据文件 (.csv)", type=["csv"])
    st.sidebar.info(
        """
        **CSV 格式要求:**
        - 每行 = 一个时间点
        - 每列 = 一个支架(列名要与坐标文件对应)
        
        示例:
        ```
        时间, ZJ001, ZJ002, ZJ003, ...
        2023-01-01, 100.5, 98.3, 102.1, ...
        ```
        """
    )

    # 步骤2: 上传支架坐标
    st.sidebar.subheader("步骤2: 支架坐标 (重要!)")
    coord_file = st.sidebar.file_uploader(
        "上传支架坐标文件 (.csv/.xlsx)", 
        type=["csv", "xlsx", "xls"],
        help="坐标文件应包含: 支架ID, X坐标, Y坐标"
    )
    st.sidebar.info(
        """
        **坐标文件格式:**
        ```
        支架ID, X坐标, Y坐标
        ZJ001, 1000.5, 2000.3
        ZJ002, 1001.2, 2000.5
        ZJ003, 1002.0, 2001.1
        ```
        
        ⚠️ **支架ID必须与矿压数据的列名对应**
        """
    )

# 步骤3: 上传地质特征(可选) - 只在CSV模式下显示
use_geological = False  # 默认值
geo_file = None
if data_source == "上传CSV文件":
    st.sidebar.subheader("步骤3: 地质特征 (可选)")
    use_geological = st.sidebar.checkbox("融合地质特征数据", value=False)
    
if use_geological:
    # 检查是否存在默认地质特征文件
    default_geo_path = os.path.join(os.path.dirname(__file__), 'geology_features_extracted.csv')
    use_default_geo = False
    
    if os.path.exists(default_geo_path):
        use_default_geo = st.sidebar.checkbox(
            "使用提取的钻孔地质特征", 
            value=True,
            help=f"已检测到 geology_features_extracted.csv"
        )
    
    if use_default_geo:
        geo_file = default_geo_path
        st.sidebar.success("✓ 使用钻孔地质特征数据")
        st.sidebar.info(
            """
            **包含的地质特征:**
            - 总厚度 (m)
            - 煤层厚度/数量 (m/个)
            - 顶板煤层埋深 (m)
            - 平均弹性模量 (GPa)
            - 平均容重 (kN/m³)
            - 最大抗拉强度 (MPa)
            - 砂岩/泥岩占比
            """
        )
    else:
        geo_file = st.sidebar.file_uploader(
            "上传地质特征文件 (.csv/.xlsx)",
            type=["csv", "xlsx", "xls"],
            help="地质文件应包含: X坐标, Y坐标, 地质特征"
        )
        st.sidebar.info(
            """
            **地质文件格式:**
            ```
            X坐标, Y坐标, 断层距离, 煤层厚度, ...
            1000.5, 2000.3, 50.2, 3.5, ...
            1001.0, 2000.5, 48.5, 3.6, ...
            ```
            
            系统会根据距离将地质特征映射到支架位置
            """
        )

# 邻接矩阵生成方式
st.sidebar.header("邻接矩阵设置")
adj_method = st.sidebar.selectbox(
    "邻接矩阵生成方式",
    ["chain", "grid", "distance", "knn", "full", "upload"],
    format_func=lambda x: {
        "chain": "链式结构 (相邻支架连接)",
        "grid": "网格结构 (2D排列)",
        "distance": "距离阈值 (需要坐标)",
        "knn": "K近邻 (需要坐标)",
        "full": "全连接 (所有节点互连)",
        "upload": "上传自定义邻接矩阵"
    }[x]
)

# 根据选择的方法显示额外参数
adj_params = {}
if adj_method == "grid":
    adj_params['rows'] = st.sidebar.number_input("网格行数", min_value=1, value=10)
elif adj_method == "knn":
    adj_params['k'] = st.sidebar.number_input("K值(近邻数量)", min_value=1, value=3)
    st.sidebar.warning("K近邻方法需要支架坐标信息,暂时使用随机坐标")

adj_file = None
if adj_method == "upload":
    adj_file = st.sidebar.file_uploader("上传邻接矩阵文件 (.npy或.csv)", type=["npy", "csv"])

# --- 主界面 ---
if data_source == "使用预处理数据集" and npz_file:
    st.header("1. 加载预处理数据集")
    
    try:
        # 加载预处理的序列数据
        X, y_final, support_ids, feature_names = load_processed_sequence_data(npz_file)
        
        st.write(f"**数据形状:**")
        st.write(f"- 样本数量: {X.shape[0]:,}")
        st.write(f"- 序列长度: {X.shape[1]} (历史步数)")
        st.write(f"- 特征维度: {X.shape[2]}")
        st.write(f"- 标签形状: {y_final.shape}")
        
        NUM_SAMPLES = X.shape[0]
        SEQ_LEN = X.shape[1]
        NUM_FEATURES = X.shape[2]
        PRED_LEN = y_final.shape[1]
        
        # 获取支架信息
        unique_supports = np.unique(support_ids)
        NUM_NODES = len(unique_supports)
        
        st.success(f"✅ 成功加载数据集！包含 {NUM_NODES} 个支架，{NUM_SAMPLES:,} 个训练样本")
        
        # 显示特征列表
        with st.expander("📋 查看特征列表"):
            st.write(f"共 {len(feature_names)} 个特征:")
            for i, fname in enumerate(feature_names, 1):
                st.write(f"{i}. {fname}")
        
        # 加载坐标
        coords_array = None
        if coord_file:
            if isinstance(coord_file, str):
                coord_df = pd.read_csv(coord_file)
            else:
                coord_df = load_coordinate_file(coord_file)
            
            coords_array = coord_df[['x', 'y']].values
            st.write(f"**支架坐标:** {coords_array.shape}")
        else:
            # 使用默认线性坐标
            coords_array = np.column_stack([np.arange(NUM_NODES), np.zeros(NUM_NODES)])
            st.info("使用默认线性坐标")
        
        # 数据分割
        st.header("2. 数据分割")
        
        st.info("""
        💡 **数据量充足**：当前有 195,836 个样本，数据量充足适合深度学习。
        
        建议比例：训练 70% / 验证 15% / 测试 15%
        """)
        
        train_ratio = st.slider("训练集比例", 0.5, 0.9, 0.7, 0.05)
        val_ratio = st.slider("验证集比例", 0.05, 0.3, 0.15, 0.05)
        
        train_end = int(NUM_SAMPLES * train_ratio)
        val_end = int(NUM_SAMPLES * (train_ratio + val_ratio))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("训练集", f"{train_end:,}", f"{train_ratio*100:.0f}%")
        with col2:
            st.metric("验证集", f"{val_end-train_end:,}", f"{val_ratio*100:.0f}%")
        with col3:
            st.metric("测试集", f"{NUM_SAMPLES-val_end:,}", f"{(1-train_ratio-val_ratio)*100:.0f}%")
        
        # 转换为图数据格式
        # 由于预处理数据已经是 (num_samples, seq_len, num_features)
        # 我们需要重塑为 (num_samples_per_support, num_supports, seq_len, num_features)
        
        st.header("3. 图结构构建")
        
        # 生成邻接矩阵
        adj_method = st.selectbox(
            "邻接矩阵生成方式",
            ["distance", "knn", "chain", "full"],
            help="distance: 基于坐标距离 | knn: K近邻 | chain: 链式 | full: 全连接"
        )
        
        adj_params = {}
        if adj_method == "knn":
            adj_params['k'] = st.number_input("K值(近邻数量)", min_value=1, value=5, max_value=20)
        elif adj_method == "distance":
            adj_params['threshold'] = st.number_input("距离阈值(米)", min_value=0.1, value=5.0, step=0.5)
        
        adj_mx = generate_adjacency_matrix(
            NUM_NODES,
            method=adj_method,
            coords=coords_array,
            **adj_params
        )
        
        # 显示图信息
        num_edges = np.sum(adj_mx > 0)
        avg_degree = num_edges / NUM_NODES
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("节点数", NUM_NODES)
            st.metric("平均度数", f"{avg_degree:.1f}")
        with col2:
            st.metric("边数", int(num_edges))
            st.metric("图密度", f"{num_edges/(NUM_NODES*(NUM_NODES-1))*100:.2f}%")
        
        # 可视化邻接矩阵
        with st.expander("🔍 查看邻接矩阵"):
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(adj_mx, cmap='Blues', aspect='auto')
            ax.set_title("邻接矩阵")
            ax.set_xlabel("支架 ID")
            ax.set_ylabel("支架 ID")
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)
        
        # 模型训练部分
        st.header("4. 模型训练")
        
        # 模型选择
        model_type = st.radio(
            "选择模型类型",
            ["LSTM (推荐)", "STGCN (图神经网络)"],
            help="LSTM 适用于稀疏数据，STGCN 适用于密集图数据"
        )
        
        st.info(f"""
        **{'✅ LSTM 模型' if model_type.startswith('LSTM') else '⚠️ STGCN 模型'}**
        
        {'- 不使用图结构，直接序列预测' if model_type.startswith('LSTM') else '- 使用图结构进行空间-时间联合建模'}
        {'- 适合稀疏数据（当前数据每样本只有1个节点）' if model_type.startswith('LSTM') else '- 适合密集图数据（所有节点都有值）'}
        {'- 训练速度快，效果稳定' if model_type.startswith('LSTM') else '- 需要完整的图结构信息'}
        """)
        
        # 训练参数
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.number_input("训练轮数", min_value=1, value=50, max_value=500)
            batch_size = st.number_input("批次大小", min_value=16, value=64, max_value=512, step=16)
        with col2:
            learning_rate = st.number_input("学习率", min_value=0.0001, value=0.001, max_value=0.1, format="%.4f", step=0.0001)
            hidden_dim = st.number_input("隐藏层维度", min_value=16, value=64, max_value=256, step=16)
        
        # 优化建议
        with st.expander("💡 训练优化建议"):
            st.markdown("""
            **如果效果不好，可以尝试：**
            
            1. **增加训练轮数** → 改为 100-200 轮
            2. **调整学习率** → 尝试 0.0001-0.005 之间
            3. **增大批次** → 改为 128 或 256（如果显存够）
            4. **更换图结构** → 试试 KNN (K=3-10) 而不是 distance
            5. **调整距离阈值** → 如果用 distance 方法，试试 3-15 米
            
            **当前优化：**
            - ✅ 逐特征归一化（避免不同尺度特征的影响）
            - ✅ 批处理验证（避免显存溢出）
            - ✅ 动态学习率调整（可考虑添加学习率衰减）
            
            **理想指标：**
            - MAE < 10 MPa
            - RMSE < 15 MPa  
            - R² > 0.5
            """)
        
        if st.button("🚀 开始训练", type="primary"):
            try:
                st.success("开始训练STGCN模型...")
                
                # 1. 数据切分
                st.write("### 步骤1: 数据切分")
                X_train = X[:train_end]
                y_train = y_final[:train_end]
                train_support_ids = support_ids[:train_end]
                
                X_val = X[train_end:val_end]
                y_val = y_final[train_end:val_end]
                val_support_ids = support_ids[train_end:val_end]
                
                X_test = X[val_end:]
                y_test = y_final[val_end:]
                test_support_ids = support_ids[val_end:]
                
                st.write(f"✓ 训练集: {len(X_train):,} 样本")
                st.write(f"✓ 验证集: {len(X_val):,} 样本")
                st.write(f"✓ 测试集: {len(X_test):,} 样本")
                
                # 获取邻接矩阵
                A_hat = adj_mx
                
                # 2. 数据准备
                st.write("### 步骤2: 准备GPU计算")
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                st.info(f"使用设备: {device}")
                
                # 转换数据格式 - STGCN需要 (Batch, Features, Nodes, SeqLen)
                # 当前 X_train: (samples, seq_len, features)
                # 需要按 support_id 重组为图结构
                
                # 为每个样本找到对应的support索引
                unique_supports_list = np.unique(support_ids)
                support_to_idx = {sup_id: idx for idx, sup_id in enumerate(unique_supports_list)}
                num_nodes = len(unique_supports_list)
                
                st.write(f"图节点数: {num_nodes}")
                
                # 创建训练数据批次
                def prepare_batch_data(X_data, y_data, support_data, num_nodes):
                    """将序列数据转换为STGCN所需的图结构批次"""
                    batch_size = len(X_data)
                    seq_len = X_data.shape[1]
                    num_features = X_data.shape[2]
                    
                    # 初始化批次张量 (Batch, Features, Nodes, SeqLen)
                    batch_X = np.zeros((batch_size, num_features, num_nodes, seq_len))
                    batch_y = np.zeros((batch_size, 1, num_nodes, 1))
                    
                    for i in range(batch_size):
                        sup_id = support_data[i]
                        node_idx = support_to_idx[sup_id]
                        
                        # X: (seq_len, features) -> (features, 1, seq_len)
                        # 将单个支架的数据放到对应节点位置
                        batch_X[i, :, node_idx, :] = X_data[i].T
                        batch_y[i, 0, node_idx, 0] = y_data[i, 0]
                    
                    return batch_X, batch_y
                
                # 转换训练数据
                st.write("### 步骤3: 数据归一化与格式转换")
                
                # ⭐ 改进的归一化策略
                st.write("正在进行数据归一化...")
                
                # 对整个训练集计算统计量（包括特征和目标）
                # 方案：只对目标值归一化，特征保持原始尺度
                y_mean = y_train.mean()
                y_std = y_train.std()
                if y_std < 1e-6:
                    y_std = 1.0
                
                # MinMax 归一化目标值到 [0, 1]
                y_min = y_train.min()
                y_max = y_train.max()
                y_range = y_max - y_min
                if y_range < 1e-6:
                    y_range = 1.0
                
                # 使用 MinMax 而不是 Z-Score
                y_train_normalized = (y_train - y_min) / y_range
                y_val_normalized = (y_val - y_min) / y_range
                
                # 特征归一化：逐特征 MinMax
                X_train_normalized = X_train.copy()
                X_val_normalized = X_val.copy()
                
                for feat_idx in range(X_train.shape[2]):
                    feat_min = X_train[:, :, feat_idx].min()
                    feat_max = X_train[:, :, feat_idx].max()
                    feat_range = feat_max - feat_min
                    if feat_range < 1e-6:
                        feat_range = 1.0
                    
                    X_train_normalized[:, :, feat_idx] = (X_train[:, :, feat_idx] - feat_min) / feat_range
                    X_val_normalized[:, :, feat_idx] = (X_val[:, :, feat_idx] - feat_min) / feat_range
                
                st.write(f"✓ MinMax归一化完成 (y范围: {y_min:.2f} - {y_max:.2f} MPa)")
                
                # 根据模型类型准备数据
                if model_type.startswith("LSTM"):
                    # LSTM: 直接使用序列数据，不需要图结构
                    st.write("### 步骤4: 准备序列数据（LSTM模式）")
                    
                    # 数据已经是 (samples, seq_len, features) 格式，直接使用
                    train_X_tensor = torch.FloatTensor(X_train_normalized)
                    train_y_tensor = torch.FloatTensor(y_train_normalized).unsqueeze(1)  # (N, 1)
                    val_X_tensor = torch.FloatTensor(X_val_normalized)
                    val_y_tensor = torch.FloatTensor(y_val_normalized).unsqueeze(1)
                    
                    st.write(f"训练集: X {train_X_tensor.shape}, y {train_y_tensor.shape}")
                    st.write(f"验证集: X {val_X_tensor.shape}, y {val_y_tensor.shape}")
                    
                else:
                    # STGCN: 需要图结构
                    st.write("### 步骤4: 转换为图数据格式（STGCN模式）")
                
                with st.spinner("转换训练数据格式..."):
                    if model_type.startswith("STGCN"):
                        train_X_graph, train_y_graph = prepare_batch_data(
                            X_train_normalized, y_train_normalized, train_support_ids, num_nodes
                        )
                        val_X_graph, val_y_graph = prepare_batch_data(
                            X_val_normalized, y_val_normalized, val_support_ids, num_nodes
                        )
                        
                        # 转为torch张量
                        train_X_tensor = torch.FloatTensor(train_X_graph)
                        train_y_tensor = torch.FloatTensor(train_y_graph)
                        val_X_tensor = torch.FloatTensor(val_X_graph)
                        val_y_tensor = torch.FloatTensor(val_y_graph)
                        A_hat_tensor = torch.FloatTensor(A_hat).to(device)
                        
                        st.write(f"训练集: X {train_X_tensor.shape}, y {train_y_tensor.shape}")
                        st.write(f"验证集: X {val_X_tensor.shape}, y {val_y_tensor.shape}")
                
                # 初始化模型
                seq_len = X_train.shape[1]
                pred_len = 1
                num_features = X_train.shape[2]
                
                if model_type.startswith("LSTM"):
                    model = SimpleLSTM(
                        num_features=num_features,
                        hidden_dim=hidden_dim * 2,  # LSTM 用更大的隐藏层
                        num_layers=2
                    ).to(device)
                else:
                    model = STGCN(
                        num_nodes=num_nodes,
                        num_features=num_features,
                        seq_len=seq_len,
                        pred_len=pred_len,
                        Kt=3
                    ).to(device)
                
                st.write(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
                
                # 定义损失函数和优化器
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                
                # 学习率调度器 (验证损失不下降时降低学习率)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=10
                )
                
                # 早停参数
                early_stop_patience = 20
                early_stop_counter = 0
                
                # 训练循环
                progress_bar = st.progress(0)
                status_text = st.empty()
                metrics_placeholder = st.empty()
                
                # 存储损失历史
                train_losses = []
                val_losses = []
                best_val_loss = float('inf')
                
                # 创建DataLoader (训练和验证都使用批处理)
                from torch.utils.data import TensorDataset, DataLoader
                train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                
                val_dataset = TensorDataset(val_X_tensor, val_y_tensor)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                
                start_time = time.time()
                
                for epoch in range(epochs):
                    # 训练阶段
                    model.train()
                    epoch_train_loss = 0
                    batch_count = 0
                    
                    for batch_X, batch_y in train_loader:
                        # 将数据移到GPU
                        batch_X = batch_X.to(device)
                        batch_y = batch_y.to(device)
                        
                        optimizer.zero_grad()
                        
                        # 前向传播
                        if model_type.startswith("LSTM"):
                            outputs = model(batch_X)  # (B, 1)
                            loss = criterion(outputs, batch_y)
                        else:
                            outputs = model(batch_X, A_hat_tensor)  # (B, 1, N, 1)
                            loss = criterion(outputs, batch_y)
                        
                        # 反向传播
                        loss.backward()
                        optimizer.step()
                        
                        epoch_train_loss += loss.item()
                        batch_count += 1
                    
                    avg_train_loss = epoch_train_loss / batch_count
                    train_losses.append(avg_train_loss)
                    
                    # 验证阶段 (使用批处理避免显存溢出)
                    model.eval()
                    val_loss_sum = 0
                    all_preds = []
                    all_targets = []
                    val_batch_count = 0
                    
                    with torch.no_grad():
                        for val_batch_X, val_batch_y in val_loader:
                            val_batch_X = val_batch_X.to(device)
                            val_batch_y = val_batch_y.to(device)
                            
                            if model_type.startswith("LSTM"):
                                val_batch_outputs = model(val_batch_X)
                            else:
                                val_batch_outputs = model(val_batch_X, A_hat_tensor)
                            
                            # 累积损失（归一化空间）
                            batch_loss = criterion(val_batch_outputs, val_batch_y).item()
                            val_loss_sum += batch_loss * len(val_batch_X)
                            
                            # 收集预测和真实值用于后续计算
                            all_preds.append(val_batch_outputs.cpu())
                            all_targets.append(val_batch_y.cpu())
                            
                            val_batch_count += len(val_batch_X)
                            
                            # 清理显存
                            del val_batch_X, val_batch_y, val_batch_outputs
                            torch.cuda.empty_cache()
                        
                        # 合并所有批次
                        all_preds = torch.cat(all_preds, dim=0)
                        all_targets = torch.cat(all_targets, dim=0)
                        
                        # 反归一化到原始尺度 (MinMax 反变换)
                        all_preds_original = all_preds * y_range + y_min
                        all_targets_original = all_targets * y_range + y_min
                        
                        # 计算原始尺度的指标
                        mae = torch.mean(torch.abs(all_preds_original - all_targets_original)).item()
                        rmse = torch.sqrt(torch.mean((all_preds_original - all_targets_original)**2)).item()
                        
                        # R² (在原始尺度计算)
                        y_mean_original = torch.mean(all_targets_original)
                        ss_tot = torch.sum((all_targets_original - y_mean_original)**2)
                        ss_res = torch.sum((all_targets_original - all_preds_original)**2)
                        r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0
                        
                        # 计算平均损失
                        val_loss = val_loss_sum / val_batch_count
                        val_losses.append(val_loss)
                    
                    # 保存最佳模型
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), 'best_stgcn_model.pth')
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                    
                    # 学习率调度
                    scheduler.step(val_loss)
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    # 早停检查
                    if early_stop_counter >= early_stop_patience:
                        st.warning(f"⚠️ 验证损失连续 {early_stop_patience} 轮未改善，提前停止训练")
                        break
                    
                    # 更新进度
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    
                    elapsed = time.time() - start_time
                    eta = elapsed / (epoch + 1) * (epochs - epoch - 1)
                    
                    status_text.text(
                        f"Epoch {epoch+1}/{epochs} | "
                        f"训练损失: {avg_train_loss:.4f} | "
                        f"验证损失: {val_loss:.4f} | "
                        f"学习率: {current_lr:.6f} | "
                        f"已用时: {elapsed:.1f}s | ETA: {eta:.1f}s"
                    )
                    
                    # 每10个epoch更新一次指标
                    if (epoch + 1) % 10 == 0 or epoch == 0:
                        metrics_placeholder.markdown(f"""
                        ### 当前指标
                        - **训练损失**: {avg_train_loss:.6f}
                        - **验证损失**: {val_loss:.6f}
                        - **MAE**: {mae:.4f} MPa
                        - **RMSE**: {rmse:.4f} MPa
                        - **R²**: {r2:.4f}
                        """)
                
                # 训练完成
                st.success("✅ 训练完成！")
                st.balloons()
                
                # 绘制损失曲线
                st.subheader("📈 训练历史")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(train_losses, label='训练损失', alpha=0.8)
                ax.plot(val_losses, label='验证损失', alpha=0.8)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss (MSE)')
                ax.set_title('训练过程')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # 最终评估
                st.subheader("🎯 最终评估结果")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("最佳验证损失", f"{best_val_loss:.6f}")
                with col2:
                    st.metric("MAE", f"{mae:.4f} MPa")
                with col3:
                    st.metric("RMSE", f"{rmse:.4f} MPa")
                with col4:
                    st.metric("R²", f"{r2:.4f}")
                
                st.info("最佳模型已保存到: best_stgcn_model.pth")
                
                # 预测示例 (使用小批次避免显存溢出)
                st.subheader("🔮 预测示例")
                model.load_state_dict(torch.load('best_stgcn_model.pth'))
                model.eval()
                
                # 随机选择几个验证样本
                num_examples = min(5, len(val_X_tensor))
                indices = np.random.choice(len(val_X_tensor), num_examples, replace=False)
                
                with torch.no_grad():
                    example_X = val_X_tensor[indices].to(device)
                    example_y_true = val_y_tensor[indices].to(device)
                    
                    if model_type.startswith("LSTM"):
                        example_y_pred = model(example_X).unsqueeze(1)  # (B, 1)
                    else:
                        example_y_pred = model(example_X, A_hat_tensor)
                
                # 创建对比表
                comparison_data = []
                for i, idx in enumerate(indices):
                    sup_id = val_support_ids[idx]
                    
                    if model_type.startswith("LSTM"):
                        # LSTM: 直接输出标量
                        true_val_normalized = example_y_true[i, 0].cpu().item()
                        pred_val_normalized = example_y_pred[i, 0].cpu().item()
                    else:
                        # STGCN: 从图结构中提取
                        node_idx = support_to_idx[sup_id]
                        true_val_normalized = example_y_true[i, 0, node_idx, 0].cpu().item()
                        pred_val_normalized = example_y_pred[i, 0, node_idx, 0].cpu().item()
                    
                    # 反归一化到原始尺度 (MinMax 反变换)
                    true_val = true_val_normalized * y_range + y_min
                    pred_val = pred_val_normalized * y_range + y_min
                    error = abs(pred_val - true_val)
                    
                    comparison_data.append({
                        '支架编号': sup_id,
                        '真实值 (MPa)': f"{true_val:.2f}",
                        '预测值 (MPa)': f"{pred_val:.2f}",
                        '误差 (MPa)': f"{error:.2f}",
                        '误差率': f"{error/abs(true_val)*100:.1f}%" if abs(true_val) > 1e-6 else "N/A"
                    })
                
                st.table(pd.DataFrame(comparison_data))
                
                # 清理显存
                del example_X, example_y_true, example_y_pred
                torch.cuda.empty_cache()
                
            except Exception as e:
                st.error(f"训练过程出错: {e}")
                import traceback
                st.code(traceback.format_exc())
        
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        import traceback
        st.code(traceback.format_exc())

elif data_source == "上传CSV文件" and data_file:
    st.header("1. 数据加载与对齐")
    
    # 载入数据
    try:
        # 加载矿压数据
        data, column_names = load_csv_data(data_file)
        
        st.write(f"**矿压数据形状:** {data.shape}")
        st.write(f"- 时间步数: {data.shape[0]}")
        st.write(f"- 支架数量: {data.shape[1]}")
        st.write(f"- 特征数: {data.shape[2]}")
        
        NUM_SAMPLES, NUM_NODES, NUM_FEATURES = data.shape
        
        # 显示支架列表
        with st.expander("📋 查看支架列表"):
            st.write(column_names)
        
        # 坐标对齐
        coords_array = None
        if coord_file:
            st.subheader("🗺️ 坐标对齐")
            try:
                coord_df = load_coordinate_file(coord_file)
                st.write("**坐标文件预览:**")
                st.dataframe(coord_df.head(10))
                
                # 对齐坐标
                coords_array, alignment_info = align_data_with_coordinates(column_names, coord_df)
                
                # 显示对齐结果
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("总支架数", alignment_info['total_supports'])
                with col2:
                    st.metric("成功匹配", alignment_info['matched'], 
                             delta=f"{alignment_info['matched']/alignment_info['total_supports']*100:.1f}%")
                with col3:
                    st.metric("维度", "3D" if alignment_info['has_z'] else "2D")
                
                if alignment_info['missing']:
                    st.warning(f"⚠️ 以下 {len(alignment_info['missing'])} 个支架未找到坐标: {', '.join(alignment_info['missing'][:5])}" + 
                              ("..." if len(alignment_info['missing']) > 5 else ""))
                else:
                    st.success("✅ 所有支架坐标对齐成功!")
                
                # 可视化支架分布
                st.subheader("支架空间分布")
                fig_scatter_data = pd.DataFrame({
                    'X坐标': coords_array[:, 0],
                    'Y坐标': coords_array[:, 1],
                    '支架': column_names
                })
                st.scatter_chart(fig_scatter_data, x='X坐标', y='Y坐标', size=20)
                
            except Exception as e:
                st.error(f"坐标对齐失败: {e}")
                st.info("将使用自动生成的坐标")
        else:
            st.warning("⚠️ 未上传坐标文件,将使用线性排列的默认坐标")
            # 生成默认坐标(线性排列)
            coords_array = np.column_stack([np.arange(NUM_NODES), np.zeros(NUM_NODES)])
        
        # 地质特征融合
        geo_features = None
        if use_geological and geo_file and coords_array is not None:
            st.subheader("🌍 地质特征融合")
            try:
                geo_features, feature_names = load_geological_features(geo_file, coords_array)
                st.write(f"**地质特征形状:** {geo_features.shape}")
                st.write(f"**特征名称:** {feature_names}")
                
                # 显示地质特征统计
                geo_df_display = pd.DataFrame(geo_features, columns=feature_names)
                st.write("**地质特征统计:**")
                st.dataframe(geo_df_display.describe())
                
                # 将地质特征添加到数据中
                # 将地质特征扩展到所有时间步
                geo_features_expanded = np.tile(geo_features[np.newaxis, :, :], (NUM_SAMPLES, 1, 1))
                # (num_samples, num_nodes, geo_features)
                
                # 合并矿压数据和地质特征
                data = np.concatenate([data, geo_features_expanded], axis=-1)
                NUM_FEATURES = data.shape[2]
                
                st.success(f"✅ 地质特征已融合! 总特征数: {NUM_FEATURES}")
                
            except Exception as e:
                st.error(f"地质特征加载失败: {e}")
                st.info("将仅使用矿压数据进行训练")
        
        # 生成或加载邻接矩阵
        st.header("2. 图结构构建")
        if adj_method == "upload" and adj_file:
            # 上传自定义邻接矩阵
            if adj_file.name.endswith('.npy'):
                adj_mx = np.load(adj_file)
            else:  # CSV
                adj_mx = pd.read_csv(adj_file).values
            st.write(f"**邻接矩阵 (上传) 形状:** {adj_mx.shape}")
        else:
            # 自动生成邻接矩阵
            if adj_method == "knn" and coords_array is not None:
                # 使用真实坐标生成 KNN
                adj_params['coords'] = coords_array
                st.info(f"✅ 使用真实支架坐标生成 K={adj_params.get('k', 3)} 近邻图")
            elif adj_method == "distance" and coords_array is not None:
                # 使用距离阈值
                adj_params['coords'] = coords_array
                threshold = st.slider("距离阈值 (单位:米)", 1.0, 100.0, 10.0)
                adj_params['threshold'] = threshold
                st.info(f"✅ 使用真实坐标生成距离图 (阈值={threshold}米)")
            elif adj_method in ["knn", "distance"] and coords_array is None:
                st.warning("⚠️ 需要坐标文件才能使用距离相关方法,将使用随机坐标")
                adj_params['coords'] = np.random.rand(NUM_NODES, 2)
            
            adj_mx = generate_adjacency_matrix(NUM_NODES, adj_method, **adj_params)
            st.write(f"**邻接矩阵 (自动生成):** {adj_mx.shape}")
            st.info(f"使用 **{adj_method}** 方法生成邻接矩阵")
        
        # 验证邻接矩阵
        if NUM_NODES != adj_mx.shape[0] or NUM_NODES != adj_mx.shape[1]:
            st.error(f"数据与邻接矩阵的节点数不匹配! (数据: {NUM_NODES}, 邻接矩阵: {adj_mx.shape[0]})")
        else:
            st.success("数据文件和邻接矩阵加载/生成成功!")
            
            # 显示邻接矩阵的连接统计
            num_edges = np.sum(adj_mx) / 2  # 除以2因为是无向图
            st.write(f"**图结构统计:**")
            st.write(f"- 总边数: {int(num_edges)}")
            st.write(f"- 平均度数: {np.sum(adj_mx, axis=1).mean():.2f}")
            
            # 可视化邻接矩阵
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("邻接矩阵可视化")
                st.image(adj_mx, use_container_width=True, clamp=True, caption="白色=连接,黑色=无连接")
            
            with col2:
                st.subheader("节点 0 的数据预览")
                chart_data = pd.DataFrame(data[:min(500, len(data)), 0, 0], columns=['Node 0, Feature 0'])
                st.line_chart(chart_data)

            # --- 训练模块 ---
            st.header("3. 模型训练")
            if st.button("开始训练"):
                
                with st.spinner("正在初始化模型和数据..."):
                    # 1. 计算 A_hat
                    A_hat = calculate_normalized_laplacian(adj_mx)
                    
                    # 2. 生成 Dataloaders
                    train_loader, val_loader, test_loader, scaler = generate_dataloader(
                        data, BATCH_SIZE, SEQ_LEN, PRED_LEN
                    )
                    
                    # 3. 初始化模型
                    device = torch.device(DEVICE)
                    model = STGCN(NUM_NODES, NUM_FEATURES, SEQ_LEN, PRED_LEN).to(device)
                    optimizer = optim.Adam(model.parameters(), lr=LR)
                    loss_fn = nn.MSELoss()
                    
                    st.success("模型和数据初始化完成！开始训练...")
                
                # 准备实时显示
                status_placeholder = st.empty()
                loss_chart_placeholder = st.empty()
                loss_df = pd.DataFrame(columns=["Epoch", "Train Loss", "Val Loss"])

                start_time = time.time()
                
                for epoch in range(1, EPOCHS + 1):
                    # 训练
                    train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, A_hat)
                    
                    # 验证
                    val_loss = evaluate(model, val_loader, loss_fn, device, A_hat)
                    
                    # 更新状态
                    elapsed = time.time() - start_time
                    status_text = f"""
                    **Epoch: {epoch}/{EPOCHS}**
                    - 训练损失 (Train Loss): {train_loss:.6f}
                    - 验证损失 (Val Loss): {val_loss:.6f}
                    - 已用时间 (Elapsed): {elapsed:.2f}s
                    """
                    status_placeholder.markdown(status_text)
                    
                    # 更新图表
                    new_loss_row = pd.DataFrame({
                        "Epoch": [epoch],
                        "Train Loss": [train_loss],
                        "Val Loss": [val_loss]
                    })
                    loss_df = pd.concat([loss_df, new_loss_row], ignore_index=True)
                    loss_chart_placeholder.line_chart(loss_df.set_index("Epoch"))

                st.success("模型训练完成!")
                
                # --- 结果展示 ---
                st.header("4. 训练结果")
                st.subheader("最终损失曲线")
                st.line_chart(loss_df.set_index("Epoch"))
                
                st.subheader("模型在测试集上的表现 (随机抽样)")
                try:
                    # 从测试集获取一个批次
                    x_test_batch, y_test_batch = next(iter(test_loader))
                    x_test_batch, y_test_batch = x_test_batch.to(device), y_test_batch.to(device)
                    A_hat_tensor = torch.tensor(A_hat).to(device)
                    
                    model.eval()
                    with torch.no_grad():
                        y_pred = model(x_test_batch, A_hat_tensor) # (B, T_out, N, F_out)
                        y_pred = y_pred.permute(0, 3, 2, 1) # (B, F_out, N, T_out)

                    # 反归一化
                    y_pred_real = (y_pred.cpu().numpy() * scaler['std']) + scaler['mean']
                    y_test_real = (y_test_batch.cpu().numpy() * scaler['std']) + scaler['mean']
                    
                    # 选择一个样本和一个节点进行比较 (Batch 0, Node 0)
                    pred_series = y_pred_real[0, :, 0, 0] # 预测值
                    true_series = y_test_real[0, :, 0, 0] # 真实值
                    
                    if PRED_LEN == 1:
                        st.write("预测值 (Test Sample 0, Node 0):", pred_series[0])
                        st.write("真实值 (Test Sample 0, Node 0):", true_series[0])
                    else:
                        result_df = pd.DataFrame({
                            'Predicted': pred_series,
                            'True': true_series
                        }, index=[f't+{i+1}' for i in range(PRED_LEN)])
                        
                        st.dataframe(result_df)
                        st.line_chart(result_df)
                        
                except Exception as e:
                    st.error(f"在测试集上评估时出错: {e}")

    except Exception as e:
        st.error(f"加载数据时出错: {e}")
        st.error("请确保您上传了正确格式的 CSV 文件。")
        st.info("""
        **CSV 文件格式提示:**
        - 每行代表一个时间点
        - 每列代表一个支架(监测点)
        - 可以包含时间列(会自动识别并移除)
        """)
else:
    st.info("请在左侧边栏上传您的 CSV 矿压数据文件以开始。")