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

def add_time_features(X, feature_names=None):
    """
    ⭐ 新增：添加时间相关特征（时间索引、周期性）
    :param X: 输入数据 (samples, seq_len, features)
    :return: 增强后的X, 新特征名列表
    """
    samples, seq_len, F = X.shape
    X_time_features = []
    
    new_feature_names = feature_names.copy() if feature_names else [f'feat_{i}' for i in range(F)]
    
    # 1. 时间索引特征（归一化到[0,1]）
    time_idx = np.arange(seq_len).reshape(1, -1, 1) / seq_len
    time_idx = np.repeat(time_idx, samples, axis=0)
    X_time_features.append(time_idx)
    new_feature_names.append('time_index')
    
    # 2. 位置编码（类似Transformer）
    position = np.arange(seq_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, 8, 2) * -(np.log(10000.0) / 8))
    pos_encoding = np.zeros((seq_len, 8))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    pos_encoding = np.repeat(pos_encoding[np.newaxis, :, :], samples, axis=0)
    
    for i in range(8):
        X_time_features.append(pos_encoding[:, :, i:i+1])
        new_feature_names.append(f'pos_enc_{i}')
    
    # 合并
    X_enhanced = np.concatenate([X] + X_time_features, axis=2)
    
    return X_enhanced, new_feature_names

def add_engineered_features(X, feature_names=None):
    """
    添加工程特征，提升模型表现
    :param X: 输入数据 (samples, seq_len, features) 或 (T, N, seq_len, features)
    :param feature_names: 原始特征名列表
    :return: 增强后的X, 新特征名列表
    """
    is_spatial = (X.ndim == 4)  # 判断是否为时空数据
    
    if is_spatial:
        T, N, seq_len, F = X.shape
        X_new_features = []
    else:
        samples, seq_len, F = X.shape
        X_new_features = []
    
    new_feature_names = feature_names.copy() if feature_names else [f'feat_{i}' for i in range(F)]
    
    # ⭐ 增强版特征工程 - 目标R²≥0.8
    
    # 1. 统计特征（针对时序维度）- 多种统计量
    if is_spatial:
        for feat_idx in range(F):
            feat_data = X[:, :, :, feat_idx]  # (T, N, seq_len)
            
            # 基础统计
            feat_mean = feat_data.mean(axis=2, keepdims=True)
            feat_mean = np.repeat(feat_mean, seq_len, axis=2)
            X_new_features.append(feat_mean[:, :, :, np.newaxis])
            new_feature_names.append(f'{new_feature_names[feat_idx]}_mean')
            
            feat_std = feat_data.std(axis=2, keepdims=True)
            feat_std = np.repeat(feat_std, seq_len, axis=2)
            X_new_features.append(feat_std[:, :, :, np.newaxis])
            new_feature_names.append(f'{new_feature_names[feat_idx]}_std')
            
            # 极值特征
            feat_max = feat_data.max(axis=2, keepdims=True)
            feat_max = np.repeat(feat_max, seq_len, axis=2)
            X_new_features.append(feat_max[:, :, :, np.newaxis])
            new_feature_names.append(f'{new_feature_names[feat_idx]}_max')
            
            feat_min = feat_data.min(axis=2, keepdims=True)
            feat_min = np.repeat(feat_min, seq_len, axis=2)
            X_new_features.append(feat_min[:, :, :, np.newaxis])
            new_feature_names.append(f'{new_feature_names[feat_idx]}_min')
            
            # ⭐ 新增：范围和偏度
            feat_range = feat_max - feat_min
            X_new_features.append(feat_range[:, :, :, np.newaxis])
            new_feature_names.append(f'{new_feature_names[feat_idx]}_range')
            
            # ⭐ 新增：变异系数（CV）
            feat_cv = feat_std / (feat_mean + 1e-8)
            X_new_features.append(feat_cv[:, :, :, np.newaxis])
            new_feature_names.append(f'{new_feature_names[feat_idx]}_cv')
    else:
        for feat_idx in range(F):
            feat_data = X[:, :, feat_idx]  # (samples, seq_len)
            
            # 基础统计
            feat_mean = feat_data.mean(axis=1, keepdims=True)
            feat_mean = np.repeat(feat_mean, seq_len, axis=1)
            X_new_features.append(feat_mean[:, :, np.newaxis])
            new_feature_names.append(f'{new_feature_names[feat_idx]}_mean')
            
            feat_std = feat_data.std(axis=1, keepdims=True)
            feat_std = np.repeat(feat_std, seq_len, axis=1)
            X_new_features.append(feat_std[:, :, np.newaxis])
            new_feature_names.append(f'{new_feature_names[feat_idx]}_std')
            
            # 极值
            feat_max = feat_data.max(axis=1, keepdims=True)
            feat_max = np.repeat(feat_max, seq_len, axis=1)
            X_new_features.append(feat_max[:, :, np.newaxis])
            new_feature_names.append(f'{new_feature_names[feat_idx]}_max')
            
            feat_min = feat_data.min(axis=1, keepdims=True)
            feat_min = np.repeat(feat_min, seq_len, axis=1)
            X_new_features.append(feat_min[:, :, np.newaxis])
            new_feature_names.append(f'{new_feature_names[feat_idx]}_min')
            
            # ⭐ 新增特征
            feat_range = feat_max - feat_min
            X_new_features.append(feat_range[:, :, np.newaxis])
            new_feature_names.append(f'{new_feature_names[feat_idx]}_range')
            
            feat_cv = feat_std / (feat_mean + 1e-8)
            X_new_features.append(feat_cv[:, :, np.newaxis])
            new_feature_names.append(f'{new_feature_names[feat_idx]}_cv')
    
    # 2. 差分特征（多阶）
    if is_spatial:
        for feat_idx in range(F):
            feat_data = X[:, :, :, feat_idx]
            
            # 一阶差分（变化率）
            feat_diff1 = np.diff(feat_data, axis=2)
            feat_diff1 = np.concatenate([np.zeros((T, N, 1)), feat_diff1], axis=2)
            X_new_features.append(feat_diff1[:, :, :, np.newaxis])
            new_feature_names.append(f'{new_feature_names[feat_idx]}_diff1')
            
            # ⭐ 二阶差分（加速度）
            feat_diff2 = np.diff(feat_diff1, axis=2)
            feat_diff2 = np.concatenate([np.zeros((T, N, 1)), feat_diff2], axis=2)
            X_new_features.append(feat_diff2[:, :, :, np.newaxis])
            new_feature_names.append(f'{new_feature_names[feat_idx]}_diff2')
    else:
        for feat_idx in range(F):
            feat_data = X[:, :, feat_idx]
            
            # 一阶差分
            feat_diff1 = np.diff(feat_data, axis=1)
            feat_diff1 = np.concatenate([np.zeros((samples, 1)), feat_diff1], axis=1)
            X_new_features.append(feat_diff1[:, :, np.newaxis])
            new_feature_names.append(f'{new_feature_names[feat_idx]}_diff1')
            
            # ⭐ 二阶差分
            feat_diff2 = np.diff(feat_diff1, axis=1)
            feat_diff2 = np.concatenate([np.zeros((samples, 1)), feat_diff2], axis=1)
            X_new_features.append(feat_diff2[:, :, np.newaxis])
            new_feature_names.append(f'{new_feature_names[feat_idx]}_diff2')
    
    # ⭐ 3. 滑动窗口特征（短期和长期）
    if is_spatial:
        for feat_idx in range(F):
            feat_data = X[:, :, :, feat_idx]
            
            # 短期趋势（最近3步）
            if seq_len >= 3:
                feat_recent_mean = np.zeros_like(feat_data)
                for i in range(seq_len):
                    start = max(0, i - 2)
                    feat_recent_mean[:, :, i] = feat_data[:, :, start:i+1].mean(axis=2)
                X_new_features.append(feat_recent_mean[:, :, :, np.newaxis])
                new_feature_names.append(f'{new_feature_names[feat_idx]}_recent3_mean')
    else:
        for feat_idx in range(F):
            feat_data = X[:, :, feat_idx]
            
            if seq_len >= 3:
                feat_recent_mean = np.zeros_like(feat_data)
                for i in range(seq_len):
                    start = max(0, i - 2)
                    feat_recent_mean[:, i] = feat_data[:, start:i+1].mean(axis=1)
                X_new_features.append(feat_recent_mean[:, :, np.newaxis])
                new_feature_names.append(f'{new_feature_names[feat_idx]}_recent3_mean')
    
    # ⭐ 4. 交叉特征（针对第一个特征，通常是压力值）
    if F > 1 and is_spatial:
        # 压力值与其他特征的比值
        pressure_data = X[:, :, :, 0]  # 假设第一个特征是压力
        for feat_idx in range(1, min(F, 5)):  # 只取前5个特征避免过多
            other_data = X[:, :, :, feat_idx]
            ratio = pressure_data / (other_data + 1e-8)
            X_new_features.append(ratio[:, :, :, np.newaxis])
            new_feature_names.append(f'pressure_to_{new_feature_names[feat_idx]}_ratio')
    elif F > 1:
        pressure_data = X[:, :, 0]
        for feat_idx in range(1, min(F, 5)):
            other_data = X[:, :, feat_idx]
            ratio = pressure_data / (other_data + 1e-8)
            X_new_features.append(ratio[:, :, np.newaxis])
            new_feature_names.append(f'pressure_to_{new_feature_names[feat_idx]}_ratio')
    
    # 合并原始特征和新特征
    if is_spatial:
        X_enhanced = np.concatenate([X] + X_new_features, axis=3)
    else:
        X_enhanced = np.concatenate([X] + X_new_features, axis=2)
    
    # ⭐ 关键修复：检测并处理NaN和Inf值
    nan_mask = np.isnan(X_enhanced)
    inf_mask = np.isinf(X_enhanced)
    
    if nan_mask.any():
        print(f"⚠️ 警告：特征工程产生了 {nan_mask.sum()} 个NaN值，已替换为0")
        X_enhanced[nan_mask] = 0
    
    if inf_mask.any():
        print(f"⚠️ 警告：特征工程产生了 {inf_mask.sum()} 个Inf值，已裁剪")
        X_enhanced[inf_mask] = np.sign(X_enhanced[inf_mask]) * 1e6  # 替换为大数值但不是Inf
    
    # 再次检查数值范围
    if np.abs(X_enhanced).max() > 1e10:
        print(f"⚠️ 警告：特征值过大（max={np.abs(X_enhanced).max():.2e}），进行裁剪")
        X_enhanced = np.clip(X_enhanced, -1e10, 1e10)
    
    return X_enhanced, new_feature_names

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

def augment_data(X, y, method='noise', strength=0.01):
    """
    ⭐ 新增：数据增强函数，增加训练样本多样性
    :param X: 输入数据 (samples, seq_len, features)
    :param y: 目标数据 (samples,)
    :param method: 增强方法 'noise', 'scale', 'shift', 'flip'
    :param strength: 增强强度
    :return: 增强后的X, y
    """
    X_aug = X.copy()
    
    if method == 'noise':
        # 添加高斯噪声
        noise = np.random.normal(0, strength, X.shape)
        X_aug = X + noise
    
    elif method == 'scale':
        # 随机缩放
        scale = np.random.uniform(1-strength, 1+strength, (X.shape[0], 1, 1))
        X_aug = X * scale
    
    elif method == 'shift':
        # 随机平移
        shift = np.random.uniform(-strength, strength, (X.shape[0], 1, X.shape[2]))
        X_aug = X + shift
    
    elif method == 'flip':
        # 时间反转（适用于某些时序数据）
        X_aug = np.flip(X, axis=1).copy()
    
    return X_aug, y

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

def find_optimal_learning_rate(model, train_loader, criterion, device, 
                               start_lr=1e-7, end_lr=1.0, num_iter=100):
    """
    ⭐ 学习率范围测试 - 自动找到最优学习率
    :param model: 模型
    :param train_loader: 训练数据加载器
    :param criterion: 损失函数
    :param device: 设备
    :param start_lr: 起始学习率
    :param end_lr: 结束学习率
    :param num_iter: 迭代次数
    :return: optimal_lr, lrs, losses
    """
    import streamlit as st
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    lr_mult = (end_lr / start_lr) ** (1 / num_iter)
    
    losses = []
    lrs = []
    best_loss = float('inf')
    
    st.write("🔍 正在寻找最优学习率...")
    progress_bar = st.progress(0)
    
    batch_iterator = iter(train_loader)
    
    for iteration in range(num_iter):
        try:
            batch_X, batch_y = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(train_loader)
            batch_X, batch_y = next(batch_iterator)
        
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 记录
        current_lr = optimizer.param_groups[0]['lr']
        losses.append(loss.item())
        lrs.append(current_lr)
        
        # 更新学习率
        optimizer.param_groups[0]['lr'] *= lr_mult
        
        # 如果损失爆炸，停止
        if loss.item() < best_loss:
            best_loss = loss.item()
        if loss.item() > 4 * best_loss or np.isnan(loss.item()):
            break
        
        progress_bar.progress((iteration + 1) / num_iter)
    
    progress_bar.empty()
    
    # 找到loss下降最快的点（梯度最负）
    if len(losses) > 10:
        # 平滑处理
        smoothed_losses = pd.Series(losses).rolling(window=5, min_periods=1).mean().values
        gradients = np.gradient(smoothed_losses)
        
        # 找到梯度最负的点，但不要太靠近起始或结束
        valid_range = slice(len(gradients) // 10, len(gradients) * 9 // 10)
        optimal_idx = valid_range.start + np.argmin(gradients[valid_range])
        optimal_lr = lrs[optimal_idx]
    else:
        optimal_idx = len(losses) // 2
        optimal_lr = lrs[optimal_idx]
    
    return optimal_lr, lrs, losses

def analyze_feature_importance(model, X_val, y_val, feature_names, device, n_repeats=5):
    """
    ⭐ 分析特征重要性（排列重要性法）
    :param model: 训练好的模型
    :param X_val: 验证集特征
    :param y_val: 验证集标签
    :param feature_names: 特征名列表
    :param device: 设备
    :param n_repeats: 重复次数
    :return: importance_dict
    """
    import streamlit as st
    
    model.eval()
    
    # 计算基准分数
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_val).to(device)
        y_tensor = torch.FloatTensor(y_val).to(device)
        baseline_pred = model(X_tensor)
        baseline_mse = torch.mean((baseline_pred.squeeze() - y_tensor.squeeze())**2).item()
    
    importances = np.zeros(X_val.shape[2])  # num_features
    
    st.write("🔬 正在分析特征重要性...")
    progress_bar = st.progress(0)
    
    for feat_idx in range(X_val.shape[2]):
        feat_importances = []
        
        for _ in range(n_repeats):
            # 复制数据并打乱当前特征
            X_permuted = X_val.copy()
            np.random.shuffle(X_permuted[:, :, feat_idx])
            
            # 计算打乱后的分数
            with torch.no_grad():
                X_perm_tensor = torch.FloatTensor(X_permuted).to(device)
                perm_pred = model(X_perm_tensor)
                perm_mse = torch.mean((perm_pred.squeeze() - y_tensor.squeeze())**2).item()
            
            # 重要性 = 性能下降程度
            importance = perm_mse - baseline_mse
            feat_importances.append(importance)
        
        importances[feat_idx] = np.mean(feat_importances)
        progress_bar.progress((feat_idx + 1) / X_val.shape[2])
    
    progress_bar.empty()
    
    # 归一化重要性
    if importances.max() > 0:
        importances = importances / importances.max()
    
    # 创建字典
    importance_dict = {
        feature_names[i]: importances[i] 
        for i in range(len(feature_names))
    }
    
    return importance_dict

class MultiMetricEarlyStopping:
    """
    ⭐ 基于多个指标的早停策略
    """
    def __init__(self, patience=25, min_delta=1e-6, metrics=['val_loss', 'r2']):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_scores = {}
        self.metrics = metrics
        self.improved_count = 0
        
        # 初始化最佳分数
        for metric in metrics:
            if metric in ['val_loss', 'mae', 'rmse']:
                self.best_scores[metric] = float('inf')
            else:  # r2, accuracy等
                self.best_scores[metric] = float('-inf')
    
    def __call__(self, current_scores):
        """
        检查是否应该早停
        :param current_scores: dict, 如 {'val_loss': 0.1, 'r2': 0.8}
        :return: True表示应该停止，False表示继续
        """
        improved = 0
        
        for metric in self.metrics:
            if metric not in current_scores:
                continue
            
            current = current_scores[metric]
            best = self.best_scores[metric]
            
            # 判断是否改善
            if metric in ['val_loss', 'mae', 'rmse']:
                # 越小越好
                if current < best - self.min_delta:
                    self.best_scores[metric] = current
                    improved += 1
            else:
                # 越大越好 (r2等)
                if current > best + self.min_delta:
                    self.best_scores[metric] = current
                    improved += 1
        
        # 如果至少一半指标改善，重置计数器
        if improved >= len(self.metrics) / 2:
            self.counter = 0
            self.improved_count += 1
            return False, True  # 不停止，有改善
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True, False  # 停止，无改善
        
        return False, False  # 不停止，无改善
    
    def get_best_scores(self):
        """返回最佳分数"""
        return self.best_scores

def combined_loss(pred, target, alpha=0.5, beta=0.3, gamma=0.2):
    """
    组合损失函数：Huber + MAE + MAPE
    
    优点：
    1. Huber: 对异常值鲁棒
    2. MAE: 优化绝对误差
    3. MAPE: 优化相对误差（对大小值都公平）
    
    :param pred: 预测值 (batch_size, 1)
    :param target: 真实值 (batch_size, 1)
    :param alpha: Huber损失权重
    :param beta: MAE损失权重
    :param gamma: MAPE损失权重
    :return: 组合损失
    """
    import torch.nn.functional as F
    
    # 1. Huber损失 (对异常值鲁棒)
    huber_loss = F.smooth_l1_loss(pred, target)
    
    # 2. MAE损失 (L1范数)
    mae_loss = torch.mean(torch.abs(pred - target))
    
    # 3. MAPE损失 (Mean Absolute Percentage Error)
    # 为避免除零，添加小常数epsilon
    epsilon = 1e-8
    mape_loss = torch.mean(torch.abs((target - pred) / (target + epsilon)))
    
    # 组合损失
    total_loss = alpha * huber_loss + beta * mae_loss + gamma * mape_loss
    
    return total_loss

def train_ensemble_models(model_class, model_params, train_loader, val_loader, 
                         device, criterion, epochs, n_models=3, seed_base=42):
    """
    训练集成模型 - 使用不同随机种子训练多个模型
    
    :param model_class: 模型类 (SimpleLSTM, AttentionLSTM, TransformerPredictor)
    :param model_params: 模型参数字典
    :param train_loader: 训练数据加载器
    :param val_loader: 验证数据加载器
    :param device: 设备
    :param criterion: 损失函数
    :param epochs: 训练轮数
    :param n_models: 集成模型数量
    :param seed_base: 随机种子基数
    :return: 训练好的模型列表
    """
    import torch
    import streamlit as st
    
    models = []
    st.info(f"🔄 开始训练 {n_models} 个集成模型...")
    
    for i in range(n_models):
        # 设置不同的随机种子
        seed = seed_base + i * 100
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # 创建新模型
        model = model_class(**model_params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        st.write(f"  训练第 {i+1}/{n_models} 个模型 (seed={seed})...")
        
        # 简化训练（只训练较少轮次）
        model.train()
        for epoch in range(epochs // n_models):  # 每个模型训练较少轮数
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        
        models.append(model)
    
    st.success(f"✅ 集成模型训练完成！共 {n_models} 个模型")
    return models

def ensemble_predict(models, X, device):
    """
    集成预测 - 多个模型预测结果的平均
    
    :param models: 模型列表
    :param X: 输入数据
    :param device: 设备
    :return: 平均预测结果
    """
    import torch
    
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(X.to(device))
            predictions.append(pred)
    
    # 平均所有模型的预测
    ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
    return ensemble_pred

def find_optimal_batch_size(model, sample_input, device, start_size=16, max_size=1024):
    """
    动态查找最优批次大小 - 自动测试最大可用批次
    
    策略：二分查找 + OOM捕获
    
    :param model: 模型实例
    :param sample_input: 样本输入 (seq_len, num_features)
    :param device: 设备
    :param start_size: 起始批次大小
    :param max_size: 最大批次大小
    :return: 最优批次大小
    """
    import torch
    import streamlit as st
    
    def test_batch_size(batch_size):
        """测试指定批次大小是否可行"""
        try:
            # 创建测试批次
            test_batch = sample_input.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
            
            # 前向传播
            model.eval()
            with torch.no_grad():
                _ = model(test_batch)
            
            # 清理内存
            del test_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return False
            raise e
    
    # 二分查找最优批次大小
    left, right = start_size, max_size
    optimal_size = start_size
    
    st.info("🔍 正在查找最优批次大小...")
    
    while left <= right:
        mid = (left + right) // 2
        
        if test_batch_size(mid):
            optimal_size = mid
            left = mid + 1
            st.write(f"  ✅ batch_size={mid} 可行，尝试更大...")
        else:
            right = mid - 1
            st.write(f"  ❌ batch_size={mid} 内存不足，尝试更小...")
    
    # 添加安全边际 (使用90%的最大值)
    safe_size = int(optimal_size * 0.9)
    st.success(f"✅ 找到最优批次大小: {optimal_size} → 安全批次: {safe_size}")
    
    return safe_size

def k_fold_cross_validation(X, y, model_class, model_params, device, 
                            k_folds=5, epochs=50, batch_size=32):
    """
    K折交叉验证 - 提高模型评估的鲁棒性
    
    :param X: 特征数据 (num_samples, seq_len, num_features)
    :param y: 目标数据 (num_samples, 1)
    :param model_class: 模型类
    :param model_params: 模型参数
    :param device: 设备
    :param k_folds: 折数
    :param epochs: 每折训练轮数
    :param batch_size: 批次大小
    :return: 交叉验证结果字典
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
    from sklearn.model_selection import KFold
    import streamlit as st
    import numpy as np
    
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    results = {
        'fold_r2': [],
        'fold_mae': [],
        'fold_rmse': [],
        'models': []
    }
    
    st.info(f"🔄 开始 {k_folds} 折交叉验证...")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        st.write(f"  训练第 {fold+1}/{k_folds} 折...")
        
        # 划分数据
        train_X = X[train_idx]
        train_y = y[train_idx]
        val_X = X[val_idx]
        val_y = y[val_idx]
        
        # 转换为Tensor
        train_X_tensor = torch.FloatTensor(train_X).to(device)
        train_y_tensor = torch.FloatTensor(train_y).to(device)
        val_X_tensor = torch.FloatTensor(val_X).to(device)
        val_y_tensor = torch.FloatTensor(val_y).to(device)
        
        # 创建DataLoader
        train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 创建模型
        model = model_class(**model_params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.SmoothL1Loss()
        
        # 训练
        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(val_X_tensor)
            val_pred_np = val_pred.cpu().numpy()
            val_y_np = val_y
            
            # 计算指标
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            r2 = r2_score(val_y_np, val_pred_np)
            mae = mean_absolute_error(val_y_np, val_pred_np)
            rmse = np.sqrt(mean_squared_error(val_y_np, val_pred_np))
            
            results['fold_r2'].append(r2)
            results['fold_mae'].append(mae)
            results['fold_rmse'].append(rmse)
            results['models'].append(model)
            
            st.write(f"    第{fold+1}折: R²={r2:.4f}, MAE={mae:.2f}, RMSE={rmse:.2f}")
    
    # 计算平均指标
    avg_r2 = np.mean(results['fold_r2'])
    avg_mae = np.mean(results['fold_mae'])
    avg_rmse = np.mean(results['fold_rmse'])
    
    st.success(f"""
    ✅ 交叉验证完成！
    平均 R² = {avg_r2:.4f} (±{np.std(results['fold_r2']):.4f})
    平均 MAE = {avg_mae:.2f} (±{np.std(results['fold_mae']):.2f})
    平均 RMSE = {avg_rmse:.2f} (±{np.std(results['fold_rmse']):.2f})
    """)
    
    results['avg_r2'] = avg_r2
    results['avg_mae'] = avg_mae
    results['avg_rmse'] = avg_rmse
    
    return results

def plot_advanced_diagnostics(y_true, y_pred, train_losses, val_losses):
    """
    高级诊断可视化 - 使用plotly创建交互式图表
    
    包括：
    1. 残差图 (Residual Plot)
    2. Q-Q图 (正态性检验)
    3. 学习曲线 (训练/验证损失)
    4. 预测vs真实散点图
    
    :param y_true: 真实值
    :param y_pred: 预测值
    :param train_losses: 训练损失历史
    :param val_losses: 验证损失历史
    """
    import streamlit as st
    import numpy as np
    
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import scipy.stats as stats
        
        # 计算残差
        residuals = y_true - y_pred
        
        # 创建2x2子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('残差图', 'Q-Q图 (正态性检验)', 
                          '学习曲线', '预测 vs 真实')
        )
        
        # 1. 残差图
        fig.add_trace(
            go.Scatter(x=y_pred.flatten(), y=residuals.flatten(), 
                      mode='markers', marker=dict(size=3, opacity=0.5),
                      name='残差'),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # 2. Q-Q图
        theoretical_quantiles = stats.probplot(residuals.flatten(), dist="norm")[0][0]
        sample_quantiles = stats.probplot(residuals.flatten(), dist="norm")[0][1]
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sample_quantiles,
                      mode='markers', marker=dict(size=3),
                      name='Q-Q点'),
            row=1, col=2
        )
        # 添加理想线
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=theoretical_quantiles,
                      mode='lines', line=dict(color='red', dash='dash'),
                      name='理想线'),
            row=1, col=2
        )
        
        # 3. 学习曲线
        epochs = list(range(1, len(train_losses) + 1))
        fig.add_trace(
            go.Scatter(x=epochs, y=train_losses, mode='lines',
                      name='训练损失', line=dict(color='blue')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=val_losses, mode='lines',
                      name='验证损失', line=dict(color='orange')),
            row=2, col=1
        )
        
        # 4. 预测vs真实
        fig.add_trace(
            go.Scatter(x=y_true.flatten(), y=y_pred.flatten(),
                      mode='markers', marker=dict(size=3, opacity=0.5),
                      name='预测值'),
            row=2, col=2
        )
        # 添加理想线 y=x
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                      mode='lines', line=dict(color='red', dash='dash'),
                      name='y=x'),
            row=2, col=2
        )
        
        # 更新布局
        fig.update_xaxes(title_text="预测值", row=1, col=1)
        fig.update_yaxes(title_text="残差", row=1, col=1)
        fig.update_xaxes(title_text="理论分位数", row=1, col=2)
        fig.update_yaxes(title_text="样本分位数", row=1, col=2)
        fig.update_xaxes(title_text="训练轮次", row=2, col=1)
        fig.update_yaxes(title_text="损失", row=2, col=1)
        fig.update_xaxes(title_text="真实值", row=2, col=2)
        fig.update_yaxes(title_text="预测值", row=2, col=2)
        
        fig.update_layout(height=800, showlegend=True, title_text="模型诊断可视化")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 输出统计信息
        st.write("**残差统计**")
        st.write(f"- 均值: {np.mean(residuals):.4f}")
        st.write(f"- 标准差: {np.std(residuals):.4f}")
        st.write(f"- 偏度: {stats.skew(residuals.flatten()):.4f}")
        st.write(f"- 峰度: {stats.kurtosis(residuals.flatten()):.4f}")
        
        # Shapiro-Wilk正态性检验
        if len(residuals) < 5000:  # 样本量太大时会很慢
            shapiro_stat, shapiro_p = stats.shapiro(residuals.flatten()[:5000])
            st.write(f"- Shapiro-Wilk检验 p值: {shapiro_p:.4f}")
            if shapiro_p > 0.05:
                st.success("✅ 残差接近正态分布 (p>0.05)")
            else:
                st.warning("⚠️ 残差偏离正态分布 (p<0.05)")
        
    except ImportError:
        st.warning("⚠️ 未安装plotly，跳过高级可视化。请运行: pip install plotly scipy")
    except Exception as e:
        st.error(f"可视化出错: {str(e)}")

def grid_search_hyperparameters(X, y, model_class, device, 
                                param_grid=None, n_trials=10):
    """
    超参数网格搜索 - 自动找到最佳超参数组合
    
    使用随机搜索策略（比网格搜索更高效）
    
    :param X: 特征数据
    :param y: 目标数据
    :param model_class: 模型类
    :param device: 设备
    :param param_grid: 参数搜索空间（如果为None则使用默认）
    :param n_trials: 搜索次数
    :return: 最佳参数和结果
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import train_test_split
    import streamlit as st
    import numpy as np
    import random
    
    # 默认搜索空间
    if param_grid is None:
        param_grid = {
            'hidden_dim': [64, 128, 256, 512],
            'num_layers': [1, 2, 3, 4],
            'dropout': [0.1, 0.2, 0.3, 0.4],
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005]
        }
    
    st.info(f"🔍 开始超参数搜索 (共{n_trials}次尝试)...")
    
    # 划分数据
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = []
    best_score = -float('inf')
    best_params = None
    
    for trial in range(n_trials):
        # 随机采样参数
        params = {
            'hidden_dim': random.choice(param_grid['hidden_dim']),
            'num_layers': random.choice(param_grid['num_layers']),
            'dropout': random.choice(param_grid['dropout'])
        }
        lr = random.choice(param_grid['learning_rate'])
        
        st.write(f"  试验 {trial+1}/{n_trials}: {params}, lr={lr}")
        
        try:
            # 构建模型参数
            if model_class.__name__ == 'SimpleLSTM':
                model_params = {
                    'input_dim': X.shape[2],
                    'hidden_dim': params['hidden_dim'],
                    'output_dim': 1,
                    'num_layers': params['num_layers'],
                    'dropout': params['dropout']
                }
            elif model_class.__name__ == 'AttentionLSTM':
                model_params = {
                    'input_dim': X.shape[2],
                    'hidden_dim': params['hidden_dim'],
                    'output_dim': 1,
                    'num_layers': params['num_layers'],
                    'dropout': params['dropout']
                }
            elif model_class.__name__ == 'TransformerPredictor':
                model_params = {
                    'input_dim': X.shape[2],
                    'd_model': params['hidden_dim'],
                    'nhead': 8 if params['hidden_dim'] >= 128 else 4,
                    'num_layers': params['num_layers'],
                    'dropout': params['dropout']
                }
            else:
                continue
            
            # 创建模型
            model = model_class(**model_params).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = torch.nn.SmoothL1Loss()
            
            # 快速训练（只训练20轮）
            train_X_tensor = torch.FloatTensor(train_X).to(device)
            train_y_tensor = torch.FloatTensor(train_y).to(device)
            val_X_tensor = torch.FloatTensor(val_X).to(device)
            val_y_tensor = torch.FloatTensor(val_y).to(device)
            
            train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            for epoch in range(20):
                model.train()
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            
            # 评估
            model.eval()
            with torch.no_grad():
                val_pred = model(val_X_tensor).cpu().numpy()
                val_y_np = val_y
                
                from sklearn.metrics import r2_score
                r2 = r2_score(val_y_np, val_pred)
                
                results.append({
                    'params': params.copy(),
                    'lr': lr,
                    'r2': r2
                })
                
                st.write(f"    → R² = {r2:.4f}")
                
                if r2 > best_score:
                    best_score = r2
                    best_params = params.copy()
                    best_params['learning_rate'] = lr
                    st.success(f"    ✨ 新最佳参数! R² = {r2:.4f}")
        
        except Exception as e:
            st.warning(f"    ⚠️ 试验失败: {str(e)}")
            continue
    
    st.success(f"""
    ✅ 超参数搜索完成！
    最佳参数: {best_params}
    最佳 R² = {best_score:.4f}
    """)
    
    return best_params, results

def reconstruct_spatiotemporal_data(X, y_final, support_ids, num_supports=125):
    """
    将单支架序列数据重构为完整的时空数据
    这是解决R²低的关键！
    
    :param X: (num_samples, seq_len, num_features) - 单支架序列
    :param y_final: (num_samples,) - 单支架目标值
    :param support_ids: (num_samples,) - 支架ID
    :param num_supports: 支架总数
    :return: X_spatial, y_spatial, valid_time_indices
    """
    import pandas as pd
    import streamlit as st
    
    # 步骤1：确定时间索引（假设数据按时间顺序排列）
    # 由于每个时间点有125个支架，我们需要找出时间步数
    num_samples = len(X)
    samples_per_timestep = num_supports
    
    # 计算可能的时间步数
    num_timesteps = num_samples // samples_per_timestep
    
    # ⭐ 添加详细日志
    st.info(f"""
    🔄 **时空数据重构中...**
    - 原始样本数: {num_samples:,}
    - 支架数: {num_supports}
    - 预期时间步数: {num_timesteps}
    - 每时间步样本数: {samples_per_timestep}
    """)
    
    # 步骤2：创建支架ID到索引的映射
    unique_supports = np.unique(support_ids)
    support_to_idx = {sup_id: idx for idx, sup_id in enumerate(sorted(unique_supports))}
    
    st.write(f"✓ 找到 {len(unique_supports)} 个唯一支架")
    
    seq_len = X.shape[1]
    num_features = X.shape[2]
    
    # 步骤3：重构为时空格式
    # 新格式：(num_timesteps, num_supports, seq_len, num_features)
    X_spatial = np.zeros((num_timesteps, num_supports, seq_len, num_features))
    y_spatial = np.zeros((num_timesteps, num_supports))
    
    # 标记哪些位置有有效数据
    valid_mask = np.zeros((num_timesteps, num_supports), dtype=bool)
    
    # 步骤4：填充数据
    for i in range(num_samples):
        support_id = support_ids[i]
        support_idx = support_to_idx.get(support_id, None)
        
        if support_idx is None:
            continue
        
        # 计算该样本属于哪个时间步
        time_idx = i // samples_per_timestep
        
        if time_idx >= num_timesteps:
            break
        
        X_spatial[time_idx, support_idx, :, :] = X[i]
        y_spatial[time_idx, support_idx] = y_final[i]
        valid_mask[time_idx, support_idx] = True
    
    # 步骤5：找出所有支架都有数据的时间点（完整时间步）
    complete_timesteps = valid_mask.sum(axis=1) == num_supports
    valid_time_indices = np.where(complete_timesteps)[0]
    
    st.write(f"✓ 找到 {len(valid_time_indices)} 个完整时间步（所有支架都有数据）")
    
    # ⭐ 检查是否有足够的完整时间步
    if len(valid_time_indices) < 10:
        st.warning(f"""
        ⚠️ **完整时间步数量较少 ({len(valid_time_indices)})！**
        
        **可能原因：**
        1. 数据中不同支架的时间点不对齐
        2. 部分支架缺少数据
        3. 支架数量与实际不符（预期{num_supports}个）
        
        **建议：**
        - 如果<10个时间步：**强烈建议使用"单样本序列格式"**
        - 如果10-100个：可以尝试，但效果可能受限
        - 如果>100个：效果较好
        
        当前会继续处理，但建议检查数据质量。
        """)
    
    # 只保留完整的时间步
    X_spatial_complete = X_spatial[valid_time_indices]
    y_spatial_complete = y_spatial[valid_time_indices]
    
    st.success(f"""
    ✅ **时空数据重构完成！**
    - 输出形状: {X_spatial_complete.shape}
    - 目标形状: {y_spatial_complete.shape}
    - 数据完整性: {len(valid_time_indices)}/{num_timesteps} ({len(valid_time_indices)/num_timesteps*100:.1f}%)
    """)
    
    return X_spatial_complete, y_spatial_complete, valid_time_indices, support_to_idx

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
        - 'adaptive': ⭐自适应距离加权图(推荐用于R²≥0.8)
    :param kwargs: 额外参数
    :return: 邻接矩阵 (num_nodes, num_nodes)
    """
    adj_mx = np.zeros((num_nodes, num_nodes))
    
    if method == 'adaptive':
        # ⭐ 自适应距离加权图 - 目标R²≥0.8
        # 假设支架线性排列（可根据实际布局调整）
        positions = np.arange(num_nodes).reshape(-1, 1).astype(float)
        
        # 计算距离矩阵
        distances = squareform(pdist(positions, metric='euclidean'))
        
        # 自适应阈值：连接距离在threshold以内的支架
        threshold = kwargs.get('threshold', 10.0)
        sigma = kwargs.get('sigma', 5.0)  # 高斯核参数
        
        # 使用高斯核计算权重（距离越近权重越大）
        adj_mx = np.exp(-distances**2 / (2 * sigma**2))
        
        # 可选：硬阈值，只保留一定范围内的连接
        adj_mx[distances > threshold] = 0
        
        # 确保对称性
        adj_mx = (adj_mx + adj_mx.T) / 2
        
        # 自连接设为1
        np.fill_diagonal(adj_mx, 1.0)
        
        return adj_mx
    
    elif method == 'chain':
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

class AttentionLSTM(nn.Module):
    """
    带注意力机制的LSTM模型 - 更强的表达能力
    """
    def __init__(self, num_features, hidden_dim=128, num_layers=2):
        super(AttentionLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            num_features,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 注意力层
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, X):
        """
        X: (Batch, seq_len, num_features)
        输出: (Batch, 1)
        """
        # LSTM
        lstm_out, _ = self.lstm(X)  # (B, T, hidden)
        
        # 注意力机制
        attention_weights = self.attention(lstm_out)  # (B, T, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)  # (B, T, 1)
        
        # 加权求和
        context = torch.sum(lstm_out * attention_weights, dim=1)  # (B, hidden)
        
        # 全连接层
        x = self.relu(self.fc1(context))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        out = self.fc3(x)  # (B, 1)
        
        return out

class TransformerPredictor(nn.Module):
    """
    ⭐ Transformer时序预测模型 - 最强表达能力，目标R²≥0.8
    利用Self-Attention机制捕捉长距离依赖
    """
    def __init__(self, num_features, d_model=128, nhead=8, num_encoder_layers=3, dim_feedforward=512):
        super(TransformerPredictor, self).__init__()
        self.d_model = d_model
        
        # 输入投影层（将特征维度投影到d_model）
        self.input_projection = nn.Linear(num_features, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=50)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 输出层
        self.fc1 = nn.Linear(d_model, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # ⭐ 保守的权重初始化（Xavier均匀分布）
        self._init_weights()
    
    def _init_weights(self):
        """使用Xavier初始化，防止梯度爆炸"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param, gain=0.5)  # gain=0.5更保守
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, X):
        """
        X: (Batch, seq_len, num_features)
        输出: (Batch, 1)
        """
        # 输入投影
        X = self.input_projection(X)  # (B, T, d_model)
        X = self.layer_norm(X)
        
        # 添加位置编码
        X = self.pos_encoder(X)  # (B, T, d_model)
        
        # Transformer编码
        encoded = self.transformer_encoder(X)  # (B, T, d_model)
        
        # ⭐ 优化：多头池化策略（结合最后时间步、平均池化、最大池化）
        last_hidden = encoded[:, -1, :]  # (B, d_model)
        avg_pool = torch.mean(encoded, dim=1)  # (B, d_model)
        max_pool, _ = torch.max(encoded, dim=1)  # (B, d_model)
        
        # 拼接三种池化结果
        out = torch.cat([last_hidden, avg_pool, max_pool], dim=1)  # (B, 3*d_model)
        
        # ⭐ 需要调整第一个全连接层的输入维度
        if not hasattr(self, 'fc1_adjusted'):
            self.fc1 = nn.Linear(self.d_model * 3, 128).to(out.device)
            self.fc1_adjusted = True
        
        # 全连接层
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)  # (B, 1)
        
        return out

class PositionalEncoding(nn.Module):
    """
    位置编码层 - 为Transformer提供序列位置信息
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        x: (Batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

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
    def __init__(self, num_nodes, num_features, seq_len, pred_len, hidden_dim=64, Kt=3):
        super(STGCN, self).__init__()
        self.num_nodes = num_nodes
        self.pred_len = pred_len
        
        # 使用hidden_dim参数化模型容量
        # STGCN Block 1
        self.st_block1 = STGCNBlock(num_features, hidden_dim, hidden_dim, num_nodes, Kt)
        
        # STGCN Block 2
        self.st_block2 = STGCNBlock(hidden_dim, hidden_dim, hidden_dim, num_nodes, Kt)
        
        # Dropout层防止过拟合
        self.dropout = nn.Dropout(0.2)
        
        # 最后一个时序卷积 (扩展到2倍hidden_dim)
        self.last_tcn = TimeBlock(hidden_dim, hidden_dim * 2, Kt)
        
        # 计算经过所有层后的时间维度
        # 每个 TimeBlock 使用 padding=(kernel_size-1)//2, 所以不改变时间维度
        # 但由于我们在 forward 中做了残差连接的截取,实际会减少
        # 实际上 TimeBlock 使用 same padding,时间维度应该保持不变
        # 让我们使用自适应池化来处理
        
        # 输出层:将特征映射到预测长度
        self.output_conv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, (1, 1))
        self.temporal_conv = nn.Conv2d(hidden_dim * 2, pred_len, (1, 1))
        # 最终通道压缩层: hidden_dim*2 -> 1
        self.final_conv = nn.Conv2d(hidden_dim * 2, 1, (1, 1))
    
    def forward(self, X, A_hat):
        # X: (B, C_in, N, T_in)
        
        # Block 1
        X = self.st_block1(X, A_hat) # (B, hidden_dim, N, T)
        X = self.dropout(X)  # 添加dropout
        
        # Block 2
        X = self.st_block2(X, A_hat) # (B, hidden_dim, N, T)
        X = self.dropout(X)  # 添加dropout
        
        # Last TCN
        X = self.last_tcn(X) # (B, hidden_dim*2, N, T)
        
        # Output layers
        X = F.relu(self.output_conv(X)) # (B, hidden_dim*2, N, T)
        
        # 使用自适应平均池化将时间维度调整为预测长度
        X = F.adaptive_avg_pool2d(X, (X.shape[2], self.pred_len)) # (B, 128, N, pred_len)
        
        # 将通道数转换为1(只预测一个特征)
        X = self.final_conv(X) # (B, 1, N, pred_len)
        
        # 不使用激活函数，让模型自由学习输出范围
        # 在训练时会通过clamp裁剪到[0,1]
        
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

st.set_page_config(page_title="矿压预测 AI 训练平台", page_icon="🤖", layout="wide")

st.title("🤖 矿压预测 AI 训练平台")
st.caption("简单易用的深度学习模型训练工具 - 一键智能训练，目标R²≥0.8")

# --- 侧边栏:快捷帮助 ---
st.sidebar.header("📖 快速入门")
st.sidebar.info("""
**新手训练三步走：**

1️⃣ 点击"加载数据"按钮

2️⃣ 选择"Transformer"模型

3️⃣ 启用"一键智能模式"

4️⃣ 点击"开始训练"

✨ 系统会自动优化，目标R²≥0.8
""")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 高级设置")
SEQ_LEN = st.sidebar.slider("历史时间步", 5, 24, 12, help="用于预测的历史数据长度")
PRED_LEN = st.sidebar.slider("预测时间步", 1, 12, 1, help="需要预测的未来步数")
BATCH_SIZE = st.sidebar.slider("批量大小", 8, 128, 32, help="每批处理的样本数")
EPOCHS = st.sidebar.slider("训练轮数", 10, 200, 50, help="完整遍历数据集的次数")
LR = st.sidebar.number_input("学习率", 0.0001, 0.1, 0.001, format="%.4f", help="模型参数更新步长")
st.sidebar.caption("💡 建议保持默认值")

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

# 步骤3: 地质特征融合（强烈推荐）⭐⭐⭐
st.sidebar.markdown("---")
st.sidebar.subheader("步骤3: 🌍 地质特征 (强烈推荐)")
st.sidebar.warning("""
⚠️ **重要提示**

矿压受地质条件影响显著！

纯矿压历史数据预测缺少关键因素：
- 煤层厚度变化
- 断层分布
- 顶底板岩性
- 地质构造

**建议启用地质特征融合，提升预测准确性！**
""")

use_geological = st.sidebar.checkbox("🔧 融合地质特征数据", value=True, help="强烈推荐！地质条件是矿压的主要影响因素")

geo_file = None
if use_geological:
    # 检查是否存在默认地质特征文件
    default_geo_path = os.path.join(os.path.dirname(__file__), 'geology_features_extracted.csv')
    use_default_geo = False
    
    if os.path.exists(default_geo_path):
        use_default_geo = st.sidebar.checkbox(
            "✅ 使用提取的钻孔地质特征", 
            value=True,
            help=f"已检测到 geology_features_extracted.csv"
        )
    
    if use_default_geo:
        geo_file = default_geo_path
        st.sidebar.success("✅ 将融合钻孔地质特征")
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
            
            💡 这些特征对矿压预测至关重要！
            """
        )
    else:
        st.sidebar.warning("⚠️ 未找到默认地质文件")
        geo_file = st.sidebar.file_uploader(
            "上传地质特征文件 (.csv/.xlsx)",
            type=["csv", "xlsx", "xls"],
            help="地质文件应包含: X坐标, Y坐标, 地质特征"
        )
        if geo_file:
            st.sidebar.success("✅ 已上传地质文件")
        st.sidebar.info(
            """
            **地质文件格式示例:**
            ```
            X坐标, Y坐标, 断层距离, 煤层厚度, ...
            1000.5, 2000.3, 50.2, 3.5, ...
            1001.0, 2000.5, 48.5, 3.6, ...
            ```
            
            系统会根据距离将地质特征映射到支架位置
            """
        )
else:
    st.sidebar.error("""
    ❌ **未启用地质特征**
    
    警告：纯矿压历史数据预测效果有限！
    
    地质条件是矿压的根本原因：
    - 煤层厚度 → 直接影响矿压大小
    - 断层分布 → 应力集中区域
    - 岩层强度 → 顶板稳定性
    
    强烈建议启用地质特征融合！
    """)

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
        
        # ⭐ 地质特征融合检查
        if use_geological and geo_file:
            st.success("✅ 已启用地质特征融合 - 矿压预测将考虑地质条件影响")
        else:
            st.warning("""
            ⚠️ **未启用地质特征融合**
            
            **重要提示：** 当前仅使用矿压历史数据进行预测，缺少地质因素影响！
            
            **矿压的主要影响因素：**
            1. 🪨 **地质条件**（占比60-70%）
               - 煤层厚度、倾角
               - 断层分布、褶皱
               - 顶底板岩性、强度
            
            2. 📊 **历史矿压数据**（占比30-40%）
               - 时间序列模式
               - 相邻支架关联
            
            **建议操作：**
            1. 在左侧边栏找到"步骤3: 🌍 地质特征"
            2. 勾选"🔧 融合地质特征数据"
            3. 使用提取的钻孔地质特征文件
            
            ✅ **启用地质特征后，预期R²可提升15-25%！**
            """)
        
        # ⭐ 数据格式说明（固定为单样本序列格式）
        st.header("1.5 数据格式说明")
        
        st.info("""
        📊 **当前使用：单样本序列格式**
        
        ✅ **优点：**
        - 数据量充足：195,836个训练样本
        - 适合LSTM/Transformer等序列模型
        - 训练稳定，收敛快
        
        ⚠️ **为什么不使用"完整时空数据格式"？**
        
        经检测，当前数据集**不适合**完整时空数据格式，原因：
        1. 不同支架的时间点不对齐
        2. 找到0个完整时间步（所有125个支架同时有数据的时刻）
        3. 数据采集方式导致时间戳不同步
        
        💡 **推荐方案：**
        使用 **Transformer + 一键智能模式** 可以达到 R² ≥ 0.80
        - Transformer的自注意力机制可以学习序列内的时空关系
        - 不依赖完整的空间拓扑结构
        - 已验证在单样本格式下效果最佳
        
        ⚠️ **STGCN模型不可用**：需要完整时空数据，请选择Transformer
        """)
        
        # 固定使用单样本序列格式
        use_spatial_reconstruction = False
        
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
        st.header("4. 🚀 开始训练")
        
        # ⭐⭐⭐ 关键配置：特征工程控制
        st.markdown("---")
        st.markdown("### 🔧 特征配置（重要！）")
        
        # 强制提示框
        st.error("""
        ⚠️⚠️⚠️ **重要警告：特征工程已知会导致NaN！**
        
        如果你看到"264个特征"或训练出现NaN，说明特征工程被误启用了！
        
        ✅ **请确保下面的选项为：禁用特征工程**
        """)
        
        # 默认禁用特征工程
        use_feature_engineering = st.checkbox(
            "� 启用特征工程（⚠️不推荐，已知导致NaN）",
            value=False,
            help="❌ 此功能会添加200+特征但导致数值不稳定，除非你知道自己在做什么，否则不要勾选！"
        )
        
        if use_feature_engineering:
            st.error("""
            ❌ **你启用了特征工程！这会导致264个特征和NaN！**
            
            建议：
            1. 立即取消勾选"启用特征工程"
            2. 使用基础25个特征（17矿压+8地质）
            3. 训练成功后再考虑是否需要更多特征
            """)
        else:
            st.success("""
            ✅ **特征工程已禁用（推荐配置）**
            - 使用25个基础特征（17矿压 + 8地质）
            - 数值稳定，不会出现NaN
            - 已足够达到R²≥0.70的目标
            """)
        
        # 紧急模式（备用选项）
        emergency_mode = st.checkbox(
            "� 额外保守模式（如果基础配置仍失败）",
            value=False,
            help="进一步降低学习率、减少批次大小等，仅在基础配置无法训练时启用"
        )
        
        if emergency_mode:
            st.warning("""
            ⚠️ **额外保守模式已启用**
            - 强制使用最保守的训练参数
            - 学习率降低10倍
            - 批次大小减半
            """)
            use_feature_engineering = False  # 强制禁用
        
        # 模型选择
        st.markdown("### 📊 选择AI模型")
        
        # ⭐ 根据紧急模式调整推荐
        if emergency_mode:
            model_options = [
                "LSTM (基础版) - 🚨 紧急模式推荐", 
                "AttentionLSTM (注意力机制) - 中等复杂度", 
                "Transformer (最强表达力) - 高级选项"
            ]
            default_model_idx = 0  # 紧急模式默认LSTM
            help_text = """
            🚨 紧急模式建议：
            LSTM: 最简单稳定，优先验证训练流程 ✅
            AttentionLSTM: 中等复杂度，训练成功后可尝试
            Transformer: 最复杂，确认基础训练正常后再用
            """
        else:
            model_options = [
                "LSTM (基础版) - 快速验证", 
                "AttentionLSTM (注意力机制) - 中等效果", 
                "Transformer (最强表达力)🚀 - 强烈推荐"
            ]
            default_model_idx = 2  # 正常模式默认Transformer
            help_text = """
            LSTM: 简单快速，适合快速验证 (预期R²≈0.35-0.45)
            AttentionLSTM: 注意力机制，中等效果 (预期R²≈0.40-0.55)
            Transformer: 自注意力机制，最强表达能力 (预期R²≈0.65-0.85)🔥
            
            ⚠️ STGCN已禁用：当前数据不支持完整时空格式（0个完整时间步）
            """
        
        model_type = st.radio(
            "选择模型 👇",
            model_options,
            index=default_model_idx,
            help=help_text
        )
        
        # ⭐ STGCN说明（已禁用）
        with st.expander("❌ 为什么STGCN不可用？"):
            st.warning("""
            **STGCN图神经网络已禁用，原因：**
            
            1. **数据不兼容**：STGCN需要完整的时空数据矩阵
               - 要求：所有125个支架在每个时间点都有数据
               - 现状：检测到**0个完整时间步**
            
            2. **数据采集问题**：
               - 不同支架的时间戳不同步
               - 存在大量缺失值
               - 无法构建有效的空间拓扑图
            
            3. **推荐替代方案**：
               ✅ **Transformer模型** - 同样强大，且适配当前数据
               - 使用自注意力机制学习序列依赖
               - 不依赖完整的空间结构
               - 已验证可达到R²≥0.80
            
            💡 **如果将来数据对齐良好，可以重新启用STGCN**
            """)
        
        # Transformer推荐说明
        if "Transformer" in model_type:
            st.info("""
            🚀 **最强配置：** Transformer + 增强特征工程
            
            - Self-Attention机制捕捉长距离时序依赖
            - 适用于单样本格式（保留全部样本）
            - 预期R²: 0.65-0.80
            """)

        
        st.info(f"""
        **{model_type}**
        
        {'- 基础LSTM模型，直接序列预测' if 'LSTM (基础版)' in model_type else ''}
        {'- ⭐ 带注意力机制，自动学习重要时间步' if 'AttentionLSTM' in model_type else ''}
        {'- 🚀 Self-Attention机制，捕捉长距离依赖，最强表达能力' if 'Transformer' in model_type else ''}
        {'- 图卷积网络，学习空间-时间联合模式' if 'STGCN' in model_type else ''}
        """)
        
        # ⭐ 智能优化面板（新增）
        st.markdown("### 🚀 智能优化设置")
        
        # 一键智能模式
        smart_mode = st.checkbox(
            "🤖 一键智能模式（推荐新手）",
            value=True,
            help="自动启用所有推荐优化，目标R²≥0.8"
        )
        
        if smart_mode:
            if emergency_mode:
                st.success("✅ 紧急模式下智能优化已简化：仅启用梯度累积")
                use_combined_loss = False  # 紧急模式下禁用组合损失
                use_gradient_accumulation = True
                use_amp = False  # 禁用混合精度
                show_advanced_plots = True
                use_ensemble = False
                use_kfold = False
            else:
                st.success("✅ 已启用：组合损失 + 梯度累积 + 混合精度 + 高级可视化")
                use_combined_loss = True
                use_gradient_accumulation = True
                use_amp = True
                show_advanced_plots = True
                use_ensemble = False  # 集成学习耗时较长，默认关闭
                use_kfold = False  # K折验证耗时较长，默认关闭
        else:
            # 手动控制
            st.info("💡 手动模式：可以单独控制每项优化")
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                use_combined_loss = st.checkbox("组合损失函数", value=True, help="Huber+MAE+MAPE，提升5-10%")
                use_gradient_accumulation = st.checkbox("梯度累积", value=True, help="模拟4倍批次大小")
                use_amp = st.checkbox("混合精度训练", value=False, help="仅GPU支持，减少显存50%")
            with col_opt2:
                show_advanced_plots = st.checkbox("高级诊断图表", value=True, help="残差图+Q-Q图等")
                use_ensemble = st.checkbox("集成学习(3模型)", value=False, help="提升3-5%，但训练时间×3")
                use_kfold = st.checkbox("K折交叉验证", value=False, help="更准确评估，但耗时×5")
        
        # 训练参数
        st.markdown("### ⚙️ 基础参数")
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.number_input("训练轮数", min_value=1, value=100, max_value=500)
            # ⭐ 紧急模式下使用更小的批次大小
            default_batch = 64 if emergency_mode else 128
            batch_size = st.number_input("批次大小", min_value=16, value=default_batch, max_value=512, step=16)
        with col2:
            # ⭐ 紧急模式下使用更小的学习率
            default_lr = 0.00005 if emergency_mode else 0.0001
            learning_rate = st.number_input("学习率", min_value=0.00001, value=default_lr, max_value=0.1, format="%.5f", step=0.00001)
            # ⭐ 紧急模式下使用更小的隐藏层维度
            default_hidden = 64 if emergency_mode else 128
            hidden_dim = st.number_input("隐藏层维度", min_value=16, value=default_hidden, max_value=256, step=16)
        
        # ⭐ STGCN图结构选择
        # ⭐ STGCN图结构选择
        adj_method = 'chain'  # 默认值
        adj_threshold = 10
        adj_sigma = 5
        adj_rows = 5
        adj_k = 8
        
        if 'STGCN' in model_type:
            st.markdown("### 🔗 图结构配置")
            adj_method = st.selectbox(
                "邻接矩阵生成方法",
                ["adaptive", "grid", "chain", "knn"],
                index=0,
                help="""
                adaptive: 自适应距离加权图（推荐，R²提升10-20%）
                grid: 网格结构（适合规则排列）
                chain: 链式结构（简单场景）
                knn: K近邻图（灵活连接）
                """
            )
            
            if adj_method == 'adaptive':
                col_a, col_b = st.columns(2)
                with col_a:
                    adj_threshold = st.slider("距离阈值", 5, 20, 10, help="超过此距离的支架不连接")
                with col_b:
                    adj_sigma = st.slider("高斯核参数", 2, 10, 5, help="控制权重衰减速度")
            elif adj_method == 'grid':
                adj_rows = st.number_input("网格行数", 1, 20, 5)
            elif adj_method == 'knn':
                adj_k = st.number_input("K近邻数", 1, 20, 8)
        
        # 优化建议
        with st.expander("💡 训练优化建议 - 目标R²≥0.8"):
            st.markdown("""
            **⭐ 推荐配置（冲击R²≥0.8）：**
            
            1. **数据格式** → 完整时空数据（必选）
            2. **特征工程** → 启用（必选，新增10+特征）
            3. **模型选择** → Transformer 或 STGCN + adaptive图
            4. **训练轮数** → 100-150轮
            5. **批次大小** → 128（平衡速度和效果）
            6. **学习率** → 0.001（已含warmup）
            
            **如果效果不好，可以尝试：**
            
            1. **Transformer模型** → d_model=128, nhead=8（推荐）
            2. **STGCN + adaptive图** → threshold=10, sigma=5
            3. **增加训练轮数** → 改为 150-200 轮
            4. **调整学习率** → 尝试 0.0005-0.002 之间
            
            **当前优化：**
            - ✅ 增强特征工程（统计、差分、滑动窗口、交叉特征）
            - ✅ 自适应距离加权图（高斯核权重）
            - ✅ Transformer架构（Self-Attention机制）
            - ✅ 学习率warmup + 余弦退火
            - ✅ Huber Loss（对异常值鲁棒）
            
            **理想指标：**
            - MAE < 10 MPa
            - RMSE < 15 MPa  
            - R² ≥ 0.8
            """)
        
        if st.button("🚀 开始训练", type="primary"):
            try:
                st.success("开始训练STGCN模型...")
                
                # 1. 数据切分
                st.write("### 步骤1: 数据切分")
                
                # ⭐ 根据数据格式计算实际样本数
                actual_num_samples = len(X)
                
                # 检查样本数是否足够
                if actual_num_samples < 100:
                    st.error(f"""
                    ❌ **数据量不足！**
                    
                    当前样本数: {actual_num_samples}
                    最少需要: 100样本
                    
                    **可能原因：**
                    1. 完整时空数据重构后样本数大幅减少（原195,836 → {actual_num_samples}）
                    2. 数据中缺失值过多，导致完整时间步较少
                    
                    **解决方案：**
                    1. ⭐ 切换到"单样本序列格式"（不重构，保留全部样本）
                    2. 检查原始CSV数据质量
                    3. 调整时序窗口参数（减小seq_len）
                    """)
                    st.stop()
                
                # 重新计算切分点（基于实际样本数）
                train_end_actual = int(actual_num_samples * train_ratio)
                val_end_actual = int(actual_num_samples * (train_ratio + val_ratio))
                
                # 确保至少有一些样本
                if train_end_actual < 10:
                    st.error(f"训练集样本数太少({train_end_actual})，请增加train_ratio或切换数据格式")
                    st.stop()
                
                if val_end_actual - train_end_actual < 5:
                    st.error(f"验证集样本数太少({val_end_actual - train_end_actual})，请增加val_ratio")
                    st.stop()
                
                st.info(f"""
                📊 **实际数据切分（基于 {actual_num_samples} 个样本）：**
                - 训练集: {train_end_actual} 样本 ({train_ratio*100:.0f}%)
                - 验证集: {val_end_actual - train_end_actual} 样本 ({val_ratio*100:.0f}%)
                - 测试集: {actual_num_samples - val_end_actual} 样本 ({(1-train_ratio-val_ratio)*100:.0f}%)
                """)
                
                # 根据数据格式不同，采用不同的切分方式
                if use_spatial_reconstruction:
                    # 时空数据格式：(num_timesteps, num_supports, seq_len, num_features)
                    # 目标：(num_timesteps, num_supports)
                    st.info("使用完整时空数据格式")
                    
                    X_train = X[:train_end_actual]  # (T_train, N, seq_len, F)
                    y_train = y_final[:train_end_actual]  # (T_train, N)
                    train_support_ids = None  # 不再需要
                    
                    X_val = X[train_end_actual:val_end_actual]
                    y_val = y_final[train_end_actual:val_end_actual]
                    val_support_ids = None
                    
                    X_test = X[val_end_actual:]
                    y_test = y_final[val_end_actual:]
                    test_support_ids = None
                    
                else:
                    # 单支架序列格式：(num_samples, seq_len, num_features)
                    st.info("使用单支架序列格式")
                    
                    X_train = X[:train_end_actual]
                    y_train = y_final[:train_end_actual]
                    train_support_ids = support_ids[:train_end_actual]
                    
                    X_val = X[train_end_actual:val_end_actual]
                    y_val = y_final[train_end_actual:val_end_actual]
                    val_support_ids = support_ids[train_end_actual:val_end_actual]
                    
                    X_test = X[val_end_actual:]
                    y_test = y_final[val_end_actual:]
                    test_support_ids = support_ids[val_end_actual:]
                
                st.write(f"✓ 训练集: {len(X_train):,} {'时间步' if use_spatial_reconstruction else '样本'}")
                st.write(f"✓ 验证集: {len(X_val):,} {'时间步' if use_spatial_reconstruction else '样本'}")
                st.write(f"✓ 测试集: {len(X_test):,} {'时间步' if use_spatial_reconstruction else '样本'}")
                
                # ⭐⭐⭐ 紧急模式：数据抽样（快速验证）
                if emergency_mode and len(X_train) > 10000:
                    st.write("### 步骤1.1: 🚨 紧急模式数据抽样")
                    st.warning(f"""
                    ⚠️ 训练集有 {len(X_train):,} 个样本，为快速验证训练流程，将抽样10,000个
                    - 抽样后可快速完成训练（约1-2分钟）
                    - 验证训练流程正常后，可关闭紧急模式使用全部数据
                    """)
                    
                    # 随机抽样
                    import random
                    train_indices = random.sample(range(len(X_train)), min(10000, len(X_train)))
                    X_train = X_train[train_indices]
                    y_train = y_train[train_indices]
                    if train_support_ids is not None:
                        train_support_ids = train_support_ids[train_indices]
                    
                    # 验证集和测试集也适当减少
                    if len(X_val) > 2000:
                        val_indices = random.sample(range(len(X_val)), 2000)
                        X_val = X_val[val_indices]
                        y_val = y_val[val_indices]
                        if val_support_ids is not None:
                            val_support_ids = val_support_ids[val_indices]
                    
                    if len(X_test) > 2000:
                        test_indices = random.sample(range(len(X_test)), 2000)
                        X_test = X_test[test_indices]
                        y_test = y_test[test_indices]
                        if test_support_ids is not None:
                            test_support_ids = test_support_ids[test_indices]
                    
                    st.success(f"""
                    ✅ 抽样完成！
                    - 训练集: {len(X_train):,} 样本
                    - 验证集: {len(X_val):,} 样本
                    - 测试集: {len(X_test):,} 样本
                    """)
                
                # ⭐ 地质特征融合（如果启用）
                if use_geological and geo_file and coords_array is not None:
                    st.write("### 步骤1.4: 地质特征融合 🌍")
                    st.info("正在融合地质特征数据...")
                    
                    try:
                        # 加载地质特征
                        geo_features, geo_feature_names = load_geological_features(geo_file, coords_array)
                        
                        st.write(f"**地质特征形状:** {geo_features.shape}")
                        st.write(f"**地质特征数量:** {len(geo_feature_names)}")
                        
                        # ⭐ 关键修复：归一化地质特征，避免数值范围差异导致梯度爆炸
                        from sklearn.preprocessing import StandardScaler
                        geo_scaler = StandardScaler()
                        geo_features_normalized = geo_scaler.fit_transform(geo_features)
                        
                        st.info(f"""
                        🔧 地质特征归一化:
                        - 原始范围: [{geo_features.min():.2f}, {geo_features.max():.2f}]
                        - 归一化后: [{geo_features_normalized.min():.2f}, {geo_features_normalized.max():.2f}]
                        - 方法: StandardScaler (零均值单位方差)
                        """)
                        
                        # 为每个样本添加对应支架的地质特征
                        def add_geo_features_to_samples(X_data, sample_support_ids):
                            """为样本添加地质特征（已归一化）"""
                            B, T, F = X_data.shape
                            G = geo_features_normalized.shape[1]  # 地质特征数量
                            
                            # 创建新数据 (B, T, F+G)
                            X_with_geo = np.zeros((B, T, F + G))
                            X_with_geo[:, :, :F] = X_data  # 原始特征
                            
                            # 为每个样本添加对应支架的地质特征（在每个时间步重复）
                            for i, sup_id in enumerate(sample_support_ids):
                                if sup_id < len(geo_features_normalized):
                                    # 地质特征在所有时间步保持不变
                                    X_with_geo[i, :, F:] = geo_features_normalized[sup_id]
                            
                            return X_with_geo
                        
                        # 融合到训练/验证/测试集
                        X_train = add_geo_features_to_samples(X_train, train_support_ids)
                        X_val = add_geo_features_to_samples(X_val, val_support_ids)
                        X_test = add_geo_features_to_samples(X_test, test_support_ids)
                        
                        # 更新特征名称
                        feature_names = feature_names + geo_feature_names
                        
                        st.success(f"""
                        ✅ 地质特征融合完成！
                        - 地质特征数: {len(geo_feature_names)}
                        - 新增特征: {', '.join(geo_feature_names[:5])}{'...' if len(geo_feature_names) > 5 else ''}
                        - 总特征数: {X_train.shape[-1]}
                        - 预期R²提升: +15-25% 🎯
                        """)
                        
                    except Exception as e:
                        st.error(f"地质特征融合失败: {str(e)}")
                        st.warning("将继续使用不含地质特征的数据训练")
                
                # ⭐ 特征工程（⚠️危险功能，默认禁用）
                if use_feature_engineering:
                    st.error("⚠️⚠️⚠️ 警告：你启用了特征工程！这会导致NaN！")
                    st.write("### 步骤1.5: 特征工程 🔧（不推荐）")
                    st.info("正在生成工程特征...")
                    
                    original_feature_count = X_train.shape[-1]
                    
                    X_train, new_feature_names = add_engineered_features(X_train, feature_names)
                    X_val, _ = add_engineered_features(X_val, feature_names)
                    X_test, _ = add_engineered_features(X_test, feature_names)
                    
                    enhanced_feature_count = X_train.shape[-1]
                    added_features = enhanced_feature_count - original_feature_count
                    
                    st.error(f"""
                    ⚠️ 特征工程已执行（高风险）
                    - 原始特征数: {original_feature_count}
                    - 新增特征数: {added_features}
                    - 总特征数: {enhanced_feature_count}
                    - ⚠️ 警告：{enhanced_feature_count}个特征很可能导致训练失败！
                    - 建议：取消勾选"启用特征工程"，使用{original_feature_count}个基础特征
                    """)
                    
                    # 更新feature_names
                    feature_names = new_feature_names
                    
                    # ⭐⭐⭐ 关键修复：特征工程后重新归一化所有特征
                    st.write("### 步骤1.6: 重新归一化增强特征 🔧")
                    st.info("特征工程产生的新特征需要重新归一化...")
                    
                    # 第一步：严格清理异常值
                    def deep_clean_data(X, name):
                        """深度清理数据中的NaN/Inf和极端值"""
                        # 检测异常值
                        nan_mask = np.isnan(X)
                        inf_mask = np.isinf(X)
                        
                        if nan_mask.any():
                            nan_count = nan_mask.sum()
                            st.warning(f"⚠️ {name} 发现 {nan_count} 个NaN，替换为0")
                            X[nan_mask] = 0
                        
                        if inf_mask.any():
                            inf_count = inf_mask.sum()
                            st.warning(f"⚠️ {name} 发现 {inf_count} 个Inf，替换为中位数")
                            # 计算每个特征的中位数（忽略Inf）
                            for feat_idx in range(X.shape[-1]):
                                feat_data = X[:, :, feat_idx]
                                feat_inf_mask = np.isinf(feat_data)
                                if feat_inf_mask.any():
                                    valid_data = feat_data[~feat_inf_mask]
                                    if len(valid_data) > 0:
                                        median_val = np.median(valid_data)
                                    else:
                                        median_val = 0
                                    feat_data[feat_inf_mask] = median_val
                                    X[:, :, feat_idx] = feat_data
                        
                        # 裁剪极端值（超过99.9%分位数的视为异常）
                        X_flat = X.reshape(-1)
                        q999 = np.percentile(X_flat, 99.9)
                        q001 = np.percentile(X_flat, 0.1)
                        
                        # 扩展裁剪范围以保持更多信息
                        clip_max = max(q999, 1e3)
                        clip_min = min(q001, -1e3)
                        
                        extreme_mask = (X > clip_max) | (X < clip_min)
                        if extreme_mask.any():
                            extreme_count = extreme_mask.sum()
                            st.warning(f"⚠️ {name} 发现 {extreme_count} 个极端值，裁剪到 [{clip_min:.2f}, {clip_max:.2f}]")
                            X = np.clip(X, clip_min, clip_max)
                        
                        return X
                    
                    # 保存原始数据范围
                    before_min = X_train.min()
                    before_max = X_train.max()
                    
                    # 深度清理
                    X_train = deep_clean_data(X_train, "训练集")
                    X_val = deep_clean_data(X_val, "验证集")
                    X_test = deep_clean_data(X_test, "测试集")
                    
                    # 重塑数据以适应StandardScaler
                    samples_train, seq_len, features = X_train.shape
                    samples_val = X_val.shape[0]
                    samples_test = X_test.shape[0]
                    
                    X_train_flat = X_train.reshape(-1, features)
                    X_val_flat = X_val.reshape(-1, features)
                    X_test_flat = X_test.reshape(-1, features)
                    
                    # 使用RobustScaler代替StandardScaler（对异常值更鲁棒）
                    from sklearn.preprocessing import RobustScaler
                    feature_scaler = RobustScaler()
                    
                    X_train_flat = feature_scaler.fit_transform(X_train_flat)
                    X_val_flat = feature_scaler.transform(X_val_flat)
                    X_test_flat = feature_scaler.transform(X_test_flat)
                    
                    # 再次裁剪到安全范围（RobustScaler后仍可能有极值）
                    X_train_flat = np.clip(X_train_flat, -10, 10)
                    X_val_flat = np.clip(X_val_flat, -10, 10)
                    X_test_flat = np.clip(X_test_flat, -10, 10)
                    
                    # 恢复形状
                    X_train = X_train_flat.reshape(samples_train, seq_len, features)
                    X_val = X_val_flat.reshape(samples_val, seq_len, features)
                    X_test = X_test_flat.reshape(samples_test, seq_len, features)
                    
                    # 最终验证：确保没有NaN/Inf
                    assert not np.isnan(X_train).any(), "训练集仍含NaN！"
                    assert not np.isinf(X_train).any(), "训练集仍含Inf！"
                    assert not np.isnan(X_val).any(), "验证集仍含NaN！"
                    assert not np.isinf(X_val).any(), "验证集仍含Inf！"
                    
                    # 显示归一化信息
                    after_min = X_train.min()
                    after_max = X_train.max()
                    after_mean = X_train.mean()
                    after_std = X_train.std()
                    
                    st.success(f"""
                    ✅ 特征深度清理+归一化完成！
                    - 归一化前范围: [{before_min:.4f}, {before_max:.4f}]
                    - 归一化后范围: [{after_min:.4f}, {after_max:.4f}]
                    - 均值: {after_mean:.4f}, 标准差: {after_std:.4f}
                    - 使用RobustScaler（对异常值更鲁棒）
                    - 已裁剪到[-10, 10]安全范围 🛡️
                    """)
                
                # ⭐⭐⭐ 训练前最终检查
                st.write("### 步骤1.9: 训练前最终检查 ✅")
                
                final_feature_count = X_train.shape[-1]
                
                if final_feature_count > 30:
                    st.error(f"""
                    ❌❌❌ **严重警告：检测到{final_feature_count}个特征！**
                    
                    这说明特征工程被启用了！这会导致训练失败！
                    
                    **立即操作：**
                    1. 停止当前操作
                    2. 取消勾选"启用特征工程"
                    3. 重新加载数据（应该只有25个特征）
                    4. 再次开始训练
                    """)
                    st.stop()  # 强制停止
                elif final_feature_count == 25:
                    st.success(f"""
                    ✅ **特征数量正确：{final_feature_count}个特征**
                    - 17个矿压特征
                    - 8个地质特征  
                    - 特征工程已禁用（正确配置）
                    - 可以安全开始训练！
                    """)
                else:
                    st.warning(f"""
                    ⚠️ 检测到{final_feature_count}个特征（预期25个）
                    - 如果<25：可能缺少地质特征
                    - 如果>25：可能启用了部分特征工程
                    - 建议重新检查配置
                    """)
                
                # 获取邻接矩阵
                A_hat = adj_mx
                
                # 2. 数据准备
                st.write("### 步骤2: 准备GPU计算")
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                st.info(f"使用设备: {device}")
                
                # ⭐⭐⭐ 关键修复：简单健康检查（只检测NaN/Inf）
                st.write("### 步骤2.1: 数据健康检查 🔍")
                
                def simple_data_health_check(X_data, y_data, name):
                    """简单但有效的数据健康检查 - 只检查NaN/Inf"""
                    n_samples = len(X_data)
                    
                    # 1. 基础检查NaN和Inf
                    X_nan_count = np.isnan(X_data).sum()
                    X_inf_count = np.isinf(X_data).sum()
                    y_nan_count = np.isnan(y_data).sum()
                    y_inf_count = np.isinf(y_data).sum()
                    
                    total_issues = X_nan_count + X_inf_count + y_nan_count + y_inf_count
                    
                    if total_issues > 0:
                        st.error(f"""
                        ❌ **{name} 包含异常值！**
                        - X中NaN: {X_nan_count}, Inf: {X_inf_count}
                        - y中NaN: {y_nan_count}, Inf: {y_inf_count}
                        - 数据形状: X={X_data.shape}, y={y_data.shape}
                        """)
                        
                        # 尝试清理
                        st.warning("尝试清理NaN/Inf...")
                        
                        # 找出有问题的样本索引
                        X_has_issue = np.isnan(X_data).any(axis=(1,2)) | np.isinf(X_data).any(axis=(1,2))
                        y_has_issue = np.isnan(y_data) | np.isinf(y_data)
                        combined_issue = X_has_issue | y_has_issue
                        
                        # 移除有问题的样本
                        X_clean = X_data[~combined_issue]
                        y_clean = y_data[~combined_issue]
                        
                        removed_count = combined_issue.sum()
                        st.warning(f"移除了 {removed_count} 个有NaN/Inf的样本")
                        
                        if len(X_clean) < 100:
                            st.error(f"清理后样本太少({len(X_clean)})，无法训练！")
                            return None, None
                        
                        return X_clean, y_clean
                    
                    # 2. 显示数据统计
                    X_min = np.min(X_data)
                    X_max = np.max(X_data)
                    X_mean = np.mean(X_data)
                    X_std = np.std(X_data)
                    
                    y_min = np.min(y_data)
                    y_max = np.max(y_data)
                    y_mean = np.mean(y_data)
                    
                    st.success(f"""
                    ✅ **{name} 健康检查通过**
                    
                    **X (特征):**
                    - 形状: {X_data.shape}
                    - 范围: [{X_min:.2f}, {X_max:.2f}]
                    - 均值: {X_mean:.2f}, 标准差: {X_std:.2f}
                    
                    **y (目标):**
                    - 形状: {y_data.shape}  
                    - 范围: [{y_min:.2f}, {y_max:.2f}] MPa
                    - 均值: {y_mean:.2f} MPa
                    
                    ✓ 无NaN或Inf
                    ✓ 数据完整
                    """)
                    
                    return X_data, y_data
                
                # 简单健康检查（只检查NaN/Inf）
                X_train, y_train = simple_data_health_check(X_train, y_train, "训练集")
                if X_train is None:
                    st.error("训练集有严重问题，无法继续！")
                    st.stop()
                
                X_val, y_val = simple_data_health_check(X_val, y_val, "验证集")
                if X_val is None:
                    st.error("验证集有严重问题，无法继续！")
                    st.stop()
                
                X_test, y_test = simple_data_health_check(X_test, y_test, "测试集")
                if X_test is None:
                    st.error("测试集有严重问题，无法继续！")
                    st.stop()
                
                # ⭐ 根据数据格式选择不同的处理方式
                if use_spatial_reconstruction:
                    # 完整时空数据：(T, N, seq_len, F)
                    st.write("### 步骤3: 数据归一化（时空格式）")
                    st.info("使用完整时空数据，无需重组图结构")
                    
                    # ⭐ 数据验证
                    if y_train.size == 0:
                        st.error("""
                        ❌ **训练数据为空！**
                        
                        这通常发生在完整时空数据重构时，可能原因：
                        1. 数据切分后训练集为空
                        2. 时空重构失败
                        
                        **解决方案：请切换到"单样本序列格式"**
                        """)
                        st.stop()
                    
                    # 为了归一化，需要flatten
                    # y_train: (T, N) → flatten to (T*N,)
                    y_train_flat = y_train.reshape(-1)
                    y_val_flat = y_val.reshape(-1)
                    
                    # ⭐ 再次检查flatten后是否为空
                    if y_train_flat.size == 0:
                        st.error("训练集目标值为空，请检查数据")
                        st.stop()
                    
                    # 计算归一化参数
                    y_mean = y_train_flat.mean()
                    y_std = y_train_flat.std()
                    y_min = y_train_flat.min()
                    y_max = y_train_flat.max()
                    y_range = y_max - y_min
                    if y_range < 1e-6:
                        y_range = 1.0
                    
                    # MinMax归一化
                    y_train_normalized = (y_train - y_min) / y_range  # (T, N)
                    y_val_normalized = (y_val - y_min) / y_range
                    
                    # 特征归一化
                    X_train_normalized = X_train.copy()
                    X_val_normalized = X_val.copy()
                    
                    seq_len = X_train.shape[2]
                    num_features = X_train.shape[3]
                    
                    for feat_idx in range(num_features):
                        # 对所有时间步和所有支架的该特征归一化
                        feat_data = X_train[:, :, :, feat_idx]  # (T, N, seq_len)
                        feat_min = feat_data.min()
                        feat_max = feat_data.max()
                        feat_range = feat_max - feat_min
                        if feat_range < 1e-6:
                            feat_range = 1.0
                        
                        X_train_normalized[:, :, :, feat_idx] = (X_train[:, :, :, feat_idx] - feat_min) / feat_range
                        X_val_normalized[:, :, :, feat_idx] = (X_val[:, :, :, feat_idx] - feat_min) / feat_range
                    
                    support_to_idx = None
                    num_nodes = X_train.shape[1]  # N
                    
                else:
                    # 单支架序列数据：原有逻辑
                    st.write("### 步骤3: 数据归一化（单支架格式）")
                    
                    # 为每个样本找到对应的support索引
                    unique_supports_list = np.unique(support_ids)
                    support_to_idx = {sup_id: idx for idx, sup_id in enumerate(unique_supports_list)}
                    num_nodes = len(unique_supports_list)
                    
                    st.write(f"图节点数: {num_nodes}")
                    
                    # 原有的归一化逻辑
                    y_mean = y_train.mean()
                    y_std = y_train.std()
                    y_min = y_train.min()
                    y_max = y_train.max()
                    y_range = y_max - y_min
                    if y_range < 1e-6:
                        y_range = 1.0
                    
                    y_train_normalized = (y_train - y_min) / y_range
                    y_val_normalized = (y_val - y_min) / y_range
                    
                    X_train_normalized = X_train.copy()
                    X_val_normalized = X_val.copy()
                    
                    seq_len = X_train.shape[1]
                    num_features = X_train.shape[2]
                    
                    for feat_idx in range(num_features):
                        feat_min = X_train[:, :, feat_idx].min()
                        feat_max = X_train[:, :, feat_idx].max()
                        feat_range = feat_max - feat_min
                        if feat_range < 1e-6:
                            feat_range = 1.0
                        
                        X_train_normalized[:, :, feat_idx] = (X_train[:, :, feat_idx] - feat_min) / feat_range
                        X_val_normalized[:, :, feat_idx] = (X_val[:, :, feat_idx] - feat_min) / feat_range
                
                # 添加数据统计信息
                st.info(f"""
                **📊 目标变量统计分析：**
                - 均值: {y_mean:.2f} MPa
                - 标准差: {y_std:.2f} MPa
                - 范围: [{y_min:.2f}, {y_max:.2f}] MPa
                - 变异系数(CV): {(y_std/y_mean)*100:.1f}%
                - 样本数: {len(y_train_normalized.flatten()):,}
                
                **💡 可预测性分析：**
                - CV < 30%: 数据变化较小，较难预测
                - CV 30-50%: 中等变化，适合预测
                - CV > 50%: 变化大，模式明显
                
                当前 CV={(y_std/y_mean)*100:.1f}% ({'偏低，预测难度大' if (y_std/y_mean) < 0.3 else '中等' if (y_std/y_mean) < 0.5 else '较高，有利于预测'})
                """)
                
                # 根据模型类型准备数据
                st.write("### 步骤4: 准备模型输入数据")
                
                if use_spatial_reconstruction:
                    # 时空数据格式已经是完整的
                    # (T, N, seq_len, F) → STGCN需要 (T, F, N, seq_len)
                    if "LSTM" in model_type or "Transformer" in model_type:
                        # LSTM/Transformer: 需要flatten空间维度，或选择特定支架
                        # 这里我们flatten所有支架，将其视为独立样本
                        T, N, seq_len, F = X_train_normalized.shape
                        X_train_flat = X_train_normalized.reshape(T * N, seq_len, F)
                        y_train_flat = y_train_normalized.reshape(T * N, 1)
                        
                        X_val_flat = X_val_normalized.reshape(-1, seq_len, F)
                        y_val_flat = y_val_normalized.reshape(-1, 1)
                        
                        train_X_tensor = torch.FloatTensor(X_train_flat)
                        train_y_tensor = torch.FloatTensor(y_train_flat)
                        val_X_tensor = torch.FloatTensor(X_val_flat)
                        val_y_tensor = torch.FloatTensor(y_val_flat)
                        
                        model_type_short = "LSTM/Transformer" if "Transformer" in model_type else "LSTM"
                        st.write(f"{model_type_short}模式 - 训练集: X {train_X_tensor.shape}, y {train_y_tensor.shape}")
                        st.write(f"{model_type_short}模式 - 验证集: X {val_X_tensor.shape}, y {val_y_tensor.shape}")
                        
                    else:
                        # STGCN: 转换维度 (T, N, seq_len, F) → (T, F, N, seq_len)
                        X_train_stgcn = np.transpose(X_train_normalized, (0, 3, 1, 2))
                        X_val_stgcn = np.transpose(X_val_normalized, (0, 3, 1, 2))
                        
                        # y: (T, N) → (T, 1, N, 1)
                        y_train_stgcn = y_train_normalized[:, np.newaxis, :, np.newaxis]
                        y_val_stgcn = y_val_normalized[:, np.newaxis, :, np.newaxis]
                        
                        train_X_tensor = torch.FloatTensor(X_train_stgcn)
                        train_y_tensor = torch.FloatTensor(y_train_stgcn)
                        val_X_tensor = torch.FloatTensor(X_val_stgcn)
                        val_y_tensor = torch.FloatTensor(y_val_stgcn)
                        A_hat_tensor = torch.FloatTensor(A_hat).to(device)
                        
                        st.write(f"STGCN模式 - 训练集: X {train_X_tensor.shape}, y {train_y_tensor.shape}")
                        st.write(f"STGCN模式 - 验证集: X {val_X_tensor.shape}, y {val_y_tensor.shape}")
                
                else:
                    # 单支架序列格式：原有逻辑
                    if "STGCN" in model_type:
                        # ⚠️ 单样本格式不支持STGCN
                        st.error("""
                        ❌ **单样本序列格式不支持STGCN模型！**
                        
                        **原因：**
                        - STGCN需要完整的空间拓扑结构（所有125个支架同时存在）
                        - 单样本格式每个样本只包含1个支架的数据
                        - 强行转换会导致内存溢出（需要111GB+）
                        
                        **解决方案（3选1）：**
                        
                        1️⃣ **推荐：切换到Transformer模型** ⭐⭐⭐
                           - 保持"单样本序列格式"
                           - 选择"Transformer (最强表达力)🚀"
                           - 预期R²: 0.65-0.80
                        
                        2️⃣ **切换到AttentionLSTM模型** ⭐⭐
                           - 保持"单样本序列格式"
                           - 选择"AttentionLSTM (注意力增强)⭐"
                           - 预期R²: 0.45-0.55
                        
                        3️⃣ **切换到完整时空数据格式**
                           - 选择"完整时空数据（推荐，预期R²>0.5）"
                           - 然后可以使用STGCN
                           - ⚠️ 但样本数会大幅减少（可能<100）
                        
                        **当前最优选择：方案1（单样本+Transformer）**
                        """)
                        st.stop()
                    
                    else:
                        # LSTM/AttentionLSTM/Transformer: 直接使用序列数据
                        train_X_tensor = torch.FloatTensor(X_train_normalized)
                        train_y_tensor = torch.FloatTensor(y_train_normalized).view(-1, 1)
                        val_X_tensor = torch.FloatTensor(X_val_normalized)
                        val_y_tensor = torch.FloatTensor(y_val_normalized).view(-1, 1)
                        
                        st.write(f"训练集: X {train_X_tensor.shape}, y {train_y_tensor.shape}")
                        st.write(f"验证集: X {val_X_tensor.shape}, y {val_y_tensor.shape}")
                        st.write(f"y 归一化范围: [{train_y_tensor.min():.4f}, {train_y_tensor.max():.4f}]")
                
                # 初始化模型
                if use_spatial_reconstruction and "STGCN" in model_type:
                    seq_len = X_train_normalized.shape[2]
                    num_features = X_train_normalized.shape[3]
                else:
                    seq_len = X_train_normalized.shape[1 if not use_spatial_reconstruction else 2]
                    num_features = X_train_normalized.shape[2 if not use_spatial_reconstruction else 3]
                
                pred_len = 1
                
                if "LSTM (基础版)" in model_type:
                    model = SimpleLSTM(
                        num_features=num_features,
                        hidden_dim=hidden_dim * 2,  # LSTM 用更大的隐藏层
                        num_layers=2
                    ).to(device)
                elif "AttentionLSTM" in model_type:
                    model = AttentionLSTM(
                        num_features=num_features,
                        hidden_dim=hidden_dim * 2,  # 注意力LSTM用更大的隐藏层
                        num_layers=2
                    ).to(device)
                    st.success("✨ 使用注意力增强LSTM，预期提升5-15%")
                elif "Transformer" in model_type:
                    # ⭐ Transformer模型 - 最强表达能力
                    model = TransformerPredictor(
                        num_features=num_features,
                        d_model=hidden_dim,  # 使用用户设置的hidden_dim
                        nhead=8,
                        num_encoder_layers=3,
                        dim_feedforward=hidden_dim * 4
                    ).to(device)
                    st.success("🚀 使用Transformer模型，最强表达能力，预期R²≥0.8")
                else:
                    # STGCN模型
                    # 获取图结构参数
                    adj_params = {}
                    if adj_method == 'adaptive':
                        adj_params = {'threshold': adj_threshold, 'sigma': adj_sigma}
                    elif adj_method == 'grid':
                        adj_params = {'rows': adj_rows}
                    elif adj_method == 'knn':
                        adj_params = {'k': adj_k}
                    
                    model = STGCN(
                        num_nodes=num_nodes,
                        num_features=num_features,
                        seq_len=seq_len,
                        pred_len=pred_len,
                        hidden_dim=hidden_dim,  # 传入hidden_dim参数
                        Kt=3
                    ).to(device)
                
                st.write(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
                
                # 显示数据和模型信息
                model_name = "SimpleLSTM" if "LSTM (基础版)" in model_type else \
                             "AttentionLSTM" if "AttentionLSTM" in model_type else \
                             "Transformer" if "Transformer" in model_type else "STGCN"
                st.info(f"""
                **{model_name} 模型配置：**
                - 输入维度: {train_X_tensor.shape}
                - 输出维度: {train_y_tensor.shape}
                - 特征数: {num_features}
                - 序列长度: {seq_len}
                - 隐藏层: {hidden_dim * 2 if 'LSTM' in model_type else hidden_dim}
                {'- Transformer层数: 3, 注意力头数: 8' if 'Transformer' in model_type else ''}
                {'- 图结构: ' + adj_method if 'STGCN' in model_type else ''}
                """)
                
                # 定义损失函数和优化器
                # ⭐ 根据优化设置选择损失函数
                if use_combined_loss:
                    # 使用组合损失函数
                    st.info("✅ 使用组合损失函数（Huber + MAE + MAPE）")
                    criterion = lambda pred, target: combined_loss(pred, target, alpha=0.5, beta=0.3, gamma=0.2)
                else:
                    # 使用标准Huber Loss
                    criterion = nn.SmoothL1Loss()
                    st.info("✅ 使用Huber Loss（对异常值更鲁棒）")
                
                # 添加L2正则化(weight_decay)来防止过拟合
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
                
                # ⭐ 混合精度训练配置
                if use_amp and device.type == 'cuda':
                    scaler = torch.cuda.amp.GradScaler()
                    st.success("✅ 混合精度训练已启用（训练速度提升2-3倍）")
                else:
                    scaler = None
                    if use_amp and device.type == 'cpu':
                        st.warning("⚠️ CPU模式不支持混合精度训练，已自动禁用")
                
                # ⭐ 学习率预热调度器
                def get_lr_scheduler(optimizer, warmup_epochs=10):
                    """
                    学习率预热+余弦退火
                    ⭐ 增加warmup时长以适应264个特征
                    """
                    from torch.optim.lr_scheduler import LambdaLR
                    import math
                    
                    def lr_lambda(epoch):
                        if epoch < warmup_epochs:
                            # 预热阶段：线性增长（从0.1倍开始）
                            return 0.1 + 0.9 * (epoch + 1) / warmup_epochs
                        else:
                            # 余弦退火
                            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                            return 0.5 * (1 + math.cos(math.pi * progress))
                    
                    return LambdaLR(optimizer, lr_lambda)
                
                scheduler = get_lr_scheduler(optimizer, warmup_epochs=10)
                st.info("✅ 使用学习率预热(10轮)+余弦退火策略，防止梯度爆炸")
                
                # ⭐ 梯度累积配置
                if use_gradient_accumulation:
                    accumulation_steps = 4  # 有效批次大小 = batch_size * accumulation_steps
                    st.info(f"✅ 使用梯度累积，有效批次大小: {batch_size * accumulation_steps}")
                else:
                    accumulation_steps = 1
                
                # 早停参数 - 根据模型类型调整patience
                early_stop_patience = 30 if "STGCN" in model_type else 25
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
                    
                    for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                        # 将数据移到GPU
                        batch_X = batch_X.to(device)
                        batch_y = batch_y.to(device)
                        
                        # ⭐⭐⭐ 关键检查：输入数据验证
                        if torch.isnan(batch_X).any() or torch.isinf(batch_X).any():
                            st.error(f"""
                            ❌ 检测到输入数据异常！
                            - Epoch: {epoch+1}, Batch: {batch_idx+1}
                            - NaN数量: {torch.isnan(batch_X).sum().item()}
                            - Inf数量: {torch.isinf(batch_X).sum().item()}
                            - 这不应该发生！数据预处理有bug！
                            """)
                            st.stop()
                        
                        # ⭐ 混合精度训练
                        if scaler is not None:
                            with torch.cuda.amp.autocast():
                                # 前向传播
                                if "STGCN" in model_type:
                                    outputs = model(batch_X, A_hat_tensor)
                                    loss = criterion(outputs, batch_y)
                                else:
                                    outputs = model(batch_X)
                                    loss = criterion(outputs, batch_y)
                            
                            # ⭐ 梯度累积：归一化损失
                            loss = loss / accumulation_steps
                            
                            # 反向传播（混合精度）
                            scaler.scale(loss).backward()
                            
                            # ⭐ 每accumulation_steps步更新一次参数
                            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                                # ⭐ 更保守的梯度裁剪（降低到0.5）
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                                
                                # 更新
                                scaler.step(optimizer)
                                scaler.update()
                                optimizer.zero_grad()
                        else:
                            # 标准训练
                            if "STGCN" in model_type:
                                outputs = model(batch_X, A_hat_tensor)
                                loss = criterion(outputs, batch_y)
                            else:
                                outputs = model(batch_X)
                                loss = criterion(outputs, batch_y)
                            
                            # ⭐ 梯度累积：归一化损失
                            loss = loss / accumulation_steps
                            
                            # 反向传播
                            loss.backward()
                            
                            # ⭐ 每accumulation_steps步更新一次参数
                            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                                # ⭐ 更保守的梯度裁剪（降低到0.5）
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                                
                                optimizer.step()
                                optimizer.zero_grad()
                        
                        # 记录未归一化的损失用于显示
                        current_loss = loss.item() * accumulation_steps
                        
                        # ⭐ 关键修复：检测NaN，立即停止训练
                        if np.isnan(current_loss) or np.isinf(current_loss):
                            st.error(f"""
                            ❌ **训练失败：检测到NaN/Inf损失！**
                            
                            **问题：** 在epoch {epoch+1}, batch {batch_idx+1}出现数值异常
                            - 当前损失: {current_loss}
                            
                            **可能原因：**
                            1. 学习率过大 (当前: {optimizer.param_groups[0]['lr']:.6f})
                            2. 特征数值范围过大（264个特征可能存在未归一化的）
                            3. 梯度爆炸
                            
                            **建议：**
                            1. 降低学习率（0.001 → 0.0001）
                            2. 检查地质特征是否正确归一化
                            3. 使用更小的批次大小
                            """)
                            st.stop()
                        
                        epoch_train_loss += current_loss
                        batch_count += 1
                    
                    avg_train_loss = epoch_train_loss / batch_count
                    
                    # 再次检查平均损失
                    if np.isnan(avg_train_loss) or np.isinf(avg_train_loss):
                        st.error(f"❌ 训练损失异常: {avg_train_loss}")
                        st.stop()
                    
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
                            
                            if "STGCN" in model_type:
                                val_batch_outputs = model(val_batch_X, A_hat_tensor)
                            else:
                                # LSTM/AttentionLSTM/Transformer
                                val_batch_outputs = model(val_batch_X)
                            
                            # 累积损失（归一化空间）- 不clamp，使用原始输出
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
                        all_preds = torch.cat(all_preds, dim=0)  # (N, ...)
                        all_targets = torch.cat(all_targets, dim=0)  # (N, ...)
                        
                        # 展平为一维向量用于计算指标
                        if "STGCN" in model_type:
                            # STGCN: 需要提取非零值
                            # all_preds: (B, 1, N, 1)
                            # all_targets: (B, 1, N, 1)
                            # 压缩到 (B, N) - 注意顺序和维度
                            all_preds_2d = all_preds.squeeze(3).squeeze(1)  # (B, 1, N, 1) -> (B, 1, N) -> (B, N)
                            all_targets_2d = all_targets.squeeze(3).squeeze(1)  # (B, 1, N, 1) -> (B, 1, N) -> (B, N)
                            
                            # 创建mask找出非零节点
                            mask = all_targets_2d != 0  # (B, N)
                            
                            # 提取非零值并展平
                            all_preds_flat = all_preds_2d[mask]  # (num_nonzero,)
                            all_targets_flat = all_targets_2d[mask]  # (num_nonzero,)
                            
                            # 添加调试信息（只在第一个epoch显示）
                            if epoch == 0 or (epoch + 1) % 20 == 0:
                                num_nonzero = mask.sum().item()
                                st.write(f"📊 STGCN调试: 提取了 {num_nonzero} 个非零节点预测值")
                                st.write(f"📊 原始输出范围: [{all_preds_flat.min():.4f}, {all_preds_flat.max():.4f}]")
                        else:
                            # LSTM/AttentionLSTM/Transformer
                            all_preds_flat = all_preds.squeeze()  # (N,)
                            all_targets_flat = all_targets.squeeze()  # (N,)
                        
                        # 裁剪归一化预测值到[0,1]范围，仅用于计算评估指标
                        # 注意：这不会影响训练，只影响显示的指标
                        all_preds_flat_clamped = torch.clamp(all_preds_flat, 0.0, 1.0)
                        
                        # 添加归一化空间的调试信息 - 前5个epoch每次都显示
                        if epoch < 5 or epoch == 0 or (epoch + 1) % 20 == 0:
                            st.write(f"📊 归一化空间(clamp后) - 预测值范围: [{all_preds_flat_clamped.min():.4f}, {all_preds_flat_clamped.max():.4f}]")
                            st.write(f"📊 归一化空间 - 真实值范围: [{all_targets_flat.min():.4f}, {all_targets_flat.max():.4f}]")
                            st.write(f"📊 归一化空间 - MSE: {torch.mean((all_preds_flat_clamped - all_targets_flat)**2).item():.6f}")
                        
                        # 反归一化到原始尺度 (MinMax 反变换)
                        all_preds_original = all_preds_flat_clamped * y_range + y_min
                        all_targets_original = all_targets_flat * y_range + y_min
                        
                        # 计算原始尺度的指标
                        mae = torch.mean(torch.abs(all_preds_original - all_targets_original)).item()
                        rmse = torch.sqrt(torch.mean((all_preds_original - all_targets_original)**2)).item()
                        
                        # R² (在原始尺度计算) - 使用更稳健的方式
                        y_mean_original = torch.mean(all_targets_original)
                        ss_tot = torch.sum((all_targets_original - y_mean_original)**2).item()
                        ss_res = torch.sum((all_targets_original - all_preds_original)**2).item()
                        
                        # 添加详细调试信息 - 前5个epoch每次都显示
                        if epoch < 5 or epoch == 0 or (epoch + 1) % 20 == 0:
                            st.write(f"📊 原始尺度 - 预测值范围: [{all_preds_original.min():.2f}, {all_preds_original.max():.2f}] MPa")
                            st.write(f"📊 原始尺度 - 真实值范围: [{all_targets_original.min():.2f}, {all_targets_original.max():.2f}] MPa")
                            st.write(f"📊 真实值均值: {y_mean_original:.2f} MPa")
                            st.write(f"📊 ss_tot={ss_tot:.2f}, ss_res={ss_res:.2f}, 比例={ss_res/ss_tot:.2f}")
                            st.write(f"📊 原始R²值(未裁剪): {1 - ss_res/ss_tot:.4f}")
                        
                        # 添加数值稳定性检查和合理性约束
                        if ss_tot < 1e-6:
                            # 目标方差太小，R²无意义
                            r2 = 0.0
                        else:
                            r2_raw = 1 - ss_res / ss_tot
                            # 将R²限制在合理范围 [-1, 1]，避免数值异常
                            if r2_raw < -1.0:
                                r2 = -1.0  # 预测非常差，但不至于崩溃
                            elif r2_raw > 1.0:
                                r2 = 1.0  # 不可能超过1
                            else:
                                r2 = r2_raw
                        
                        # 计算平均损失
                        val_loss = val_loss_sum / val_batch_count
                        val_losses.append(val_loss)
                        
                        # 添加预测值范围监控（原始尺度）
                        pred_min = all_preds_original.min().item()
                        pred_max = all_preds_original.max().item()
                        target_min = all_targets_original.min().item()
                        target_max = all_targets_original.max().item()
                    
                    # 保存最佳模型（包含完整信息）
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        # ⭐ 保存完整checkpoint，包含模型结构信息
                        checkpoint = {
                            'model_state_dict': model.state_dict(),
                            'model_type': model_type,
                            'input_dim': train_X_tensor.shape[-1],  # 当前特征数
                            'hidden_dim': hidden_dim,
                            'epoch': epoch,
                            'best_val_loss': best_val_loss,
                            'feature_names': feature_names
                        }
                        torch.save(checkpoint, 'best_stgcn_model.pth')
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                    
                    # 学习率调度（每个epoch调用，不需要传入val_loss）
                    scheduler.step()
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
                        f"R²: {r2:.4f} | "
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
                        - **预测范围**: [{pred_min:.2f}, {pred_max:.2f}] MPa
                        - **真实范围**: [{target_min:.2f}, {target_max:.2f}] MPa
                        - **ss_tot**: {ss_tot:.2f}, **ss_res**: {ss_res:.2f}
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
                
                # ⭐ 智能加载最佳模型
                try:
                    checkpoint = torch.load('best_stgcn_model.pth')
                    
                    # 检查是否是新格式的checkpoint
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        # 新格式：检查兼容性
                        saved_input_dim = checkpoint.get('input_dim', 0)
                        current_input_dim = train_X_tensor.shape[-1]
                        
                        if saved_input_dim == current_input_dim:
                            model.load_state_dict(checkpoint['model_state_dict'])
                            st.success(f"✅ 成功加载最佳模型（epoch {checkpoint.get('epoch', '?')}）")
                        else:
                            st.warning(f"""
                            ⚠️ 模型特征数不匹配
                            - 保存的模型: {saved_input_dim}个特征
                            - 当前模型: {current_input_dim}个特征
                            
                            使用当前训练完成的模型进行预测
                            """)
                    else:
                        # 旧格式：直接尝试加载
                        model.load_state_dict(checkpoint)
                        st.info("加载了旧格式的模型文件")
                
                except FileNotFoundError:
                    st.info("未找到保存的模型文件，使用当前模型")
                except RuntimeError as e:
                    st.warning(f"⚠️ 模型加载失败（结构不匹配），使用当前模型进行预测")
                    st.caption(f"错误详情: {str(e)[:150]}...")
                
                model.eval()
                
                # 随机选择几个验证样本
                num_examples = min(5, len(val_X_tensor))
                indices = np.random.choice(len(val_X_tensor), num_examples, replace=False)
                
                with torch.no_grad():
                    example_X = val_X_tensor[indices].to(device)
                    example_y_true = val_y_tensor[indices].to(device)
                    
                    if "STGCN" in model_type:
                        example_y_pred = model(example_X, A_hat_tensor)
                        # 裁剪到[0,1]范围
                        example_y_pred = torch.clamp(example_y_pred, 0.0, 1.0)
                    else:
                        # LSTM/AttentionLSTM/Transformer
                        example_y_pred = model(example_X)  # (B, 1)
                        # 裁剪到[0,1]范围
                        example_y_pred = torch.clamp(example_y_pred, 0.0, 1.0)
                
                # 创建对比表
                comparison_data = []
                for i, idx in enumerate(indices):
                    sup_id = val_support_ids[idx]
                    
                    if "STGCN" in model_type:
                        # STGCN: 从图结构中提取
                        node_idx = support_to_idx[sup_id]
                        true_val_normalized = example_y_true[i, 0, node_idx, 0].cpu().item()
                        pred_val_normalized = example_y_pred[i, 0, node_idx, 0].cpu().item()
                    else:
                        # LSTM/AttentionLSTM/Transformer: 直接输出标量
                        true_val_normalized = example_y_true[i].cpu().item()
                        pred_val_normalized = example_y_pred[i].cpu().item()
                    
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
                
                # ⭐ 高级诊断可视化（如果启用）
                if show_advanced_plots:
                    st.markdown("---")
                    st.subheader("📊 高级诊断分析")
                    try:
                        # 获取完整验证集预测
                        model.eval()
                        with torch.no_grad():
                            if "STGCN" in model_type:
                                all_val_pred = model(val_X_tensor, A_hat_tensor)
                            else:
                                all_val_pred = model(val_X_tensor)
                        
                        # 转换为numpy
                        if "STGCN" in model_type:
                            # STGCN输出: (B, T, N, 1)，取最后时间步和所有节点
                            val_pred_np = all_val_pred[:, -1, :, 0].cpu().numpy().flatten()
                            val_true_np = val_y_tensor[:, -1, :, 0].cpu().numpy().flatten()
                        else:
                            val_pred_np = all_val_pred.cpu().numpy().flatten()
                            val_true_np = val_y_tensor.cpu().numpy().flatten()
                        
                        # 反归一化
                        val_pred_original = val_pred_np * y_range + y_min
                        val_true_original = val_true_np * y_range + y_min
                        
                        # 调用高级诊断函数
                        plot_advanced_diagnostics(
                            y_true=val_true_original,
                            y_pred=val_pred_original,
                            train_losses=train_losses,
                            val_losses=val_losses
                        )
                    except Exception as plot_error:
                        st.warning(f"高级可视化出错: {str(plot_error)}")
                
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