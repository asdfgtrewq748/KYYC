# -*- coding: utf-8 -*-
"""
极简数据加载器 - 零特征工程，纯粹加载NPZ
作用：只做数据加载和基本张量转换，避免任何数值风险
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class SafeDataLoader:
    """防弹数据加载器 - 内置全面数值检查"""
    
    def __init__(self, npz_path='processed_data/sequence_dataset.npz'):
        """
        参数:
            npz_path: NPZ文件路径
        """
        self.npz_path = npz_path
        self.scaler_X = None
        self.scaler_y = None
        
    def load_and_split(self, train_ratio=0.7, val_ratio=0.15, random_seed=42):
        """
        加载数据并切分，包含完整的数值安全检查
        
        返回:
            (X_train, y_train, X_val, y_val, X_test, y_test)
            全部为 numpy.ndarray 格式
        """
        print("=" * 70)
        print("📂 步骤1: 加载数据文件")
        print("=" * 70)
        
        # 加载NPZ
        data = np.load(self.npz_path, allow_pickle=True)
        X = data['X']  # 形状: (N, seq_len, features)
        y = data['y_final']  # 形状: (N, 1) - 使用末阻力作为目标
        
        print(f"✓ 原始数据形状: X={X.shape}, y={y.shape}")
        print(f"✓ 特征数: {X.shape[-1]}")
        
        # 关键检查1: NaN/Inf
        print("\n🔍 数值健康检查...")
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("❌ X中包含NaN或Inf！")
        if np.isnan(y).any() or np.isinf(y).any():
            raise ValueError("❌ y中包含NaN或Inf！")
        print("✓ 无NaN/Inf")
        
        # 关键检查2: 数值范围
        print(f"✓ X范围: [{X.min():.2f}, {X.max():.2f}]")
        print(f"✓ y范围: [{y.min():.2f}, {y.max():.2f}] MPa")
        
        # 切分数据
        print("\n" + "=" * 70)
        print("✂️ 步骤2: 切分训练/验证/测试集")
        print("=" * 70)
        
        np.random.seed(random_seed)
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        print(f"✓ 训练集: {len(X_train):,} 样本 ({train_ratio*100:.0f}%)")
        print(f"✓ 验证集: {len(X_val):,} 样本 ({val_ratio*100:.0f}%)")
        print(f"✓ 测试集: {len(X_test):,} 样本 ({(1-train_ratio-val_ratio)*100:.0f}%)")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def normalize_data(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        标准化数据（均值0，标准差1）
        
        注意：只用训练集拟合，然后应用到验证/测试集
        """
        print("\n" + "=" * 70)
        print("📊 步骤3: 数据标准化")
        print("=" * 70)
        
        # 重塑X用于标准化
        n_train, seq_len, n_features = X_train.shape
        X_train_flat = X_train.reshape(-1, n_features)
        X_val_flat = X_val.reshape(-1, n_features)
        X_test_flat = X_test.reshape(-1, n_features)
        
        # 标准化X
        self.scaler_X = StandardScaler()
        X_train_norm = self.scaler_X.fit_transform(X_train_flat)
        X_val_norm = self.scaler_X.transform(X_val_flat)
        X_test_norm = self.scaler_X.transform(X_test_flat)
        
        # 恢复形状
        X_train_norm = X_train_norm.reshape(X_train.shape)
        X_val_norm = X_val_norm.reshape(X_val.shape)
        X_test_norm = X_test_norm.reshape(X_test.shape)
        
        print(f"✓ X标准化完成: 均值≈0, 标准差≈1")
        print(f"  训练集范围: [{X_train_norm.min():.2f}, {X_train_norm.max():.2f}]")
        
        # 标准化y
        self.scaler_y = StandardScaler()
        y_train_norm = self.scaler_y.fit_transform(y_train)
        y_val_norm = self.scaler_y.transform(y_val)
        y_test_norm = self.scaler_y.transform(y_test)
        
        print(f"✓ y标准化完成")
        print(f"  原始范围: [{y_train.min():.2f}, {y_train.max():.2f}] MPa")
        print(f"  归一化范围: [{y_train_norm.min():.2f}, {y_train_norm.max():.2f}]")
        
        # 最终检查
        for name, X_norm in [('训练', X_train_norm), ('验证', X_val_norm), ('测试', X_test_norm)]:
            if np.isnan(X_norm).any() or np.isinf(X_norm).any():
                raise ValueError(f"❌ {name}集X标准化后出现NaN/Inf！")
        
        print("✓ 标准化后无NaN/Inf")
        
        return X_train_norm, y_train_norm, X_val_norm, y_val_norm, X_test_norm, y_test_norm
    
    def create_dataloaders(self, X_train, y_train, X_val, y_val, X_test, y_test,
                          batch_size=32, num_workers=0):
        """
        创建PyTorch DataLoader
        
        参数:
            batch_size: 批次大小
            num_workers: 数据加载线程数（Windows建议设为0）
        
        返回:
            (train_loader, val_loader, test_loader)
        """
        print("\n" + "=" * 70)
        print("🔄 步骤4: 创建DataLoader")
        print("=" * 70)
        
        # 转换为Tensor
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)
        X_test_t = torch.FloatTensor(X_test)
        y_test_t = torch.FloatTensor(y_test)
        
        print(f"✓ Tensor转换完成")
        print(f"  训练集: X={X_train_t.shape}, y={y_train_t.shape}")
        
        # 创建Dataset
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        test_dataset = TensorDataset(X_test_t, y_test_t)
        
        # 创建DataLoader
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True  # 加速GPU传输
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"✓ DataLoader创建完成")
        print(f"  批次大小: {batch_size}")
        print(f"  训练批次数: {len(train_loader)}")
        print(f"  验证批次数: {len(val_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def inverse_transform_y(self, y_norm):
        """反标准化y（用于最终预测结果）"""
        if self.scaler_y is None:
            raise ValueError("必须先调用normalize_data()！")
        return self.scaler_y.inverse_transform(y_norm)


def quick_load(batch_size=32, data_path='processed_data/sequence_dataset.npz'):
    """
    一键加载函数 - 最简单的使用方式
    
    使用示例:
        train_loader, val_loader, test_loader, loader = quick_load(batch_size=32)
        
        for X_batch, y_batch in train_loader:
            # 训练代码
            pass
    
    返回:
        train_loader, val_loader, test_loader, loader对象
    """
    loader = SafeDataLoader(data_path)
    
    # 加载和切分
    X_train, y_train, X_val, y_val, X_test, y_test = loader.load_and_split()
    
    # 标准化
    X_train, y_train, X_val, y_val, X_test, y_test = loader.normalize_data(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    # 创建DataLoader
    train_loader, val_loader, test_loader = loader.create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=batch_size
    )
    
    print("\n" + "=" * 70)
    print("✅ 数据准备完成！可以开始训练")
    print("=" * 70)
    
    return train_loader, val_loader, test_loader, loader


if __name__ == '__main__':
    """测试数据加载"""
    print("\n🧪 测试数据加载器...\n")
    
    # 快速加载
    train_loader, val_loader, test_loader, loader = quick_load(batch_size=32)
    
    # 测试一个批次
    print("\n🔍 测试批次数据...")
    for X_batch, y_batch in train_loader:
        print(f"✓ 批次形状: X={X_batch.shape}, y={y_batch.shape}")
        print(f"✓ X范围: [{X_batch.min():.2f}, {X_batch.max():.2f}]")
        print(f"✓ y范围: [{y_batch.min():.2f}, {y_batch.max():.2f}]")
        
        # 检查NaN
        if torch.isnan(X_batch).any() or torch.isinf(X_batch).any():
            print("❌ X批次包含NaN/Inf！")
        else:
            print("✓ X批次健康")
        
        if torch.isnan(y_batch).any() or torch.isinf(y_batch).any():
            print("❌ y批次包含NaN/Inf！")
        else:
            print("✓ y批次健康")
        
        break  # 只测试第一个批次
    
    print("\n✅ 数据加载器测试通过！")
