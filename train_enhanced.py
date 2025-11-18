"""
增强模型训练脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import time
import json
from enhanced_model import EnhancedGeoPressureModel, count_parameters
from simple_dataloader import SafeDataLoader

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device, scaler_y=None):
    """验证模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            output = model(X_batch)
            loss = criterion(output, y_batch)
            
            total_loss += loss.item()
            all_preds.append(output.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # 反标准化
    if scaler_y is not None:
        all_preds = scaler_y.inverse_transform(all_preds)
        all_targets = scaler_y.inverse_transform(all_targets)
    
    # 计算指标
    mae = np.mean(np.abs(all_preds - all_targets))
    mse = np.mean((all_preds - all_targets) ** 2)
    rmse = np.sqrt(mse)
    
    # R²
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return total_loss / len(val_loader), mae, rmse, r2


def main():
    print("=" * 70)
    print("🚀 增强版模型训练 - 改进地质特征融合")
    print("=" * 70)
    
    # ==================== 配置 ====================
    BATCH_SIZE = 512
    EPOCHS = 300
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n训练设备: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # ==================== 加载数据 ====================
    print("\n" + "=" * 70)
    print("📊 数据加载")
    print("=" * 70)
    
    loader = SafeDataLoader('processed_data/sequence_dataset.npz')
    X_train, y_train, X_val, y_val, X_test, y_test = loader.load_and_split(
        train_ratio=0.7, val_ratio=0.15, random_seed=42
    )
    
    X_train, y_train, X_val, y_val, X_test, y_test = loader.normalize_data(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    train_loader, val_loader, test_loader = loader.create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=BATCH_SIZE, num_workers=0
    )
    
    # ==================== 创建模型 ====================
    print("\n" + "=" * 70)
    print("🏗️ 模型初始化")
    print("=" * 70)
    
    model = EnhancedGeoPressureModel(
        seq_len=5,
        num_pressure_features=6,
        num_geology_features=9,
        num_time_features=2,
        hidden_dim=128,
        num_lstm_layers=2,
        num_attn_heads=4,
        dropout=0.3
    ).to(DEVICE)
    
    print(f"模型参数量: {count_parameters(model):,}")
    print(f"模型结构:")
    print(model)
    
    # ==================== 训练配置 ====================
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-3
    )
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    print(f"\n优化器: AdamW")
    print(f"学习率: {LEARNING_RATE}")
    print(f"损失函数: HuberLoss")
    print(f"学习率调度: OneCycleLR")
    
    # ==================== 训练循环 ====================
    print("\n" + "=" * 70)
    print("🎯 开始训练")
    print("=" * 70)
    
    best_r2 = -float('inf')
    best_epoch = 0
    patience = 50
    no_improve_count = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': [],
        'val_r2': [],
        'lr': []
    }
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # 训练
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, DEVICE)
        
        # 验证
        val_loss, val_mae, val_rmse, val_r2 = validate(
            model, val_loader, criterion, DEVICE, loader.scaler_y
        )
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        history['val_r2'].append(val_r2)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"MAE: {val_mae:.2f} | "
                  f"R²: {val_r2:.4f}")
        
        # 保存最佳模型
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_epoch = epoch + 1
            no_improve_count = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_r2': val_r2,
                'val_mae': val_mae,
                'model_config': {
                    'seq_len': 5,
                    'num_pressure_features': 6,
                    'num_geology_features': 9,
                    'num_time_features': 2,
                    'hidden_dim': 128,
                    'num_lstm_layers': 2,
                    'num_attn_heads': 4,
                    'dropout': 0.3
                }
            }, 'enhanced_best_model.pth')
        else:
            no_improve_count += 1
        
        # 早停
        if no_improve_count >= patience:
            print(f"\n早停触发！已 {patience} 个epoch无改善")
            break
    
    total_time = time.time() - start_time
    
    # ==================== 测试集评估 ====================
    print("\n" + "=" * 70)
    print("📈 测试集评估")
    print("=" * 70)
    
    # 加载最佳模型
    checkpoint = torch.load('enhanced_best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_mae, test_rmse, test_r2 = validate(
        model, test_loader, criterion, DEVICE, loader.scaler_y
    )
    
    print(f"\n最佳模型 (Epoch {best_epoch}):")
    print(f"  R² Score: {test_r2:.4f}")
    print(f"  MAE: {test_mae:.2f} MPa")
    print(f"  RMSE: {test_rmse:.2f} MPa")
    print(f"  训练时间: {total_time/60:.1f} 分钟")
    
    # ==================== 保存历史 ====================
    history['best_epoch'] = best_epoch
    history['best_r2'] = float(best_r2)
    history['test_r2'] = float(test_r2)
    history['test_mae'] = float(test_mae)
    history['test_rmse'] = float(test_rmse)
    history['total_time_minutes'] = total_time / 60
    
    # 转换numpy类型为Python原生类型
    for key in history:
        if isinstance(history[key], list):
            history[key] = [float(x) if isinstance(x, (np.floating, np.integer)) else x 
                          for x in history[key]]
    
    with open('enhanced_training_history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    print("\n✓ 训练历史已保存到 enhanced_training_history.json")
    print("✓ 最佳模型已保存到 enhanced_best_model.pth")
    
    print("\n" + "=" * 70)
    print("🎉 训练完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
