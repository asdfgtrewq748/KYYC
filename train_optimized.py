"""
优化模型训练脚本
针对R²=0.5问题的改进训练
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import json
import time
from datetime import datetime
from optimized_model import SimpleButEffectiveModel
from simple_dataloader import SafeDataLoader

def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    batch_count = 0
    
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = loss_fn(predictions, y_batch)
        
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)}: Loss={loss.item():.4f}, AvgLoss={total_loss/batch_count:.4f}")
    
    return total_loss / batch_count

def validate(model, val_loader, loss_fn, device):
    """验证"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)
            
            total_loss += loss.item()
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # 计算R²
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    # 计算MAE (反归一化到原始尺度)
    mae = np.mean(np.abs(targets - predictions))
    mae_mpa = mae * 17.31  # 近似反归一化
    
    return total_loss / len(val_loader), r2, mae_mpa

def main():
    print("="*70)
    print("🚀 优化模型训练系统")
    print("="*70)
    
    # ============ 配置参数 ============
    EPOCHS = 200
    BATCH_SIZE = 512
    LEARNING_RATE = 0.001  # 提高学习率
    WEIGHT_DECAY = 1e-3  # 增强正则化
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n⚙️ 训练配置:")
    print(f"  设备: {DEVICE}")
    print(f"  训练轮数: {EPOCHS}")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  权重衰减: {WEIGHT_DECAY}")
    
    # ============ 加载数据 ============
    print("\n" + "="*70)
    print("📂 加载训练数据")
    print("="*70)
    
    loader = SafeDataLoader(npz_path='processed_data/sequence_dataset.npz')
    data_tuple = loader.load_and_split()
    
    # load_and_split返回的是归一化后的6个数组
    X_train, y_train, X_val, y_val, X_test, y_test = data_tuple
    
    print(f"\n数据形状检查:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # 创建DataLoader
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"\n✓ 数据加载完成")
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")
    
    # ============ 创建模型 ============
    print("\n" + "="*70)
    print("🏗️ 创建优化模型")
    print("="*70)
    
    model = SimpleButEffectiveModel(
        seq_len=5,
        num_pressure_features=6,
        num_geology_features=9,
        num_time_features=2,
        hidden_dim=128,
        num_lstm_layers=2,
        dropout=0.3
    ).to(DEVICE)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n✓ 模型创建成功")
    print(f"  模型参数量: {param_count:,}")
    print(f"  模型类型: SimpleButEffectiveModel (LSTM-based)")
    
    # ============ 训练设置 ============
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # 使用OneCycleLR - 更激进的学习率策略
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # 30%的时间用于warmup
        anneal_strategy='cos'
    )
    
    # ============ 开始训练 ============
    print("\n" + "="*70)
    print("🎯 开始训练")
    print("="*70)
    
    best_val_loss = float('inf')
    best_r2 = -float('inf')
    patience_counter = 0
    MAX_PATIENCE = 40
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_r2': [],
        'val_mae': [],
        'learning_rate': []
    }
    
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        
        print(f"\n{'='*70}")
        print(f"📈 Epoch {epoch} - 训练阶段")
        print(f"{'='*70}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        
        print(f"\n✓ Epoch {epoch} 训练完成 - 平均损失: {train_loss:.4f}")
        
        print(f"\n{'='*70}")
        print(f"🔍 Epoch {epoch} - 验证阶段")
        print(f"{'='*70}\n")
        
        val_loss, r2, mae_mpa = validate(model, val_loader, loss_fn, DEVICE)
        
        print(f"✓ 验证完成")
        print(f"  验证损失: {val_loss:.4f}")
        print(f"  R² 分数: {r2:.4f}")
        print(f"  MAE: {mae_mpa:.2f} MPa")
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(r2)
        history['val_mae'].append(mae_mpa)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # 保存最佳模型
        if r2 > best_r2:
            best_r2 = r2
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'r2': r2,
                'mae': mae_mpa,
                'model_config': {
                    'seq_len': 5,
                    'num_pressure_features': 6,
                    'num_geology_features': 9,
                    'num_time_features': 2,
                    'hidden_dim': 128,
                    'num_lstm_layers': 2,
                    'dropout': 0.3
                }
            }
            
            torch.save(checkpoint, 'optimized_best_model.pth')
            print(f"\n🎉 最佳模型已保存！")
            print(f"  验证损失: {val_loss:.4f}")
            print(f"  R²: {r2:.4f}")
            print(f"  MAE: {mae_mpa:.2f} MPa")
        else:
            patience_counter += 1
        
        # 学习率调度（每个batch调用一次）
        current_lr = optimizer.param_groups[0]['lr']
        
        elapsed = (time.time() - start_time) / 60
        
        print(f"\n📊 Epoch {epoch} 总结:")
        print(f"  训练损失: {train_loss:.4f}")
        print(f"  验证损失: {val_loss:.4f} (最佳: {best_val_loss:.4f})")
        print(f"  R²: {r2:.4f} (最佳: {best_r2:.4f})")
        print(f"  学习率: {current_lr:.6f}")
        print(f"  已用时间: {elapsed:.1f} 分钟")
        print(f"  耐心值: {patience_counter}/{MAX_PATIENCE}")
        
        # 早停
        if patience_counter >= MAX_PATIENCE:
            print(f"\n⏹️ 早停触发！{MAX_PATIENCE}轮未改善")
            break
    
    # ============ 训练完成 ============
    total_time = (time.time() - start_time) / 60
    
    print(f"\n{'='*70}")
    print("✅ 训练完成！")
    print(f"{'='*70}")
    print(f"  总训练时间: {total_time:.1f} 分钟")
    print(f"  最佳验证损失: {best_val_loss:.4f}")
    print(f"  最佳R²: {best_r2:.4f}")
    print(f"  模型保存位置: optimized_best_model.pth")
    
    # 保存训练历史（转换numpy类型为Python原生类型）
    history_serializable = {
        'train_loss': [float(x) for x in history['train_loss']],
        'val_loss': [float(x) for x in history['val_loss']],
        'val_r2': [float(x) for x in history['val_r2']],
        'val_mae': [float(x) for x in history['val_mae']],
        'learning_rate': [float(x) for x in history['learning_rate']]
    }
    
    with open('optimized_training_history.json', 'w') as f:
        json.dump(history_serializable, f, indent=2)
    print(f"  训练历史: optimized_training_history.json")
    
    # ============ 测试集评估 ============
    print(f"\n{'='*70}")
    print("🔬 测试集评估")
    print(f"{'='*70}\n")
    
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 加载最佳模型
    checkpoint = torch.load('optimized_best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_r2, test_mae = validate(model, test_loader, loss_fn, DEVICE)
    
    print(f"✓ 测试集结果:")
    print(f"  R²: {test_r2:.4f}")
    print(f"  MAE: {test_mae:.2f} MPa")
    
    print(f"\n{'='*70}")
    print("🎉 所有流程完成！")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
