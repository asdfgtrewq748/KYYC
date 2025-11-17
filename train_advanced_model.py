# -*- coding: utf-8 -*-
"""
高级地质感知模型训练脚本
========================================
使用新的AdvancedGeoPressureModel进行训练
"""

import torch
import numpy as np
import time
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from simple_dataloader import SafeDataLoader
from advanced_geo_pressure_model import AdvancedGeoPressureModel


def calculate_r2(y_true, y_pred):
    """计算R²分数"""
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2.item()


def train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0.0
    total_samples = 0
    
    print(f"\n{'='*70}")
    print(f"📈 Epoch {epoch} - 训练阶段")
    print(f"{'='*70}")
    
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        # 移到GPU
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        try:
            # 前向传播
            pred = model(X_batch, return_attention=False)
            
            # 计算损失
            loss = loss_fn(pred, y_batch)
            
            # 检查损失
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ Batch {batch_idx}: 损失为NaN/Inf，跳过")
                continue
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新参数
            optimizer.step()
            
            # 累计损失
            total_loss += loss.item() * len(X_batch)
            total_samples += len(X_batch)
            
            # 打印进度
            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                avg_loss = total_loss / total_samples
                print(f"  Batch {batch_idx+1}/{len(train_loader)}: "
                      f"Loss={loss.item():.4f}, AvgLoss={avg_loss:.4f}")
        
        except Exception as e:
            print(f"❌ Batch {batch_idx} 训练失败: {e}")
            continue
    
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    
    print(f"\n✓ Epoch {epoch} 训练完成 - 平均损失: {avg_loss:.4f}")
    
    return avg_loss


def validate(model, val_loader, loss_fn, device, data_loader, epoch):
    """验证模型"""
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    print(f"\n{'='*70}")
    print(f"🔍 Epoch {epoch} - 验证阶段")
    print(f"{'='*70}")
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # 前向传播
            pred = model(X_batch, return_attention=False)
            
            # 计算损失
            loss = loss_fn(pred, y_batch)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            total_loss += loss.item() * len(X_batch)
            
            # 收集预测值
            all_preds.append(pred.cpu())
            all_targets.append(y_batch.cpu())
    
    # 计算指标
    avg_loss = total_loss / len(val_loader.dataset)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    r2 = calculate_r2(all_targets, all_preds)
    
    # 反标准化
    all_preds_mpa = data_loader.inverse_transform_y(all_preds.numpy())
    all_targets_mpa = data_loader.inverse_transform_y(all_targets.numpy())
    
    mae_mpa = np.mean(np.abs(all_preds_mpa - all_targets_mpa))
    
    print(f"\n✓ 验证完成")
    print(f"  验证损失: {avg_loss:.4f}")
    print(f"  R² 分数: {r2:.4f}")
    print(f"  MAE: {mae_mpa:.2f} MPa")
    
    return avg_loss, r2, mae_mpa


def main():
    """主训练流程"""
    print("\n" + "="*70)
    print("🚀 高级地质感知矿压预测模型 - 训练系统")
    print("="*70)
    
    # ============ 配置参数 ============
    EPOCHS = 300
    BATCH_SIZE = 256  # 减小批次增加更新频率
    LEARNING_RATE = 0.0001  # 提高学习率
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n⚙️ 训练配置:")
    print(f"  设备: {DEVICE}")
    print(f"  训练轮数: {EPOCHS}")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  学习率: {LEARNING_RATE}")
    
    # ============ 加载数据 ============
    print(f"\n{'='*70}")
    print("📂 加载训练数据")
    print("="*70)
    
    data_loader = SafeDataLoader('processed_data/sequence_dataset.npz')
    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.load_and_split(
        train_ratio=0.7,
        val_ratio=0.15,
        random_seed=42
    )
    
    # 标准化
    X_train_norm, y_train_norm, X_val_norm, y_val_norm, X_test_norm, y_test_norm = \
        data_loader.normalize_data(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # 创建DataLoader
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train_norm),
        torch.FloatTensor(y_train_norm)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val_norm),
        torch.FloatTensor(y_val_norm)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    print(f"\n✓ 数据加载完成")
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")
    
    # ============ 创建模型 ============
    print(f"\n{'='*70}")
    print("🏗️ 创建模型")
    print("="*70)
    
    model = AdvancedGeoPressureModel(
        seq_len=5,
        num_pressure_features=6,
        num_geology_features=9,
        num_time_features=2,
        hidden_dim=512,  # 大幅增加容量
        num_stgcn_layers=4,
        num_heads=16,
        dropout=0.2
    )
    
    model = model.to(DEVICE)
    
    print(f"\n✓ 模型创建成功")
    print(f"  模型参数量: {model.count_parameters():,}")
    print(f"  模型类型: AdvancedGeoPressureModel")
    
    # ============ 训练配置 ============
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=5e-4,  # 增强正则化
        betas=(0.9, 0.999)
    )
    
    # 余弦退火学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=30,
        T_mult=2,
        eta_min=1e-6
    )
    
    # 使用Huber Loss，对异常值更鲁棒
    loss_fn = torch.nn.HuberLoss(delta=1.0)
    
    # ============ 训练循环 ============
    print(f"\n{'='*70}")
    print("🎯 开始训练")
    print("="*70)
    
    best_val_loss = float('inf')
    best_r2 = -float('inf')
    patience_counter = 0
    MAX_PATIENCE = 50  # 给更多时间优化
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_r2': [],
        'val_mae': [],
        'learning_rate': []
    }
    
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE, epoch)
        
        # 验证
        val_loss, val_r2, val_mae = validate(model, val_loader, loss_fn, DEVICE, data_loader, epoch)
        
        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(val_r2)
        history['val_mae'].append(val_mae)
        history['learning_rate'].append(current_lr)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_r2 = val_r2
            patience_counter = 0
            
            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_r2': val_r2,
                'val_mae': val_mae,
                'model_config': {
                    'seq_len': 5,
                    'num_pressure_features': 6,
                    'num_geology_features': 9,
                    'num_time_features': 2,
                    'hidden_dim': 512,
                    'num_stgcn_layers': 4,
                    'num_heads': 16
                }
            }, 'advanced_best_model.pth')
            
            print(f"\n🎉 最佳模型已保存！")
            print(f"  验证损失: {val_loss:.4f}")
            print(f"  R²: {val_r2:.4f}")
            print(f"  MAE: {val_mae:.2f} MPa")
        else:
            patience_counter += 1
        
        # 早停检查
        if patience_counter >= MAX_PATIENCE:
            print(f"\n⏹️ 早停触发！{MAX_PATIENCE}轮未改善")
            break
        
        # 显示进度
        elapsed = time.time() - start_time
        print(f"\n📊 Epoch {epoch} 总结:")
        print(f"  训练损失: {train_loss:.4f}")
        print(f"  验证损失: {val_loss:.4f} (最佳: {best_val_loss:.4f})")
        print(f"  R²: {val_r2:.4f} (最佳: {best_r2:.4f})")
        print(f"  学习率: {current_lr:.6f}")
        print(f"  已用时间: {elapsed/60:.1f} 分钟")
        print(f"  耐心值: {patience_counter}/{MAX_PATIENCE}")
    
    # ============ 训练完成 ============
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print("✅ 训练完成！")
    print("="*70)
    print(f"  总训练时间: {total_time/60:.1f} 分钟")
    print(f"  最佳验证损失: {best_val_loss:.4f}")
    print(f"  最佳R²: {best_r2:.4f}")
    print(f"  模型保存位置: advanced_best_model.pth")
    
    # 保存训练历史（转换numpy类型为Python类型）
    history_to_save = {
        'train_loss': [float(x) for x in history['train_loss']],
        'val_loss': [float(x) for x in history['val_loss']],
        'val_r2': [float(x) for x in history['val_r2']],
        'val_mae': [float(x) for x in history['val_mae']],
        'learning_rate': [float(x) for x in history['learning_rate']]
    }
    with open('advanced_training_history.json', 'w', encoding='utf-8') as f:
        json.dump(history_to_save, f, indent=2, ensure_ascii=False)
    print(f"  训练历史: advanced_training_history.json")
    
    # ============ 测试集评估 ============
    print(f"\n{'='*70}")
    print("🔬 测试集评估")
    print("="*70)
    
    model.eval()
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_test_norm),
        torch.FloatTensor(y_test_norm)
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            pred = model(X_batch, return_attention=False)
            test_preds.append(pred.cpu())
            test_targets.append(y_batch.cpu())
    
    test_preds = torch.cat(test_preds, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    
    test_r2 = calculate_r2(test_targets, test_preds)
    test_preds_mpa = data_loader.inverse_transform_y(test_preds.numpy())
    test_targets_mpa = data_loader.inverse_transform_y(test_targets.numpy())
    test_mae = np.mean(np.abs(test_preds_mpa - test_targets_mpa))
    
    print(f"\n✓ 测试集结果:")
    print(f"  R²: {test_r2:.4f}")
    print(f"  MAE: {test_mae:.2f} MPa")
    
    print("\n" + "="*70)
    print("🎉 所有流程完成！")
    print("="*70)


if __name__ == '__main__':
    main()
