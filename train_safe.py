# -*- coding: utf-8 -*-
"""
安全训练脚本 - 命令行版本
特点：
1. 完全独立，不依赖STGCN.py
2. 零特征工程
3. 全面数值保护
4. 实时监控NaN
"""

import torch
import numpy as np
import time
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from simple_dataloader import quick_load
from stable_transformer import StableTransformer, SafeLoss, SafeOptimizer


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
    nan_count = 0
    
    print(f"\n{'='*70}")
    print(f"📈 Epoch {epoch} - 训练阶段")
    print(f"{'='*70}")
    
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        # 移到GPU
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        try:
            # 前向传播（带NaN检查）
            pred = model(X_batch, check_nan=True)
            
            # 计算损失
            loss = loss_fn(pred, y_batch)
            
            # 检查损失是否为NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ Batch {batch_idx}: 损失为NaN/Inf，跳过此批次")
                nan_count += 1
                if nan_count > 10:
                    raise ValueError("❌ 连续10个批次出现NaN，训练中止！")
                continue
            
            # 反向传播和优化
            grad_norm = optimizer.step(loss)
            
            # 累计损失
            total_loss += loss.item() * len(X_batch)
            total_samples += len(X_batch)
            
            # 打印进度（每50个批次）
            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                avg_loss = total_loss / total_samples
                print(f"  Batch {batch_idx+1}/{len(train_loader)}: "
                      f"Loss={loss.item():.4f}, "
                      f"AvgLoss={avg_loss:.4f}, "
                      f"GradNorm={grad_norm:.4f}")
        
        except Exception as e:
            print(f"❌ Batch {batch_idx} 训练失败: {e}")
            nan_count += 1
            if nan_count > 10:
                raise
            continue
    
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    
    print(f"\n✓ Epoch {epoch} 训练完成")
    print(f"  平均损失: {avg_loss:.4f}")
    print(f"  NaN批次数: {nan_count}/{len(train_loader)}")
    
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
            
            # 前向传播（不检查NaN，提速）
            pred = model(X_batch, check_nan=False)
            
            # 计算损失
            loss = loss_fn(pred, y_batch)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print("⚠️ 验证时出现NaN损失")
                continue
            
            total_loss += loss.item() * len(X_batch)
            
            # 收集预测值和真实值
            all_preds.append(pred.cpu())
            all_targets.append(y_batch.cpu())
    
    # 计算平均损失
    avg_loss = total_loss / len(val_loader.dataset)
    
    # 计算R²
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    r2 = calculate_r2(all_targets, all_preds)
    
    # 反标准化（转换为原始MPa）
    all_preds_mpa = data_loader.inverse_transform_y(all_preds.numpy())
    all_targets_mpa = data_loader.inverse_transform_y(all_targets.numpy())
    
    # 计算MAE（MPa）
    mae_mpa = np.mean(np.abs(all_preds_mpa - all_targets_mpa))
    
    print(f"\n✓ 验证完成")
    print(f"  验证损失: {avg_loss:.4f}")
    print(f"  R² 分数: {r2:.4f}")
    print(f"  MAE: {mae_mpa:.2f} MPa")
    
    return avg_loss, r2, mae_mpa


def train_model(config):
    """
    主训练函数
    
    参数:
        config: 训练配置字典
    """
    print("\n" + "="*70)
    print("🚀 安全训练系统启动")
    print("="*70)
    print(f"\n配置信息:")
    print(f"  学习率: {config['lr']}")
    print(f"  批次大小: {config['batch_size']}")
    print(f"  训练轮数: {config['epochs']}")
    print(f"  梯度裁剪: {config['max_grad_norm']}")
    print(f"  设备: {config['device']}")
    
    # 1. 加载数据
    print("\n" + "="*70)
    print("📂 加载数据")
    print("="*70)
    
    train_loader, val_loader, test_loader, data_loader = quick_load(
        batch_size=config['batch_size'],
        data_path=config['data_path']
    )
    
    # 获取数据维度
    X_sample, _ = next(iter(train_loader))
    input_dim = X_sample.shape[-1]
    seq_len = X_sample.shape[1]
    
    print(f"\n✓ 数据加载完成")
    print(f"  输入维度: {input_dim}")
    print(f"  序列长度: {seq_len}")
    
    # 2. 创建模型
    print("\n" + "="*70)
    print("🏗️ 创建模型")
    print("="*70)
    
    model = StableTransformer(
        input_dim=input_dim,
        seq_len=seq_len,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout']
    ).to(config['device'])
    
    print(f"✓ Transformer模型创建成功")
    print(f"  参数量: {model.count_parameters():,}")
    
    # 3. 创建损失函数和优化器
    loss_fn = SafeLoss()
    optimizer = SafeOptimizer(
        model,
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        max_grad_norm=config['max_grad_norm']
    )
    
    print(f"✓ 优化器创建成功")
    print(f"  类型: AdamW")
    print(f"  权重衰减: {config['weight_decay']}")
    
    # 4. 训练循环
    print("\n" + "="*70)
    print("🎯 开始训练")
    print("="*70)
    
    best_r2 = -float('inf')
    best_epoch = 0
    patience = config['patience']
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_r2': [],
        'val_mae': []
    }
    
    start_time = time.time()
    
    for epoch in range(1, config['epochs'] + 1):
        epoch_start = time.time()
        
        try:
            # 训练
            train_loss = train_one_epoch(
                model, train_loader, optimizer, loss_fn, 
                config['device'], epoch
            )
            
            # 验证
            val_loss, val_r2, val_mae = validate(
                model, val_loader, loss_fn, config['device'],
                data_loader, epoch
            )
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_r2'].append(val_r2)
            history['val_mae'].append(val_mae)
            
            # 保存最佳模型
            if val_r2 > best_r2:
                best_r2 = val_r2
                best_epoch = epoch
                patience_counter = 0
                
                # 保存模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.optimizer.state_dict(),
                    'val_r2': val_r2,
                    'val_mae': val_mae,
                    'config': config
                }, config['save_path'])
                
                print(f"\n🌟 新的最佳模型！R² = {val_r2:.4f}")
            else:
                patience_counter += 1
                print(f"\n  (无改进，patience: {patience_counter}/{patience})")
            
            # 早停
            if patience_counter >= patience:
                print(f"\n⏹️ 早停触发！已连续{patience}轮无改进")
                break
            
            epoch_time = time.time() - epoch_start
            print(f"\n  Epoch用时: {epoch_time:.1f}秒")
        
        except Exception as e:
            print(f"\n❌ Epoch {epoch} 失败: {e}")
            print("继续下一轮...")
            continue
    
    total_time = time.time() - start_time
    
    # 5. 训练总结
    print("\n" + "="*70)
    print("✅ 训练完成")
    print("="*70)
    print(f"\n总结:")
    print(f"  总用时: {total_time/60:.1f} 分钟")
    print(f"  最佳Epoch: {best_epoch}")
    print(f"  最佳R²: {best_r2:.4f}")
    print(f"  最佳MAE: {min(history['val_mae']):.2f} MPa")
    
    # 6. 最终测试
    print("\n" + "="*70)
    print("🧪 最终测试")
    print("="*70)
    
    # 加载最佳模型（PyTorch 2.9兼容）
    checkpoint = torch.load(config['save_path'], weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 测试
    test_loss, test_r2, test_mae = validate(
        model, test_loader, loss_fn, config['device'],
        data_loader, epoch=-1
    )
    
    print(f"\n最终测试结果:")
    print(f"  测试R²: {test_r2:.4f}")
    print(f"  测试MAE: {test_mae:.2f} MPa")
    
    # 7. 保存训练历史
    history_path = Path(config['save_path']).parent / 'training_history.json'
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump({
            'history': history,
            'best_epoch': best_epoch,
            'best_val_r2': best_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'total_time_minutes': total_time / 60,
            'config': config
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 训练历史已保存: {history_path}")
    
    return model, history


def main():
    """主函数"""
    
    # 训练配置
    config = {
        # 数据
        'data_path': 'processed_data/sequence_dataset.npz',
        'batch_size': 32,
        
        # 模型
        'hidden_dim': 128,
        'num_layers': 3,
        'num_heads': 8,
        'dropout': 0.1,
        
        # 优化
        'lr': 0.0001,  # 保守的学习率
        'weight_decay': 1e-5,
        'max_grad_norm': 1.0,  # 梯度裁剪
        
        # 训练
        'epochs': 50,
        'patience': 10,  # 早停耐心值
        
        # 设备
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # 保存
        'save_path': 'safe_best_model.pth'
    }
    
    # 打印配置
    print("\n" + "="*70)
    print("⚙️ 训练配置")
    print("="*70)
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 开始训练
    try:
        model, history = train_model(config)
        
        print("\n" + "="*70)
        print("🎉 训练成功完成！")
        print("="*70)
        print(f"\n模型已保存: {config['save_path']}")
        print(f"可以使用以下命令加载模型:")
        print(f"  checkpoint = torch.load('{config['save_path']}')")
        print(f"  model.load_state_dict(checkpoint['model_state_dict'])")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"\n\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
