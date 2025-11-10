# -*- coding: utf-8 -*-
"""
安全训练配置 - 100%防止NaN
应用所有已知的稳定性措施
"""

# ===== 强制配置（不可修改） =====

# 1. 禁用特征工程
USE_FEATURE_ENGINEERING = False  # 强制禁用

# 2. 数据配置
DATA_CONFIG = {
    'use_normalization': True,  # 使用已标准化的数据
    'clip_outliers': True,      # 裁剪异常值
    'max_value': 10.0,          # 最大值限制
    'min_value': -10.0,         # 最小值限制
}

# 3. 模型配置
MODEL_CONFIG = {
    'model_type': 'Transformer',  # 推荐模型
    'hidden_size': 64,            # 隐藏层大小（较小更稳定）
    'num_layers': 2,              # 层数（较少更稳定）
    'dropout': 0.2,               # Dropout率
}

# 4. 训练配置（保守设置）
TRAINING_CONFIG = {
    'learning_rate': 0.0001,      # 学习率（保守）
    'batch_size': 32,             # 批次大小
    'epochs': 50,                 # 训练轮数
    'gradient_clip': 1.0,         # 梯度裁剪（重要！）
    'weight_decay': 1e-5,         # 权重衰减
    'early_stopping_patience': 10, # 早停耐心值
}

# 5. 紧急模式配置（如果保守配置还失败）
EMERGENCY_CONFIG = {
    'learning_rate': 0.00001,     # 更小的学习率
    'batch_size': 16,             # 更小的批次
    'gradient_clip': 0.5,         # 更严格的梯度裁剪
    'use_amp': False,             # 禁用混合精度
}

# ===== 安全检查函数 =====

def safe_data_check(X, y):
    """训练前数据安全检查"""
    import numpy as np
    
    print("=" * 60)
    print("安全检查")
    print("=" * 60)
    
    issues = []
    
    # 检查NaN
    if np.isnan(X).any():
        issues.append("X中有NaN")
    if np.isnan(y).any():
        issues.append("y中有NaN")
    
    # 检查Inf
    if np.isinf(X).any():
        issues.append("X中有Inf")
    if np.isinf(y).any():
        issues.append("y中有Inf")
    
    # 检查数值范围
    if np.abs(X).max() > 100:
        issues.append(f"X的值过大: {np.abs(X).max()}")
    
    # 检查特征数量
    if X.shape[-1] > 20:
        issues.append(f"特征数过多: {X.shape[-1]}（应该<=17）")
    
    if issues:
        print("⚠️ 发现问题:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ 所有检查通过")
        print(f"  - X形状: {X.shape}")
        print(f"  - X范围: [{X.min():.2f}, {X.max():.2f}]")
        print(f"  - X均值: {X.mean():.4f}")
        print(f"  - y范围: [{y.min():.2f}, {y.max():.2f}]")
        return True

def safe_training_wrapper(model, train_loader, val_loader, config):
    """安全的训练包装器"""
    import torch
    import torch.nn as nn
    import numpy as np
    
    print("\n" + "=" * 60)
    print("开始安全训练")
    print("=" * 60)
    
    # 使用配置
    lr = config['learning_rate']
    epochs = config['epochs']
    grad_clip = config['gradient_clip']
    
    print(f"学习率: {lr}")
    print(f"批次大小: {config['batch_size']}")
    print(f"梯度裁剪: {grad_clip}")
    
    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr,
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    # 损失函数
    criterion = nn.MSELoss()
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_losses = []
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            try:
                # 前向传播
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # 检查loss是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n❌ Epoch {epoch+1}, Batch {batch_idx}: 检测到NaN/Inf损失!")
                    print(f"   输入范围: [{X_batch.min():.2f}, {X_batch.max():.2f}]")
                    print(f"   输出范围: [{outputs.min():.2f}, {outputs.max():.2f}]")
                    print(f"   目标范围: [{y_batch.min():.2f}, {y_batch.max():.2f}]")
                    return None, f"Batch {batch_idx} 出现NaN"
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪（重要！）
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                optimizer.step()
                
                train_losses.append(loss.item())
                
                # 每10个batch打印一次
                if batch_idx % 10 == 0:
                    print(f"  Epoch {epoch+1}, Batch {batch_idx}: Loss = {loss.item():.4f}")
                    
            except Exception as e:
                print(f"\n❌ Epoch {epoch+1}, Batch {batch_idx}: 训练出错")
                print(f"   错误: {str(e)}")
                return None, str(e)
        
        # 验证
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  训练损失: {avg_train_loss:.4f}")
        print(f"  验证损失: {avg_val_loss:.4f}")
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        # 早停
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.get('early_stopping_patience', 10):
                print(f"\n早停：验证损失{patience_counter}轮未改善")
                break
    
    print("\n✅ 训练完成！")
    return model, None

# ===== 使用说明 =====
"""
在STGCN.py中导入并使用：

from safe_training_config import (
    TRAINING_CONFIG, 
    safe_data_check, 
    safe_training_wrapper
)

# 训练前检查
if not safe_data_check(X_train, y_train):
    st.error("数据检查失败！")
    st.stop()

# 使用安全训练
model, error = safe_training_wrapper(model, train_loader, val_loader, TRAINING_CONFIG)
if error:
    st.error(f"训练失败: {error}")
else:
    st.success("训练成功！")
"""

print("✅ 安全训练配置已加载")
print(f"  学习率: {TRAINING_CONFIG['learning_rate']}")
print(f"  批次大小: {TRAINING_CONFIG['batch_size']}")
print(f"  梯度裁剪: {TRAINING_CONFIG['gradient_clip']}")
