"""
极简训练脚本 - 绕过Streamlit的复杂逻辑
直接训练，10分钟完成，确保成功
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
import time

print("="*70)
print(" 🚀 极简LSTM训练脚本 - 无任何花哨功能 ")
print("="*70)

# ======================== 1. 加载数据 ========================
print("\n📂 步骤1: 加载数据...")
npz_path = os.path.join('processed_data', 'sequence_dataset.npz')

if not os.path.exists(npz_path):
    print(f"❌ 数据文件不存在: {npz_path}")
    exit(1)

data = np.load(npz_path, allow_pickle=True)
X = data['X']  # (195836, 5, 17)
y = data['y_final']  # (195836, 1)

print(f"✅ 数据加载成功")
print(f"   样本数: {X.shape[0]:,}")
print(f"   序列长度: {X.shape[1]}")
print(f"   特征数: {X.shape[2]}")
print(f"   数据范围: [{X.min():.2f}, {X.max():.2f}]")

# 检查特征数
if X.shape[2] != 17:
    print(f"⚠️ 警告：特征数为{X.shape[2]}，预期17")
    response = input("是否继续？(y/n): ")
    if response.lower() != 'y':
        exit(0)

# ======================== 2. 数据切分 ========================
print("\n✂️ 步骤2: 切分数据集...")
n_samples = len(X)
train_end = int(n_samples * 0.7)
val_end = int(n_samples * 0.85)

X_train = X[:train_end]
y_train = y[:train_end]
X_val = X[train_end:val_end]
y_val = y[train_end:val_end]
X_test = X[val_end:]
y_test = y[val_end:]

print(f"✅ 数据切分完成")
print(f"   训练集: {len(X_train):,} 样本")
print(f"   验证集: {len(X_val):,} 样本")
print(f"   测试集: {len(X_test):,} 样本")

# ======================== 3. 数据归一化 ========================
print("\n🔧 步骤3: 数据归一化...")

# 重塑为2D进行归一化
n_train, seq_len, n_feat = X_train.shape
X_train_flat = X_train.reshape(-1, n_feat)
X_val_flat = X_val.reshape(-1, n_feat)
X_test_flat = X_test.reshape(-1, n_feat)

# 使用RobustScaler（对异常值鲁棒）
X_scaler = RobustScaler()
X_train_flat = X_scaler.fit_transform(X_train_flat)
X_val_flat = X_scaler.transform(X_val_flat)
X_test_flat = X_scaler.transform(X_test_flat)

# 裁剪到安全范围
X_train_flat = np.clip(X_train_flat, -10, 10)
X_val_flat = np.clip(X_val_flat, -10, 10)
X_test_flat = np.clip(X_test_flat, -10, 10)

# 恢复3D形状
X_train = X_train_flat.reshape(len(X_train), seq_len, n_feat)
X_val = X_val_flat.reshape(len(X_val), seq_len, n_feat)
X_test = X_test_flat.reshape(len(X_test), seq_len, n_feat)

# 目标值归一化
y_scaler = RobustScaler()
y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

print(f"✅ 归一化完成")
print(f"   X范围: [{X_train.min():.2f}, {X_train.max():.2f}]")
print(f"   y范围: [{y_train.min():.2f}, {y_train.max():.2f}]")

# ======================== 4. 转换为PyTorch张量 ========================
print("\n🔄 步骤4: 转换为PyTorch张量...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   设备: {device}")

X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.FloatTensor(y_train).to(device)
X_val_t = torch.FloatTensor(X_val).to(device)
y_val_t = torch.FloatTensor(y_val).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.FloatTensor(y_test).to(device)

# ======================== 5. 定义模型 ========================
print("\n🏗️ 步骤5: 构建LSTM模型（优化版）...")

class OptimizedLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3):
        super(OptimizedLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        # 添加全连接层
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        last_out = lstm_out[:, -1, :]
        # 全连接层
        out = self.fc1(last_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out.squeeze()

model = OptimizedLSTM(input_dim=n_feat, hidden_dim=128, num_layers=3).to(device)

print(f"✅ 模型构建完成（优化版）")
print(f"   输入维度: {n_feat}")
print(f"   隐藏维度: 128 (提升)")
print(f"   LSTM层数: 3 (增加)")
print(f"   全连接层: 128→64→1")
print(f"   参数总数: {sum(p.numel() for p in model.parameters()):,}")

# ======================== 6. 训练配置 ========================
print("\n⚙️ 步骤6: 配置训练（优化版）...")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=8
)

epochs = 100
batch_size = 512  # 利用GPU优势
print(f"✅ 训练配置完成（优化版）")
print(f"   训练轮数: {epochs} (增加)")
print(f"   批次大小: {batch_size} (增大)")
print(f"   初始学习率: 0.001")
print(f"   权重衰减: 1e-5 (正则化)")
print(f"   损失函数: MSE")
print(f"   优化器: Adam")
print(f"   学习率调度: ReduceLROnPlateau (patience=8)")

# ======================== 7. 训练循环 ========================
print("\n🚀 步骤7: 开始训练...")
print("="*70)

best_val_loss = float('inf')
best_val_r2 = -float('inf')
patience_counter = 0
early_stop_patience = 20  # 增加patience

start_time = time.time()

for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_losses = []
    
    # 按批次训练
    for i in range(0, len(X_train_t), batch_size):
        batch_X = X_train_t[i:i+batch_size]
        batch_y = y_train_t[i:i+batch_size]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_losses.append(loss.item())
    
    avg_train_loss = np.mean(train_losses)
    
    # 验证阶段
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_loss = criterion(val_pred, y_val_t).item()
        
        # 反归一化计算R²
        val_pred_np = val_pred.cpu().numpy().reshape(-1, 1)
        val_true_np = y_val_t.cpu().numpy().reshape(-1, 1)
        
        val_pred_orig = y_scaler.inverse_transform(val_pred_np).flatten()
        val_true_orig = y_scaler.inverse_transform(val_true_np).flatten()
        
        val_r2 = r2_score(val_true_orig, val_pred_orig)
        val_mae = mean_absolute_error(val_true_orig, val_pred_orig)
    
    # 学习率调度
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # 打印进度（每5轮或R²提升时）
    if (epoch + 1) % 5 == 0 or epoch == 0 or val_r2 > best_val_r2:
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {avg_train_loss:7.4f} | "
              f"Val Loss: {val_loss:7.4f} | "
              f"Val R²: {val_r2:6.4f} | "
              f"Val MAE: {val_mae:6.2f} MPa | "
              f"LR: {current_lr:.6f}")
    
    # Early stopping (基于R²而非loss)
    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        best_val_loss = val_loss
        patience_counter = 0
        # 保存最佳模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'best_val_loss': best_val_loss,
            'epoch': epoch + 1
        }, 'simple_best_model.pth')
        # 单独保存scaler（避免序列化问题）
        import joblib
        joblib.dump({'X_scaler': X_scaler, 'y_scaler': y_scaler}, 'simple_scalers.pkl')
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            break

training_time = time.time() - start_time
print("="*70)
print(f"✅ 训练完成！耗时: {training_time/60:.1f} 分钟")
print(f"   最佳验证R²: {best_val_r2:.4f}")
print(f"   最佳验证Loss: {best_val_loss:.4f}")

# ======================== 8. 测试集评估 ========================
print("\n📊 步骤8: 测试集评估...")

# 加载最佳模型
checkpoint = torch.load('simple_best_model.pth', weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])

# 加载scaler
import joblib
scalers = joblib.load('simple_scalers.pkl')
X_scaler = scalers['X_scaler']
y_scaler = scalers['y_scaler']

model.eval()
with torch.no_grad():
    test_pred = model(X_test_t).cpu().numpy().reshape(-1, 1)
    test_true = y_test_t.cpu().numpy().reshape(-1, 1)
    
    # 反归一化
    test_pred_orig = y_scaler.inverse_transform(test_pred).flatten()
    test_true_orig = y_scaler.inverse_transform(test_true).flatten()
    
    # 计算指标
    test_r2 = r2_score(test_true_orig, test_pred_orig)
    test_mae = mean_absolute_error(test_true_orig, test_pred_orig)
    test_rmse = np.sqrt(mean_squared_error(test_true_orig, test_pred_orig))
    test_mape = np.mean(np.abs((test_true_orig - test_pred_orig) / (test_true_orig + 1e-8))) * 100

print("="*70)
print(" 🎯 最终测试结果 ")
print("="*70)
print(f"   R² Score:     {test_r2:.4f} {'🎉' if test_r2 >= 0.60 else '⚠️' if test_r2 >= 0.50 else '❌'}")
print(f"   MAE:          {test_mae:.2f} MPa")
print(f"   RMSE:         {test_rmse:.2f} MPa")
print(f"   MAPE:         {test_mape:.2f}%")
print("="*70)

# 判断是否达标
if test_r2 >= 0.70:
    print("✅✅✅ 恭喜！R² ≥ 0.70，完美达标！")
    print("   已经超过目标，可以直接使用此模型")
elif test_r2 >= 0.60:
    print("✅ 不错！R² ≥ 0.60，基本达标！")
    print("   可以继续优化：")
    print("   - 数据预处理：重新生成更长序列（seq_len=20）")
    print("   - 模型优化：尝试Transformer或AttentionLSTM")
    print("   - 特征工程：适度添加10-20个统计特征")
elif test_r2 >= 0.50:
    print("⚠️ R² ≥ 0.50，有一定效果但不够理想")
    print("   关键问题：序列长度太短（只有5步）")
    print("   建议：")
    print("   1. 重新生成数据：seq_len=20, step=1")
    print("   2. 增加特征：添加统计特征（均值、标准差等）")
    print("   3. 更强模型：Transformer + Attention")
else:
    print("❌ R² < 0.50，效果不理想")
    print("   根本问题：")
    print("   1. 序列长度严重不足（5步太短，建议20步）")
    print("   2. 特征信息量不够（17个特征可能不足）")
    print("   3. 数据质量需要检查")

# ======================== 9. 保存结果 ========================
print("\n💾 步骤9: 保存结果...")

# 保存预测结果
results = {
    'test_true': test_true_orig,
    'test_pred': test_pred_orig,
    'metrics': {
        'r2': test_r2,
        'mae': test_mae,
        'rmse': test_rmse,
        'mape': test_mape
    }
}
np.savez('simple_results.npz', **results)

print(f"✅ 结果已保存:")
print(f"   模型: simple_best_model.pth")
print(f"   结果: simple_results.npz")

print("\n" + "="*70)
print(" 🎉 所有步骤完成！")
print("="*70)
print("\n💡 提示：")
print("   - 如果R²不理想，可以修改脚本中的超参数")
print("   - 模型已保存，可以直接用于预测")
print("   - 没有使用任何特征工程，只用了17个原始特征")
print("   - 数据已正确归一化，不会出现NaN")
