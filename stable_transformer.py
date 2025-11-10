# -*- coding: utf-8 -*-
"""
防弹Transformer模型 - 内置全面数值保护
特点：
1. 所有计算都有NaN检查
2. 自动梯度裁剪
3. 稳定的初始化
4. 保守的Dropout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class StableTransformer(nn.Module):
    """
    防弹Transformer模型
    
    特点：
    - 内置数值稳定性保护
    - 自动NaN检测
    - 梯度裁剪
    - Layer Normalization
    """
    
    def __init__(self, input_dim, seq_len, hidden_dim=128, num_layers=3, 
                 num_heads=8, dropout=0.1, output_dim=1):
        """
        参数:
            input_dim: 输入特征数
            seq_len: 序列长度
            hidden_dim: 隐藏层维度（必须能被num_heads整除）
            num_layers: Transformer层数
            num_heads: 注意力头数
            dropout: Dropout比率
            output_dim: 输出维度（默认1，单值回归）
        """
        super(StableTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 位置编码（固定）
        self.register_buffer('pos_encoding', self._generate_positional_encoding(seq_len, hidden_dim))
        
        # Transformer Encoder层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',  # GELU比ReLU更稳定
            batch_first=True,
            norm_first=True  # Pre-LN更稳定
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim)
        )
        
        # 输出层（双层MLP）
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _generate_positional_encoding(self, seq_len, hidden_dim):
        """生成正弦位置编码"""
        position = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * 
                            -(math.log(10000.0) / hidden_dim))
        
        pos_encoding = torch.zeros(seq_len, hidden_dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding.unsqueeze(0)  # (1, seq_len, hidden_dim)
    
    def _init_weights(self):
        """稳定的权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier初始化（更稳定）
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)
    
    def _check_tensor(self, x, name="tensor"):
        """检查张量是否包含NaN或Inf"""
        if torch.isnan(x).any():
            raise ValueError(f"❌ {name} 包含 NaN！")
        if torch.isinf(x).any():
            raise ValueError(f"❌ {name} 包含 Inf！")
    
    def forward(self, x, check_nan=True):
        """
        前向传播
        
        参数:
            x: 输入张量，形状 (batch_size, seq_len, input_dim)
            check_nan: 是否检查NaN（训练时建议开启）
        
        返回:
            output: 形状 (batch_size, output_dim)
        """
        if check_nan:
            self._check_tensor(x, "输入X")
        
        # 输入投影
        x = self.input_projection(x)  # (batch, seq_len, hidden_dim)
        
        if check_nan:
            self._check_tensor(x, "投影后")
        
        # 添加位置编码
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        if check_nan:
            self._check_tensor(x, "位置编码后")
        
        # Transformer编码
        x = self.transformer_encoder(x)  # (batch, seq_len, hidden_dim)
        
        if check_nan:
            self._check_tensor(x, "Transformer编码后")
        
        # 全局平均池化（避免只用最后一个时间步）
        x = x.mean(dim=1)  # (batch, hidden_dim)
        
        if check_nan:
            self._check_tensor(x, "池化后")
        
        # 输出层
        output = self.output_mlp(x)  # (batch, output_dim)
        
        if check_nan:
            self._check_tensor(output, "最终输出")
        
        return output
    
    def count_parameters(self):
        """统计模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SafeLoss(nn.Module):
    """
    安全的损失函数 - 组合Huber + MSE
    
    Huber对异常值不敏感，MSE保证平滑
    """
    
    def __init__(self, delta=1.0, huber_weight=0.7, mse_weight=0.3):
        """
        参数:
            delta: Huber损失的阈值
            huber_weight: Huber损失权重
            mse_weight: MSE损失权重
        """
        super(SafeLoss, self).__init__()
        self.delta = delta
        self.huber_weight = huber_weight
        self.mse_weight = mse_weight
        
    def forward(self, pred, target):
        """
        计算损失
        
        参数:
            pred: 预测值，形状 (batch_size, 1)
            target: 真实值，形状 (batch_size, 1)
        
        返回:
            loss: 标量
        """
        # Huber损失（对异常值鲁棒）
        huber_loss = F.smooth_l1_loss(pred, target, beta=self.delta)
        
        # MSE损失（平滑）
        mse_loss = F.mse_loss(pred, target)
        
        # 组合
        total_loss = self.huber_weight * huber_loss + self.mse_weight * mse_loss
        
        # 检查NaN
        if torch.isnan(total_loss):
            print(f"⚠️ 损失为NaN！pred范围=[{pred.min():.2f}, {pred.max():.2f}], "
                  f"target范围=[{target.min():.2f}, {target.max():.2f}]")
            # 返回一个较大但安全的值
            return torch.tensor(1e6, device=pred.device, requires_grad=True)
        
        return total_loss


class SafeOptimizer:
    """
    安全的优化器包装器
    
    功能：
    - 自动梯度裁剪
    - NaN检测
    - 学习率调度
    """
    
    def __init__(self, model, lr=0.0001, weight_decay=1e-5, max_grad_norm=1.0):
        """
        参数:
            model: PyTorch模型
            lr: 学习率
            weight_decay: L2正则化系数
            max_grad_norm: 最大梯度范数（梯度裁剪）
        """
        self.model = model
        self.max_grad_norm = max_grad_norm
        
        # AdamW优化器（带权重衰减）
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
    def step(self, loss):
        """
        执行一步优化
        
        参数:
            loss: 损失张量
        
        返回:
            grad_norm: 梯度范数（用于监控）
        """
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 检查梯度是否包含NaN
        has_nan = False
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"⚠️ {name} 的梯度包含NaN/Inf！")
                    has_nan = True
        
        if has_nan:
            print("❌ 检测到梯度NaN，跳过此次更新")
            return 0.0
        
        # 梯度裁剪
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.max_grad_norm
        )
        
        # 更新参数
        self.optimizer.step()
        
        return grad_norm.item()
    
    def get_lr(self):
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']
    
    def set_lr(self, lr):
        """设置学习率"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


if __name__ == '__main__':
    """测试模型"""
    print("\n🧪 测试Transformer模型...\n")
    
    # 创建模型
    model = StableTransformer(
        input_dim=17,
        seq_len=5,
        hidden_dim=128,
        num_layers=3,
        num_heads=8,
        dropout=0.1
    )
    
    print(f"✓ 模型创建成功")
    print(f"✓ 参数量: {model.count_parameters():,}")
    
    # 测试前向传播
    batch_size = 32
    x = torch.randn(batch_size, 5, 17)
    
    try:
        output = model(x, check_nan=True)
        print(f"✓ 前向传播成功")
        print(f"✓ 输入形状: {x.shape}")
        print(f"✓ 输出形状: {output.shape}")
        print(f"✓ 输出范围: [{output.min():.4f}, {output.max():.4f}]")
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
    
    # 测试损失函数
    loss_fn = SafeLoss()
    target = torch.randn(batch_size, 1)
    loss = loss_fn(output, target)
    print(f"✓ 损失计算成功: {loss.item():.4f}")
    
    # 测试优化器
    optimizer = SafeOptimizer(model, lr=0.0001)
    grad_norm = optimizer.step(loss)
    print(f"✓ 优化步骤成功")
    print(f"✓ 梯度范数: {grad_norm:.4f}")
    
    print("\n✅ 模型测试通过！")
