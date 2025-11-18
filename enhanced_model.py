"""
增强版模型 - 改进地质特征融合机制
核心改进：
1. 地质特征注意力融合
2. 地质-压力交互建模
3. 多尺度地质特征提取
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GeologyAttentionFusion(nn.Module):
    """地质特征注意力融合模块"""
    def __init__(self, geo_dim=9, hidden_dim=128, num_heads=4):
        super(GeologyAttentionFusion, self).__init__()
        
        # 多头注意力机制
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=geo_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(geo_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, geo_features):
        """
        geo_features: (batch, geo_dim)
        返回: (batch, hidden_dim)
        """
        # 扩展维度用于自注意力
        geo_expanded = geo_features.unsqueeze(1)  # (batch, 1, geo_dim)
        
        # 自注意力（学习特征间的关系）
        attn_out, attn_weights = self.multihead_attn(
            geo_expanded, geo_expanded, geo_expanded
        )
        attn_out = attn_out.squeeze(1)  # (batch, geo_dim)
        
        # 残差连接
        geo_refined = geo_features + attn_out
        
        # 特征提取
        geo_encoded = self.feature_extractor(geo_refined)
        
        return geo_encoded


class GeoPressureInteraction(nn.Module):
    """地质-压力交互建模模块"""
    def __init__(self, pressure_dim=128, geo_dim=128, output_dim=128):
        super(GeoPressureInteraction, self).__init__()
        
        # 双线性交互
        self.bilinear = nn.Bilinear(pressure_dim, geo_dim, output_dim)
        
        # 门控机制（控制地质影响的强度）
        self.gate = nn.Sequential(
            nn.Linear(pressure_dim + geo_dim, output_dim),
            nn.Sigmoid()
        )
        
        # 输出变换
        self.output_transform = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
    def forward(self, pressure_features, geo_features):
        """
        pressure_features: (batch, pressure_dim)
        geo_features: (batch, geo_dim)
        返回: (batch, output_dim)
        """
        # 双线性交互
        interaction = self.bilinear(pressure_features, geo_features)
        
        # 门控融合
        concat = torch.cat([pressure_features, geo_features], dim=-1)
        gate = self.gate(concat)
        
        # 加权融合
        output = interaction * gate
        output = self.output_transform(output)
        
        return output


class EnhancedGeoPressureModel(nn.Module):
    """
    增强版矿压预测模型
    
    改进点：
    1. 地质特征注意力融合
    2. 地质-压力交互建模
    3. 多尺度特征提取
    4. 残差连接和层归一化
    """
    
    def __init__(
        self,
        seq_len=5,
        num_pressure_features=6,
        num_geology_features=9,
        num_time_features=2,
        hidden_dim=128,
        num_lstm_layers=2,
        num_attn_heads=4,
        dropout=0.3
    ):
        super(EnhancedGeoPressureModel, self).__init__()
        
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # ==================== 压力序列编码 ====================
        self.pressure_lstm = nn.LSTM(
            input_size=num_pressure_features,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # 双向LSTM输出降维
        self.pressure_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # ==================== 地质特征编码（增强版）====================
        self.geo_attention_fusion = GeologyAttentionFusion(
            geo_dim=num_geology_features,
            hidden_dim=hidden_dim,
            num_heads=num_attn_heads
        )
        
        # ==================== 时间特征编码 ====================
        self.time_encoder = nn.Sequential(
            nn.Linear(num_time_features, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ==================== 地质-压力交互 ====================
        self.geo_pressure_interaction = GeoPressureInteraction(
            pressure_dim=hidden_dim,
            geo_dim=hidden_dim,
            output_dim=hidden_dim
        )
        
        # ==================== 多尺度融合 ====================
        fusion_dim = hidden_dim + hidden_dim + hidden_dim // 2
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ==================== 预测头 ====================
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        x: (batch, seq_len, total_features)
        """
        batch_size = x.size(0)
        
        # ==================== 分离特征 ====================
        pressure_features = x[:, :, :6]  # (batch, seq, 6)
        geology_features = x[:, -1, 6:15]  # (batch, 9) 取最后时间步
        time_features = x[:, -1, 15:17]  # (batch, 2)
        
        # ==================== 1. 编码压力序列 ====================
        lstm_out, _ = self.pressure_lstm(pressure_features)
        pressure_encoded = lstm_out[:, -1, :]  # (batch, hidden_dim*2)
        pressure_encoded = self.pressure_projection(pressure_encoded)  # (batch, hidden_dim)
        
        # ==================== 2. 编码地质特征（注意力融合）====================
        geo_encoded = self.geo_attention_fusion(geology_features)  # (batch, hidden_dim)
        
        # ==================== 3. 编码时间特征 ====================
        time_encoded = self.time_encoder(time_features)  # (batch, hidden_dim//2)
        
        # ==================== 4. 地质-压力交互 ====================
        interaction_features = self.geo_pressure_interaction(
            pressure_encoded, geo_encoded
        )  # (batch, hidden_dim)
        
        # ==================== 5. 多尺度融合 ====================
        # 融合：交互特征 + 时间特征
        fused = torch.cat([interaction_features, time_encoded], dim=-1)
        fused = self.fusion_layer(fused)  # (batch, hidden_dim*2)
        
        # ==================== 6. 预测 ====================
        output = self.predictor(fused)
        
        return output


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 70)
    print("🚀 增强版模型测试 - 改进地质特征融合")
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
    )
    
    print(f"\n模型参数量: {count_parameters(model):,}")
    
    # 测试前向传播
    batch_size = 64
    seq_len = 5
    num_features = 17
    
    x = torch.randn(batch_size, seq_len, num_features)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"\n前向传播测试:")
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"✓ 模型结构正常")
    
    print("\n" + "=" * 70)
    print("🎯 关键改进点:")
    print("=" * 70)
    print("1. 地质特征注意力融合 - 学习特征间关系")
    print("2. 地质-压力交互建模 - 双线性交互+门控机制")
    print("3. LayerNorm替代BatchNorm - 更稳定")
    print("4. 残差连接 - 梯度流动更顺畅")
    print("5. 多头注意力 - 捕获多角度地质信息")
    print("=" * 70)
