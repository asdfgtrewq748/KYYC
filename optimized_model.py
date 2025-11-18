"""
优化方案：简化模型+增强特征
基于诊断结果的针对性改进
"""

import torch
import torch.nn as nn
import numpy as np

class SimpleButEffectiveModel(nn.Module):
    """
    简化但高效的矿压预测模型
    
    核心思路：
    1. 使用LSTM提取时序特征（比Transformer更适合时序数据）
    2. 简化地质特征处理
    3. 使用残差连接和BatchNorm稳定训练
    4. 大幅减少参数量，避免过拟合
    """
    
    def __init__(
        self,
        seq_len=5,
        num_pressure_features=6,
        num_geology_features=9,
        num_time_features=2,
        hidden_dim=128,
        num_lstm_layers=2,
        dropout=0.3
    ):
        super(SimpleButEffectiveModel, self).__init__()
        
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # 压力特征LSTM编码
        self.pressure_lstm = nn.LSTM(
            input_size=num_pressure_features,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # 地质特征编码器
        self.geo_encoder = nn.Sequential(
            nn.Linear(num_geology_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 时间特征编码
        self.time_encoder = nn.Linear(num_time_features, hidden_dim // 2)
        
        # 特征融合（压力LSTM输出是双向的，所以是hidden_dim*2）
        fusion_dim = hidden_dim * 2 + hidden_dim + hidden_dim // 2
        
        # 预测头（更深的网络）
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            
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
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        x: (batch, seq_len, total_features)
        """
        batch_size = x.size(0)
        
        # 分离特征
        pressure_features = x[:, :, :6]  # (batch, seq, 6)
        geology_features = x[:, -1, 6:15]  # (batch, 9) 取最后时间步
        time_features = x[:, -1, 15:17]  # (batch, 2)
        
        # 1. LSTM编码压力序列
        lstm_out, (h_n, c_n) = self.pressure_lstm(pressure_features)
        # lstm_out: (batch, seq, hidden_dim*2)
        # 取最后时间步
        pressure_encoded = lstm_out[:, -1, :]  # (batch, hidden_dim*2)
        
        # 2. 编码地质特征
        geo_encoded = self.geo_encoder(geology_features)  # (batch, hidden_dim)
        
        # 3. 编码时间特征
        time_encoded = self.time_encoder(time_features)  # (batch, hidden_dim//2)
        
        # 4. 融合所有特征
        fused = torch.cat([pressure_encoded, geo_encoded, time_encoded], dim=-1)
        
        # 5. 预测
        output = self.predictor(fused)
        
        return output


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    print("="*70)
    print("🚀 简化优化模型测试")
    print("="*70)
    
    model = SimpleButEffectiveModel(
        seq_len=5,
        num_pressure_features=6,
        num_geology_features=9,
        num_time_features=2,
        hidden_dim=128,
        num_lstm_layers=2,
        dropout=0.3
    )
    
    print(f"\n模型参数量: {count_parameters(model):,}")
    print(f"相比原模型(20,668,673)减少: {(1 - count_parameters(model)/20668673)*100:.1f}%")
    
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
    
    print("\n" + "="*70)
    print("关键改进点:")
    print("="*70)
    print("1. LSTM替代Transformer - 更适合短时序数据")
    print("2. 双向LSTM - 捕获前后文信息")
    print("3. 参数量大幅减少 - 避免过拟合")
    print("4. BatchNorm + Dropout - 增强泛化能力")
    print("5. Kaiming初始化 - 更好的训练起点")
    print("="*70)
