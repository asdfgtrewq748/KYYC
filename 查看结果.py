# -*- coding: utf-8 -*-
"""查看模型训练结果"""

import torch
from pathlib import Path

model_path = 'advanced_best_model.pth'

if not Path(model_path).exists():
    print("模型文件不存在，请先完成训练")
    exit(1)

checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

print("\n" + "="*60)
print("模型训练结果")
print("="*60)
print(f"\n训练轮次: Epoch {checkpoint['epoch']}")
print(f"验证R²: {checkpoint['val_r2']:.4f}")
print(f"验证MAE: {checkpoint['val_mae']:.2f} MPa")
print(f"验证损失: {checkpoint['val_loss']:.4f}")

print(f"\n模型配置:")
config = checkpoint['model_config']
print(f"  隐藏维度: {config['hidden_dim']}")
print(f"  STGCN层: {config['num_stgcn_layers']}")
print(f"  注意力头: {config['num_heads']}")

print(f"\n模型文件: {model_path}")
print(f"文件大小: {Path(model_path).stat().st_size / 1024 / 1024:.1f} MB")
print("="*60 + "\n")
