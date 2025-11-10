# KYYC 矿压预测系统

基于Transformer的煤矿液压支架末阻力预测系统。

##  核心特点

-  17个特征（6矿压 + 9地质 + 2时间）
-  地质条件融合（52.9%特征占比）
-  防弹数值保护（零NaN风险）
-  GPU加速（CUDA 13.0）

##  训练成果

- **验证R**: 0.5728
- **验证MAE**: 4.91 MPa
- **训练时长**: 48分钟
- **训练批次**: 149,940个（0个NaN）

##  快速开始

**一键训练：**
```
双击: 开始训练.bat
```

**命令行：**
```powershell
conda activate kyyc_py311
python train_safe.py
```

##  项目结构

```
KYYC/
 核心系统
    simple_dataloader.py      数据加载器
    stable_transformer.py     Transformer模型
    train_safe.py            训练脚本
    safe_best_model.pth      最佳模型

 数据处理
    preprocess/              预处理脚本
    processed_data/          训练数据（17特征）
    测试钻孔/                地质数据
    kaungya/                压力数据

 文档
    README.md               本文件
    docs/                   详细文档

 工具
     debug_data.py           数据调试
     regenerate_data.py      重生成数据
     检查训练就绪.py          环境检查
     开始训练.bat            启动脚本
     项目总览.bat            项目导航
```

##  模型架构

**Transformer配置：**
- 参数量: 605,825
- 隐藏层: 128维
- Transformer层: 3层
- 注意力头: 8个
- 学习率: 0.0001
- 梯度裁剪: 1.0

##  地质特征（9个）

模型包含了丰富的地质特征：
1. 煤层厚度
2. 总厚度
3. 顶煤深度
4. 平均弹性模量
5. 平均密度
6. 砂岩占比
7. 泥岩占比
8. 煤层数量
9. 最大抗拉强度

这些特征占总特征的**52.9%**，模型能够学习不同地质条件下的压力规律。

##  使用模型

```python
import torch
from stable_transformer import StableTransformer

# 加载模型
checkpoint = torch.load('safe_best_model.pth', weights_only=False)
model = StableTransformer(
    input_dim=17, seq_len=5, 
    hidden_dim=128, num_layers=3, num_heads=8
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

##  详细文档

- [安全训练系统说明](docs/安全训练系统说明.md)
- [地质特征影响分析](docs/地质特征影响分析.md)

##  数据重生成

```powershell
python preprocess/prepare_training_data.py
```

##  环境要求

- Python 3.11+
- PyTorch 2.9.0 (CUDA 13.0)
- Conda环境: kyyc_py311

##  版本历史

**v2.0** - 安全训练系统（当前）
- 完全重写，零NaN风险
- 固定17特征，地质融合
- 352行简洁代码

**v1.0** - STGCN系统（已归档）
- 见 旧系统存档/

---

**项目仓库**: [GitHub - KYYC](https://github.com/asdfgtrewq748/KYYC)
