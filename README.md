# KYYC 矿压预测系统 - 安全训练版本

##  项目简介

基于Transformer的煤矿液压支架末阻力预测系统。

**核心特点：**
-  17个标准化特征（矿压 + 地质）
-  防弹数值保护（零NaN风险）
-  自动化训练流程
-  GPU加速（CUDA 13.0）

---

##  快速开始

**一键启动：**
```
双击: 开始训练.bat
```

**命令行：**
```powershell
conda activate kyyc_py311
python train_safe.py
```

---

##  项目结构

```
KYYC/
 数据处理
    preprocess/              预处理脚本
    processed_data/          处理后数据（17特征）
    测试钻孔/                钻孔地质数据
    kaungya/                 原始压力数据

 训练系统
    simple_dataloader.py     数据加载器
    stable_transformer.py    Transformer模型
    train_safe.py           训练脚本
    开始训练.bat             启动脚本

 文档
    README.md               本文件
    安全训练系统说明.md      详细说明
    requirements.txt        Python依赖

 旧系统存档/                  旧版STGCN系统
```

---

##  数据说明

**输入：** 17个特征（8矿压 + 9地质） 5时间步  
**输出：** 末阻力预测值（0-60 MPa）  
**数据集：** 195,836样本（训练70% | 验证15% | 测试15%）

---

##  模型架构

- **模型：** Transformer（3层，8注意力头）
- **参数量：** 622,209
- **训练配置：** lr=0.0001, batch=32, 梯度裁剪=1.0

---

##  预期结果

```
训练时长: 45-60分钟（GPU）
验证R: 0.75-0.80
测试R: 0.70-0.78
MAE: 5-7 MPa
```

---

##  版本历史

**v2.0 - 安全训练系统（当前）**
- 完全重写，零NaN风险
- 固定17特征
- 352行简洁代码

**v1.0 - STGCN系统（已归档）**
- Streamlit界面
- 4063行代码
- 见 `旧系统存档/`

---

详细文档请查看：`安全训练系统说明.md`
