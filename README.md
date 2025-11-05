# STGCN 矿压预测模型 🎯

基于时空图卷积网络 (Spatio-Temporal Graph Convolutional Network) 的煤矿支架压力预测系统。

## 📋 项目简介

本项目实现了一个完整的矿压预测系统，整合了**钻孔地质数据**和**支架工作循环数据**，使用 STGCN 模型来预测煤矿工作面支架的末阻力变化。系统提供了友好的 Streamlit Web 界面，支持预处理数据集直接加载，实现从数据预处理到模型训练的完整工作流。

**✨ 特别说明**: 本项目完美支持 NVIDIA RTX 50系显卡 (Blackwell 架构)，使用 PyTorch 2.9.0 + CUDA 13.0!

## ✨ 主要特性

- 🚀 **GPU 加速**: 完美支持 RTX 50系/40系/30系等 NVIDIA 显卡
- �️ **地质特征融合**: 从19个钻孔数据中提取地质特征，映射到125个支架节点
- 🔄 **工作循环建模**: 基于初撑力-末阻力数据提取工作循环特征，无需插值
- 📊 **预处理数据集**: 支持加载预处理的NPZ格式数据集（195,836个训练样本）
- 🔗 **多种图结构**: 支持距离、KNN、链式、全连接等多种邻接矩阵生成方式
- � **实时训练监控**: 可视化训练过程、损失曲线和评估指标
- 🎯 **多特征输入**: 17维特征（6个压力特征 + 9个地质特征 + 2个时间特征）
- 🖥️ **友好界面**: 基于 Streamlit 的 Web 界面，一键启动训练

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.9.0+ (支持 CUDA 13.0)
- NVIDIA GPU (可选，但强烈推荐)
- 至少 8GB RAM (推荐 16GB+)

### 1. 安装依赖

#### 方法 A: 使用 conda (推荐)

```bash
# 创建环境
conda create -n kan python=3.8 -y
conda activate kan

# 安装 PyTorch (GPU版 - CUDA 13.0)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# 安装其他依赖
pip install streamlit pandas scipy numpy matplotlib
```

#### 方法 B: 使用 requirements.txt

```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

### 2. 数据预处理（首次使用）

如果需要从原始数据开始：

```bash
# 1. 提取地质特征（从19个钻孔数据）
python preprocess/geology_features.py

# 2. 准备训练数据集（合并初撑力和末阻力，提取工作循环特征）
python preprocess/prepare_training_data.py
```

这将生成 `processed_data/sequence_dataset.npz`，包含：
- 195,836 个训练样本
- 125 个支架节点
- 17 维特征向量

### 3. 启动训练界面

#### Windows (推荐)

```powershell
# 双击运行
启动训练界面.bat
```

或手动运行：
```powershell
conda activate kan
streamlit run STGCN.py
```

#### Linux/Mac

```bash
conda activate kan
streamlit run STGCN.py
```

### 4. 开始训练

1. 打开浏览器访问 `http://localhost:8501`
2. 在侧边栏选择 **"使用预处理数据集"**
3. 系统自动检测并加载 `processed_data/sequence_dataset.npz`
4. 调整训练参数：
   - 数据分割比例（训练/验证/测试）
   - 图结构类型（knn/distance/chain/full）
   - 训练轮数、批次大小、学习率等
5. 点击 **"🚀 开始训练"**
6. 实时查看训练进度和指标

## 📊 数据说明

### 输入数据

#### 1. 钻孔地质数据（测试钻孔/）
- **文件**: BK-1.csv 至 BK-63.csv（共19个有效钻孔）
- **内容**: 地层序号、岩层名称、厚度、弹性模量、容重、抗拉强度等
- **用途**: 提取9个地质特征，映射到支架位置

#### 2. 初撑力数据（kaungya/初撑力数据1-9 (2).csv）
- **记录数**: 140,925 条
- **字段**: 编号, 工作面, 支架编号, 初撑力, 时间
- **说明**: 支架工作循环开始时的压力值

#### 3. 末阻力数据（kaungya/末阻力数据1-9 (2).csv）
- **记录数**: 140,925 条
- **字段**: 编号, 工作面, 支架编号, 末阻力, 时间
- **说明**: 支架工作循环结束时的压力值（预测目标）

### 预处理数据集

**文件**: `processed_data/sequence_dataset.npz`

**数据结构**:
```python
{
    'X': (195836, 5, 17),      # 输入序列：[样本数, 历史步长, 特征数]
    'y_final': (195836, 1),    # 标签：末阻力预测值
    'support_ids': (195836,),  # 支架编号
    'feature_names': (17,)     # 特征名称列表
}
```

**17维特征详解**:

| 类别 | 特征名称 | 说明 |
|------|---------|------|
| **压力特征** | 初撑力值 | 工作循环初始压力 (MPa) |
| | 末阻力值 | 工作循环最终压力 (MPa) |
| | 压力增量 | 末阻力 - 初撑力 (MPa) |
| | 压力增长率 | (末阻力-初撑力)/初撑力 × 100% |
| | 循环时长 | 单个工作循环持续时间 (秒) |
| | 压力变化速率 | 压力增量/循环时长 (MPa/秒) |
| **地质特征** | geo_total_thickness_m | 总厚度 (米) |
| | geo_coal_thickness_m | 煤层总厚度 (米) |
| | geo_coal_seam_count | 煤层数量 |
| | geo_depth_to_top_coal_m | 顶煤深度 (米) |
| | geo_avg_elastic_modulus_GPa | 平均弹性模量 (GPa) |
| | geo_avg_density_kN_m3 | 平均密度 (kN/m³) |
| | geo_max_tensile_MPa | 最大抗拉强度 (MPa) |
| | geo_prop_sandstone | 砂岩占比 (0-1) |
| | geo_prop_mudstone | 泥岩占比 (0-1) |
| **时间特征** | 小时 | 一天中的小时 (0-23) |
| | 星期几 | 一周中的第几天 (0-6) |

### 数据统计

- **支架数量**: 125 个
- **工作循环总数**: 196,461 个
- **训练样本**: 195,836 个（5步历史 → 1步预测）
- **时间跨度**: 2025-01-08 至 2025-09-16（约8个月）
- **标签范围**: 0-60 MPa
- **平均每支架样本**: 1,567 个

## 🔗 图结构选项

系统支持多种邻接矩阵生成方式，适用于不同的空间布局：

1. **distance（距离阈值）**: 
   - 基于支架坐标，距离小于阈值的节点相连
   - 需要提供 `support_coordinates.csv`
   - 适合：已知精确支架位置

2. **knn（K近邻）**: 
   - 每个节点连接最近的 K 个邻居
   - 推荐 K=5
   - 适合：工作面支架均匀分布

3. **chain（链式）**: 
   - 节点按顺序连接成链状
   - 适合：支架一字排列

4. **full（全连接）**: 
   - 所有节点两两相连
   - 计算量大，适合小规模图（<50节点）

## 📁 项目结构

```
KYYC/
├── STGCN.py                              # 主训练程序（Streamlit界面）
├── 启动训练界面.bat                       # Windows启动脚本
├── 训练指南.md                            # 详细使用文档
├── README.md                             # 项目说明
├── requirements.txt                      # Python依赖列表
├── .gitignore                            # Git忽略文件
│
├── preprocess/                           # 数据预处理脚本
│   ├── geology_features.py              # 地质特征提取
│   └── prepare_training_data.py         # 训练数据准备
│
├── processed_data/                       # 预处理后的数据
│   ├── sequence_dataset.npz             # 训练数据集（195,836样本）
│   ├── support_coordinates.csv          # 支架坐标（125个）
│   ├── merged_pressure_data.csv         # 合并压力数据
│   └── dataset_summary.json             # 数据集摘要信息
│
├── 测试钻孔/                              # 钻孔地质数据（19个文件）
│   └── BK-*.csv                         # 钻孔地层信息
│
├── kaungya/                              # 原始压力数据
│   ├── 初撑力数据1-9 (2).csv            # 初撑力记录（140,925条）
│   └── 末阻力数据1-9 (2).csv            # 末阻力记录（140,925条）
│
└── geology_features_extracted.csv        # 提取的地质特征（19钻孔）
```

## 🛠️ 技术栈

- **深度学习框架**: PyTorch 2.9.0 (CUDA 13.0)
- **Web 界面**: Streamlit
- **数据处理**: NumPy, Pandas
- **科学计算**: SciPy
- **可视化**: Matplotlib

## 📝 模型架构

**STGCN (Spatial-Temporal Graph Convolutional Network)** 模型由以下部分组成：

```
输入: (Batch, 17 features, 125 nodes, 5 time_steps)
  ↓
┌─────────────────────────────────────┐
│  STGCNBlock 1                       │
│  ├─ 时序卷积 (Temporal Conv)        │
│  ├─ 图卷积 (Graph Conv)             │
│  └─ 时序卷积 (Temporal Conv)        │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│  STGCNBlock 2                       │
│  ├─ 时序卷积 (Temporal Conv)        │
│  ├─ 图卷积 (Graph Conv)             │
│  └─ 时序卷积 (Temporal Conv)        │
└─────────────────────────────────────┘
  ↓
最后时序卷积层 (Last TCN)
  ↓
输出映射层 (1x1 Conv)
  ↓
输出: (Batch, 1, 125 nodes, 1) - 末阻力预测
```

**关键组件**:

1. **时序卷积块 (TimeBlock)**: 
   - 使用门控线性单元 (GLU)
   - 捕获时间序列特征

2. **图卷积层 (GraphConvolution)**:
   - 利用邻接矩阵传播节点信息
   - 捕获空间依赖关系

3. **ST-Block (时空卷积块)**:
   - 组合时序和空间特征
   - 残差连接加速训练

## ⚙️ 训练参数说明

| 参数 | 说明 | 默认值 | 推荐范围 |
|------|------|--------|---------|
| **训练轮数** | 模型训练的总轮数（epochs） | 50 | 50-300 |
| **批次大小** | 每批训练样本数量（batch_size） | 64 | 32-128 |
| **学习率** | Adam优化器学习率 | 0.001 | 0.0001-0.01 |
| **隐藏层维度** | 模型隐藏层特征数 | 64 | 32-128 |
| **训练集比例** | 用于训练的数据比例 | 0.7 | 0.6-0.8 |
| **验证集比例** | 用于验证的数据比例 | 0.15 | 0.1-0.2 |

**推荐配置方案**:

**快速测试** (10-20分钟):
```
epochs: 50
batch_size: 64
learning_rate: 0.001
```

**标准训练** (30-60分钟):
```
epochs: 150
batch_size: 32
learning_rate: 0.0005
```

**高精度训练** (1-2小时):
```
epochs: 300
batch_size: 128
learning_rate: 0.0001
```

## 📈 训练输出与评估

### 实时监控指标

训练过程中会实时显示：

- **训练损失** (MSE): 训练集均方误差
- **验证损失** (MSE): 验证集均方误差
- **MAE**: 平均绝对误差（单位：MPa）
- **RMSE**: 均方根误差（单位：MPa）
- **R²**: 决定系数（拟合优度，0-1）

### 输出文件

- `best_stgcn_model.pth`: 最佳验证性能的模型权重

### 可视化结果

1. **损失曲线图**: 训练/验证损失随epoch变化
2. **邻接矩阵热力图**: 图结构可视化
3. **预测对比表**: 真实值 vs 预测值示例

## 🚨 常见问题

### Q1: 训练速度慢怎么办？
**A**: 
- 确认使用GPU（界面会显示"使用设备: cuda:0"）
- 减小batch_size（64→32）
- 减少epochs（150→50）
- 使用更少的历史步长

### Q2: 显存不足 (CUDA out of memory)
**A**: 
- 减小batch_size（64→32→16）
- 减小hidden_dim（64→32）
- 关闭其他占用GPU的程序

### Q3: 验证损失不下降
**A**: 
- 降低学习率（0.001→0.0005）
- 增加训练轮数
- 尝试不同的图结构（knn vs distance）
- 检查数据预处理是否正确

### Q4: 预测精度不理想
**A**: 
- 增加训练轮数（50→150）
- 调整图结构参数（K值、距离阈值）
- 检查特征工程（地质特征映射是否合理）
- 尝试不同的学习率

### Q5: 如何使用训练好的模型？
**A**: 
- 模型保存在 `best_stgcn_model.pth`
- 可以加载模型进行预测
- 详见 `训练指南.md`

## 📖 详细文档

- **训练指南**: 查看 `训练指南.md` 了解完整的训练流程
- **数据预处理**: 查看 `preprocess/` 目录下的脚本
- **模型代码**: 查看 `STGCN.py` 中的模型定义

## 🔬 技术细节

### 数据预处理策略

本项目采用**工作循环特征工程**方法而非线性插值：

**为什么不插值？**
- 初撑力和末阻力之间的中间过程是非线性的
- 插值会引入虚假的时间序列规律
- 违背物理实际（支架压力受多因素影响）

**工作循环建模优势**：
- 将每个工作循环作为原子单位
- 提取物理意义明确的特征（压力增量、增长率、变化速率）
- 符合煤矿实际操作流程

### 地质特征映射

使用 KNN (K-Nearest Neighbors) 方法：
- 每个支架映射到最近的钻孔
- 继承该钻孔的9个地质特征
- 将地质数据作为静态背景信息融入模型

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

如有改进建议，请：
1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

## 📄 许可证

MIT License

## 👥 作者

KYYC 项目团队

## 🙏 致谢

- 感谢所有贡献者
- STGCN 模型基于论文: "Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting"
- 数据来源：某煤矿工作面实际监测数据

## 📞 联系方式

- GitHub Issues: [提交问题](https://github.com/asdfgtrewq748/KYYC/issues)
- 项目地址: https://github.com/asdfgtrewq748/KYYC

---

**更新日期**: 2025年11月  
**版本**: v1.0  
**注意**: 本项目用于学习和研究目的，实际应用时请根据具体情况调整模型参数和架构。
