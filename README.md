# STGCN 矿压预测模型 🎯

基于时空图卷积网络 (Spatio-Temporal Graph Convolutional Network) 的矿压预测系统。

## 📋 项目简介

本项目实现了一个完整的矿压预测系统,使用 STGCN 模型来预测煤矿工作面支架的压力变化。系统提供了友好的 Streamlit Web 界面,支持 CSV 格式的数据输入和多种邻接矩阵生成方式。

**✨ 特别说明**: 本项目完美支持 NVIDIA RTX 50系显卡 (Blackwell 架构),使用 PyTorch 2.9.0 + CUDA 13.0!

## ✨ 主要特性

- � **GPU 加速**: 完美支持 RTX 50系/40系/30系等 NVIDIA 显卡
- 🔄 **CSV 格式输入**: 直接上传 CSV 格式的矿压数据
- 🔗 **多种邻接矩阵**: 链式、网格、全连接、K近邻或自定义上传
- 📊 **实时监控**: 可视化训练过程和损失曲线
- 🎯 **灵活参数**: 可调整历史时间步、预测时间步、批量大小等
- 🖥️ **友好界面**: 基于 Streamlit 的 Web 界面
- 📈 **结果可视化**: 对比真实值和预测值

## 🚀 快速开始

### 环境要求

- Python 3.11+
- PyTorch 2.9.0+ (支持 CUDA 13.0)
- NVIDIA GPU (可选,但强烈推荐)

### 1. 安装依赖

#### 方法 A: 使用 conda (推荐)

```bash
# 创建环境
conda create -n kyyc_py311 python=3.11 -y
conda activate kyyc_py311

# 安装 PyTorch (GPU版 - CUDA 13.0)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# 安装其他依赖
pip install streamlit pandas scipy openpyxl
```

#### 方法 B: 使用 pip

```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

### 2. 启动应用

#### Windows

```powershell
.\启动应用_GPU版.ps1
```

#### Linux/Mac

```bash
python -m streamlit run STGCN.py
```

### 3. 上传数据开始训练

1. 打开浏览器访问 `http://localhost:8501`
2. 上传矿压数据 CSV 文件
3. 上传支架坐标文件 (可选)
4. 选择训练参数
5. 点击"开始训练"

## 📊 示例数据

生成示例数据:

```bash
python -m streamlit run STGCN.py
```

### 4. 使用应用

1. 在浏览器中打开 http://localhost:8501
2. 上传 CSV 数据文件
3. 选择邻接矩阵生成方式
4. 调整模型参数
5. 点击"开始训练"

## 📊 数据格式

### CSV 数据文件格式

#### 格式 1: 带时间列 (推荐)

```csv
时间, 支架1, 支架2, 支架3, ...
2023-01-01 00:00, 100.5, 98.3, 102.1, ...
2023-01-01 01:00, 101.2, 99.1, 103.5, ...
```

#### 格式 2: 不带时间列

```csv
支架1, 支架2, 支架3, ...
100.5, 98.3, 102.1, ...
101.2, 99.1, 103.5, ...
```

**要求:**
- 每行代表一个时间点
- 每列代表一个支架/监测点
- 数值为压力读数

## 🔗 邻接矩阵选项

1. **链式结构 (推荐)**: 适用于工作面支架一字排列
2. **网格结构**: 适用于支架呈网格状排列
3. **全连接**: 所有支架之间都有关联
4. **K近邻**: 基于空间距离连接最近的K个支架
5. **自定义上传**: 上传自己的邻接矩阵文件

## 📁 项目结构

```
KYYC/
├── STGCN.py                              # 主程序文件
├── generate_sample_data.py               # 示例数据生成脚本
├── 数据格式说明.md                        # 详细的数据格式文档
├── .gitignore                            # Git 忽略文件
├── README.md                             # 项目说明文档
├── sample_mine_pressure_with_time.csv    # 示例数据(带时间)
├── sample_mine_pressure_no_time.csv      # 示例数据(不带时间)
└── sample_adjacency_matrix.npy           # 示例邻接矩阵
```

## 🛠️ 技术栈

- **深度学习框架**: PyTorch
- **Web 框架**: Streamlit
- **数据处理**: NumPy, Pandas
- **科学计算**: SciPy

## 📝 模型架构

STGCN 模型由以下部分组成:

1. **时空卷积块 (ST-Block)**: 
   - 时序卷积层 (Temporal Convolution)
   - 图卷积层 (Graph Convolution)
   - 批量归一化 (Batch Normalization)

2. **时序卷积块 (Time Block)**:
   - 门控线性单元 (GLU - Gated Linear Unit)

3. **输出层**:
   - 全连接层 (使用 1x1 卷积实现)

## ⚙️ 参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| 历史时间步 | 用于预测的历史数据长度 | 12-24 |
| 预测时间步 | 预测未来的时间步数 | 1-3 |
| 批量大小 | 每批训练样本数量 | 32-64 |
| 训练轮数 | 模型训练的总轮数 | 50-100 |
| 学习率 | 优化器学习率 | 0.001 |

## 📈 性能优化建议

1. **使用 GPU**: 如果有 NVIDIA GPU,选择 CUDA 设备加速训练
2. **调整批量大小**: 根据显存大小调整 batch_size
3. **减少数据量**: 如果训练太慢,可以减少历史时间步或采样数据
4. **Early Stopping**: 观察验证损失,如果不再下降可以提前停止

## 🤝 贡献

欢迎提交 Issue 和 Pull Request!

## 📄 许可证

MIT License

## 👥 作者

KYYC 项目团队

## 📞 联系方式

如有问题,请提交 Issue 或联系项目维护者。

---

**注意**: 本项目仅用于学习和研究目的,实际应用时请根据具体情况调整模型参数和架构。
