# DNN项目 - IC/IMAC信道功率分配优化

本项目实现了基于深度神经网络(DNN)的无线通信系统功率分配优化，包括干扰信道(IC)和干扰多址接入信道(IMAC)两种场景。项目对比了DNN方法与传统WMMSE算法的性能。

##  项目概述

### IC (干扰信道) 部分
- **数据生成**: 使用WMMSE算法生成高斯干扰信道数据
- **模型训练**: 训练DNN模型学习最优功率分配策略
- **性能对比**: 与WMMSE算法进行和速率性能对比

### IMAC (干扰多址接入信道) 部分  
- **大规模数据集生成**: 生成100万组IMAC数据集
- **DNN模型训练**: 基于TensorFlow 1.x的深度神经网络训练
- **流式处理**: 支持批量数据处理和内存优化

##  项目结构

```
DNN/
├── requirements.txt          # 项目依赖
├── README.md                # 项目说明文档
├── IC/                      # IC信道相关代码
│   ├── code_IC/             # IC核心代码
│   │   ├── WMMSE_data_gennerate.py      # WMMSE数据生成
│   │   ├── DNN_train.py                 # DNN模型训练
│   │   └── function_wmmse_powercontrol.py  # WMMSE算法实现
│   ├── data_IC/             # 生成的IC数据存储
│   └── results_IC/          # IC实验结果
└── IMAC/                    # IMAC信道相关代码
    ├── generate_imac_dataset.py         # IMAC数据集生成
    ├── train_imac_dnn_tf1.py           # IMAC DNN训练 (TensorFlow 1.x)
    ├── check_tf1_environment.py        # TensorFlow环境检查
    ├── dnn_predictions_tf1_streaming/   # DNN预测结果
    └── imac_dnn_results_tf1_streaming/  # 训练曲线和性能对比图
```

##  快速开始

### 1. 环境准备

```bash
# 克隆项目 (如果从远程获取)
git clone <repository-url>
cd DNN

# 安装依赖
pip install -r requirements.txt
```

### 2. 环境检查 (IMAC部分)

```bash
# 检查TensorFlow 1.x环境是否正确配置
cd IMAC
python check_tf1_environment.py
```

### 3. IC信道实验

```bash
cd IC/code_IC

# 1. 生成IC数据
python WMMSE_data_gennerate.py

# 2. 训练DNN模型并对比性能
python DNN_train.py
```

### 4. IMAC信道实验

```bash
cd IMAC

# 1. 生成IMAC数据集 (100万样本)
python generate_imac_dataset.py

# 2. 训练IMAC DNN模型
python train_imac_dnn_tf1.py
```

##  核心功能详解

### IC信道 (IC/)

#### 数据生成 (`WMMSE_data_gennerate.py`)
- 生成高斯干扰信道矩阵
- 使用WMMSE算法计算最优功率分配
- 保存训练和测试数据集

#### DNN训练 (`DNN_train.py`)  
- 实现深度神经网络模型
- 训练功率分配优化网络
- 生成和速率性能对比图
- 支持CDF曲线绘制

#### 关键特性
- **信道类型**: 高斯干扰信道
- **优化目标**: 最大化和速率
- **对比基准**: WMMSE算法
- **可视化**: CDF性能曲线

### IMAC信道 (IMAC/)

#### 数据集生成 (`generate_imac_dataset.py`)
- **大规模生成**: 支持100万样本数据集
- **多进程优化**: 并行数据生成提升效率
- **批量存储**: 分批保存数据避免内存溢出
- **性能监控**: 实时显示生成进度

#### DNN训练 (`train_imac_dnn_tf1.py`)
- **TensorFlow 1.x**: 基于TF1的完整实现
- **改进早停**: 防止过拟合的早停机制
- **流式处理**: 支持大规模数据的批量训练
- **可视化**: 训练曲线和性能对比图

#### 关键参数
- **基站数**: 24个
- **用户数**: 3个  
- **覆盖半径**: 100m
- **噪声功率**: 1
- **网络结构**: [200, 80, 80] 隐藏层

## 🔧 配置说明

### TensorFlow版本要求
- **推荐**: TensorFlow 1.15.0
- **兼容**: TensorFlow 2.x (需启用兼容模式)
- **GPU支持**: 需要相应CUDA和cuDNN版本

### 内存和存储需求
- **IMAC数据集**: ~10GB (100万样本)
- **训练内存**: 建议16GB以上
- **GPU内存**: 建议4GB以上

### 数据路径配置
默认数据保存路径: `/root/autodl-tmp/IMAC/`
可在代码中修改为适合的路径。

## 📈 结果分析

### 输出文件
- **训练曲线**: 损失函数和准确率变化
- **性能对比图**: DNN vs WMMSE和速率对比
- **CDF曲线**: 累积分布函数性能分析
- **模型文件**: 训练好的DNN模型权重

### 性能指标
- **和速率 (Sum Rate)**: 系统总吞吐量
- **训练损失**: 模型收敛情况
- **测试精度**: 泛化性能评估

##  故障排除

### 常见问题

1. **TensorFlow版本冲突**
   ```bash
   # 卸载现有版本
   pip uninstall tensorflow tensorflow-gpu
   # 安装指定版本
   pip install tensorflow==1.15.0
   ```

2. **内存不足**
   - 减少批处理大小
   - 使用流式数据加载
   - 增加系统虚拟内存

3. **数据路径错误**
   - 检查数据保存路径是否存在
   - 确保有足够的存储空间
   - 修改代码中的路径配置

4. **GPU支持问题**
   ```bash
   # 检查CUDA版本
   nvidia-smi
   # 安装GPU版本TensorFlow
   pip install tensorflow-gpu==1.15.0
   ```

##  算法原理

### WMMSE算法
- **全称**: Weighted Minimum Mean Square Error
- **特点**: 迭代优化算法，保证收敛到局部最优
- **复杂度**: 较高，适合作为性能基准

### DNN方法
- **优势**: 推理速度快，适合实时应用
- **训练**: 使用WMMSE结果作为标签进行监督学习
- **网络结构**: 全连接深度神经网络

##  参考文献

本项目实现基于相关无线通信和深度学习研究论文，详细算法原理请参考相关学术文献。



如遇到技术问题，请检查：
1. Python和依赖包版本是否正确
2. 数据路径和权限设置
3. TensorFlow环境配置
4. 系统内存和存储空间

##  更新日志

- **v1.0**: 初始版本，包含IC和IMAC完整实现
- 支持TensorFlow 1.x和2.x兼容模式
- 优化大规模数据处理性能
- 增加详细的环境检查工具

---

