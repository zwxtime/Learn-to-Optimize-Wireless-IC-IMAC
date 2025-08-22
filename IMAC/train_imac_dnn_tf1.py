#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMAC DNN训练和测试代码 (TensorFlow 1.x版本) - 改进早停机制版本
基于论文要求优化，移除BatchNorm，调整训练策略，重点改进早停机制防止过拟合
"""

import numpy as np
import scipy.io as sio
import os
import time
import tensorflow as tf
from typing import Dict, List, Tuple, Any, Generator, Optional
import warnings
warnings.filterwarnings('ignore')

# 优先设置 Matplotlib 非交互式后端，避免无显示环境错误
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 确保matplotlib完全使用非交互式模式
plt.ioff()  # 关闭交互模式

# 检查TensorFlow版本
print(f"TensorFlow版本: {tf.__version__}")
if not tf.__version__.startswith('1.'):
    print("警告：当前TensorFlow版本不是1.x，建议使用TensorFlow 1.15或更低版本")

# 设置TensorFlow 1.x兼容性
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

class IMAC_DNN_TF1_Improved:
    """IMAC功率分配DNN模型 (TensorFlow 1.x版本) - 改进版"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = None):
        if hidden_dims is None:
            hidden_dims = [200, 80, 80]  # 按照论文要求：三个隐藏层，神经元数分别为200、80、80
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # 学习率占位符（支持训练时自适应调整）
        self.learning_rate = tf.compat.v1.placeholder(tf.float32, shape=(), name='learning_rate')

        # 构建网络
        self._build_network()
        
        print(f"TensorFlow 1.x DNN模型初始化完成 (改进版):")
        print(f"  输入维度: {input_dim}")
        print(f"  输出维度: {output_dim}")
        print(f"  隐藏层: {hidden_dims}")
        print(f"  特性: 移除BatchNorm，使用He初始化")
        
        # 计算参数数量
        total_params = 0
        for var in tf.compat.v1.trainable_variables():
            total_params += np.prod(var.get_shape().as_list())
        print(f"  总参数数: {total_params:,}")
    
    def _build_network(self):
        """构建神经网络 - 改进版（移除BatchNorm）"""
        # 输入占位符
        self.X = tf.compat.v1.placeholder(tf.float32, [None, self.input_dim], name='X')
        self.Y = tf.compat.v1.placeholder(tf.float32, [None, self.output_dim], name='Y')
        
        # 构建网络层
        current_layer = self.X
        
        # 隐藏层（不使用BatchNorm，按论文要求）
        for i, hidden_dim in enumerate(self.hidden_dims):
            with tf.compat.v1.variable_scope(f'hidden_{i}'):
                # 使用He初始化（适合ReLU激活函数）
                fan_in = current_layer.get_shape()[-1].value
                # He initialization: std = sqrt(2 / fan_in)
                std = np.sqrt(2.0 / fan_in)
                
                weights = tf.compat.v1.get_variable(
                    'weights', 
                    [fan_in, hidden_dim],
                    initializer=tf.compat.v1.truncated_normal_initializer(stddev=std),
                    dtype=tf.float32
                )
                biases = tf.compat.v1.get_variable(
                    'biases',
                    [hidden_dim],
                    initializer=tf.compat.v1.constant_initializer(0.0),
                    dtype=tf.float32
                )
                
                # 线性变换
                linear = tf.matmul(current_layer, weights) + biases
                
                # 直接使用ReLU激活（移除BatchNorm）
                current_layer = tf.nn.relu(linear)
        
        # 输出层（按论文要求使用特殊激活函数）
        with tf.compat.v1.variable_scope('output'):
            fan_in = current_layer.get_shape()[-1].value
            # 输出层使用更小的初始化方差
            std = np.sqrt(1.0 / fan_in)
            
            weights = tf.compat.v1.get_variable(
                'weights',
                [fan_in, self.output_dim],
                initializer=tf.compat.v1.truncated_normal_initializer(stddev=std),
                dtype=tf.float32
            )
            biases = tf.compat.v1.get_variable(
                'biases',
                [self.output_dim],
                initializer=tf.compat.v1.constant_initializer(0.0),
                dtype=tf.float32
            )
            
            # 线性输出
            linear_output = tf.matmul(current_layer, weights) + biases
            
            # 输出层激活函数：y = min(max(x, 0), Pmax) 
            # 论文中强调用于强制功率约束
            p_max = 1.0  # 最大功率约束
            self.Y_pred = tf.minimum(tf.maximum(linear_output, 0.0), p_max)
        
        # 损失函数（MSE）
        self.loss = tf.reduce_mean(tf.square(self.Y_pred - self.Y))
        
        # 优化器（RMSprop），按论文设置
        optimizer = tf.compat.v1.train.RMSPropOptimizer(
            learning_rate=self.learning_rate,
            decay=0.9,
            momentum=0.0,  # 论文中提到的RMSprop配置
            epsilon=1e-8
        )
        self.optimizer = optimizer.minimize(self.loss)

def train_imac_dnn_tf1_improved(data_path: str, train_batches: Tuple[int, int], val_batches: Tuple[int, int],
                               batch_size: int = 1000, epochs: int = 500, patience: int = 30):
    """改进的流式训练IMAC DNN模型 - 重点改进早停机制"""
    print(f"\n开始改进版流式训练TensorFlow 1.x DNN模型（改进早停机制）...")
    print(f"训练批次: {train_batches[0]}-{train_batches[1]} (共{train_batches[1]-train_batches[0]+1}批)")
    print(f"验证批次: {val_batches[0]}-{val_batches[1]} (共{val_batches[1]-val_batches[0]+1}批)")
    print(f"批大小: {batch_size}")
    print(f"最大轮数: {epochs}")
    print(f"早停耐心值: {patience} (减少以防止过拟合)")
    print(f"改进特性: 移除BatchNorm, He初始化, 改进早停机制防止过拟合")
    
    # 获取数据维度（从第一个文件）
    first_file = f"{data_path}train_batch_{train_batches[0]:03d}.mat"
    if not os.path.exists(first_file):
        raise FileNotFoundError(f"训练数据文件不存在: {first_file}")
    
    data = sio.loadmat(first_file)
    input_dim = data['X_batch'].shape[0]
    output_dim = data['Y_batch'].shape[0]
    del data
    
    print(f"数据维度: 输入={input_dim}, 输出={output_dim}")
    
    # 初始化改进的模型
    model = IMAC_DNN_TF1_Improved(input_dim, output_dim)
    
    # 创建会话
    sess = tf.compat.v1.Session()
    
    try:
        # 初始化变量
        sess.run(tf.compat.v1.global_variables_initializer())
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_vars = None
        
        # 改进的早停机制变量
        val_loss_history = []  # 记录验证损失历史
        overfitting_window = 5  # 检查过拟合的窗口大小
        min_delta = 1e-4  # 最小改善阈值（调整为更合理的值）
        overfitting_patience = 10  # 检测到过拟合后的额外耐心
        overfitting_detected = False
        
        start_time = time.time()
        # 按论文设置初始学习率
        current_lr = 0.001
        
        print(f"\n改进的早停策略:")
        print(f"  最小改善阈值: {min_delta}")
        print(f"  基础耐心值: {patience}")
        print(f"  过拟合检测窗口: {overfitting_window}")
        print(f"  过拟合检测后额外耐心: {overfitting_patience}")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # 训练阶段 - 流式加载
            train_loss = 0
            train_batches_count = 0
            
            if (epoch + 1) % 10 == 0 or epoch < 5:
                print(f"  Epoch {epoch+1}: 训练阶段...")
                
            for X_batch, Y_batch in data_generator(data_path, 'train', train_batches[0], train_batches[1], batch_size):
                # 训练
                _, batch_loss = sess.run([model.optimizer, model.loss], 
                                       feed_dict={model.X: X_batch, model.Y: Y_batch, 
                                                model.learning_rate: current_lr})
                
                train_loss += batch_loss
                train_batches_count += 1
                
                # 立即释放批次数据
                del X_batch, Y_batch
            
            avg_train_loss = train_loss / train_batches_count if train_batches_count > 0 else 0
            
            # 验证阶段 - 流式加载
            val_loss = 0
            val_batches_count = 0
            
            if (epoch + 1) % 10 == 0 or epoch < 5:
                print(f"  Epoch {epoch+1}: 验证阶段...")
                
            for X_batch, Y_batch in data_generator(data_path, 'train', val_batches[0], val_batches[1], batch_size):
                batch_val_loss = sess.run(model.loss, 
                                        feed_dict={model.X: X_batch, model.Y: Y_batch})
                
                val_loss += batch_val_loss
                val_batches_count += 1
                
                # 立即释放批次数据
                del X_batch, Y_batch
            
            avg_val_loss = val_loss / val_batches_count if val_batches_count > 0 else 0
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_loss_history.append(avg_val_loss)
            
            epoch_time = time.time() - epoch_start
            
            # 改进的早停检查
            improved = False
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                improved = True
                # 保存最佳模型参数
                best_model_vars = sess.run(tf.compat.v1.trainable_variables())
                if (epoch + 1) % 5 == 0 or epoch < 10:
                    print(f"    验证损失改善: {avg_val_loss:.6f} (最佳)")
            else:
                patience_counter += 1
            
            # 检测过拟合趋势
            if not overfitting_detected and len(val_loss_history) >= overfitting_window:
                # 检查最近几轮验证损失是否呈上升趋势
                recent_val_losses = val_loss_history[-overfitting_window:]
                
                # 计算趋势：如果最近的损失平均值比之前的平均值高
                if len(val_loss_history) > overfitting_window:
                    earlier_avg = np.mean(val_loss_history[-(2*overfitting_window):-overfitting_window])
                    recent_avg = np.mean(recent_val_losses)
                    
                    # 如果验证损失明显上升，检测为过拟合
                    if recent_avg > earlier_avg + min_delta:
                        overfitting_detected = True
                        print(f"    *** 检测到过拟合趋势！验证损失从 {earlier_avg:.6f} 上升到 {recent_avg:.6f} ***")
                        print(f"    *** 启动过拟合模式，耐心值重置为 {overfitting_patience} ***")
                        patience_counter = 0  # 重置计数器，但使用更小的耐心值
            
            # 学习率衰减策略（更积极）
            if patience_counter > 0 and patience_counter % 8 == 0:  # 每8轮无改善时衰减
                old_lr = current_lr
                current_lr = max(current_lr * 0.7, 1e-6)  # 更大的衰减率
                print(f"    学习率衰减: {old_lr:.6f} -> {current_lr:.6f}")
            
            # 打印进度（每5轮或前10轮）
            if (epoch + 1) % 5 == 0 or epoch < 10:
                status = "过拟合检测中" if overfitting_detected else "正常训练"
                effective_patience = overfitting_patience if overfitting_detected else patience
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"训练损失: {avg_train_loss:.6f} | "
                      f"验证损失: {avg_val_loss:.6f} | "
                      f"学习率: {current_lr:.6f} | "
                      f"耐心: {patience_counter}/{effective_patience} | "
                      f"状态: {status} | "
                      f"时间: {epoch_time:.1f}s")
            
            # 早停条件检查
            effective_patience = overfitting_patience if overfitting_detected else patience
            if patience_counter >= effective_patience:
                if overfitting_detected:
                    print(f"\n早停触发！检测到过拟合且验证损失在 {overfitting_patience} 轮内未改善")
                else:
                    print(f"\n早停触发！验证损失在 {patience} 轮内未改善超过 {min_delta}")
                break
            
            # 额外的保护机制：如果验证损失爆炸式增长
            if len(val_loss_history) > 5:
                recent_val_loss = val_loss_history[-1]
                early_val_loss = val_loss_history[-5]
                if recent_val_loss > early_val_loss * 1.5:  # 增长超过50%
                    print(f"\n验证损失爆炸式增长检测！从 {early_val_loss:.6f} 增长到 {recent_val_loss:.6f}")
                    print(f"触发紧急早停机制")
                    break
            
            # 检查内存使用
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                if memory_mb > 2000:  # 超过2GB时警告
                    print(f"    警告：内存使用较高: {memory_mb:.1f} MB")
            except ImportError:
                pass
        
        # 恢复最佳模型
        if best_model_vars:
            for var, best_var in zip(tf.compat.v1.trainable_variables(), best_model_vars):
                var.load(best_var, sess)
            print(f"已恢复最佳模型 (验证损失: {best_val_loss:.6f})")
        
        total_time = time.time() - start_time
        print(f"\n改进版流式训练完成！总耗时: {total_time/60:.1f}分钟")
        print(f"最佳验证损失: {best_val_loss:.6f}")
        print(f"实际训练轮数: {len(train_losses)}")
        print(f"过拟合检测: {'是' if overfitting_detected else '否'}")
        
        return model, train_losses, val_losses, sess
        
    except Exception as e:
        # 如果训练过程中出错，确保关闭会话
        sess.close()
        raise e

def data_generator(data_path: str, dataset_type: str, start_batch: int, end_batch: int, 
                  batch_size: int = 1000, return_H: bool = False) -> Generator[Any, None, None]:
    """流式数据生成器，避免内存不足"""
    if (end_batch - start_batch + 1) <= 10:
        print(f"创建{dataset_type}集数据生成器 (批次 {start_batch}-{end_batch})...")
    
    # 查找批次文件
    batch_files = []
    batch_idx = start_batch
    while True:
        filename = f"{data_path}{dataset_type}_batch_{batch_idx:03d}.mat"
        if not os.path.exists(filename):
            break
        batch_files.append((batch_idx, filename))
        batch_idx += 1
        if end_batch and batch_idx > end_batch:
            break
    
    if not batch_files:
        raise FileNotFoundError(f"未找到{dataset_type}集数据文件")
    
    if (end_batch - start_batch + 1) <= 10:
        print(f"找到 {len(batch_files)} 个批次文件")
    
    # 流式生成数据
    for batch_idx, filename in batch_files:
        try:
            # 加载当前批次
            data = sio.loadmat(filename)
            H_batch = data['H_batch'].astype(np.float32)
            X_batch = data['X_batch'].astype(np.float32)
            Y_batch = data['Y_batch'].astype(np.float32)
            
            # 立即释放原始数据
            del data
            
            # 分批返回数据
            num_samples = X_batch.shape[1]
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                
                X_subset = X_batch[:, i:end_idx].T
                Y_subset = Y_batch[:, i:end_idx].T
                if return_H:
                    H_subset = H_batch[:, i:end_idx].T
                    yield X_subset, Y_subset, H_subset
                else:
                    yield X_subset, Y_subset
            
            # 释放批次数据
            del H_batch, X_batch, Y_batch
            
        except Exception as e:
            print(f"加载批次 {batch_idx} 时出错: {e}")
            continue

def evaluate_dnn_performance_tf1_improved(model_path: str, data_path: str, test_batches: Tuple[int, int],
                                         var_noise: float = 1.0):
    """改进版流式评估DNN模型性能 - 从文件加载模型"""
    print(f"\n开始改进版流式评估TensorFlow 1.x DNN模型性能...")
    print(f"从路径加载模型: {model_path}")
    
    # 读取模型信息
    model_info_path = os.path.join(model_path, 'model_info.txt')
    if not os.path.exists(model_info_path):
        raise FileNotFoundError(f"模型信息文件不存在: {model_info_path}")
    
    # 解析模型信息
    model_info = {}
    with open(model_info_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                if key == 'hidden_dims':
                    # 解析列表格式 [200, 80, 80]
                    import ast
                    model_info[key] = ast.literal_eval(value)
                elif key in ['input_dim', 'output_dim']:
                    model_info[key] = int(value)
                else:
                    model_info[key] = value
    
    print(f"模型信息: 输入维度={model_info['input_dim']}, 输出维度={model_info['output_dim']}")
    print(f"隐藏层: {model_info['hidden_dims']}")
    
    # 创建新的TensorFlow会话和图
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    
    try:
        # 恢复模型
        model_file_path = os.path.join(model_path, 'model')
        saver = tf.compat.v1.train.import_meta_graph(f"{model_file_path}.meta")
        saver.restore(sess, model_file_path)
        
        print(f"模型成功从 {model_file_path} 恢复")
        
        # 获取输入输出张量
        graph = tf.compat.v1.get_default_graph()
        X = graph.get_tensor_by_name("X:0")
        Y_pred = graph.get_tensor_by_name("output/Minimum:0")  # 输出层的激活函数结果
        
        print(f"成功获取模型张量: X={X.shape}, Y_pred={Y_pred.shape}")
        
        # DNN预测（流式处理）
        y_dnn_pred = []
        batch_size = 1000
        total_prediction_time = 0
        total_samples = 0
        
        print("开始流式DNN预测，统计预测时间...")
        for X_batch, Y_batch in data_generator(data_path, 'test', test_batches[0], test_batches[1], batch_size):
            # 记录预测开始时间
            start_time = time.time()
            batch_pred = sess.run(Y_pred, feed_dict={X: X_batch})
            batch_time = time.time() - start_time
            
            # 累加时间和样本数
            total_prediction_time += batch_time
            total_samples += X_batch.shape[0]
            
            y_dnn_pred.append(batch_pred)
            
            # 立即释放批次数据
            del X_batch, Y_batch
            
            # 打印进度
            if len(y_dnn_pred) % 20 == 0:
                print(f"  预测进度: {len(y_dnn_pred)} 批次")
        
        y_dnn_pred = np.vstack(y_dnn_pred)
        
        # 计算平均预测时间
        avg_prediction_time = total_prediction_time / total_samples
        print(f"\nDNN预测时间统计:")
        print(f"  总预测时间: {total_prediction_time:.4f} 秒")
        print(f"  总样本数: {total_samples:,}")
        print(f"  平均每样本预测时间: {avg_prediction_time*1000:.4f} 毫秒")
        
        # 保存预测结果
        save_predictions_tf1_improved(y_dnn_pred, None, 'dnn_predictions_tf1_improved', avg_prediction_time)
        
        # 计算各种算法的和速率（流式处理）
        print("计算和速率...")
        
        # 权重因子（假设所有用户权重相等）
        K = y_dnn_pred.shape[1]  # 用户数量
        alpha = np.ones(K) / K
        
        # 初始化结果数组
        rates_dnn = []
        rates_wmmse = []
        rates_random = []
        rates_maxpower = []
        
        # 重新加载测试数据进行性能评估（带H）
        sample_count = 0
        for X_batch, Y_batch, H_batch in data_generator(data_path, 'test', test_batches[0], test_batches[1], batch_size, return_H=True):
            batch_size_actual = X_batch.shape[0]
            
            # 获取对应的DNN预测
            start_idx = sample_count
            end_idx = sample_count + batch_size_actual
            y_dnn_batch = y_dnn_pred[start_idx:end_idx]
            
            # 计算每个样本的和速率
            for i in range(batch_size_actual):
                H_i = H_batch[i, :].reshape(K, K)
                
                p_dnn = y_dnn_batch[i, :]
                rate_dnn = compute_sum_rate(H_i, p_dnn, var_noise, alpha)
                rates_dnn.append(rate_dnn)
                
                p_wmmse = Y_batch[i, :]
                rate_wmmse = compute_sum_rate(H_i, p_wmmse, var_noise, alpha)
                rates_wmmse.append(rate_wmmse)
                
                p_random = np.random.uniform(0, 1, K)
                rate_random = compute_sum_rate(H_i, p_random, var_noise, alpha)
                rates_random.append(rate_random)
                
                p_maxpower = np.ones(K)
                rate_maxpower = compute_sum_rate(H_i, p_maxpower, var_noise, alpha)
                rates_maxpower.append(rate_maxpower)
            
            sample_count += batch_size_actual
            
            # 释放批次数据
            del X_batch, Y_batch, H_batch
            
            if sample_count % 2000 == 0:
                print(f"  性能评估进度: {sample_count} 样本")
        
        # 转为numpy数组
        rates_dnn = np.asarray(rates_dnn)
        rates_wmmse = np.asarray(rates_wmmse)
        rates_random = np.asarray(rates_random)
        rates_maxpower = np.asarray(rates_maxpower)
        
        # 计算性能指标
        dnn_vs_wmmse_ratio = float(np.mean(rates_dnn / np.maximum(rates_wmmse, 1e-12)))
        
        results = {
            'rates_dnn': rates_dnn,
            'rates_wmmse': rates_wmmse,
            'rates_random': rates_random,
            'rates_maxpower': rates_maxpower,
            'dnn_vs_wmmse_ratio': dnn_vs_wmmse_ratio,
            'avg_prediction_time': avg_prediction_time,
            'num_test_samples': total_samples,
            'mean_rates': {
                'DNN': float(np.mean(rates_dnn)) if rates_dnn.size else 0.0,
                'WMMSE': float(np.mean(rates_wmmse)) if rates_wmmse.size else 0.0,
                'Random': float(np.mean(rates_random)) if rates_random.size else 0.0,
                'MaxPower': float(np.mean(rates_maxpower)) if rates_maxpower.size else 0.0
            }
        }
        
        print(f"\n改进版流式性能评估完成:")
        print(f"测试样本数: {total_samples:,}")
        print(f"DNN vs WMMSE 性能比: {dnn_vs_wmmse_ratio:.4f} ({dnn_vs_wmmse_ratio*100:.2f}%)")
        print(f"平均每样本预测时间: {avg_prediction_time*1000:.4f} 毫秒")
        print(f"实时性能: {1/avg_prediction_time:.1f} 样本/秒")
        
        return results
        
    except Exception as e:
        print(f"模型评估过程中出错: {e}")
        raise e
    finally:
        # 确保关闭会话
        sess.close()
        print("会话已关闭")

def compute_sum_rate(H: np.ndarray, p: np.ndarray, var_noise: float, alpha: np.ndarray) -> float:
    """计算加权和速率"""
    K = len(p)
    sum_rate = 0
    
    for k in range(K):
        # 计算用户k的SINR
        signal_power = H[k, k] * p[k]
        interference_power = 0
        
        for j in range(K):
            if j != k:
                interference_power += H[k, j] * p[j]
        
        sinr = signal_power / (interference_power + var_noise)
        
        # 计算用户k的速率
        rate_k = np.log2(1 + sinr)
        
        # 加权累加
        sum_rate += alpha[k] * rate_k
    
    return sum_rate

def plot_performance_comparison_improved(results, save_path: str = None):
    """绘制改进版性能对比图"""
    print(f"\n生成改进版性能对比图...")
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 提取速率数据
    rates_dnn = results['rates_dnn']
    rates_wmmse = results['rates_wmmse']
    rates_random = results['rates_random']
    rates_maxpower = results['rates_maxpower']
    
    # 排序用于CDF计算
    rates_dnn_sorted = np.sort(rates_dnn)
    rates_wmmse_sorted = np.sort(rates_wmmse)
    rates_random_sorted = np.sort(rates_random)
    rates_maxpower_sorted = np.sort(rates_maxpower)
    
    # CDF概率
    n = len(rates_dnn)
    cdf_prob = np.arange(1, n + 1) / n
    
    # 绘制CDF曲线
    ax.plot(rates_wmmse_sorted, cdf_prob, '--', color='purple', linewidth=2, label='WMMSE')
    ax.plot(rates_dnn_sorted, cdf_prob, '-', color='blue', linewidth=2, label='DNN (Improved Early-Stop)')
    ax.plot(rates_maxpower_sorted, cdf_prob, '--', color='orange', linewidth=2, label='Max Power')
    ax.plot(rates_random_sorted, cdf_prob, ':', color='lightblue', linewidth=2, label='Random Power')
    
    # 设置图形属性
    ax.set_xlabel('Sum-Rate (bit/sec)', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('IMAC Channel: Sum-Rate Performance CDF (Improved Early-Stopping)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # 添加性能比标注
    dnn_vs_wmmse = results['dnn_vs_wmmse_ratio']
    ax.text(0.02, 0.98, f'DNN vs WMMSE: {dnn_vs_wmmse:.4f} ({dnn_vs_wmmse*100:.2f}%)', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 添加改进信息
    ax.text(0.02, 0.88, 'Improvements:\n- Advanced Early Stopping\n- Overfitting Detection\n- Adaptive Patience', 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # 保存图形
    if save_path:
        # 保存PNG格式
        png_path = save_path
        plt.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
        print(f"改进版性能对比图已保存到: {png_path}")
        
        # 保存PDF格式
        pdf_path = save_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"改进版性能对比图已保存到: {pdf_path}")
    
    # 关闭图形以释放内存
    plt.close(fig)

def plot_training_curves_improved(train_losses: List[float], val_losses: List[float], save_path: str = None):
    """绘制改进版训练损失曲线，突出显示过拟合检测"""
    print(f"\n生成改进版训练损失曲线...")
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    epochs = range(1, len(train_losses) + 1)
    
    # 绘制损失曲线
    ax.plot(epochs, train_losses, '-o', color='blue', linewidth=2, markersize=2, label='训练损失')
    ax.plot(epochs, val_losses, '-s', color='red', linewidth=2, markersize=2, label='验证损失')
    
    # 检测过拟合点（验证损失开始明显上升的点）
    overfitting_window = 5
    if len(val_losses) > overfitting_window * 2:
        for i in range(overfitting_window, len(val_losses) - overfitting_window):
            earlier_avg = np.mean(val_losses[i-overfitting_window:i])
            recent_avg = np.mean(val_losses[i:i+overfitting_window])
            if recent_avg > earlier_avg * 1.02:  # 2%的增长
                ax.axvline(x=i+1, color='orange', linestyle='--', alpha=0.7, label='潜在过拟合点')
                break
    
    # 找到最佳验证损失点
    best_epoch = np.argmin(val_losses) + 1
    best_val_loss = min(val_losses)
    ax.axvline(x=best_epoch, color='green', linestyle=':', alpha=0.8, label=f'最佳验证损失 (Epoch {best_epoch})')
    ax.plot(best_epoch, best_val_loss, 'go', markersize=8, label=f'最佳点: {best_val_loss:.6f}')
    
    # 设置图形属性
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('损失 (MSE)', fontsize=12)
    ax.set_title('IMAC DNN 训练损失曲线 (改进早停机制)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # 设置y轴为对数刻度（如果损失差异很大）
    if max(train_losses) / min(train_losses) > 10:
        ax.set_yscale('log')
        ax.set_ylabel('损失 (MSE, 对数刻度)', fontsize=12)
    
    # 添加训练统计信息
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    total_epochs = len(train_losses)
    
    info_text = f'训练统计:\n'
    info_text += f'总轮数: {total_epochs}\n'
    info_text += f'最佳验证损失: {best_val_loss:.6f} (Epoch {best_epoch})\n'
    info_text += f'最终训练损失: {final_train_loss:.6f}\n'
    info_text += f'最终验证损失: {final_val_loss:.6f}\n'
    info_text += f'改进: 过拟合检测'
    
    ax.text(0.02, 0.98, info_text, 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图形
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
        print(f"改进版训练损失曲线已保存到: {save_path}")
        
        # 也保存PDF格式
        pdf_path = save_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"改进版训练损失曲线已保存到: {pdf_path}")
    
    # 关闭图形以释放内存
    plt.close(fig)

def save_model_tf1_improved(model: IMAC_DNN_TF1_Improved, sess: tf.compat.v1.Session, save_path: str):
    """保存改进版TensorFlow 1.x模型"""
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 保存TensorFlow格式
    tf_save_path = os.path.join(save_path, 'model')
    saver = tf.compat.v1.train.Saver()
    saver.save(sess, tf_save_path)
    print(f"改进版TensorFlow模型已保存到: {tf_save_path}")
    
    # 保存模型信息
    model_info = {
        'input_dim': model.input_dim,
        'output_dim': model.output_dim,
        'hidden_dims': model.hidden_dims,
        'model_type': 'IMAC_DNN_TF1_Improved_EarlyStopping',
        'improvements': 'No BatchNorm, He Initialization, Advanced Early Stopping, Overfitting Detection'
    }
    
    info_path = os.path.join(save_path, 'model_info.txt')
    with open(info_path, 'w', encoding='utf-8') as f:
        for key, value in model_info.items():
            f.write(f"{key}: {value}\n")
    print(f"改进版模型信息已保存到: {info_path}")
    
    print(f"改进版模型完整保存到目录: {save_path}")

def save_predictions_tf1_improved(y_pred: np.ndarray, y_true: Optional[np.ndarray], save_path: str, avg_prediction_time: float = None):
    """保存改进版DNN预测结果"""
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 保存预测结果为MAT格式
    mat_path = os.path.join(save_path, 'predictions_improved_earlystop.mat')
    mat_data = {
        'y_pred': y_pred,
        'model_type': 'IMAC_DNN_TF1_Improved_EarlyStopping',
        'prediction_time': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    if y_true is not None:
        mat_data['y_true'] = y_true
    if avg_prediction_time is not None:
        mat_data['avg_prediction_time_per_sample'] = avg_prediction_time
        mat_data['avg_prediction_time_per_sample_ms'] = avg_prediction_time * 1000
    
    sio.savemat(mat_path, mat_data)
    print(f"改进版预测结果已保存到: {mat_path}")

def main():
    """主函数 - 改进早停机制版本"""
    print("=" * 80)
    print("IMAC DNN训练和测试程序 (TensorFlow 1.x版本) - 改进早停机制版本")
    print("主要改进: 移除BatchNorm, He初始化, 高级早停机制, 过拟合检测")
    print("=" * 80)
    
    # 数据路径
    data_path = '/root/autodl-tmp/IMAC/'
    
    if not os.path.exists(data_path):
        print(f"错误：数据路径不存在: {data_path}")
        return
    
    try:
        # 使用更多训练数据（您有198个batch）
        print("\n使用改进版流式训练模式，重点改进早停机制防止过拟合...")
        
        # 训练参数（使用更多批次，按论文要求）
        train_batches = (1, 160)        # 训练集：前160批（约80%）
        val_batches = (160, 180)        # 验证集：20批（约10%）
        test_batches = (1, 2)       # 测试集：剩余18批（约10%）
        
        print(f"改进版训练配置（重点改进早停机制）:")
        print(f"  总批次数: 198")
        print(f"  训练批次: {train_batches[0]}-{train_batches[1]} ({train_batches[1]-train_batches[0]+1}批, {(train_batches[1]-train_batches[0]+1)/198*100:.1f}%)")
        print(f"  验证批次: {val_batches[0]}-{val_batches[1]} ({val_batches[1]-val_batches[0]+1}批, {(val_batches[1]-val_batches[0]+1)/198*100:.1f}%)")
        print(f"  测试批次: {test_batches[0]}-{test_batches[1]} ({test_batches[1]-test_batches[0]+1}批, {(test_batches[1]-test_batches[0]+1)/198*100:.1f}%)")
        print(f"  早停改进: 降低耐心值(50->30), 过拟合检测, 验证损失趋势分析")
        
        # # 改进版流式训练DNN模型（重点改进早停机制）
        # print("\n开始改进版流式训练DNN模型（重点改进早停机制）...")
        # model, train_losses, val_losses, sess = train_imac_dnn_tf1_improved(
        #     data_path, train_batches, val_batches, 
        #     batch_size=1000, epochs=500, patience=30  # 降低patience防止过拟合
        # )
        
        # # 创建结果保存目录
        # results_dir = 'imac_dnn_results_tf1_improved_earlystop'
        # os.makedirs(results_dir, exist_ok=True)
        # print(f"\n结果将保存到目录: {results_dir}")
        
        # # 绘制训练损失曲线（突出显示过拟合检测）
        # print("\n生成改进版训练损失曲线（突出显示过拟合检测）...")
        # training_curves_path = os.path.join(results_dir, 'imac_training_curves_tf1_improved_earlystop.png')
        # plot_training_curves_improved(train_losses, val_losses, training_curves_path)

        # # 保存训练好的模型
        # print("\n保存改进版模型...")
        # model_save_path = os.path.join(results_dir, 'model')
        # save_model_tf1_improved(model, sess, model_save_path)

        # 改进版流式评估模型性能
        print("\n开始改进版流式评估模型性能...")
        # 模型路径
        model_path = 'imac_dnn_results_tf1_improved_earlystop/model'
        results = evaluate_dnn_performance_tf1_improved(model_path, data_path, test_batches)
        
      
       
        
        # 绘制性能对比图
        print("\n生成改进版性能对比图...")
        results_dir = 'imac_dnn_results_tf1_improved_earlystop'
        plot_save_path = os.path.join(results_dir, 'imac_performance_comparison_tf1_improved_earlystop.png')
        plot_performance_comparison_improved(results, plot_save_path)
        
        # 输出最终性能结果
        print("\n" + "=" * 80)
        print("最终性能结果 (改进早停机制版本)")
        print("=" * 80)
        print(f"IMAC场景 - 改进版流式训练（重点改进早停机制）:")
        print(f"DNN vs WMMSE 性能比: {results['dnn_vs_wmmse_ratio']:.4f}")
        print(f"DNN性能: {results['dnn_vs_wmmse_ratio']*100:.2f}% of WMMSE")
        
        # 与论文目标对比
        target_performance = 0.9302  # 论文中IMAC的性能目标
        performance_gap = (results['dnn_vs_wmmse_ratio'] - target_performance) * 100
        print(f"论文目标: 93.02%")
        print(f"性能差距: {performance_gap:+.2f}个百分点")
        
        print(f"\n平均和速率 (bit/sec):")
        for name, rate in results['mean_rates'].items():
            print(f"  {name:8s}: {rate:.4f}")
        
        # 输出预测时间统计
        print(f"\n预测性能统计:")
        print(f"  测试集样本数: {results.get('num_test_samples', 0):,}")
        print(f"  平均每样本预测时间: {results.get('avg_prediction_time', 0)*1000:.4f} 毫秒")
        print(f"  实时性能: {1/results.get('avg_prediction_time', 1):.1f} 样本/秒")
        
        print(f"\n改进版模型评估完成！")
        print(f"主要改进措施 (重点改进早停机制):")
        print(f"  - 移除BatchNorm（避免过度正则化）")
        print(f"  - 使用He初始化（适合ReLU）")
        print(f"  - 降低早停耐心值（50->30，防止过拟合）")
        print(f"  - 增加过拟合检测机制（检测验证损失上升趋势）")
        print(f"  - 自适应耐心值（检测到过拟合后进一步降低到10）")
        print(f"  - 验证损失爆炸检测（紧急早停机制）")
        print(f"  - 更积极的学习率衰减（每8轮，衰减率70%）")
        print(f"  - 更合理的改善阈值（1e-4）")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()