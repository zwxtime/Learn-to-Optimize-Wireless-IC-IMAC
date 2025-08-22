import numpy as np
import scipy.io as sio
import time
import os
import matplotlib
matplotlib.use('Agg')   # 设置后端Agg
import matplotlib.pyplot as plt
import tensorflow as tf

# 禁用TensorFlow 2.x的行为
tf.compat.v1.disable_v2_behavior()

def calculate_sum_rate(H, p, noise_power=1.0):
    """
    计算高斯IC信道的和速率
    参数:
        H: 信道矩阵, 形状为 (K, K, num_samples)
        p: 功率分配矩阵, 形状为 (K, num_samples)
        noise_power: 噪声功率 (默认1.0)
    返回:
        sum_rates: 每个样本的和速率, 形状为 (num_samples,)
    """
    K, _, num_samples = H.shape
    sum_rates = np.zeros(num_samples)
    
    for i in range(num_samples):
        rate = 0.0
        for k in range(K):
            signal_power = np.abs(H[k, k, i])**2 * p[k, i]
            interference = 0.0
            for j in range(K):
                if j != k:
                    interference += np.abs(H[k, j, i])**2 * p[j, i]
            sinr = signal_power / (interference + noise_power)
            rate += np.log2(1 + sinr)
        sum_rates[i] = rate
    
    return sum_rates
def plot_cdf(data_dict, title, save_path, K):
    """
    绘制和速率的CDF曲线 - 论文风格版本
    参数:
        data_dict: 字典, 键为算法名称, 值为和速率数组
        title: 图表标题
        save_path: 图片保存路径
        K: 用户数
    """
    # 设置matplotlib参数以获得更好的图形质量
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.8,
        'lines.linewidth': 2.0
    })
    
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    # 定义颜色和线型映射 - 与论文图片一致
    style_map = {
        'WMMSE': {'color': '#E6B3E6', 'linestyle': '--', 'linewidth': 2.5},  # 浅紫色虚线
        'DNN': {'color': '#4169E1', 'linestyle': '-', 'linewidth': 2.5},     # 蓝色实线
        'Max Power': {'color': '#FF8C00', 'linestyle': '-.', 'linewidth': 2.5}, # 橙色点划线
        'Random Power': {'color': '#32CD32', 'linestyle': ':', 'linewidth': 3.0}  # 绿色点线，稍粗一点
    }
    
    # 为每种算法绘制CDF曲线
    for label, rates in data_dict.items():
        if label in style_map:
            sorted_rates = np.sort(rates)
            cdf = np.arange(1, len(sorted_rates)+1) / len(sorted_rates)
            
            style = style_map[label]
            ax.plot(sorted_rates, cdf, 
                   label=label,
                   color=style['color'],
                   linestyle=style['linestyle'],
                   linewidth=style['linewidth'])
    
    # 设置坐标轴标签
    ax.set_xlabel('sum-rate (bit/sec)', fontsize=14, fontweight='normal')
    ax.set_ylabel('cumulative probability', fontsize=14, fontweight='normal')
    
    # 设置标题
    ax.set_title('Empirical CDF', fontsize=16, fontweight='bold', pad=15)
    
    # 设置坐标轴范围
    max_rate = max([np.max(rates) for rates in data_dict.values()])
    ax.set_xlim(0, max_rate * 1.05)
    ax.set_ylim(0, 1)
    
    # 设置坐标轴刻度
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # 根据最大值设置x轴刻度
    if max_rate <= 7:
        ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
    else:
        # 动态设置x轴刻度
        x_ticks = np.linspace(0, int(max_rate), min(8, int(max_rate)+1))
        ax.set_xticks(x_ticks)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linewidth=0.8)
    
    # 设置图例 - 位置和样式与论文一致
    legend = ax.legend(loc='lower right', 
                      fontsize=11,
                      frameon=True,
                      fancybox=True,
                      framealpha=0.9,
                      edgecolor='black',
                      facecolor='white')
    legend.get_frame().set_linewidth(1.0)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    fig_path = os.path.join(save_path, f'CDF_GaussianIC_K{K}.png')
    eps_path = os.path.join(save_path, f'CDF_GaussianIC_K{K}.eps')
    pdf_path = os.path.join(save_path, f'CDF_GaussianIC_K{K}.pdf')
    
    # 保存多种格式
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(eps_path, format='eps', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.show()  # 显示图形
    plt.close()
    
    print(f"CDF plot saved: {fig_path}")
    print(f"EPS format saved: {eps_path}")
    print(f"PDF format saved: {pdf_path}")

# 如果你想要更进一步的定制，可以使用这个高级版本
def plot_cdf_advanced(data_dict, title, save_path, K):
    """
    绘制和速率的CDF曲线 - 高级定制版本
    """
    # 设置全局字体和样式
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.linewidth': 0.8,
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': False,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'black',
        'figure.facecolor': 'white'
    })
    
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    # 更精确的颜色匹配
    style_map = {
        'WMMSE': {'color': '#D8BFD8', 'linestyle': '--', 'linewidth': 2.8, 'alpha': 0.9},
        'DNN': {'color': '#1E90FF', 'linestyle': '-', 'linewidth': 2.8, 'alpha': 1.0},
        'Max Power': {'color': '#FF6347', 'linestyle': '-.', 'linewidth': 2.5, 'alpha': 0.9},
        'Random Power': {'color': '#32CD32', 'linestyle': ':', 'linewidth': 3.2, 'alpha': 0.8}
    }
def ini_weights(n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output):
    """权重初始化 - 使用Glorot初始化"""
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=np.sqrt(2.0/(n_input+n_hidden_1)))),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=np.sqrt(2.0/(n_hidden_1+n_hidden_2)))),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev=np.sqrt(2.0/(n_hidden_2+n_hidden_3)))),
        'out': tf.Variable(tf.random_normal([n_hidden_3, n_output], stddev=np.sqrt(2.0/(n_hidden_3+n_output))))
    }
    biases = {
        'b1': tf.Variable(tf.zeros([n_hidden_1])),
        'b2': tf.Variable(tf.zeros([n_hidden_2])),
        'b3': tf.Variable(tf.zeros([n_hidden_3])),
        'out': tf.Variable(tf.zeros([n_output]))
    }
    return weights, biases

def multilayer_perceptron(x, weights, biases, is_training, P_max=1.0):
    """DNN网络结构 - 添加功率约束输出"""
    # Layer 1
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    
    # Layer 2
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    
    # Layer 3
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    
    # Output layer with power constraint (0 ≤ p ≤ P_max)
    out_layer = tf.add(tf.matmul(layer_3, weights['out']), biases['out'])
    out_layer = tf.minimum(tf.maximum(out_layer, 0), P_max)
    return out_layer

def train(X, Y, location, P_max=1.0, training_epochs=300, batch_size=1000, LR=0.001, 
          n_hidden_1=200, n_hidden_2=80, n_hidden_3=80, traintestsplit=0.01, patience=10):
    """DNN训练函数 - 添加早停机制和功率约束"""
    num_total = X.shape[1]
    num_val = int(num_total * traintestsplit)
    num_train = num_total - num_val
    n_input = X.shape[0]
    n_output = Y.shape[0]
    
    X_train = np.transpose(X[:, 0:num_train])
    Y_train = np.transpose(Y[:, 0:num_train])
    X_val = np.transpose(X[:, num_train:num_total])
    Y_val = np.transpose(Y[:, num_train:num_total])
    
    print(f'Train samples: {num_train}, Validation samples: {num_val}')

    # 创建占位符
    x = tf.compat.v1.placeholder(tf.float32, [None, n_input])
    y = tf.compat.v1.placeholder(tf.float32, [None, n_output])
    is_training = tf.compat.v1.placeholder(tf.bool)
    learning_rate = tf.compat.v1.placeholder(tf.float32, [])
    
    # 初始化权重和构建网络
    weights, biases = ini_weights(n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output)
    pred = multilayer_perceptron(x, weights, biases, is_training, P_max)
    
    # 损失函数和优化器
    cost = tf.reduce_mean(tf.square(pred - y))
    optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate, decay=0.9).minimize(cost)
    
    # 初始化变量和保存器
    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()
    
    # 训练记录
    MSETime = np.zeros((training_epochs, 3))
    best_val_loss = float('inf')
    best_epoch = 0
    wait = 0  # 早停计数器
    
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        start_time = time.time()
        
        for epoch in range(training_epochs):
            # 洗牌数据
            indices = np.arange(num_train)
            np.random.shuffle(indices)
            
            # Mini-batch 训练
            avg_train_loss = 0
            num_batches = 0
            
            for start_idx in range(0, num_train, batch_size):
                end_idx = min(start_idx + batch_size, num_train)
                batch_idx = indices[start_idx:end_idx]
                
                # 执行优化
                _, batch_loss = sess.run([optimizer, cost], feed_dict={
                    x: X_train[batch_idx], 
                    y: Y_train[batch_idx],
                    learning_rate: LR,
                    is_training: True
                })
                avg_train_loss += batch_loss
                num_batches += 1
            
            avg_train_loss /= num_batches
            
            # 计算验证损失
            val_loss = sess.run(cost, feed_dict={
                x: X_val, 
                y: Y_val,
                is_training: False
            })
            
            # 记录指标
            MSETime[epoch, 0] = avg_train_loss
            MSETime[epoch, 1] = val_loss
            MSETime[epoch, 2] = time.time() - start_time
            
            # 打印进度
            if epoch % max(1, int(training_epochs/10)) == 0:
                print(f'Epoch {epoch}: Train Loss={avg_train_loss:.6f}, Val Loss={val_loss:.6f}')
            
            # 早停机制
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                wait = 0
                # 保存最佳模型
                saver.save(sess, location + '_best')
            else:
                wait += 1
                if wait >= patience:


                    
                    print(f'Validation loss not improving for {wait} epochs. Early stopping at epoch {epoch}')
                    print(f'Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}')
                    break
        
        # 恢复最佳模型
        saver.restore(sess, location + '_best')
        saver.save(sess, location)
        
        # 最终验证损失
        final_val_loss = sess.run(cost, feed_dict={x: X_val, y: Y_val, is_training: False})
        print(f"Final validation loss: {final_val_loss:.6f}")
        print(f"Total training time: {time.time()-start_time:.2f} seconds")
        
        # 保存训练指标
        sio.savemat(f'MSETime_{n_output}_{batch_size}_{int(LR*10000)}.mat', {
            'train': MSETime[:, 0],
            'validation': MSETime[:, 1],
            'time': MSETime[:, 2]
        })
    
    return 0

def test(X, model_location, save_name, n_input, n_output, n_hidden_1=200, 
         n_hidden_2=80, n_hidden_3=80, P_max=1.0, binary=0):
    """DNN测试函数 - 添加功率约束"""
    tf.compat.v1.reset_default_graph()
    x = tf.compat.v1.placeholder(tf.float32, [None, n_input])
    is_training = tf.compat.v1.placeholder(tf.bool)
    
    # 初始化权重和构建网络
    weights, biases = ini_weights(n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output)
    pred = multilayer_perceptron(x, weights, biases, is_training, P_max)
    
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, model_location)
        start_time = time.time()
        y_pred = sess.run(pred, feed_dict={x: np.transpose(X), is_training: False})
        test_time = time.time() - start_time
        
        # 对于高斯IC信道，功率分配通常是二值的
        if binary == 1:
            y_pred = np.where(y_pred >= 0.5, P_max, 0)
        
        # 保存预测结果
        sio.savemat(save_name, {'pred': y_pred})
    
    return test_time

def load_all_data(data_folder):
    """加载所有批次的数据"""
    files = [f for f in os.listdir(data_folder) if f.startswith("data_batch_")]
    files.sort()

    X_list, Y_list = [], []
    for file in files:
        print(f"Loading: {file}")
        data = np.load(os.path.join(data_folder, file))
        X_list.append(data['X'])
        Y_list.append(data['Y'])

    X_all = np.concatenate(X_list, axis=1)
    Y_all = np.concatenate(Y_list, axis=1)
    print(f"Data loaded. Total samples: {X_all.shape[1]:,}")
    return X_all, Y_all

def split_data(X, Y, train_samples=990000, test_samples=10000):
    """
    分割数据为训练集和测试集
    """
    total_samples = X.shape[1]

    if train_samples + test_samples > total_samples:
        raise ValueError(f"Requested samples {train_samples + test_samples} exceed available {total_samples}")

    # 训练集
    X_train = X[:, :train_samples]
    Y_train = Y[:, :train_samples]

    # 测试集
    X_test = X[:, train_samples:train_samples + test_samples]
    Y_test = Y[:, train_samples:train_samples + test_samples]

    print(f"Data split:")
    print(f"Training set: {X_train.shape[1]:,} samples")
    print(f"Testing set: {X_test.shape[1]:,} samples")

    return X_train, Y_train, X_test, Y_test

def train_gaussian_ic(data_folder, save_path="./", model_name="gaussian_ic_model", K=10):
    # 参数设置（与论文一致）
    training_epochs = 300  # 训练轮数
    batch_size = 1000      # 批大小
    learning_rate = 0.001  # 学习率
    train_samples = 990000 # 训练样本数
    test_samples = 10000   # 测试样本数
    P_max = 1.0            # 最大发射功率

    # 网络结构参数（与论文一致）
    n_hidden_1 = 200
    n_hidden_2 = 80
    n_hidden_3 = 80
    traintestsplit = 0.01  # 验证集比例

    print('=' * 60)
    print('Training DNN for Gaussian IC Channel')
    print(f'Users: {K}, Training samples: {train_samples:,}, Test samples: {test_samples:,}')
    print(f'Epochs: {training_epochs}, Batch size: {batch_size}')
    print('=' * 60)

    # 1. 加载数据
    print("\nStep 1: Loading data")
    X_all, Y_all = load_all_data(data_folder)

    # 2. 分割数据
    print("\nStep 2: Splitting data")
    X_train, Y_train, X_test, Y_test = split_data(X_all, Y_all, train_samples, test_samples)

    # 3. 创建模型保存目录
    model_folder = os.path.join(save_path, "DNNmodel")
    os.makedirs(model_folder, exist_ok=True)
    model_location = os.path.join(model_folder, f"{model_name}.ckpt")

#     # 4. 训练模型
#     print("\nStep 3: Training DNN model")
#     print(f"Model will be saved to: {model_location}")

#     train_start_time = time.time()

#     train(X_train, Y_train, model_location,
#           P_max=P_max,
#           training_epochs=training_epochs,
#           batch_size=batch_size,
#           LR=learning_rate,
#           n_hidden_1=n_hidden_1,
#           n_hidden_2=n_hidden_2,
#           n_hidden_3=n_hidden_3,
#           traintestsplit=traintestsplit,
#           patience=10)

#     total_train_time = time.time() - train_start_time
#     print(f"\nTraining completed! Total time: {total_train_time / 60:.1f} minutes")

    # 5. 测试模型
    print("\nStep 4: Testing model performance")
    prediction_file = os.path.join(save_path, f"Prediction_GaussianIC_K{K}")
    test_time = test(X_test, model_location, prediction_file,
                     n_input=K*K, n_output=K,
                     n_hidden_1=n_hidden_1,
                     n_hidden_2=n_hidden_2,
                     n_hidden_3=n_hidden_3,
                     P_max=P_max,
                     binary=1)  # 高斯IC信道使用二值功率分配

    print(f"DNN inference time: {test_time:.4f} sec for {test_samples:,} samples")
    print(f"Average per-sample time: {test_time / test_samples * 1000:.4f} ms")
    
    # 6. 性能比较和绘图
    print("\nStep 5: Performance comparison and plotting")
    # 定义预测结果文件路径
    #prediction_file = os.path.join(save_path, f"Prediction_GaussianIC_K{K}")
    # 加载预测结果
    dnn_pred = sio.loadmat(prediction_file)['pred'].T  # 转置为(K, num_samples)
    
    # 将信道数据重塑为(K, K, num_samples)
    H_test = X_test.reshape((K, K, -1), order='F')
    
    # 计算各种方案的和速率
    print("Calculating sum-rates for different schemes...")
    
    # DNN方案
    dnn_rates = calculate_sum_rate(H_test, dnn_pred)
    
    # WMMSE方案（使用标签作为真实值）
    wmmse_rates = calculate_sum_rate(H_test, Y_test)
    
    # 最大功率方案（所有用户都使用最大功率）
    max_power = P_max * np.ones_like(Y_test)
    max_power_rates = calculate_sum_rate(H_test, max_power)
    
    # 随机功率方案
    random_power = np.random.uniform(0, P_max, size=Y_test.shape)
    random_rates = calculate_sum_rate(H_test, random_power)
    
    # 创建数据字典
    rates_data = {
        'WMMSE': wmmse_rates,
        'DNN': dnn_rates,
        'Max Power': max_power_rates,
        'Random Power': random_rates
    }
    
    # 计算并打印DNN相对于WMMSE的准确率
    dnn_accuracy = np.mean(dnn_rates / wmmse_rates)
    print(f"DNN achieves {dnn_accuracy*100:.2f}% of WMMSE sum-rate on average")
    
    # 绘制CDF曲线
    plot_cdf(rates_data, "Sum-Rate Performance Comparison", save_path, K)
    
    # 7. 绘制训练曲线
    print("\nStep 6: Plotting training curves")
    #plot_training_curves(K, batch_size, learning_rate, save_path)

    print("\n" + "=" * 60)
    print("Training pipeline completed!")
    print(f"Model saved at: {model_location}")
    print(f"Predictions saved: {prediction_file}.mat")
    print("=" * 60)

if __name__ == "__main__":
    # 配置参数 - 高斯IC信道
    data_folder = "/root/mycode/data_IC"  # 数据文件夹路径
    save_path = "/root/mycode/results_IC"  # 结果保存路径
    model_name = "gaussian_ic_model"  # 模型名称
    K = 10  # 用户数（高斯IC信道）

    # 检查数据文件夹是否存在
    if not os.path.exists(data_folder):
        print(f"Error: Data folder not found: {data_folder}")
        print("Please generate data first or check path settings")
    else:
        # 确保结果目录存在
        os.makedirs(save_path, exist_ok=True)
        
        # 开始训练
        train_gaussian_ic(data_folder, save_path, model_name, K)