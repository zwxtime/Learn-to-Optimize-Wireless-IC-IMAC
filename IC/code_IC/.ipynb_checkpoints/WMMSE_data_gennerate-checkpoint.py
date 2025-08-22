import numpy as np
import math
import time
import os

# 原始WMMSE算法实现（保持不变）
def WMMSE_sum_rate(p_int, H, Pmax, var_noise):
    K = np.size(p_int)
    vnew = 0
    b = np.sqrt(p_int)
    f = np.zeros(K)
    w = np.zeros(K)
    for i in range(K):
        f[i] = H[i, i] * b[i] / (np.square(H[i, :]) @ np.square(b) + var_noise)
        w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
        vnew = vnew + math.log2(w[i])

    VV = np.zeros(100)
    for iter in range(100):
        vold = vnew
        for i in range(K):
            btmp = w[i] * f[i] * H[i, i] / sum(w * np.square(f) * np.square(H[:, i]))
            b[i] = min(btmp, np.sqrt(Pmax)) + max(btmp, 0) - btmp

        vnew = 0
        for i in range(K):
            f[i] = H[i, i] * b[i] / ((np.square(H[i, :])) @ (np.square(b)) + var_noise)
            w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
            vnew = vnew + math.log2(w[i])

        VV[iter] = vnew
        if vnew - vold <= 1e-3:
            break

    p_opt = np.square(b)
    return p_opt

# 原始高斯IC数据生成函数（保持不变）
def generate_Gaussian(K, num_H, Pmax=1, Pmin=0, seed=2017):
    print('Generate Data ... (seed = %d)' % seed)
    np.random.seed(seed)
    Pini = Pmax*np.ones(K)
    var_noise = 1
    X=np.zeros((K**2,num_H))
    Y=np.zeros((K,num_H))
    total_time = 0.0
    for loop in range(num_H):
        CH = 1/np.sqrt(2)*(np.random.randn(K,K)+1j*np.random.randn(K,K))
        H=abs(CH)
        X[:,loop] = np.reshape(H, (K**2,), order="F")
        H=np.reshape(X[:,loop], (K,K), order="F")
        mid_time = time.time()
        Y[:,loop] = WMMSE_sum_rate(Pini, H, Pmax, var_noise)
        total_time = total_time + time.time() - mid_time
    return X, Y, total_time

def generate_and_save_data(K=5, total_samples=1000000, batch_size=100000, save_path="./"):
    """分批生成并保存数据到指定路径"""
    
    # 创建保存文件夹
    data_folder = os.path.join(save_path, "WMMSE_Data")
    os.makedirs(data_folder, exist_ok=True)
    
    num_batches = (total_samples + batch_size - 1) // batch_size
    print(f"开始生成 {total_samples:,} 个样本，分 {num_batches} 批处理")
    
    start_time = time.time()
    
    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, total_samples - batch_idx * batch_size)
        seed = 2017 + batch_idx
        
        # 生成数据
        X_batch, Y_batch, _ = generate_Gaussian(K, current_batch_size, seed=seed)
        
        # 保存数据
        filename = f"data_batch_{batch_idx+1:03d}.npz"
        filepath = os.path.join(data_folder, filename)
        np.savez_compressed(filepath, X=X_batch, Y=Y_batch)
        
        print(f"批次 {batch_idx+1}/{num_batches} 完成，已保存 {filename}")
    
    total_time = time.time() - start_time
    print(f"全部完成！总耗时: {total_time/60:.1f} 分钟")
    print(f"数据保存在: {data_folder}")

def load_all_data(data_folder):
    """加载所有批次的数据"""
    files = [f for f in os.listdir(data_folder) if f.startswith("data_batch_")]
    files.sort()
    
    X_list, Y_list = [], []
    for file in files:
        data = np.load(os.path.join(data_folder, file))
        X_list.append(data['X'])
        Y_list.append(data['Y'])
    
    X_all = np.concatenate(X_list, axis=1)
    Y_all = np.concatenate(Y_list, axis=1)
    return X_all, Y_all

if __name__ == "__main__":
    # 参数设置
    K = 10                    # 用户数
    total_samples = 1000000  # 总样本数
    batch_size = 100000      # 每批样本数
    save_path = "E:/data"         # 保存路径，改成你的U盘路径如 "E:/"
    
    # 生成数据
    generate_and_save_data(K, total_samples, batch_size, save_path)