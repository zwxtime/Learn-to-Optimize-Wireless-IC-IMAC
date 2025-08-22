#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的Python代码，用于大规模生成IMAC数据集
针对100万样本生成进行了性能优化
包含批量存储和性能监控功能

MATLAB代码转Python实现
包含WMMSE优化算法和信道生成功能
"""

# 为避免与多进程冲突以及降低过度线程化，必须在导入numpy前设置环境变量
import os as _os
_os.environ.setdefault('OMP_NUM_THREADS', '1')
_os.environ.setdefault('MKL_NUM_THREADS', '1')
_os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
_os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np
import scipy.io as sio
import os
import time
from typing import Tuple, Dict, Any, List
import warnings
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings('ignore')

def generate_large_imac_dataset(workers: int = 1, max_iter: int = 150, tolerance: float = 1e-4, skip_estimate: bool = False,
                               batch_size: int = 10000, compress: bool = True, verbose_worker: bool = False, start_batch: int = 1):
    """主函数：生成大规模IMAC数据集"""
    # 参数设置
    total_samples = 1000000
    train_samples = 990000
    test_samples = 10000
    
    # IMAC参数
    num_BS = 24
    num_User = 3
    R = 100
    minR_ratio = 0.2
    var_noise = 1
    
    # 存储路径
    save_path = '/root/autodl-tmp/IMAC/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print('开始生成IMAC数据集...', flush=True)
    print(f'总样本数: {total_samples} (训练集: {train_samples}, 测试集: {test_samples})', flush=True)
    print(f'批大小: {batch_size}', flush=True)
    print(f'存储路径: {save_path}', flush=True)
    
    # 预估生成时间
    estimated_time = None
    if not skip_estimate:
        print('\n=== 预估生成时间 ===', flush=True)
        estimated_time = estimate_generation_time(
            num_BS, num_User, R, minR_ratio, var_noise, total_samples, batch_size,
            max_iter=max_iter, tolerance=tolerance
        )
    
    # 生成训练集
    print('\n=== 生成训练集 ===', flush=True)
    generate_dataset_batches(
        total_samples=train_samples,
        batch_size=batch_size,
        num_BS=num_BS,
        num_User=num_User,
        R=R,
        minR_ratio=minR_ratio,
        var_noise=var_noise,
        save_path=save_path,
        dataset_type='train',
        estimated_time_per_sample=(estimated_time['time_per_sample'] if estimated_time else None),
        workers=workers,
        max_iter=max_iter,
        tolerance=tolerance,
        compress=compress,
        verbose_worker=verbose_worker,
        start_batch=start_batch,
    )
    
    # 生成测试集
    print('\n=== 生成测试集 ===', flush=True)
    generate_dataset_batches(
        total_samples=test_samples,
        batch_size=batch_size,
        num_BS=num_BS,
        num_User=num_User,
        R=R,
        minR_ratio=minR_ratio,
        var_noise=var_noise,
        save_path=save_path,
        dataset_type='test',
        estimated_time_per_sample=(estimated_time['time_per_sample'] if estimated_time else None),
        workers=workers,
        max_iter=max_iter,
        tolerance=tolerance,
        compress=compress,
        verbose_worker=verbose_worker,
        start_batch=1,  # 测试集总是从第1批开始
    )
    
    print('\n数据集生成完成！')

def estimate_generation_time(num_BS: int, num_User: int, R: int, minR_ratio: float, 
                           var_noise: float, total_samples: int, batch_size: int,
                           max_iter: int = 150, tolerance: float = 1e-4) -> Dict[str, float]:
    """预估数据生成时间"""
    print('正在运行小规模测试以预估生成时间...')
    
    # 测试参数（多个规模）
    test_samples = [10, 50, 100]  # 不同的测试样本数
    test_results = []
    
    for test_num in test_samples:
        print(f'  测试 {test_num} 个样本...', end='')
        
        # 运行多次取平均值以提高准确性
        times = []
        for trial in range(3):  # 运行3次
            start_time = time.time()
            _, _, _, wmmse_time = Generate_IMAC_function_optimized(
                num_BS, num_User, test_num, R, minR_ratio, 
                42 + trial, var_noise, max_iter=max_iter, tolerance=tolerance  # 不同的随机种子
            )
            total_time = time.time() - start_time
            times.append({
                'total': total_time,
                'wmmse': wmmse_time,
                'avg_per_sample': total_time / test_num,
                'avg_wmmse_per_sample': wmmse_time / test_num
            })
        
        # 计算平均值
        avg_total = np.mean([t['total'] for t in times])
        avg_wmmse = np.mean([t['wmmse'] for t in times])
        avg_per_sample = avg_total / test_num
        avg_wmmse_per_sample = avg_wmmse / test_num
        
        test_results.append({
            'samples': test_num,
            'avg_per_sample': avg_per_sample,
            'avg_wmmse_per_sample': avg_wmmse_per_sample,
            'wmmse_ratio': avg_wmmse / avg_total
        })
        
        print(f' 平均每样本: {avg_per_sample:.4f}秒')
    
    # 使用线性回归预测（考虑到可能的非线性关系）
    samples_array = np.array([r['samples'] for r in test_results])
    times_array = np.array([r['avg_per_sample'] for r in test_results])
    wmmse_times_array = np.array([r['avg_wmmse_per_sample'] for r in test_results])
    
    # 计算加权平均（更重视大样本的结果）
    weights = samples_array / np.sum(samples_array)
    estimated_time_per_sample = np.average(times_array, weights=weights)
    estimated_wmmse_per_sample = np.average(wmmse_times_array, weights=weights)
    
    # 添加缓冲因子（考虑大规模运行时的额外开销）
    buffer_factor = 1.1  # 10%的缓冲
    estimated_time_per_sample *= buffer_factor
    estimated_wmmse_per_sample *= buffer_factor
    
    # 计算总时间预估
    total_estimated_time = estimated_time_per_sample * total_samples
    total_wmmse_time = estimated_wmmse_per_sample * total_samples
    
    # 计算批次数和I/O时间
    num_batches = int(np.ceil(total_samples / batch_size))
    estimated_io_time = num_batches * 2  # 假设每批保存需要2秒
    total_with_io = total_estimated_time + estimated_io_time
    
    # 显示预估结果
    print('\n时间预估结果:')
    print(f'  每样本平均时间: {estimated_time_per_sample:.4f}秒')
    print(f'  其中WMMSE时间: {estimated_wmmse_per_sample:.4f}秒 ({estimated_wmmse_per_sample/estimated_time_per_sample*100:.1f}%)')
    print(f'  ')
    print(f'  总计算时间: {total_estimated_time/3600:.1f}小时 ({total_estimated_time/60:.1f}分钟)')
    print(f'  其中WMMSE时间: {total_wmmse_time/3600:.1f}小时')
    print(f'  预估I/O时间: {estimated_io_time/60:.1f}分钟')
    print(f'  总预估时间: {total_with_io/3600:.1f}小时 ({total_with_io/60:.1f}分钟)')
    print(f'  ')
    print(f'  批次数量: {num_batches}')
    print(f'  预计完成时间: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + total_with_io))}')
    
    # 显示进度提醒
    milestones = [0.1, 0.25, 0.5, 0.75, 0.9]
    print(f'  ')
    print('进度里程碑预估:')
    for milestone in milestones:
        milestone_time = total_with_io * milestone
        completion_time = time.time() + milestone_time
        print(f'    {milestone*100:3.0f}%完成: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(completion_time))} '
              f'(约{milestone_time/3600:.1f}小时后)')
    
    # 性能建议
    print(f'  ')
    print('性能建议:')
    if estimated_time_per_sample > 0.01:
        print('  ⚠️  每样本时间较长，建议在性能较好的机器上运行')
    if total_with_io > 12*3600:  # 超过12小时
        print('  ⚠️  预计生成时间超过12小时，建议分批运行或降低样本数')
    if estimated_wmmse_per_sample / estimated_time_per_sample > 0.8:
        print('  💡 WMMSE算法占用大部分时间，可考虑算法优化')
    
    print('  ✅ 可以调整batch_size来平衡内存使用和I/O效率')
    
    # 询问用户是否继续
    try:
        user_input = input('\n是否继续生成完整数据集？(y/n): ').lower().strip()
        if user_input not in ['y', 'yes', '是', '']:
            print('用户取消生成，程序退出。')
            exit(0)
    except KeyboardInterrupt:
        print('\n用户中断，程序退出。')
        exit(0)
    
    return {
        'time_per_sample': estimated_time_per_sample,
        'wmmse_per_sample': estimated_wmmse_per_sample,
        'total_time': total_with_io,
        'compute_time': total_estimated_time,
        'io_time': estimated_io_time
    }

def generate_dataset_batches(total_samples: int, batch_size: int, num_BS: int, 
                           num_User: int, R: int, minR_ratio: float, 
                           var_noise: float, save_path: str, dataset_type: str,
                           estimated_time_per_sample: float = None,
                           workers: int = 1,
                           max_iter: int = 150,
                           tolerance: float = 1e-4,
                           compress: bool = True,
                           verbose_worker: bool = False,
                           start_batch: int = 1):
    """批量生成数据集（支持并行、多参数控制）"""
    num_batches = int(np.ceil(total_samples / batch_size))
    total_wmmse_time = 0
    total_generation_time = 0
    
    # 用于自适应时间预测
    actual_times = []
    
    print(f'开始生成 {dataset_type}集: {total_samples} 样本，分 {num_batches} 批', flush=True)
    if start_batch > 1:
        print(f'断点续传: 从第 {start_batch} 批开始 (跳过前 {start_batch-1} 批)', flush=True)
        remaining_samples = total_samples - (start_batch - 1) * batch_size
        if remaining_samples > 0:
            print(f'剩余样本数: {remaining_samples}', flush=True)
    if estimated_time_per_sample:
        estimated_total_time = estimated_time_per_sample * total_samples
        print(f'基于预估，预计总时间: {estimated_total_time/60:.1f}分钟', flush=True)
    print('-' * 60, flush=True)
    
    overall_start_time = time.time()
    
    if workers <= 1:
        for batch_idx in range(start_batch, num_batches + 1):
            if batch_idx == num_batches:
                current_batch_size = total_samples - (batch_idx - 1) * batch_size
            else:
                current_batch_size = batch_size

            print(f'[批次 {batch_idx:3d}/{num_batches}] 正在生成 {current_batch_size:5d} 样本...', end='', flush=True)
            stats = _generate_and_save_batch(
                batch_idx=batch_idx,
                current_batch_size=current_batch_size,
                num_BS=num_BS,
                num_User=num_User,
                R=R,
                minR_ratio=minR_ratio,
                var_noise=var_noise,
                save_path=save_path,
                dataset_type=dataset_type,
                max_iter=max_iter,
                tolerance=tolerance,
                compress=compress,
                verbose_worker=verbose_worker,
            )

            # 累计时间统计
            total_wmmse_time += stats['wmmse_time']
            total_generation_time += stats['generation_time']

            # 记录实际时间用于自适应预测
            actual_time_per_sample = stats['generation_time'] / current_batch_size
            actual_times.append(actual_time_per_sample)

            # 完成度
            completed_samples = batch_idx * batch_size if batch_idx < num_batches else total_samples
            progress_percent = (completed_samples / total_samples) * 100
            print(f' 完成! [{progress_percent:5.1f}%]', flush=True)

            # 显示时间信息
            print(f'         生成时间: {stats["generation_time"]:6.2f}秒 | 保存时间: {stats["save_time"]:5.2f}秒 | 每样本: {actual_time_per_sample:.4f}秒', flush=True)

            # 自适应时间预测
            if len(actual_times) >= 3:
                weights = np.array([0.2, 0.3, 0.5])
                recent_avg = np.average(actual_times[-3:], weights=weights)
            elif len(actual_times) >= 2:
                recent_avg = np.mean(actual_times[-2:])
            else:
                recent_avg = actual_times[0] if actual_times else estimated_time_per_sample

            if batch_idx < num_batches and recent_avg:
                remaining_samples = total_samples - completed_samples
                estimated_remaining_time = remaining_samples * recent_avg
                completion_time = time.time() + estimated_remaining_time
                completion_str = time.strftime("%H:%M:%S", time.localtime(completion_time))
                print(f'         剩余时间: {estimated_remaining_time/60:6.1f}分钟 | 预计完成: {completion_str}', flush=True)
                if estimated_time_per_sample and len(actual_times) >= 2:
                    performance_ratio = recent_avg / estimated_time_per_sample
                    if performance_ratio < 0.9:
                        print(f'         🚀 性能比预期好 {(1-performance_ratio)*100:.0f}%', flush=True)
                    elif performance_ratio > 1.1:
                        print(f'         🐌 性能比预期慢 {(performance_ratio-1)*100:.0f}%', flush=True)

            file_size_mb = os.path.getsize(stats['filename']) / (1024 * 1024)
            print(f'         文件大小: {file_size_mb:6.1f}MB | 文件: {os.path.basename(stats["filename"])}', flush=True)
            print('', flush=True)
    else:
        # 并行执行
        print(f'并行启用: {workers} 进程', flush=True)
        completed_samples = (start_batch - 1) * batch_size  # 已完成的样本数
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for batch_idx in range(start_batch, num_batches + 1):
                if batch_idx == num_batches:
                    current_batch_size = total_samples - (batch_idx - 1) * batch_size
                else:
                    current_batch_size = batch_size
                print(f'提交批次 {batch_idx:3d}/{num_batches} | 样本 {current_batch_size}', flush=True)
                futures.append(executor.submit(
                    _generate_and_save_batch,
                    batch_idx,
                    current_batch_size,
                    num_BS,
                    num_User,
                    R,
                    minR_ratio,
                    var_noise,
                    save_path,
                    dataset_type,
                    max_iter,
                    tolerance,
                    compress,
                    verbose_worker,
                ))

            for idx, future in enumerate(as_completed(futures), start=1):
                try:
                    stats = future.result()
                except Exception as e:
                    print(f'子进程异常: {e}', flush=True)
                    raise
                completed_samples += stats['current_batch_size']
                total_wmmse_time += stats['wmmse_time']
                total_generation_time += stats['generation_time']

                progress_percent = (completed_samples / total_samples) * 100
                print(f'[批次 {stats["batch_idx"]:3d}/{num_batches}] 完成! [{progress_percent:5.1f}%] | '
                      f'生成: {stats["generation_time"]:6.2f}s | 保存: {stats["save_time"]:5.2f}s | '
                      f'每样本: {stats["generation_time"]/stats["current_batch_size"]:.4f}s | '
                      f'文件: {os.path.basename(stats["filename"])}', flush=True)

                # 自适应时间预测更新
                actual_times.append(stats['generation_time'] / stats['current_batch_size'])
                if len(actual_times) >= 3:
                    weights = np.array([0.2, 0.3, 0.5])
                    recent_avg = np.average(actual_times[-3:], weights=weights)
                elif len(actual_times) >= 2:
                    recent_avg = np.mean(actual_times[-2:])
                else:
                    recent_avg = actual_times[0] if actual_times else estimated_time_per_sample

                if completed_samples < total_samples and recent_avg:
                    remaining_samples = total_samples - completed_samples
                    estimated_remaining_time = remaining_samples * recent_avg
                    completion_time = time.time() + estimated_remaining_time
                    completion_str = time.strftime("%H:%M:%S", time.localtime(completion_time))
                    print(f'         剩余时间: {estimated_remaining_time/60:6.1f}分钟 | 预计完成: {completion_str}', flush=True)
                    if estimated_time_per_sample and len(actual_times) >= 2:
                        performance_ratio = recent_avg / estimated_time_per_sample
                        if performance_ratio < 0.9:
                            print(f'         🚀 性能比预期好 {(1-performance_ratio)*100:.0f}%', flush=True)
                        elif performance_ratio > 1.1:
                            print(f'         🐌 性能比预期慢 {(performance_ratio-1)*100:.0f}%', flush=True)
                print('', flush=True)
    
    # 总体统计
    overall_time = time.time() - overall_start_time
    avg_wmmse_time_total = total_wmmse_time / total_samples
    avg_generation_time_total = total_generation_time / total_samples
    avg_overall_time = overall_time / total_samples
    
    print('=' * 60)
    print(f'{dataset_type.upper()}集生成完成! 🎉')
    print(f'总样本数: {total_samples:,}')
    print(f'总耗时: {overall_time/60:.2f}分钟 ({overall_time/3600:.2f}小时)')
    print(f'纯计算时间: {total_generation_time/60:.2f}分钟')
    print(f'平均每样本: {avg_overall_time:.4f}秒 (含I/O)')
    print(f'平均每样本: {avg_generation_time_total:.4f}秒 (纯计算)')
    print(f'平均WMMSE时间: {avg_wmmse_time_total:.4f}秒')
    print(f'WMMSE占比: {(total_wmmse_time / total_generation_time) * 100:.1f}%')
    
    # 性能评估
    if estimated_time_per_sample:
        actual_vs_estimated = avg_generation_time_total / estimated_time_per_sample
        print(f'实际vs预估: {actual_vs_estimated:.2f}x '
              f'({"快于" if actual_vs_estimated < 1 else "慢于"}预期)')
    
    print('=' * 60)

def Generate_IMAC_function_optimized(num_BS: int, num_User: int, num_H: int, 
                                   R: int, minR_ratio: float, seed: int, 
                                   var_noise: float, max_iter: int = 150, tolerance: float = 1e-4,
                                   verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """优化的IMAC函数生成"""
    np.random.seed(seed)
    K = num_User * num_BS
    
    # 预分配内存
    X = np.zeros((K * num_BS, num_H), dtype=np.float32)
    Y = np.zeros((K, num_H), dtype=np.float32)
    H = np.zeros((K * K, num_H), dtype=np.float32)
    
    # 预计算Cell结构（避免重复计算）
    Cell = setup_cell_structure(num_BS, R)
    
    total_wmmse_time = 0
    
    # 批量生成信道和计算WMMSE
    progress_interval = max(1, num_H // 20)  # 5% 一次
    for i in range(num_H):
        # 生成信道矩阵 (单样本)
        H_eq = generate_IBC_channel_optimized(num_User, R, num_BS, minR_ratio, Cell)
        H[:, i] = H_eq.reshape(-1).astype(np.float32)
        
        # 重组信道矩阵用于输入
        temp_H = reshape_channel_matrix(H_eq, num_BS, num_User)
        X[:, i] = temp_H.flatten()
        
        # 计算WMMSE（记录时间）
        wmmse_start = time.time()
        Y[:, i] = WMMSE_sum_rate_optimized(
            p_int=np.ones(K, dtype=np.float32),
            H=H_eq,
            Pmax=np.ones(K, dtype=np.float32),
            var_noise=np.float32(var_noise),
            max_iter=max_iter,
            tolerance=tolerance,
        )
        wmmse_time = time.time() - wmmse_start
        total_wmmse_time += wmmse_time
        if verbose and ((i + 1) % progress_interval == 0 or (i + 1) == num_H):
            print(f'  子任务进度: {(i+1):6d}/{num_H} 样本', flush=True)
    
    return H, X, Y, total_wmmse_time

def setup_cell_structure(Num_of_cell: int, cell_distance: int) -> Dict[str, Any]:
    """预计算Cell结构，避免在每次调用时重复计算"""
    Cell = {
        'Ncell': Num_of_cell,
        'Nintra': 3,  # 固定值
        'NintraBase': 1,  # 固定值
        'Rcell': cell_distance * 2 / np.sqrt(3)
    }
    
    # 预计算Cell位置
    Cell['Position'] = compute_cell_positions(Cell['Ncell'], Cell['Rcell'])
    
    return Cell

def compute_cell_positions(Nbs: int, Rcell: float) -> np.ndarray:
    """优化的Cell位置计算"""
    positions = np.zeros((Nbs, 2))
    
    if Nbs > 1:
        theta = np.arange(Nbs - 1) * np.pi / 3
        positions[1:, :] = np.sqrt(3) * Rcell * np.column_stack([np.cos(theta), np.sin(theta)])
    
    if Nbs > 7:
        theta1 = np.arange(-np.pi/6, 5*np.pi/3, np.pi/3)
        theta2 = np.arange(0, 5*np.pi/3, np.pi/3)
        x = np.concatenate([3*Rcell*np.cos(theta1), 2*np.sqrt(3)*Rcell*np.cos(theta2)])
        y = np.concatenate([3*Rcell*np.sin(theta1), 2*np.sqrt(3)*Rcell*np.sin(theta2)])
        
        end_idx = min(19, Nbs)
        positions[7:end_idx, :] = np.column_stack([x[:end_idx-7], y[:end_idx-7]])
    
    return positions

def reshape_channel_matrix(H_matrix: np.ndarray, num_BS: int, num_User: int) -> np.ndarray:
    """优化的信道矩阵重组"""
    K = num_User * num_BS
    temp_H = np.zeros((num_BS, K), dtype=np.float32)
    
    for l in range(num_BS):
        temp_H[l, :] = H_matrix[l * num_User, :]
    
    return temp_H.reshape(-1, 1)

def WMMSE_sum_rate_optimized(p_int: np.ndarray, H: np.ndarray, 
                           Pmax: np.ndarray, var_noise: float,
                           max_iter: int = 150, tolerance: float = 1e-4) -> np.ndarray:
    """优化的WMMSE算法（向量化与float32加速版）"""
    K = Pmax.shape[0]
    # 使用 float32 加速并减少内存
    b = np.sqrt(p_int.astype(np.float32))
    H = H.astype(np.float32)
    Pmax = Pmax.astype(np.float32)
    var_noise = np.float32(var_noise)

    # 预计算
    H_diag = np.diag(H).astype(np.float32)
    H_squared = (H * H).astype(np.float32)

    f = np.zeros(K, dtype=np.float32)
    w = np.zeros(K, dtype=np.float32)

    # 初始化一次 f,w,v
    b_squared = b * b
    denom_fw = H_squared @ b_squared + var_noise
    f = (H_diag * b) / denom_fw
    w = 1.0 / (1.0 - f * b * H_diag)
    vnew = np.sum(np.log2(w))

    for _ in range(max_iter):
        vold = vnew
        # 更新 b（完全向量化）
        weights = w * (f * f)  # 形状 (K,)
        denom_b = H_squared.T @ weights  # 形状 (K,)
        # 避免除零
        denom_b = np.where(denom_b <= 1e-12, 1e-12, denom_b).astype(np.float32)
        numerator = w * f * H_diag
        btmp = numerator / denom_b
        b = np.clip(btmp, 0.0, np.sqrt(Pmax))

        # 更新 f, w, v（向量化）
        b_squared = b * b
        denom_fw = H_squared @ b_squared + var_noise
        f = (H_diag * b) / denom_fw
        w = 1.0 / (1.0 - f * b * H_diag)
        vnew = np.sum(np.log2(w))

        if np.abs(vnew - vold) <= tolerance:
            break

    p_opt = b * b
    return p_opt.astype(np.float32)

def _generate_and_save_batch(batch_idx: int,
                            current_batch_size: int,
                            num_BS: int,
                            num_User: int,
                            R: int,
                            minR_ratio: float,
                            var_noise: float,
                            save_path: str,
                            dataset_type: str,
                            max_iter: int,
                            tolerance: float,
                            compress: bool,
                            verbose_worker: bool) -> Dict[str, Any]:
    """子进程/主进程通用：生成一个批次并保存到磁盘，返回统计信息"""
    batch_start_time = time.time()
    # 生成当前批次数据
    H_batch, X_batch, Y_batch, wmmse_time = Generate_IMAC_function_optimized(
        num_BS, num_User, current_batch_size, R, minR_ratio, batch_idx, var_noise,
        max_iter=max_iter, tolerance=tolerance, verbose=verbose_worker)

    # 保存数据
    filename = f'{save_path}{dataset_type}_batch_{batch_idx:03d}.mat'
    save_start_time = time.time()
    sio.savemat(
        filename,
        {
            'H_batch': H_batch.astype(np.float16, copy=False),
            'X_batch': X_batch.astype(np.float16, copy=False),
            'Y_batch': Y_batch.astype(np.float16, copy=False),
        },
        do_compression=bool(compress),
        long_field_names=False,
    )
    save_time = time.time() - save_start_time
    total_time = time.time() - batch_start_time
    generation_time = total_time - save_time
    return {
        'batch_idx': batch_idx,
        'current_batch_size': current_batch_size,
        'wmmse_time': wmmse_time,
        'generation_time': generation_time,
        'save_time': save_time,
        'total_time': total_time,
        'filename': filename,
    }

def initialize_wmmse_variables(b: np.ndarray, H: np.ndarray, H_diag: np.ndarray, 
                             H_squared: np.ndarray, var_noise: float, 
                             f: np.ndarray, w: np.ndarray) -> float:
    """初始化WMMSE变量"""
    K = len(b)
    b_squared = b * b
    denom = H_squared @ b_squared + var_noise
    f[:K] = (H_diag * b) / denom
    w[:K] = 1.0 / (1.0 - f * b * H_diag)
    return float(np.sum(np.log2(w)))

def update_fwv_variables(b: np.ndarray, H: np.ndarray, H_diag: np.ndarray, 
                        H_squared: np.ndarray, var_noise: float, 
                        f: np.ndarray, w: np.ndarray) -> float:
    """更新WMMSE变量"""
    b_squared = b * b
    denom = H_squared @ b_squared + var_noise
    f[:] = (H_diag * b) / denom
    w[:] = 1.0 / (1.0 - f * b * H_diag)
    return float(np.sum(np.log2(w)))

def generate_IBC_channel_optimized(Num_of_user_in_each_cell: int, cell_distance: int, 
                                 Num_of_cell: int, minR_ratio: float, 
                                 Cell: Dict[str, Any]) -> np.ndarray:
    """优化的信道生成函数"""
    T = 1
    BaseNum = 1
    UserNum = Num_of_user_in_each_cell
    CellNum = Num_of_cell
    
    # 生成用户和基站位置
    MS, BS = usergenerator_optimized(Cell, minR_ratio)
    
    # 计算大尺度衰落
    HLarge = channelsample_optimized(BS, MS, Cell)
    
    # 生成小尺度衰落
    H_small_real = np.random.randn(T, BaseNum, CellNum, UserNum, CellNum).astype(np.float32)
    H_small_imag = np.random.randn(T, BaseNum, CellNum, UserNum, CellNum).astype(np.float32)
    _scale = np.float32(1.0 / np.sqrt(np.float32(2.0)))
    H_small = (H_small_real + 1j * H_small_imag) * _scale
    
    # 合并大小尺度衰落
    total_user_Num = CellNum * UserNum
    H_eq = np.zeros((total_user_Num, total_user_Num), dtype=np.float32)
    
    k = 0
    for CellMS in range(CellNum):
        for User in range(UserNum):
            k_INF = 0
            for INFCellMS in range(CellNum):
                for INFUser in range(UserNum):
                    H_combined = (H_small[0, BaseNum-1, CellMS, INFUser, INFCellMS] * 
                                np.sqrt(HLarge[INFUser, INFCellMS, BaseNum-1, CellMS]).astype(np.float32))
                    H_eq[k, k_INF] = np.abs(H_combined).astype(np.float32)
                    k_INF += 1
            k += 1
    
    return H_eq

def usergenerator_optimized(Cell: Dict[str, Any], minR_ratio: float) -> Tuple[Dict, Dict]:
    """优化的用户生成器"""
    Ncell = Cell['Ncell']
    Nintra = Cell['Nintra']
    NintraBase = Cell['NintraBase']
    Rcellmin = minR_ratio * Cell['Rcell']
    
    MS = {'Position': [None] * Ncell}
    BS = {'Position': [None] * Ncell}
    
    # 批量生成随机数
    all_theta = np.random.rand(Nintra * Ncell) * 2 * np.pi
    theta_idx = 0
    
    # 生成用户位置
    for n in range(Ncell):
        theta = all_theta[theta_idx:theta_idx + Nintra]
        theta_idx += Nintra
        
        x, y = distrnd_optimized(Cell['Rcell'], Rcellmin, theta)
        MS['Position'][n] = np.column_stack([x + Cell['Position'][n, 0], 
                                           y + Cell['Position'][n, 1]])
    
    # 生成基站位置
    for n in range(Ncell):
        BS['Position'][n] = Cell['Position'][n, :].reshape(1, -1)
    
    return MS, BS

def distrnd_optimized(Rcell: float, Rcellmin: float, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """优化的距离生成函数"""
    MsNum = len(theta)
    R = Rcell - Rcellmin
    
    # 批量生成距离
    d = np.sum(np.random.rand(MsNum, 2), axis=1) * R
    d[d > R] = 2 * R - d[d > R]
    d = d + Rcellmin
    
    # 计算坐标
    x = d * np.cos(theta)
    y = d * np.sin(theta)
    
    return x, y

def channelsample_optimized(BS: Dict, MS: Dict, Cell: Dict[str, Any]) -> np.ndarray:
    """优化的信道采样"""
    Ncell = Cell['Ncell']
    Nintra = Cell['Nintra']
    Nbs = Cell['NintraBase']
    
    Hlarge = np.zeros((Nintra, Ncell, Nbs, Ncell), dtype=np.float32)
    
    # 预生成所有随机数
    all_randn = (np.random.randn(Nintra * Ncell * Nbs * Ncell).astype(np.float32)) * (8.0/10.0)
    randn_idx = 0
    
    for CellBS in range(Ncell):
        for CellMS in range(Ncell):
            for Base in range(Nbs):
                for User in range(Nintra):
                    d = np.linalg.norm(MS['Position'][CellMS][User, :] - 
                                     BS['Position'][CellBS][Base, :])
                    PL = 10**(all_randn[randn_idx]) * (200/d)**3
                    randn_idx += 1
                    Hlarge[User, CellMS, Base, CellBS] = PL
    
    return Hlarge

def quick_test_generation(workers: int = 1, max_iter: int = 150, tolerance: float = 1e-4):
    """快速测试生成（跳过时间预估）"""
    # 小规模测试参数
    total_samples = 100
    train_samples = 80
    test_samples = 20
    batch_size = 50
    
    # IMAC参数
    num_BS = 6
    num_User = 3
    R = 100
    minR_ratio = 0.2
    var_noise = 1
    
    # 存储路径
    save_path = '/root/autodl-tmp/test_output/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print('快速测试模式 - 生成小规模IMAC数据集...')
    print(f'总样本数: {total_samples} (训练集: {train_samples}, 测试集: {test_samples})')
    print(f'存储路径: {save_path}')
    
    start_time = time.time()
    
    # 生成训练集
    print('\n=== 生成训练集 ===')
    generate_dataset_batches(train_samples, batch_size, num_BS, num_User, R, 
                           minR_ratio, var_noise, save_path, 'train',
                           estimated_time_per_sample=None,
                           workers=workers,
                           max_iter=max_iter,
                           tolerance=tolerance)
    
    # 生成测试集
    print('\n=== 生成测试集 ===')
    generate_dataset_batches(test_samples, batch_size, num_BS, num_User, R, 
                           minR_ratio, var_noise, save_path, 'test',
                           estimated_time_per_sample=None,
                           workers=workers,
                           max_iter=max_iter,
                           tolerance=tolerance)
    
    total_time = time.time() - start_time
    print(f'\n快速测试完成! 总耗时: {total_time:.2f}秒')

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    parser = argparse.ArgumentParser(description='IMAC 数据集生成器')
    parser.add_argument('--quick', action='store_true', help='快速小规模测试生成')
    parser.add_argument('--workers', type=int, default=max(1, multiprocessing.cpu_count() - 1), help='并行进程数')
    parser.add_argument('--batch-size', type=int, default=10000, help='每批样本数')
    parser.add_argument('--max-iter', type=int, default=150, help='WMMSE 最大迭代次数')
    parser.add_argument('--tolerance', type=float, default=1e-4, help='WMMSE 收敛阈值')
    parser.add_argument('--no-compress', action='store_true', help='保存时不压缩（更快，文件更大）')
    parser.add_argument('--verbose-worker', action='store_true', help='子进程输出内部进度（排障用）')
    parser.add_argument('--estimate', action='store_true', help='运行前做时间预估（可能较慢且需要交互确认）')
    parser.add_argument('--start-batch', type=int, default=1, help='从第几批开始生成（用于断点续传）')
    args = parser.parse_args()

    if args.quick:
        quick_test_generation(workers=args.workers, max_iter=args.max_iter, tolerance=args.tolerance)
    else:
        generate_large_imac_dataset(
            workers=args.workers,
            max_iter=args.max_iter,
            tolerance=args.tolerance,
            skip_estimate=(not args.estimate),
            batch_size=args.batch_size,
            compress=(not args.no_compress),
            verbose_worker=args.verbose_worker,
            start_batch=args.start_batch,
        )

