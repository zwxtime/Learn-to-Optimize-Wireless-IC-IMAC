#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow 1.x环境检查脚本
检查当前环境是否支持TensorFlow 1.x
"""

import sys
import os

def check_python_version():
    """检查Python版本"""
    print("=" * 50)
    print("Python环境检查")
    print("=" * 50)
    
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 7:
        print("✅ Python版本符合要求 (>= 3.7)")
        return True
    else:
        print("❌ Python版本不符合要求，需要Python 3.7+")
        return False

def check_tensorflow():
    """检查TensorFlow版本"""
    print("\n" + "=" * 50)
    print("TensorFlow环境检查")
    print("=" * 50)
    
    try:
        import tensorflow as tf
        version = tf.__version__
        print(f"TensorFlow版本: {version}")
        
        if version.startswith('1.'):
            print("✅ TensorFlow 1.x版本检测成功")
            
            # 检查兼容性设置
            try:
                tf.compat.v1.disable_eager_execution()
                tf.compat.v1.disable_v2_behavior()
                print("✅ TensorFlow 1.x兼容性设置成功")
                
                # 测试基本功能
                with tf.compat.v1.Session() as sess:
                    a = tf.compat.v1.constant(1.0)
                    b = tf.compat.v1.constant(2.0)
                    c = a + b
                    result = sess.run(c)
                    print(f"✅ 基本计算测试成功: {result}")
                
                return True
                
            except Exception as e:
                print(f"❌ TensorFlow 1.x兼容性设置失败: {e}")
                return False
                
        elif version.startswith('2.'):
            print("⚠️  检测到TensorFlow 2.x版本")
            print("   建议使用TensorFlow 1.15或更低版本")
            print("   当前版本可能通过兼容性模式工作")
            
            try:
                tf.compat.v1.disable_eager_execution()
                tf.compat.v1.disable_v2_behavior()
                print("✅ TensorFlow 2.x兼容性设置成功")
                
                # 测试基本功能
                with tf.compat.v1.Session() as sess:
                    a = tf.compat.v1.constant(1.0)
                    b = tf.compat.v1.constant(2.0)
                    c = a + b
                    result = sess.run(c)
                    print(f"✅ 基本计算测试成功: {result}")
                
                return True
                
            except Exception as e:
                print(f"❌ TensorFlow 2.x兼容性设置失败: {e}")
                return False
        else:
            print("❌ 未知的TensorFlow版本")
            return False
            
    except ImportError:
        print("❌ TensorFlow未安装")
        print("   请安装TensorFlow: pip install tensorflow==1.15.0")
        return False
    except Exception as e:
        print(f"❌ TensorFlow检查失败: {e}")
        return False

def check_other_dependencies():
    """检查其他依赖"""
    print("\n" + "=" * 50)
    print("其他依赖检查")
    print("=" * 50)
    
    dependencies = [
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('matplotlib', 'Matplotlib'),
        ('sklearn', 'Scikit-learn')
    ]
    
    all_good = True
    
    for module, name in dependencies:
        try:
            if module == 'sklearn':
                import sklearn
                version = sklearn.__version__
            else:
                exec(f"import {module}")
                exec(f"version = {module}.__version__")
            
            print(f"✅ {name}: 已安装")
            
        except ImportError:
            print(f"❌ {name}: 未安装")
            all_good = False
        except Exception as e:
            print(f"⚠️  {name}: 检查失败 ({e})")
            all_good = False
    
    return all_good

def check_gpu_support():
    """检查GPU支持"""
    print("\n" + "=" * 50)
    print("GPU支持检查")
    print("=" * 50)
    
    try:
        import tensorflow as tf
        
        # 检查TensorFlow 1.x的GPU支持
        if tf.__version__.startswith('1.'):
            devices = tf.compat.v1.config.list_physical_devices('GPU')
            if devices:
                print(f"✅ 检测到 {len(devices)} 个GPU设备:")
                for device in devices:
                    print(f"   - {device.name}")
                
                # 检查CUDA版本
                try:
                    cuda_version = tf.sysconfig.get_build_info()['cuda_version']
                    print(f"✅ CUDA版本: {cuda_version}")
                except:
                    print("⚠️  无法获取CUDA版本信息")
                
                return True
            else:
                print("⚠️  未检测到GPU设备，将使用CPU")
                return False
        else:
            print("⚠️  TensorFlow 2.x GPU检查可能不准确")
            return False
            
    except Exception as e:
        print(f"❌ GPU检查失败: {e}")
        return False

def check_data_path():
    """检查数据路径"""
    print("\n" + "=" * 50)
    print("数据路径检查")
    print("=" * 50)
    
    data_path = '/root/autodl-tmp/IMAC/'
    
    if os.path.exists(data_path):
        print(f"✅ 数据路径存在: {data_path}")
        
        # 检查是否有数据文件
        train_files = [f for f in os.listdir(data_path) if f.startswith('train_batch_')]
        test_files = [f for f in os.listdir(data_path) if f.startswith('test_batch_')]
        
        print(f"   训练批次文件: {len(train_files)} 个")
        print(f"   测试批次文件: {len(test_files)} 个")
        
        if train_files and test_files:
            print("✅ 数据文件检查通过")
            return True
        else:
            print("⚠️  数据文件不完整，请先生成数据集")
            return False
    else:
        print(f"❌ 数据路径不存在: {data_path}")
        print("   请先生成IMAC数据集")
        return False

def main():
    """主函数"""
    print("TensorFlow 1.x环境检查工具")
    print("=" * 60)
    
    checks = [
        ("Python版本", check_python_version),
        ("TensorFlow", check_tensorflow),
        ("其他依赖", check_other_dependencies),
        ("GPU支持", check_gpu_support),
        ("数据路径", check_data_path)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name}检查异常: {e}")
            results.append((check_name, False))
    
    # 总结
    print("\n" + "=" * 60)
    print("检查结果总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{check_name:12s}: {status}")
    
    print(f"\n总体结果: {passed}/{total} 项检查通过")
    
    if passed == total:
        print("🎉 环境检查完全通过！可以开始训练DNN模型")
        print("\n建议运行顺序:")
        print("1. 快速测试: python quick_test_dnn_tf1.py")
        print("2. 完整训练: python train_imac_dnn_tf1.py")
    else:
        print("⚠️  环境检查未完全通过，请解决上述问题后再开始训练")
        
        if not any(name == "TensorFlow" and result for name, result in results):
            print("\nTensorFlow安装建议:")
            print("pip install tensorflow==1.15.0")
            print("或")
            print("conda install tensorflow=1.15")

if __name__ == '__main__':
    main()
