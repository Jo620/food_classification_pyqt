#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境依赖检测脚本
检查所有必需的库是否已安装，以及版本信息
"""

import sys
import subprocess

def print_header(text):
    """打印美化的标题"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_success(text):
    """打印成功信息"""
    print(f"✅ {text}")

def print_error(text):
    """打印错误信息"""
    print(f"❌ {text}")

def print_warning(text):
    """打印警告信息"""
    print(f"⚠️  {text}")

def print_info(text):
    """打印普通信息"""
    print(f"ℹ️  {text}")

def check_python_version():
    """检查Python版本"""
    print_header("1. Python 版本检查")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print_info(f"当前Python版本: {version_str}")
    
    if version.major == 3 and version.minor >= 7:
        print_success(f"Python版本符合要求 (>= 3.7)")
        return True
    else:
        print_error(f"Python版本过低，需要 >= 3.7")
        return False

def check_conda_environment():
    """检查Conda环境"""
    print_header("2. Conda 环境检查")
    try:
        result = subprocess.run(['conda', '--version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            print_success(f"Conda已安装: {result.stdout.strip()}")
            
            # 获取当前激活的环境
            result = subprocess.run(['conda', 'info', '--envs'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            print_info("当前Conda环境:")
            for line in result.stdout.split('\n'):
                if '*' in line:
                    print(f"    {line}")
            return True
        else:
            print_warning("Conda未安装或未添加到PATH")
            return False
    except FileNotFoundError:
        print_warning("Conda未安装")
        return False
    except Exception as e:
        print_error(f"检查Conda时出错: {e}")
        return False

def check_library(name, import_name=None, min_version=None):
    """
    检查单个库
    
    Args:
        name: 库的pip包名
        import_name: 导入时的模块名（如果与包名不同）
        min_version: 最低版本要求
    """
    if import_name is None:
        import_name = name
    
    try:
        module = __import__(import_name)
        
        # 尝试获取版本号
        version = "未知"
        if hasattr(module, '__version__'):
            version = module.__version__
        elif hasattr(module, 'VERSION'):
            version = module.VERSION
        elif hasattr(module, 'version'):
            if callable(module.version):
                version = module.version()
            else:
                version = module.version
        
        # 版本比较
        if min_version and version != "未知":
            try:
                from packaging import version as pkg_version
                if pkg_version.parse(version) >= pkg_version.parse(min_version):
                    print_success(f"{name:20s} ✓ (版本: {version})")
                else:
                    print_warning(f"{name:20s} ⚠ (当前: {version}, 需要: >={min_version})")
            except:
                print_success(f"{name:20s} ✓ (版本: {version})")
        else:
            print_success(f"{name:20s} ✓ (版本: {version})")
        
        return True, version
        
    except ImportError as e:
        print_error(f"{name:20s} ✗ (未安装)")
        return False, None
    except Exception as e:
        print_error(f"{name:20s} ✗ (导入错误: {str(e)[:30]}...)")
        return False, None

def check_required_libraries():
    """检查所有必需的库"""
    print_header("3. 必需库检查")
    
    # 定义所有需要检查的库
    # 格式: (pip包名, 导入名, 最低版本)
    libraries = [
        # PyQt5相关
        ("PyQt5", "PyQt5", "5.15.0"),
        ("PyQt5.QtCore", "PyQt5.QtCore", None),
        ("PyQt5.QtWidgets", "PyQt5.QtWidgets", None),
        ("PyQt5.QtGui", "PyQt5.QtGui", None),
        
        # PyTorch相关
        ("torch", "torch", "1.7.0"),
        ("torchvision", "torchvision", "0.8.0"),
        
        # 深度学习工具
        ("timm", "timm", "0.6.0"),
        
        # 图像处理
        ("Pillow", "PIL", "8.0.0"),
        
        # 数值计算
        ("numpy", "numpy", "1.19.0"),
        
        # 其他工具
        ("json", "json", None),  # 标准库
        ("os", "os", None),      # 标准库
        ("sys", "sys", None),    # 标准库
    ]
    
    results = {}
    all_installed = True
    
    for lib_info in libraries:
        if len(lib_info) == 3:
            name, import_name, min_version = lib_info
        else:
            name, import_name = lib_info
            min_version = None
        
        installed, version = check_library(name, import_name, min_version)
        results[name] = (installed, version)
        
        if not installed:
            all_installed = False
    
    return all_installed, results

def check_pytorch_cuda():
    """检查PyTorch CUDA支持"""
    print_header("4. PyTorch CUDA 支持检查")
    
    try:
        import torch
        
        print_info(f"PyTorch版本: {torch.__version__}")
        print_info(f"CUDA是否可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print_success(f"CUDA版本: {torch.version.cuda}")
            print_success(f"cuDNN版本: {torch.backends.cudnn.version()}")
            print_success(f"可用GPU数量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print_info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print_info(f"    显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        else:
            print_warning("CUDA不可用，将使用CPU模式")
            print_info("如需GPU加速，请安装CUDA版本的PyTorch")
        
        return True
        
    except ImportError:
        print_error("PyTorch未安装")
        return False
    except Exception as e:
        print_error(f"检查CUDA时出错: {e}")
        return False

def check_optional_libraries():
    """检查可选库"""
    print_header("5. 可选库检查")
    
    optional_libs = [
        ("flask", "flask", "Web版本需要"),
        ("flask-cors", "flask_cors", "Web版本需要"),
        ("matplotlib", "matplotlib", "可视化工具"),
        ("opencv-python", "cv2", "高级图像处理"),
        ("pandas", "pandas", "数据处理"),
    ]
    
    print_info("以下库为可选，不影响PyQt5版本运行:\n")
    
    for name, import_name, description in optional_libs:
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', '未知')
            print_success(f"{name:20s} ✓ (版本: {version}) - {description}")
        except ImportError:
            print_warning(f"{name:20s} ✗ (未安装) - {description}")

def generate_install_commands(results):
    """生成安装命令"""
    print_header("6. 安装建议")
    
    missing_libs = [name for name, (installed, _) in results.items() if not installed]
    
    if not missing_libs:
        print_success("所有必需库都已安装！✨")
        return
    
    print_warning("检测到以下库未安装:\n")
    
    # 分类库
    pyqt_libs = [lib for lib in missing_libs if 'PyQt5' in lib or 'Qt' in lib]
    torch_libs = [lib for lib in missing_libs if lib in ['torch', 'torchvision']]
    other_libs = [lib for lib in missing_libs if lib not in pyqt_libs and lib not in torch_libs]
    
    # PyQt5安装
    if pyqt_libs:
        print_info("📦 安装 PyQt5:")
        print("   conda install pyqt")
        print("   或")
        print("   pip install PyQt5")
        print()
    
    # PyTorch安装
    if torch_libs:
        print_info("📦 安装 PyTorch (选择合适的版本):")
        print("   CPU版本:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        print()
        print("   CUDA 11.8版本:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print()
        print("   CUDA 12.1版本:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print()
    
    # 其他库
    if other_libs:
        print_info("📦 安装其他依赖:")
        install_cmd = "pip install " + " ".join(other_libs)
        print(f"   {install_cmd}")
        print()
    
    # 一键安装命令
    print_info("📦 或使用requirements.txt一键安装:")
    print("   pip install -r requirements.txt")

def save_environment_info():
    """保存环境信息到文件"""
    print_header("7. 保存环境信息")
    
    try:
        import platform
        
        with open('environment_info.txt', 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("环境信息报告\n")
            f.write("="*60 + "\n\n")
            
            # 系统信息
            f.write("系统信息:\n")
            f.write(f"  操作系统: {platform.system()} {platform.release()}\n")
            f.write(f"  Python版本: {sys.version}\n")
            f.write(f"  架构: {platform.machine()}\n\n")
            
            # 已安装的包
            f.write("已安装的包:\n")
            result = subprocess.run(['pip', 'list'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=10)
            f.write(result.stdout)
        
        print_success("环境信息已保存到 environment_info.txt")
        return True
        
    except Exception as e:
        print_error(f"保存环境信息失败: {e}")
        return False

def main():
    """主函数"""
    print("\n" + "🔍 "*15)
    print("   Python环境依赖检测工具")
    print("   GlobalFood202 食物分类识别系统")
    print("🔍 "*15 + "\n")
    
    # 1. 检查Python版本
    python_ok = check_python_version()
    
    # 2. 检查Conda环境
    check_conda_environment()
    
    # 3. 检查必需库
    all_installed, results = check_required_libraries()
    
    # 4. 检查PyTorch CUDA
    cuda_ok = check_pytorch_cuda()
    
    # 5. 检查可选库
    check_optional_libraries()
    
    # 6. 生成安装建议
    generate_install_commands(results)
    
    # 7. 保存环境信息
    save_environment_info()
    
    # 最终总结
    print_header("✨ 检测总结")
    
    if python_ok and all_installed:
        print_success("✅ 环境检测通过！所有必需库都已安装。")
        print_info("可以运行以下命令启动应用:")
        print("   python main.py")
    else:
        print_warning("⚠️  环境检测未完全通过，请根据上述建议安装缺失的库。")
    
    print("\n" + "="*60 + "\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  检测已取消")
    except Exception as e:
        print(f"\n\n❌ 检测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
