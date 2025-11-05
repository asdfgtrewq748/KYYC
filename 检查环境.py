"""
环境检查脚本 - 验证所有依赖是否已安装
"""
import sys

print("=" * 50)
print("STGCN 环境检查")
print("=" * 50)
print()

print(f"Python 版本: {sys.version}")
print()

required_packages = {
    'torch': 'PyTorch',
    'streamlit': 'Streamlit',
    'pandas': 'Pandas',
    'numpy': 'NumPy',
    'scipy': 'SciPy',
    'matplotlib': 'Matplotlib'
}

missing = []
installed = []

for package, name in required_packages.items():
    try:
        mod = __import__(package)
        version = getattr(mod, '__version__', 'unknown')
        installed.append((name, version))
        print(f"✓ {name:<15} {version}")
    except ImportError:
        missing.append(name)
        print(f"✗ {name:<15} 未安装")

print()
print("=" * 50)

if missing:
    print(f"❌ 缺少 {len(missing)} 个依赖:")
    for pkg in missing:
        print(f"   - {pkg}")
    print()
    print("请运行以下命令安装:")
    print("  pip install streamlit pandas scipy numpy matplotlib")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130")
else:
    print("✅ 所有依赖已安装！")
    print()
    print("可以运行:")
    print("  streamlit run STGCN.py")

print("=" * 50)
