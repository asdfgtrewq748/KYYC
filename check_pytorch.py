import torch
import sys

print("=" * 70)
print("  PyTorch 安装验证")
print("=" * 70)
print()

print(f"✅ PyTorch 版本: {torch.__version__}")
print(f"✅ CUDA 编译版本: {torch.version.cuda}")
print(f"✅ CUDA 可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✅ GPU 设备: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU 数量: {torch.cuda.device_count()}")
    print()
    
    # 测试 GPU 运算
    print("🧪 测试 GPU 运算...")
    try:
        x = torch.rand(100, 100).cuda()
        y = x @ x
        z = y.cpu()
        print("✅ GPU 运算测试通过!")
        print()
        print("🎉 PyTorch + CUDA 完美工作!")
    except Exception as e:
        print(f"❌ GPU 运算失败: {e}")
else:
    print("⚠️  CUDA 不可用,仅支持 CPU")

print()
print("=" * 70)
