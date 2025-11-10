# -*- coding: utf-8 -*-
"""强制禁用特征工程 - 修复NaN问题"""
import re

print("正在修复STGCN.py，强制禁用特征工程...")

# 读取文件
with open('STGCN.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 查找并替换特征工程的checkbox
pattern1 = r'use_feature_engineering = st\.checkbox\([^)]+\)'
replacement1 = 'use_feature_engineering = False  # 强制禁用，防止NaN'

if re.search(pattern1, content):
    content = re.sub(pattern1, replacement1, content)
    print("✓ 已将特征工程checkbox替换为强制False")
else:
    print("⚠️ 未找到特征工程checkbox")

# 再检查一次是否有其他地方启用特征工程
# 在训练开始前再次强制禁用
pattern2 = r'(# ⭐ 特征工程.*?if use_feature_engineering:)'
if re.search(pattern2, content, re.DOTALL):
    # 在特征工程代码块前强制禁用
    insert_text = '\n                # ⚠️ 强制禁用特征工程，防止NaN\n                use_feature_engineering = False\n                '
    content = re.sub(
        r'(# ⭐ 特征工程\（⚠️危险功能，默认禁用\）)',
        r'\1' + insert_text,
        content
    )
    print("✓ 在特征工程代码块前添加强制禁用")

# 保存文件
with open('STGCN.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ STGCN.py 已修复")
print("\n现在特征工程已被强制禁用，不会再产生264个特征")
