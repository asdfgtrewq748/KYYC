#!/bin/bash
# 监控推送进度

echo "========================================="
echo "  Monitoring GitHub Push Progress"
echo "========================================="
echo ""

# 检查推送是否开始
echo "Checking if push is in progress..."
ps aux | grep -i "git push" | grep -v grep

echo ""
echo "Waiting 2 minutes for push to complete..."
sleep 120

echo ""
echo "=== Checking remote repository ==="
git ls-remote --heads origin

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Remote Branches Found ==="
    git ls-remote --heads origin
    echo ""
    echo "✅ SUCCESS! Visit: https://github.com/asdfgtrewq748/KYYC"
else
    echo ""
    echo "❌ FAILED: Push did not complete"
    echo ""
    echo "Try running manually:"
    echo "  git push -u origin main"
fi
