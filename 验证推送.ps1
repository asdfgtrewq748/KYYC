# 验证 GitHub 推送脚本

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "    GitHub 推送验证工具" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

# 1. 检查当前目录
Write-Host "📁 当前目录:" -ForegroundColor Yellow
Get-Location
Write-Host ""

# 2. 检查 Git 状态
Write-Host "📊 Git 状态:" -ForegroundColor Yellow
git status --short --branch
Write-Host ""

# 3. 检查最新提交
Write-Host "📝 最新提交:" -ForegroundColor Yellow
git log --oneline -3
Write-Host ""

# 4. 检查远程配置
Write-Host "🔗 远程配置:" -ForegroundColor Yellow
git remote -v
Write-Host ""

# 5. 检查代理配置
Write-Host "🌐 代理配置:" -ForegroundColor Yellow
$httpProxy = git config --global --get http.proxy
$httpsProxy = git config --global --get https.proxy
if ($httpProxy) {
    Write-Host "HTTP Proxy: $httpProxy" -ForegroundColor Green
} else {
    Write-Host "HTTP Proxy: Not configured" -ForegroundColor Red
}
if ($httpsProxy) {
    Write-Host "HTTPS Proxy: $httpsProxy" -ForegroundColor Green
} else {
    Write-Host "HTTPS Proxy: Not configured" -ForegroundColor Red
}
Write-Host ""

# 6. 检查远程分支
Write-Host "🌲 远程分支:" -ForegroundColor Yellow
try {
    $remoteBranches = git ls-remote --heads origin 2>&1
    if ($remoteBranches) {
        $remoteBranches | ForEach-Object { Write-Host $_ -ForegroundColor Green }
        Write-Host ""
        Write-Host "✅ 远程仓库有分支!" -ForegroundColor Green
    } else {
        Write-Host "⚠️  远程仓库为空" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ 无法连接到远程仓库" -ForegroundColor Red
}
Write-Host ""

# 7. 提供推送命令
Write-Host "==================================" -ForegroundColor Cyan
Write-Host "🚀 推送命令:" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "# 如果远程为空,执行:" -ForegroundColor Yellow
Write-Host "git push -u origin main" -ForegroundColor Green
Write-Host ""
Write-Host "# 访问在线仓库:" -ForegroundColor Yellow
Write-Host "https://github.com/asdfgtrewq748/KYYC" -ForegroundColor Cyan
Write-Host ""

Write-Host "按任意键退出..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
