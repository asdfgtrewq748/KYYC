# 简单验证推送状态

Write-Host "=== Git Status ===" -ForegroundColor Cyan
git status --short --branch

Write-Host "`n=== Latest Commits ===" -ForegroundColor Cyan
git log --oneline -3

Write-Host "`n=== Remote Config ===" -ForegroundColor Cyan
git remote -v

Write-Host "`n=== Proxy Config ===" -ForegroundColor Cyan
Write-Host "HTTP: $(git config --global --get http.proxy)"
Write-Host "HTTPS: $(git config --global --get https.proxy)"

Write-Host "`n=== Checking Remote Branches ===" -ForegroundColor Cyan
git ls-remote --heads origin

Write-Host "`n=== Next Steps ===" -ForegroundColor Yellow
Write-Host "If remote is empty, run: git push -u origin main"
Write-Host "Then visit: https://github.com/asdfgtrewq748/KYYC"
