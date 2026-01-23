# YOLOv12 多注意力机制训练脚本
# 依次训练 MDTA, WTA, HTA 三种注意力机制
# 每个训练的控制台输出保存到对应的 txt 文件

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "YOLOv12 多注意力机制训练脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 定义注意力类型列表
$attentionTypes = @("MDTA", "WTA", "HTA")

# 创建日志目录
$logDir = "training_logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
    Write-Host "创建日志目录: $logDir" -ForegroundColor Green
}

# 遍历每种注意力类型进行训练
foreach ($attention in $attentionTypes) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host "开始训练: $attention" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host ""
    
    # 定义日志文件路径
    $logFile = Join-Path $logDir "train_${attention}.txt"
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    # 写入训练开始信息到日志文件
    "训练开始时间: $timestamp" | Out-File -FilePath $logFile -Encoding UTF8
    "注意力类型: $attention" | Add-Content -Path $logFile -Encoding UTF8
    "=" * 80 | Add-Content -Path $logFile -Encoding UTF8
    "" | Add-Content -Path $logFile -Encoding UTF8
    
    Write-Host "使用注意力类型: $attention" -ForegroundColor Green
    Write-Host "日志文件: $logFile" -ForegroundColor Green
    Write-Host ""
    
    # 执行训练并保存输出（使用命令行参数）
    try {
        python train_yolo12_custom.py --attention $attention 2>&1 | Tee-Object -FilePath $logFile -Append
        
        $endTimestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        "" | Add-Content -Path $logFile -Encoding UTF8
        "=" * 80 | Add-Content -Path $logFile -Encoding UTF8
        "训练结束时间: $endTimestamp" | Add-Content -Path $logFile -Encoding UTF8
        
        Write-Host ""
        Write-Host "✓ $attention 训练完成" -ForegroundColor Green
        Write-Host "日志已保存到: $logFile" -ForegroundColor Green
    }
    catch {
        Write-Host ""
        Write-Host "✗ $attention 训练失败: $_" -ForegroundColor Red
        "训练失败: $_" | Add-Content -Path $logFile -Encoding UTF8
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "所有训练任务完成！" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "训练日志保存在: $logDir 目录" -ForegroundColor Green
Write-Host "- train_MDTA.txt" -ForegroundColor White
Write-Host "- train_WTA.txt" -ForegroundColor White
Write-Host "- train_HTA.txt" -ForegroundColor White
