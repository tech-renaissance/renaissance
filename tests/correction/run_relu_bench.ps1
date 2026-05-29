$ErrorActionPreference = "Continue"
$TR_EXE = "R:\renaissance\build\windows-msvc-release\bin\tests\correction\test_dl_full.exe"
$PY_EXE = "B:\Softwares\miniconda3\envs\py313\python.exe"
$PY_SCRIPT = "R:\renaissance\tests\correction\benchmark_pytorch.py"
$LOG_DIR = "R:\renaissance\tests\correction\bench_logs_relu"
$OUTPUT_FILE = "R:\renaissance\docs\RELU_TEST_D.md"
$CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin"
$CUDNN_PATH = "C:\Program Files\NVIDIA\CUDNN\v9.17\bin\13.1"

Remove-Item -Recurse -Force $LOG_DIR -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null

$env:PATH = "$CUDNN_PATH;$CUDA_PATH;" + $env:PATH

function Run-TR {
    param($mode, $run)
    $logfile = "$LOG_DIR\tr_${mode}_run${run}.txt"
    $flags = ""
    if ($mode -eq "cpu") { $flags = "--cpu" }
    elseif ($mode -eq "gpu") { $flags = "--gpu" }
    elseif ($mode -eq "amp") { $flags = "--amp" }
    Write-Host "=== TR $mode run $run ==="
    & cmd /c "`"$TR_EXE`" $flags > `"$logfile`" 2>&1"
    $lines = Get-Content $logfile
    $best = ""
    $time = ""
    foreach ($line in $lines) {
        if (-not $best -and $line -match "Best Top-1\s*:\s*([\d.]+)%") { $best = $matches[1] }
        if (-not $time -and $line -match '^\s*Time\s*:\s*([\d.]+)\s*s\s*$') { $time = $matches[1] }
    }
    Write-Host "  Time=${time}s  Acc=${best}%"
    return @{ mode = "TR $mode"; run = $run; time = $time; acc = $best }
}

function Run-PyTorch {
    param($mode, $run)
    $logfile = "$LOG_DIR\pt_${mode}_run${run}.txt"
    $flags = ""
    if ($mode -eq "cpu") { $flags = "--cpu" }
    elseif ($mode -eq "gpu") { $flags = "--gpu" }
    elseif ($mode -eq "amp") { $flags = "--amp" }
    Write-Host "=== PyTorch $mode run $run ==="
    & cmd /c "`"$PY_EXE`" `"$PY_SCRIPT`" $flags > `"$logfile`" 2>&1"
    $content = Get-Content $logfile -Raw
    $acc = ""
    $time = ""
    if ($content -match "Final Val Accuracy\s*:\s*([\d.]+)%") { $acc = $matches[1] }
    if ($content -match "Training time\s*\(4 epochs\)\s*:\s*([\d.]+)s") { $time = $matches[1] }
    Write-Host "  Time=${time}s  Acc=${acc}%"
    return @{ mode = "PyTorch $mode"; run = $run; time = $time; acc = $acc }
}

$all_results = @()

Write-Host "========== TEST D'L FULL (ReLU) 3 runs each =========="
foreach ($mode in @("cpu","gpu","amp")) {
    for ($r = 1; $r -le 3; $r++) {
        $all_results += Run-TR $mode $r
    }
}

Write-Host ""
Write-Host "========== PyTorch (ReLU) 3 runs each =========="
foreach ($mode in @("cpu","gpu","amp")) {
    for ($r = 1; $r -le 3; $r++) {
        $all_results += Run-PyTorch $mode $r
    }
}

Write-Host ""
Write-Host "========== Writing report =========="

function Get-Median { param($arr) $s = $arr | Sort-Object; return $s[1] }

function Build-Row {
    param($label, $results)
    $times = $results | ForEach-Object { [double]$_.time }
    $accs  = $results | ForEach-Object { [double]$_.acc }
    $mt = Get-Median $times
    $ma = Get-Median $accs
    return "| $label | | {0:F3} | {1:F3} | {2:F3} | {3:F3} | {4:F2}% |" -f $times[0], $times[1], $times[2], $mt, $ma
}

$md = @"
# ReLU 性能对比测试报告

> test_dl_full 与 benchmark_pytorch.py 均使用 ReLU 激活函数
> 每组 3 次独立运行，取中位数

| 框架 | 模式 | Run 1 (s) | Run 2 (s) | Run 3 (s) | 中位数 (s) | 准确率 |
|------|------|-----------|-----------|-----------|-----------|--------|
"@

foreach ($mode in @("cpu","gpu","amp")) {
    $tr_res = $all_results | Where-Object { $_.mode -like "TR $mode*" }
    $pt_res = $all_results | Where-Object { $_.mode -like "PyTorch $mode*" }
    $md += (Build-Row "TR"    $tr_res) + "`n"
    $md += (Build-Row "PyTorch" $pt_res) + "`n"
}

Set-Content -Path $OUTPUT_FILE -Value $md -Encoding UTF8
Write-Host "Done! Report: $OUTPUT_FILE"