# 下载 CUDA 运行库 wheel（ctranslate2 需要 cublas64_12.dll + cudnn64_9.dll）
# 用法: .\fetch_cuda.ps1
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
$dir = Join-Path (Split-Path -Parent $PSScriptRoot) 'wheels'
New-Item -ItemType Directory -Force -Path $dir | Out-Null

$wheels = @(
    @{ name = 'cublas'; url = 'https://mirrors.aliyun.com/pypi/packages/20/e2/fc9a0e985249d873150276d5afb02e39a66817fedbf1a385724393e505ed/nvidia_cublas_cu12-12.9.2.10-py3-none-win_amd64.whl' },
    @{ name = 'cudnn'; url = 'https://mirrors.aliyun.com/pypi/packages/6f/5c/f77147ce7e27a4e9087fb34b0539ff085c68e7093e96ee85576fe31fe064/nvidia_cudnn_cu12-9.9.0.52-py3-none-win_amd64.whl' }
)

foreach ($w in $wheels) {
    $out = Join-Path $dir (Split-Path $w.url -Leaf)
    Write-Host "[get ] $($w.name) -> $(Split-Path $out -Leaf)"
    curl.exe -L -C - --retry 10 --retry-delay 5 --retry-all-errors `
        --connect-timeout 30 --max-time 14400 --no-progress-meter `
        -o "$out" $w.url
    if ($LASTEXITCODE -eq 0) {
        Write-Host ("[ok  ] {0}  {1:N1} MB" -f $w.name, ((Get-Item $out).Length / 1MB))
    }
    else {
        Write-Host "[FAIL] $($w.name) curl exit $LASTEXITCODE"
    }
}
Write-Host "[done] wheels in $dir"
Get-ChildItem $dir | ForEach-Object { "  {0}  {1:N1} MB" -f $_.Name, ($_.Length / 1MB) }
