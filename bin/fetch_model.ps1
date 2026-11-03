# 从 hf-mirror.com 下载 faster-whisper 模型（走传统 resolve 路径 + curl 断点续传）
# 用法: .\fetch_model.ps1 -Repo Systran/faster-whisper-large-v3 -Dest D:\tools\v2t\models\faster-whisper-large-v3
param(
    [Parameter(Mandatory = $true)][string]$Repo,
    [Parameter(Mandatory = $true)][string]$Dest,
    [string]$Endpoint = 'https://hf-mirror.com'
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
New-Item -ItemType Directory -Force -Path $Dest | Out-Null

Write-Host "[list] $Endpoint/api/models/$Repo"
$api = curl.exe -sL --max-time 40 "$Endpoint/api/models/$Repo"
if ($LASTEXITCODE -ne 0 -or -not $api) { throw "无法获取文件列表" }

$files = ($api | ConvertFrom-Json).siblings.rfilename |
    Where-Object { $_ -notmatch '^\.' -and $_ -ne 'README.md' }
if (-not $files) { throw "文件列表为空" }
Write-Host "[list] 待下载 $($files.Count) 个文件: $($files -join ', ')"

$failed = @()
foreach ($f in $files) {
    $out = Join-Path $Dest $f
    $url = "$Endpoint/$Repo/resolve/main/$f"
    Write-Host "[get ] $f"

    # -C - 断点续传；--retry 反复重试；大文件慢，超时给足
    curl.exe -L -C - --retry 8 --retry-delay 3 --retry-all-errors `
        --connect-timeout 30 --max-time 7200 --no-progress-meter `
        -o "$out" "$url"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] $f (curl exit $LASTEXITCODE)"
        $failed += $f
    }
    else {
        $sz = (Get-Item $out).Length
        Write-Host ("[ok  ] {0}  {1:N1} MB" -f $f, ($sz / 1MB))
        if ($sz -eq 0) { $failed += "$f (0 字节)" }
    }
}

$total = (Get-ChildItem $Dest -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("[done] $Dest 共 {0:N1} MB" -f $total)
if ($failed) { Write-Host "[WARN] 失败: $($failed -join ', ')"; exit 1 }
exit 0
