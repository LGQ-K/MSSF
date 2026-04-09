param(
    [string]$Config = "configs/centerpoint/centerpoint_voxel005_second_secfpn_8xb4-cyclic-20e_tj4d_mm.py",
    [string]$WorkDir = "work_dirs/centerpoint_voxel005_second_secfpn_8xb4-cyclic-20e_tj4d_mm",
    [string]$PythonExe = "python",
    [string]$ExtraTestArgs = ""
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Test-Path $Config)) {
    throw "Config file not found: $Config"
}
if (-not (Test-Path $WorkDir)) {
    throw "Work dir not found: $WorkDir"
}

$bestCkpt = Get-ChildItem -Path $WorkDir -Filter "best_*.pth" -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $bestCkpt) {
    throw "No best_*.pth found. Train first and enable save_best."
}

Write-Host "Using best checkpoint:" $bestCkpt.FullName
Write-Host "Start evaluation..."

$cmd = "$PythonExe tools/test.py `"$Config`" `"$($bestCkpt.FullName)`""
if ($ExtraTestArgs -and $ExtraTestArgs.Trim().Length -gt 0) {
    $cmd = "$cmd $ExtraTestArgs"
}

Write-Host "Run command:" $cmd
Invoke-Expression $cmd
