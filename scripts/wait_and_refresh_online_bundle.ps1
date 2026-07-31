param(
    [int]$PipelineProcessId,
    [int]$PollSeconds = 60,
    [string]$Destination = '..\online_bundle',
    [string]$LogName = 'online_bundle_refresh.log'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
$checkpoint = Join-Path $root 'models\win_value_canonical_old_plus_6h_plus_8h_001.pt'
$log = Join-Path $root "logs\$LogName"
$exporter = Join-Path $root 'scripts\export_online_bundle.ps1'

try {
    "$(Get-Date -Format o) waiting checkpoint=$checkpoint pipeline_pid=$PipelineProcessId" |
        Add-Content $log
    while (-not (Test-Path $checkpoint)) {
        if (-not (Get-Process -Id $PipelineProcessId -ErrorAction SilentlyContinue)) {
            throw "pipeline process ended before final canonical checkpoint appeared"
        }
        Start-Sleep -Seconds $PollSeconds
    }
    "$(Get-Date -Format o) checkpoint ready; refreshing bundle" | Add-Content $log
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $exporter `
        -Destination $Destination 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) {
        throw "bundle exporter failed with exit code $LASTEXITCODE"
    }
    "$(Get-Date -Format o) completed" | Add-Content $log
}
catch {
    "$(Get-Date -Format o) failed: $_" | Add-Content $log
    exit 1
}
