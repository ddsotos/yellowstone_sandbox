param(
    [string]$DataPath = 'data\heuristic_value_data_6h_only',
    [string]$Checkpoint = 'models\win_value_6h_only_001.pt',
    [string]$ResumeCheckpoint = 'models\win_value_660k_1epoch.pt',
    [int]$Epochs = 1,
    [int]$BatchSize = 256,
    [string]$LogName = 'training_6h_only_from_old.log'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
$logDirectory = Join-Path $root 'logs'
$log = Join-Path $logDirectory $LogName

try {
    New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
    "$(Get-Date -Format o) data=$DataPath resume=$ResumeCheckpoint checkpoint=$Checkpoint epochs=$Epochs batch_size=$BatchSize" |
        Add-Content $log
    & python -m yellowstone.train_value `
        --data $DataPath `
        --checkpoint $Checkpoint `
        --resume $ResumeCheckpoint `
        --epochs $Epochs `
        --batch-size $BatchSize 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) {
        throw "training failed with exit code $LASTEXITCODE"
    }
    "$(Get-Date -Format o) completed checkpoint=$Checkpoint" | Add-Content $log
} catch {
    "$(Get-Date -Format o) failed: $_" | Add-Content $log
    exit 1
}
