param(
    [int]$Games = 100,
    [int]$Seed = 20260724
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
$logDirectory = Join-Path $root 'logs'
$resultDirectory = Join-Path $root 'results\evaluations'
$log = Join-Path $logDirectory 'model_comparison.log'

try {
    New-Item -ItemType Directory -Force -Path $logDirectory, $resultDirectory | Out-Null
    foreach ($model in @(
        @{ Name = 'old'; Checkpoint = 'models\win_value.pt' },
        @{ Name = 'new'; Checkpoint = 'models\win_value_660k_1epoch.pt' }
    )) {
        "$(Get-Date -Format o) model=$($model.Name) games=$Games seed=$Seed" | Add-Content $log
        & python -m yellowstone.evaluate_value `
            --checkpoint $model.Checkpoint `
            --games $Games `
            --seed $Seed `
            --output (Join-Path $resultDirectory "evaluation_$($model.Name)_same_seed.json") 2>&1 |
            Tee-Object -FilePath $log -Append
        if ($LASTEXITCODE -ne 0) {
            throw "evaluation failed for $($model.Name) with exit code $LASTEXITCODE"
        }
    }
} catch {
    "$(Get-Date -Format o) failed: $_" | Add-Content $log
    exit 1
}
