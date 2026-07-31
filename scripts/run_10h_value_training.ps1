param(
    [int]$InitialTrainingProcessId,
    [int]$Hours = 10,
    [int]$EpochsPerRound = 5
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
$deadline = (Get-Date).AddHours($Hours)
$logDirectory = Join-Path $root 'logs'
$dataDirectory = Join-Path $root 'data\heuristic_value_data'
$modelDirectory = Join-Path $root 'models'
$log = Join-Path $logDirectory 'training_10h.log'

try {
    New-Item -ItemType Directory -Force -Path $logDirectory, $modelDirectory | Out-Null
    $initial = Get-Process -Id $InitialTrainingProcessId -ErrorAction SilentlyContinue
    if ($initial) {
        Wait-Process -Id $InitialTrainingProcessId
    }

    $checkpoint = if (Test-Path (Join-Path $modelDirectory 'win_value_40.pt')) {
        Join-Path $modelDirectory 'win_value_40.pt'
    } else {
        Join-Path $modelDirectory 'win_value.pt'
    }
    $round = 1
    while ((Get-Date) -lt $deadline) {
        $next = Join-Path $modelDirectory ("win_value_10h_round_{0:D3}.pt" -f $round)
        "$(Get-Date -Format o) round=$round resume=$checkpoint" | Add-Content $log
        & python -m yellowstone.train_value `
            --data $dataDirectory `
            --checkpoint $next `
            --resume $checkpoint `
            --epochs $EpochsPerRound `
            --batch-size 256 2>&1 | Tee-Object -FilePath $log -Append
        if ($LASTEXITCODE -ne 0) {
            throw "training failed with exit code $LASTEXITCODE"
        }
        Copy-Item -LiteralPath $next -Destination (Join-Path $modelDirectory 'win_value_latest.pt') -Force
        $checkpoint = $next
        $round++
    }
    "$(Get-Date -Format o) completed rounds=$($round - 1)" | Add-Content $log
} catch {
    "$(Get-Date -Format o) failed: $_" | Add-Content $log
    exit 1
}
