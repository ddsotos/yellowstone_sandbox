param(
    [int]$Games = 1000,
    [int]$EvaluationSeed = 20260725
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

$statusPath = 'results\evaluations\epoch2_baselines.status.json'
$timingsPath = 'results\evaluations\epoch2_baselines.timings.json'
$timings = [ordered]@{}

function Write-Status {
    param([string]$Step, [string]$State, [string]$Message = '')
    $payload = [ordered]@{
        step = $Step
        state = $State
        message = $Message
        updated_at = (Get-Date).ToString('o')
        pid = $PID
    }
    $temporary = "$statusPath.tmp"
    $payload | ConvertTo-Json | Set-Content -Encoding UTF8 $temporary
    Move-Item -Force -LiteralPath $temporary -Destination $statusPath
}

function Invoke-PythonStep {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
    Write-Status -Step $Name -State 'running'
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    $watch.Stop()
    $timings[$Name] = $watch.Elapsed.TotalSeconds
    $timings | ConvertTo-Json | Set-Content -Encoding UTF8 $timingsPath
}

try {
    if (-not (Test-Path 'models\win_value_v2_generation0_197800_epoch002.pt')) {
        Invoke-PythonStep -Name 'train_v2_epoch002' -Arguments @(
            '-m', 'yellowstone.train_value_v2',
            '--data', 'data\v2_generation0_197800_tensors',
            '--checkpoint', 'models\win_value_v2_generation0_197800_epoch002.pt',
            '--epochs', '2',
            '--batch-size', '256',
            '--learning-rate', '1e-3',
            '--seed', '20260726'
        )
    }
    if (-not (Test-Path 'results\evaluations\v2_generation0_197800_epoch002_1000_same_seed_p0.json')) {
        Invoke-PythonStep -Name 'evaluate_v2_epoch002' -Arguments @(
            '-m', 'yellowstone.value_evaluation_v2',
            '--checkpoint', 'models\win_value_v2_generation0_197800_epoch002.pt',
            '--games', "$Games",
            '--seed', "$EvaluationSeed",
            '--player-index', '0',
            '--output', 'results\evaluations\v2_generation0_197800_epoch002_1000_same_seed_p0.json'
        )
    }
    if (-not (Test-Path 'models\win_value_v1_historyfix_generation0_197800_epoch002.pt')) {
        Invoke-PythonStep -Name 'train_v1_historyfix_epoch002' -Arguments @(
            '-m', 'yellowstone.train_value_historyfix',
            '--data', 'data\v1_historyfix_generation0_197800_canonical',
            '--checkpoint', 'models\win_value_v1_historyfix_generation0_197800_epoch002.pt',
            '--epochs', '2',
            '--batch-size', '256',
            '--learning-rate', '1e-3',
            '--seed', '20260727',
            '--training-games', '197800'
        )
    }
    if (-not (Test-Path 'results\evaluations\v1_historyfix_generation0_197800_epoch002_1000_same_seed_p0.json')) {
        Invoke-PythonStep -Name 'evaluate_v1_historyfix_epoch002' -Arguments @(
            '-m', 'yellowstone.evaluate_value',
            '--checkpoint', 'models\win_value_v1_historyfix_generation0_197800_epoch002.pt',
            '--games', "$Games",
            '--seed', "$EvaluationSeed",
            '--player-index', '0',
            '--adaptive-pq-pruning',
            '--approximate-new-color-neighbors',
            '--current-turn-history-only',
            '--output', 'results\evaluations\v1_historyfix_generation0_197800_epoch002_1000_same_seed_p0.json'
        )
    }
    Write-Status -Step 'complete' -State 'complete'
}
catch {
    Write-Status -Step 'failed' -State 'failed' -Message $_.Exception.Message
    throw
}
