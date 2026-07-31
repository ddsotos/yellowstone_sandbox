param(
    [int]$GamesPerSeat = 1000,
    [int]$EvaluationSeed = 20260725,
    [int]$TrainingSeed = 20260727
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUNBUFFERED = '1'

$source = 'data\v2_heuristic_one_vs_two_v1_10h_20260729'
$data = 'data\v1_original_new_88966_canonical'
$manifest = Join-Path $data 'conversion_manifest.json'
$evaluationDirectory = 'results\evaluations'
$statusPath = Join-Path $evaluationDirectory 'v1_original_new_50000_vs_88966.status.json'
$timingsPath = Join-Path $evaluationDirectory 'v1_original_new_50000_vs_88966.timings.json'
$comparisonPath = Join-Path $evaluationDirectory 'v1_original_new_50000_vs_88966.json'
$timings = [ordered]@{}
$lastCompleted = ''

New-Item -ItemType Directory -Force -Path $data, 'models', $evaluationDirectory |
    Out-Null
if (Test-Path -LiteralPath $timingsPath) {
    $existing = Get-Content -Raw -Encoding utf8 $timingsPath | ConvertFrom-Json
    foreach ($property in $existing.PSObject.Properties) {
        $timings[$property.Name] = [double]$property.Value
    }
}

function Write-AtomicJson {
    param([object]$Payload, [string]$Path)
    $temporary = "$Path.tmp"
    $Payload | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 $temporary
    Move-Item -Force -LiteralPath $temporary -Destination $Path
}

function Write-Status {
    param([string]$Step, [string]$State, [string]$Message = '')
    Write-AtomicJson -Path $statusPath -Payload ([ordered]@{
        step = $Step
        state = $State
        last_completed_step = $lastCompleted
        message = $Message
        updated_at = (Get-Date).ToString('o')
        pid = $PID
        source = $source
        data = $data
        comparison = $comparisonPath
        stdout = 'logs\v1_original_new_50000_vs_88966.stdout.log'
        stderr = 'logs\v1_original_new_50000_vs_88966.stderr.log'
    })
}

function Complete-Step {
    param([string]$Name)
    $script:lastCompleted = $Name
    Write-Status -Step $Name -State 'complete'
}

function Invoke-PythonStep {
    param([string]$Name, [string[]]$Arguments)
    Write-Status -Step $Name -State 'running'
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    $watch.Stop()
    $timings[$Name] = $watch.Elapsed.TotalSeconds
    Write-AtomicJson -Payload $timings -Path $timingsPath
    Complete-Step $Name
}

function Invoke-Evaluation {
    param(
        [int]$Size,
        [int]$PlayerIndex,
        [string]$Checkpoint
    )
    $name = "evaluate_${Size}_p${PlayerIndex}"
    $output = Join-Path $evaluationDirectory (
        "v1_original_new_${Size}_epoch001_" +
        "${GamesPerSeat}_same_seed_p${PlayerIndex}.json"
    )
    if (Test-Path -LiteralPath $output) {
        Complete-Step $name
        return
    }
    Invoke-PythonStep -Name $name -Arguments @(
        '-m', 'yellowstone.evaluate_value',
        '--checkpoint', $Checkpoint,
        '--games', "$GamesPerSeat",
        '--seed', "$EvaluationSeed",
        '--player-index', "$PlayerIndex",
        '--adaptive-pq-pruning',
        '--approximate-new-color-neighbors',
        '--output', $output
    )
}

try {
    if (-not (Test-Path -LiteralPath $manifest)) {
        Invoke-PythonStep -Name 'convert_88966' -Arguments @(
            '-m', 'yellowstone.convert_replay_v2_to_v1_original',
            '--source', $source,
            '--output', $data,
            '--expected-games', '88966',
            '--game-id-rebase', '954346',
            '--expected-source-game-id-min', '954346',
            '--expected-source-game-id-max', '1043311'
        )
    }
    else {
        Complete-Step 'convert_88966'
    }

    foreach ($size in @(50000, 88966)) {
        $checkpoint = "models\win_value_v1_original_new_${size}_epoch001.pt"
        if (-not (Test-Path -LiteralPath $checkpoint)) {
            Invoke-PythonStep -Name "train_${size}_epoch001" -Arguments @(
                '-m', 'yellowstone.train_value',
                '--data', $data,
                '--checkpoint', $checkpoint,
                '--epochs', '1',
                '--batch-size', '256',
                '--learning-rate', '1e-3',
                '--seed', "$TrainingSeed",
                '--split-game-count', '88966',
                '--train-game-id-limit', "$size",
                '--input-canonicalization', 'fast_lr_ud_color_v1',
                '--value-schema', 'yellowstone.value.v1',
                '--history-semantics', 'rolling_last_two_placements',
                '--training-games', "$size"
            )
        }
        else {
            Complete-Step "train_${size}_epoch001"
        }
    }

    foreach ($size in @(50000, 88966)) {
        $checkpoint = "models\win_value_v1_original_new_${size}_epoch001.pt"
        foreach ($playerIndex in 0..3) {
            Invoke-Evaluation `
                -Size $size `
                -PlayerIndex $playerIndex `
                -Checkpoint $checkpoint
        }
    }

    Invoke-PythonStep -Name 'summarize_comparison' -Arguments @(
        '-m', 'yellowstone.summarize_v1_new_collection',
        '--manifest', $manifest,
        '--checkpoint-50000',
        'models\win_value_v1_original_new_50000_epoch001.pt',
        '--checkpoint-88966',
        'models\win_value_v1_original_new_88966_epoch001.pt',
        '--evaluation-directory', $evaluationDirectory,
        '--timings', $timingsPath,
        '--output', $comparisonPath,
        '--games-per-seat', "$GamesPerSeat",
        '--evaluation-seed', "$EvaluationSeed",
        '--training-seed', "$TrainingSeed"
    )
    Write-Status -Step 'complete' -State 'complete'
}
catch {
    Write-Status -Step 'failed' -State 'failed' -Message $_.Exception.Message
    throw
}
