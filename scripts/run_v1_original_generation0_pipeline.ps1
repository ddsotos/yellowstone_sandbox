param(
    [int]$Games = 1000,
    [int]$EvaluationSeed = 20260725,
    [int]$TrainingSeed = 20260727
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUNBUFFERED = '1'

$statusPath = 'results\evaluations\v1_original_generation0.status.json'
$timingsPath = 'results\evaluations\v1_original_generation0.timings.json'
$baselineStatusPath = 'results\evaluations\epoch2_baselines.status.json'
$comparisonPath = 'results\evaluations\v2_v1_original_history_comparison.json'
$timings = [ordered]@{}
$lastCompleted = ''

if (Test-Path $timingsPath) {
    $existingTimings = Get-Content -Raw -Encoding utf8 $timingsPath |
        ConvertFrom-Json
    foreach ($property in $existingTimings.PSObject.Properties) {
        $timings[$property.Name] = [double]$property.Value
    }
}

function Write-Status {
    param([string]$Step, [string]$State, [string]$Message = '')
    $payload = [ordered]@{
        step = $Step
        state = $State
        last_completed_step = $lastCompleted
        message = $Message
        updated_at = (Get-Date).ToString('o')
        pid = $PID
    }
    $temporary = "$statusPath.tmp"
    $payload | ConvertTo-Json | Set-Content -Encoding UTF8 $temporary
    Move-Item -Force -LiteralPath $temporary -Destination $statusPath
}

function Complete-Step {
    param([string]$Name)
    $script:lastCompleted = $Name
    Write-Status -Step $Name -State 'complete'
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
    Complete-Step $Name
}

function Invoke-OriginalEvaluation {
    param(
        [string]$Name,
        [string]$Checkpoint,
        [int]$GameCount,
        [int]$PlayerIndex,
        [string]$Output
    )
    if (Test-Path $Output) {
        Complete-Step $Name
        return
    }
    Invoke-PythonStep -Name $Name -Arguments @(
        '-m', 'yellowstone.evaluate_value',
        '--checkpoint', $Checkpoint,
        '--games', "$GameCount",
        '--seed', "$EvaluationSeed",
        '--player-index', "$PlayerIndex",
        '--adaptive-pq-pruning',
        '--approximate-new-color-neighbors',
        '--output', $Output
    )
}

try {
    $manifest = 'data\v1_original_generation0_197800_canonical\conversion_manifest.json'
    if (-not (Test-Path $manifest)) {
        Invoke-PythonStep -Name 'convert_v1_original' -Arguments @(
            '-m', 'yellowstone.convert_replay_v2_to_v1_original',
            '--source', 'data\v2_generation0_200k_frame_features',
            '--output', 'data\v1_original_generation0_197800_canonical',
            '--expected-games', '197800',
            '--reference', 'data\v1_historyfix_generation0_197800_canonical'
        )
    }
    else {
        Complete-Step 'convert_v1_original'
    }

    $epoch1 = 'models\win_value_v1_original_generation0_197800_epoch001.pt'
    if (-not (Test-Path $epoch1)) {
        Invoke-PythonStep -Name 'train_v1_original_epoch001' -Arguments @(
            '-m', 'yellowstone.train_value',
            '--data', 'data\v1_original_generation0_197800_canonical',
            '--checkpoint', $epoch1,
            '--epochs', '1',
            '--batch-size', '256',
            '--learning-rate', '1e-3',
            '--seed', "$TrainingSeed",
            '--input-canonicalization', 'fast_lr_ud_color_v1',
            '--value-schema', 'yellowstone.value.v1',
            '--history-semantics', 'rolling_last_two_placements',
            '--training-games', '197800'
        )
    }
    else {
        Complete-Step 'train_v1_original_epoch001'
    }
    Invoke-OriginalEvaluation `
        -Name 'smoke_v1_original_epoch001' `
        -Checkpoint $epoch1 `
        -GameCount 3 `
        -PlayerIndex 0 `
        -Output 'results\smoke\v1_original_generation0_epoch001_3.json'
    Invoke-OriginalEvaluation `
        -Name 'evaluate_v1_original_epoch001' `
        -Checkpoint $epoch1 `
        -GameCount $Games `
        -PlayerIndex 0 `
        -Output 'results\evaluations\v1_original_generation0_197800_epoch001_1000_same_seed_p0.json'

    $epoch2 = 'models\win_value_v1_original_generation0_197800_epoch002.pt'
    if (-not (Test-Path $epoch2)) {
        Invoke-PythonStep -Name 'train_v1_original_epoch002' -Arguments @(
            '-m', 'yellowstone.train_value',
            '--data', 'data\v1_original_generation0_197800_canonical',
            '--checkpoint', $epoch2,
            '--epochs', '2',
            '--batch-size', '256',
            '--learning-rate', '1e-3',
            '--seed', "$TrainingSeed",
            '--input-canonicalization', 'fast_lr_ud_color_v1',
            '--value-schema', 'yellowstone.value.v1',
            '--history-semantics', 'rolling_last_two_placements',
            '--training-games', '197800'
        )
    }
    else {
        Complete-Step 'train_v1_original_epoch002'
    }
    Invoke-OriginalEvaluation `
        -Name 'smoke_v1_original_epoch002' `
        -Checkpoint $epoch2 `
        -GameCount 3 `
        -PlayerIndex 0 `
        -Output 'results\smoke\v1_original_generation0_epoch002_3.json'
    Invoke-OriginalEvaluation `
        -Name 'evaluate_v1_original_epoch002' `
        -Checkpoint $epoch2 `
        -GameCount $Games `
        -PlayerIndex 0 `
        -Output 'results\evaluations\v1_original_generation0_197800_epoch002_1000_same_seed_p0.json'

    Write-Status -Step 'wait_epoch2_baselines' -State 'running'
    while ($true) {
        if (Test-Path $baselineStatusPath) {
            $baselineStatus = Get-Content -Raw -Encoding utf8 $baselineStatusPath |
                ConvertFrom-Json
            if ($baselineStatus.state -eq 'complete') {
                break
            }
            if ($baselineStatus.state -eq 'failed') {
                throw "epoch2 baseline pipeline failed: $($baselineStatus.message)"
            }
        }
        Start-Sleep -Seconds 30
    }
    Complete-Step 'wait_epoch2_baselines'

    Invoke-PythonStep -Name 'summarize_six_conditions' -Arguments @(
        '-m', 'yellowstone.summarize_history_comparison',
        '--root', $root,
        '--output', $comparisonPath
    )
    $comparison = Get-Content -Raw -Encoding utf8 $comparisonPath |
        ConvertFrom-Json
    if ($comparison.all_seats_gate.run_all_seats) {
        $bestKey = [string]$comparison.all_seats_gate.best_original_key
        $bestCheckpoint = [string]$comparison.all_seats_gate.best_original_checkpoint
        $epochLabel = if ($bestKey.EndsWith('epoch001')) { 'epoch001' } else { 'epoch002' }
        $seats = @()
        foreach ($playerIndex in 0..3) {
            $output = "results\evaluations\v1_original_generation0_197800_${epochLabel}_1000_same_seed_p${playerIndex}.json"
            Invoke-OriginalEvaluation `
                -Name "evaluate_best_original_p${playerIndex}" `
                -Checkpoint $bestCheckpoint `
                -GameCount $Games `
                -PlayerIndex $playerIndex `
                -Output $output
            $payload = Get-Content -Raw -Encoding utf8 $output |
                ConvertFrom-Json
            $seats += [ordered]@{
                player_index = $playerIndex
                games = [int]$payload.games
                wins = [double]$payload.wins
                win_rate = [double]$payload.win_rate
                result = $output.Replace('\', '/')
            }
        }
        $allSeats = [ordered]@{
            checkpoint = $bestCheckpoint
            model_key = $bestKey
            games_per_seat = $Games
            seed = $EvaluationSeed
            adaptive_pq_pruning = $true
            approximate_new_color_neighbors = $true
            traditional_refill = $true
            seats = $seats
        }
        $allSeats | ConvertTo-Json -Depth 6 |
            Set-Content -Encoding utf8 (
                "results\evaluations\v1_original_generation0_197800_${epochLabel}_1000_same_seed_all_seats.json"
            )
        Complete-Step 'summarize_best_original_all_seats'
    }
    Write-Status -Step 'complete' -State 'complete'
}
catch {
    Write-Status -Step 'failed' -State 'failed' -Message $_.Exception.Message
    throw
}
