param(
    [string]$Source = 'D:\codex-backup\yellow_3_legacy_training_data_2026-07-27\heuristic_value_data_canonical_old',
    [int]$Games = 1000,
    [int]$EvaluationSeed = 20260725,
    [int]$TrainingSeed = 20260728
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
$env:PYTHONUNBUFFERED = '1'

$fixedData = Join-Path $root 'data\v1_historyfix_legacy_200k_canonical'
$originalCheckpoint = Join-Path $root 'models\win_value_v1_legacy_200k_epoch001.pt'
$fixedCheckpoint = Join-Path $root 'models\win_value_v1_historyfix_legacy_200k_epoch001.pt'
$resultDirectory = Join-Path $root 'results\evaluations'
New-Item -ItemType Directory -Force -Path (Split-Path $originalCheckpoint) | Out-Null
New-Item -ItemType Directory -Force -Path $resultDirectory | Out-Null

function Write-ProgressLog([string]$Message) {
    Write-Output "$(Get-Date -Format o) $Message"
}

function Invoke-Python([string[]]$Arguments) {
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "python failed ($LASTEXITCODE): $($Arguments -join ' ')"
    }
}

Write-ProgressLog 'pipeline started'
if (-not (Test-Path -LiteralPath $Source)) {
    throw "legacy source missing: $Source"
}

$fixedManifest = Join-Path $fixedData 'conversion_manifest.json'
if (-not (Test-Path -LiteralPath $fixedManifest)) {
    Write-ProgressLog 'transforming first 200000 games to fixed history'
    Invoke-Python @(
        '-m', 'yellowstone.transform_v1_historyfix',
        '--source', $Source,
        '--output', $fixedData,
        '--start-part', '0',
        '--end-part', '199900',
        '--expected-games', '200000'
    )
}

if (-not (Test-Path -LiteralPath $originalCheckpoint)) {
    Write-ProgressLog 'training original-history V1 model'
    Invoke-Python @(
        '-m', 'yellowstone.train_value',
        '--data', $Source,
        '--checkpoint', $originalCheckpoint,
        '--start-part', '0',
        '--end-part', '199900',
        '--epochs', '1',
        '--batch-size', '256',
        '--seed', "$TrainingSeed",
        '--input-canonicalization', 'fast_lr_ud_color_v1',
        '--value-schema', 'yellowstone.value.v1',
        '--history-semantics', 'rolling_last_two_placements',
        '--training-games', '200000'
    )
}

if (-not (Test-Path -LiteralPath $fixedCheckpoint)) {
    Write-ProgressLog 'training fixed-history V1 model'
    Invoke-Python @(
        '-m', 'yellowstone.train_value_historyfix',
        '--data', $fixedData,
        '--checkpoint', $fixedCheckpoint,
        '--epochs', '1',
        '--batch-size', '256',
        '--seed', "$TrainingSeed",
        '--training-games', '200000'
    )
}

function Evaluate-Model(
    [string]$Name,
    [string]$Checkpoint,
    [bool]$CurrentTurnHistoryOnly
) {
    $seats = @()
    foreach ($playerIndex in 0..3) {
        $filename = "${Name}_1000_same_seed_p${playerIndex}.json"
        $output = Join-Path $resultDirectory $filename
        if (-not (Test-Path -LiteralPath $output)) {
            Write-ProgressLog "evaluating $Name seat=$playerIndex"
            $arguments = @(
                '-m', 'yellowstone.evaluate_value',
                '--checkpoint', $Checkpoint,
                '--games', "$Games",
                '--seed', "$EvaluationSeed",
                '--player-index', "$playerIndex",
                '--adaptive-pq-pruning',
                '--approximate-new-color-neighbors',
                '--output', $output
            )
            if ($CurrentTurnHistoryOnly) {
                $arguments += '--current-turn-history-only'
            }
            Invoke-Python $arguments
        }
        $payload = Get-Content -Raw -Encoding utf8 $output | ConvertFrom-Json
        $seats += [ordered]@{
            player_index = $playerIndex
            games = [int]$payload.games
            wins = [double]$payload.wins
            win_rate = [double]$payload.win_rate
            result = "results/evaluations/$filename"
        }
    }
    $summary = [ordered]@{
        checkpoint = $Checkpoint.Replace($root + '\', '')
        training_games = 200000
        fresh_initialization = $true
        games_per_seat = $Games
        seed = $EvaluationSeed
        adaptive_pq_pruning = $true
        approximate_new_color_neighbors = $true
        current_turn_history_only = $CurrentTurnHistoryOnly
        seats = $seats
    }
    $summaryPath = Join-Path $resultDirectory "${Name}_1000_same_seed_all_seats.json"
    $summary | ConvertTo-Json -Depth 6 |
        Set-Content -LiteralPath $summaryPath -Encoding utf8
    return $summary
}

$original = Evaluate-Model 'v1_legacy_200k' $originalCheckpoint $false
$fixed = Evaluate-Model 'v1_historyfix_legacy_200k' $fixedCheckpoint $true
$oldPath = Join-Path $resultDirectory 'canonical_old_1000_same_seed_all_seats.json'
$old = $null
if (Test-Path -LiteralPath $oldPath) {
    $old = Get-Content -Raw -Encoding utf8 $oldPath | ConvertFrom-Json
}

$comparison = [ordered]@{
    generated_at = (Get-Date -Format o)
    source = $Source
    source_game_range = '0..199999'
    training_seed = $TrainingSeed
    evaluation_seed = $EvaluationSeed
    games_per_seat = $Games
    original_history_200k = $original
    fixed_history_200k = $fixed
    canonical_old_660001 = $old
}
$comparisonJson = Join-Path $resultDirectory 'v1_legacy_200k_comparison.json'
$comparison | ConvertTo-Json -Depth 10 |
    Set-Content -LiteralPath $comparisonJson -Encoding utf8

$lines = @(
    '# V1 legacy 200k comparison',
    '',
    "Generated: $($comparison.generated_at)",
    '',
    '| Model | Training games | History | Seat 0 | Seat 1 | Seat 2 | Seat 3 |',
    '|---|---:|---|---:|---:|---:|---:|',
    "| Original V1 200k | 200,000 | rolling last two placements | $([math]::Round(100*$original.seats[0].win_rate,2))% | $([math]::Round(100*$original.seats[1].win_rate,2))% | $([math]::Round(100*$original.seats[2].win_rate,2))% | $([math]::Round(100*$original.seats[3].win_rate,2))% |",
    "| History-fixed V1 200k | 200,000 | evaluated turn only | $([math]::Round(100*$fixed.seats[0].win_rate,2))% | $([math]::Round(100*$fixed.seats[1].win_rate,2))% | $([math]::Round(100*$fixed.seats[2].win_rate,2))% | $([math]::Round(100*$fixed.seats[3].win_rate,2))% |"
)
if ($null -ne $old) {
    $lines += "| Canonical old | 660,001 | rolling last two placements | $([math]::Round(100*$old.seats.'0'.win_rate,2))% | $([math]::Round(100*$old.seats.'1'.win_rate,2))% | $([math]::Round(100*$old.seats.'2'.win_rate,2))% | $([math]::Round(100*$old.seats.'3'.win_rate,2))% |"
}
$lines += @(
    '',
    'Both 200k models start from random initialization and use exactly game IDs 0..199999 for one epoch.',
    'Evaluation uses the same seed, adaptive p/q pruning, approximate new-color neighbor limits, and three heuristic opponents.'
)
$lines | Set-Content -LiteralPath (
    Join-Path $resultDirectory 'v1_legacy_200k_comparison.md'
) -Encoding utf8
Write-ProgressLog 'pipeline completed'
