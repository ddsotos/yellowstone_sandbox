param(
    [string]$Source = 'D:\codex-backup\yellow_3_legacy_training_data_2026-07-27\heuristic_value_data_canonical_old',
    [int]$TrainingSeed = 20260728,
    [int]$EvaluationSeed = 20260725,
    [int]$EvaluationGames = 1000
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
$env:PYTHONUNBUFFERED = '1'

$trainingSizes = @(1000, 5000, 10000, 20000, 50000, 100000, 150000)
$modelDirectory = Join-Path $root 'models'
$resultDirectory = Join-Path $root 'results\evaluations'
New-Item -ItemType Directory -Force -Path $modelDirectory | Out-Null
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

function Size-Label([int]$Games) {
    if ($Games % 1000 -eq 0) {
        return "$([int]($Games / 1000))k"
    }
    return "$Games"
}

if (-not (Test-Path -LiteralPath $Source)) {
    throw "legacy source missing: $Source"
}

Write-ProgressLog 'V1 learning-curve pipeline started'
$rows = @()
foreach ($trainingGames in $trainingSizes) {
    $label = Size-Label $trainingGames
    $endPart = $trainingGames - 100
    $checkpointName = "win_value_v1_legacy_${label}_epoch001.pt"
    $checkpoint = Join-Path $modelDirectory $checkpointName
    $resultName = "v1_legacy_${label}_eval1000_same_seed_p0.json"
    $result = Join-Path $resultDirectory $resultName

    if (-not (Test-Path -LiteralPath $checkpoint)) {
        Write-ProgressLog "training games=$trainingGames range=0..$($trainingGames - 1)"
        Invoke-Python @(
            '-m', 'yellowstone.train_value',
            '--data', $Source,
            '--checkpoint', $checkpoint,
            '--start-part', '0',
            '--end-part', "$endPart",
            '--epochs', '1',
            '--batch-size', '256',
            '--seed', "$TrainingSeed",
            '--input-canonicalization', 'fast_lr_ud_color_v1',
            '--value-schema', 'yellowstone.value.v1',
            '--history-semantics', 'rolling_last_two_placements',
            '--training-games', "$trainingGames"
        )
    }

    if (-not (Test-Path -LiteralPath $result)) {
        Write-ProgressLog "evaluating games=$trainingGames player_index=0"
        Invoke-Python @(
            '-m', 'yellowstone.evaluate_value',
            '--checkpoint', $checkpoint,
            '--games', "$EvaluationGames",
            '--seed', "$EvaluationSeed",
            '--player-index', '0',
            '--adaptive-pq-pruning',
            '--approximate-new-color-neighbors',
            '--output', $result
        )
    }

    $payload = Get-Content -Raw -Encoding utf8 $result | ConvertFrom-Json
    $row = [ordered]@{
        training_games = $trainingGames
        actual_training_split_games = [int]($trainingGames * 8 / 10)
        validation_games = [int]($trainingGames / 10)
        test_games = [int]($trainingGames / 10)
        epochs = 1
        checkpoint = "models/$checkpointName"
        evaluation_games = [int]$payload.games
        player_index = 0
        wins = [double]$payload.wins
        win_rate = [double]$payload.win_rate
        result = "results/evaluations/$resultName"
    }
    $rows += $row

    # Persist a valid partial summary after every completed size.
    $partial = [ordered]@{
        status = 'running'
        updated_at = (Get-Date -Format o)
        source = $Source
        source_selection = 'first N game IDs'
        history_semantics = 'rolling_last_two_placements'
        training_seed = $TrainingSeed
        evaluation_seed = $EvaluationSeed
        evaluation_games = $EvaluationGames
        player_index = 0
        adaptive_pq_pruning = $true
        approximate_new_color_neighbors = $true
        rows = $rows
    }
    $partial | ConvertTo-Json -Depth 7 |
        Set-Content -LiteralPath (
            Join-Path $resultDirectory 'v1_learning_curve_p0_summary.json'
        ) -Encoding utf8
}

$summaryPath = Join-Path $resultDirectory 'v1_learning_curve_p0_summary.json'
$summary = Get-Content -Raw -Encoding utf8 $summaryPath | ConvertFrom-Json
$summary.status = 'complete'
$summary.updated_at = Get-Date -Format o
$summary | ConvertTo-Json -Depth 7 |
    Set-Content -LiteralPath $summaryPath -Encoding utf8

$lines = @(
    '# V1 learning curve — player index 0',
    '',
    "Training seed: $TrainingSeed; evaluation seed: $EvaluationSeed; evaluation: $EvaluationGames games.",
    '',
    '| Source games | Actual train split | Win rate | Wins |',
    '|---:|---:|---:|---:|'
)
foreach ($row in $rows) {
    $lines += "| $($row.training_games) | $($row.actual_training_split_games) | $([math]::Round(100 * $row.win_rate, 2))% | $([math]::Round($row.wins, 3)) |"
}
$lines += @(
    '',
    'Each checkpoint starts from the same random initialization and trains for one epoch on the first N game IDs.',
    'The game-level 80/10/10 split means the actual optimization set contains 80% of each source size.'
)
$lines | Set-Content -LiteralPath (
    Join-Path $resultDirectory 'v1_learning_curve_p0_summary.md'
) -Encoding utf8
Write-ProgressLog 'V1 learning-curve pipeline completed'
