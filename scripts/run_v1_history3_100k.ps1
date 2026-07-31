param(
    [string]$Source = 'D:\codex-backup\yellow_3_legacy_training_data_2026-07-27\heuristic_value_data',
    [int]$TrainingSeed = 20260728,
    [int]$EvaluationSeed = 20260725,
    [int]$EvaluationGames = 1000
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
$env:PYTHONUNBUFFERED = '1'

$data = Join-Path $root 'data\v1_history3_legacy_100k_canonical'
$checkpoint = Join-Path $root 'models\win_value_v1_history3_legacy_100k_epoch001.pt'
$result = Join-Path $root 'results\evaluations\v1_history3_legacy_100k_eval1000_same_seed_p0.json'
New-Item -ItemType Directory -Force -Path (Split-Path $checkpoint) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $result) | Out-Null

function Log([string]$Message) {
    Write-Output "$(Get-Date -Format o) $Message"
}

function Run-Python([string[]]$Arguments) {
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "python failed ($LASTEXITCODE): $($Arguments -join ' ')"
    }
}

Log 'V1 history3 100k pipeline started'
if (-not (Test-Path (Join-Path $data 'conversion_manifest.json'))) {
    Log 'converting first 100000 legacy games'
    Run-Python @(
        '-m', 'yellowstone.value_history3',
        '--source', $Source,
        '--output', $data,
        '--start-part', '0',
        '--end-part', '99900',
        '--expected-games', '100000'
    )
}
if (-not (Test-Path $checkpoint)) {
    Log 'training fresh history3 model for one epoch'
    Run-Python @(
        '-m', 'yellowstone.train_value',
        '--data', $data,
        '--checkpoint', $checkpoint,
        '--epochs', '1',
        '--batch-size', '256',
        '--seed', "$TrainingSeed",
        '--context-size', '129',
        '--input-canonicalization', 'fast_lr_ud_color_v1_history3',
        '--value-schema', 'yellowstone.value.v1_history3',
        '--history-semantics', 'three_prior_completed_turns_two_slots_each',
        '--training-games', '100000'
    )
}
if (-not (Test-Path $result)) {
    Log 'evaluating player_index=0 for 1000 games'
    Run-Python @(
        '-m', 'yellowstone.evaluate_value_history3',
        '--checkpoint', $checkpoint,
        '--games', "$EvaluationGames",
        '--seed', "$EvaluationSeed",
        '--player-index', '0',
        '--output', $result
    )
}
Log 'V1 history3 100k pipeline completed'
