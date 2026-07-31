param(
    [double]$CollectionHours = 10,
    [int]$SmokeTurns = 1000,
    [int]$SmokeGamesPerSeat = 100,
    [int]$GamesPerSeat = 1000,
    [int]$EvaluationSeed = 20260725,
    [int]$TrainingSeed = 20260727
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'

$statusPath = 'results\evaluations\privileged_action_delta.status.json'
$timingsPath = 'results\evaluations\privileged_action_delta.timings.json'
$source = 'data\v2_generation0_200k_frame_features'
$continuation = 'data\v2_heuristic_one_vs_two_v1_continuation_20260729'
$criticData = 'data\privileged_state_generation0_197800_tensors'
$deltaData = 'data\action_delta_next_turn_continuation'
$criticPrefix = 'models\privileged_state_generation0_197800'
$criticSelection = 'results\evaluations\privileged_state_generation0_197800.selection.json'
$proposerSelection = 'results\evaluations\action_delta_proposer.selection.json'
$baselineCheckpoint = 'models\win_value_v1_original_generation0_197800_epoch002.pt'
$conv3Status = 'results\evaluations\v1_original_conv3.status.json'
$conv3Summary = 'results\evaluations\v1_original_conv3_generation0_197800_epoch002.json'
$conv3Checkpoint = 'models\win_value_v1_original_conv3_generation0_197800_epoch002.pt'
$smokeCheckpoint = 'models\action_delta_next_turn_smoke.pt'
$finalCheckpoint = 'models\action_delta_next_turn_10h.pt'
$summary = 'results\evaluations\privileged_action_delta_next_turn_10h.json'
$lastCompleted = ''
$timings = [ordered]@{}

New-Item -ItemType Directory -Force -Path (
    'models', 'results\evaluations', 'results\smoke', 'logs'
) | Out-Null

function Write-AtomicJson {
    param([object]$Payload, [string]$Path)
    $temporary = "$Path.tmp"
    $Payload | ConvertTo-Json -Depth 10 | Set-Content -Encoding utf8 $temporary
    Move-Item -Force -LiteralPath $temporary -Destination $Path
}

function Write-Status {
    param([string]$Step, [string]$State, [string]$Message = '')
    Write-AtomicJson -Path $statusPath -Payload ([ordered]@{
        state = $State
        step = $Step
        last_completed_step = $lastCompleted
        message = $Message
        updated_at = (Get-Date).ToString('o')
        pid = $PID
        source = $source
        continuation = $continuation
        critic_data = $criticData
        delta_data = $deltaData
        critic_selection = $criticSelection
        proposer_selection = $proposerSelection
        checkpoint = $finalCheckpoint
        summary = $summary
        stdout = 'logs\privileged_action_delta.stdout.log'
        stderr = 'logs\privileged_action_delta.stderr.log'
    })
}

function Invoke-Step {
    param([string]$Name, [string[]]$Arguments)
    Write-Status -Step $Name -State 'running'
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $attempt = 0
    do {
        $attempt++
        & python @Arguments
        $code = $LASTEXITCODE
    } while ($code -ne 0 -and $attempt -lt 2)
    if ($code -ne 0) {
        throw "$Name failed after $attempt attempts with exit code $code"
    }
    $watch.Stop()
    $timings[$Name] = $watch.Elapsed.TotalSeconds
    Write-AtomicJson -Path $timingsPath -Payload $timings
    $script:lastCompleted = $Name
    Write-Status -Step $Name -State 'complete'
}

try {
    Write-Status -Step 'waiting_for_conv3' -State 'waiting'
    while ($true) {
        $convState = $null
        $convPid = $null
        if (Test-Path -LiteralPath $conv3Status) {
            $conv = Get-Content -Raw -Encoding utf8 $conv3Status | ConvertFrom-Json
            $convState = $conv.state
            $convPid = $conv.pid
        }
        if ($convState -eq 'complete' -or $convState -eq 'failed') {
            break
        }
        if ($convPid -and -not (Get-Process -Id $convPid -ErrorAction SilentlyContinue)) {
            break
        }
        Start-Sleep -Seconds 30
        Write-Status -Step 'waiting_for_conv3' -State 'waiting'
    }

    $proposer = $baselineCheckpoint
    $reason = 'conv3_not_strictly_better_or_unverifiable'
    if (
        (Test-Path -LiteralPath $conv3Summary) -and
        (Test-Path -LiteralPath $conv3Checkpoint)
    ) {
        $comparison = Get-Content -Raw -Encoding utf8 $conv3Summary |
            ConvertFrom-Json
        if (
            [int]$comparison.conv3.all_seats_games -eq 4000 -and
            [double]$comparison.conv3.all_seats_win_rate -gt
                [double]$comparison.baseline.all_seats_win_rate
        ) {
            $proposer = $conv3Checkpoint
            $reason = 'conv3_strictly_better_all_seats'
        }
    }
    Write-AtomicJson -Path $proposerSelection -Payload ([ordered]@{
        checkpoint = $proposer
        reason = $reason
        selected_at = (Get-Date).ToString('o')
    })
    $lastCompleted = 'select_proposer'

    $criticManifest = Join-Path $criticData 'manifest.json'
    if (-not (Test-Path -LiteralPath $criticManifest)) {
        Invoke-Step 'convert_privileged_state' @(
            '-m', 'yellowstone.convert_privileged_state',
            '--source', $source, '--output', $criticData
        )
    }
    if (-not (Test-Path -LiteralPath $criticSelection)) {
        Invoke-Step 'train_privileged_state' @(
            '-m', 'yellowstone.train_privileged_state',
            '--data', $criticData,
            '--checkpoint-prefix', $criticPrefix,
            '--epochs', '2', '--batch-size', '256',
            '--learning-rate', '1e-3', '--seed', "$TrainingSeed",
            '--selection-output', $criticSelection
        )
    }
    $criticFacts = Get-Content -Raw -Encoding utf8 $criticSelection |
        ConvertFrom-Json
    $critic = [string]$criticFacts.selected.checkpoint

    $progressPath = Join-Path $deltaData 'collection_progress.json'
    $turns = 0
    if (Test-Path -LiteralPath $progressPath) {
        $turns = [int]((
            Get-Content -Raw -Encoding utf8 $progressPath | ConvertFrom-Json
        ).turns)
    }
    if ($turns -lt $SmokeTurns) {
        Invoke-Step 'collect_delta_smoke' @(
            '-m', 'yellowstone.collect_action_delta',
            '--source', $continuation, '--output', $deltaData,
            '--proposer-checkpoint', $proposer,
            '--critic-checkpoint', $critic,
            '--max-turns', "$SmokeTurns", '--shard-turns', '100'
        )
    }
    if (-not (Test-Path -LiteralPath $smokeCheckpoint)) {
        Invoke-Step 'train_delta_smoke' @(
            '-m', 'yellowstone.train_action_delta',
            '--data', $deltaData, '--checkpoint', $smokeCheckpoint,
            '--epochs', '1', '--seed', "$TrainingSeed"
        )
    }
    foreach ($player in 0..3) {
        $path = "results\smoke\action_delta_${SmokeGamesPerSeat}_p${player}.json"
        if (-not (Test-Path -LiteralPath $path)) {
            Invoke-Step "evaluate_delta_smoke_p${player}" @(
                '-m', 'yellowstone.evaluate_action_delta',
                '--proposer-checkpoint', $proposer,
                '--delta-checkpoint', $smokeCheckpoint,
                '--games', "$SmokeGamesPerSeat",
                '--seed', "$EvaluationSeed", '--player-index', "$player",
                '--output', $path
            )
        }
    }

    Invoke-Step 'collect_delta_10h' @(
        '-m', 'yellowstone.collect_action_delta',
        '--source', $continuation, '--output', $deltaData,
        '--proposer-checkpoint', $proposer,
        '--critic-checkpoint', $critic,
        '--duration-hours', "$CollectionHours", '--shard-turns', '100'
    )
    Invoke-Step 'train_delta_final' @(
        '-m', 'yellowstone.train_action_delta',
        '--data', $deltaData, '--checkpoint', $finalCheckpoint,
        '--epochs', '1', '--seed', "$TrainingSeed"
    )
    foreach ($player in 0..3) {
        $path = "results\evaluations\action_delta_${GamesPerSeat}_seed${EvaluationSeed}_p${player}.json"
        if (-not (Test-Path -LiteralPath $path)) {
            Invoke-Step "evaluate_delta_p${player}" @(
                '-m', 'yellowstone.evaluate_action_delta',
                '--proposer-checkpoint', $proposer,
                '--delta-checkpoint', $finalCheckpoint,
                '--games', "$GamesPerSeat",
                '--seed', "$EvaluationSeed", '--player-index', "$player",
                '--output', $path
            )
        }
    }
    Invoke-Step 'summarize' @(
        '-m', 'yellowstone.summarize_action_delta',
        '--proposer-selection', $proposerSelection,
        '--critic-selection', $criticSelection,
        '--delta-checkpoint', $finalCheckpoint,
        '--evaluation-directory', 'results\evaluations',
        '--output', $summary, '--games-per-seat', "$GamesPerSeat",
        '--seed', "$EvaluationSeed"
    )
    Write-Status -Step 'complete' -State 'complete'
}
catch {
    Write-Status -Step 'failed' -State 'failed' -Message $_.Exception.Message
    throw
}
