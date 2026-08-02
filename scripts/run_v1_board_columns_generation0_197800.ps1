param(
    [int]$MilestoneGames = 100,
    [int]$GamesPerSeat = 1000,
    [int]$EvaluationSeed = 20260725,
    [int]$TrainingSeed = 20260727
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUNBUFFERED = '1'

$source = 'data\v1_original_generation0_197800_canonical'
$data = 'data\v1_board_columns_generation0_197800'
$evaluationDirectory = 'results\evaluations'
$statusPath = Join-Path $evaluationDirectory 'v1_board_columns_generation0_197800.status.json'
$timingsPath = Join-Path $evaluationDirectory 'v1_board_columns_generation0_197800.timings.json'
$summaryPath = Join-Path $evaluationDirectory 'v1_board_columns_generation0_197800.json'
$timings = [ordered]@{}
$lastCompleted = ''

New-Item -ItemType Directory -Force -Path $data, 'models', $evaluationDirectory, 'logs' |
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
    $Payload | ConvertTo-Json -Depth 12 | Set-Content -Encoding utf8 $temporary
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
        summary = $summaryPath
        stdout = 'logs\v1_board_columns_generation0_197800.stdout.log'
        stderr = 'logs\v1_board_columns_generation0_197800.stderr.log'
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
        [string]$Name,
        [string]$Checkpoint,
        [int]$Games,
        [int]$PlayerIndex,
        [string]$Output
    )
    if (Test-Path -LiteralPath $Output) {
        Complete-Step $Name
        return
    }
    Invoke-PythonStep -Name $Name -Arguments @(
        '-m', 'yellowstone.evaluate_value',
        '--checkpoint', $Checkpoint,
        '--games', "$Games",
        '--seed', "$EvaluationSeed",
        '--player-index', "$PlayerIndex",
        '--adaptive-pq-pruning',
        '--approximate-new-color-neighbors',
        '--output', $Output
    )
}

function Read-Evaluation {
    param([string]$Path)
    $payload = Get-Content -Raw -Encoding utf8 $Path | ConvertFrom-Json
    return [ordered]@{
        path = $Path
        games = [int]$payload.games
        wins = [double]$payload.wins
        win_rate = [double]$payload.win_rate
        evaluated_player_one_card_turns = [int]$payload.evaluated_player_one_card_turns
        evaluated_player_two_card_turns = [int]$payload.evaluated_player_two_card_turns
        evaluated_player_one_card_turn_rate = [double]$payload.evaluated_player_one_card_turn_rate
    }
}

try {
    if (-not (Test-Path -LiteralPath (Join-Path $data 'conversion_manifest.json'))) {
        Invoke-PythonStep -Name 'convert_board_columns' -Arguments @(
            '-m', 'yellowstone.convert_v1_canonical_to_board_columns',
            '--source', $source,
            '--output', $data,
            '--expected-games', '197800'
        )
    }
    else {
        Complete-Step 'convert_board_columns'
    }

    $sizes = @(30000, 50000, 197800)
    foreach ($size in $sizes) {
        $checkpoint = "models\win_value_v1_board_columns_generation0_${size}_epoch001.pt"
        if (-not (Test-Path -LiteralPath $checkpoint)) {
            Invoke-PythonStep -Name "train_${size}_epoch001" -Arguments @(
                '-m', 'yellowstone.train_value',
                '--data', $data,
                '--checkpoint', $checkpoint,
                '--epochs', '1',
                '--batch-size', '256',
                '--learning-rate', '1e-3',
                '--seed', "$TrainingSeed",
                '--split-game-count', '197800',
                '--train-game-id-limit', "$size",
                '--input-canonicalization', 'board_columns_v1_history_none',
                '--value-schema', 'yellowstone.value.v1',
                '--history-semantics', 'none',
                '--training-games', "$size",
                '--progress-checkpoint', "models\win_value_v1_board_columns_generation0_${size}_epoch001.progress.pt"
            )
        }
        else {
            Complete-Step "train_${size}_epoch001"
        }
    }

    foreach ($size in $sizes) {
        $checkpoint = "models\win_value_v1_board_columns_generation0_${size}_epoch001.pt"
        Invoke-Evaluation `
            -Name "evaluate_${size}_p0_${MilestoneGames}" `
            -Checkpoint $checkpoint `
            -Games $MilestoneGames `
            -PlayerIndex 0 `
            -Output (Join-Path $evaluationDirectory "v1_board_columns_generation0_${size}_epoch001_${MilestoneGames}_same_seed_p0.json")
    }

    $finalCheckpoint = 'models\win_value_v1_board_columns_generation0_197800_epoch001.pt'
    foreach ($playerIndex in 0..3) {
        Invoke-Evaluation `
            -Name "evaluate_197800_p${playerIndex}_${GamesPerSeat}" `
            -Checkpoint $finalCheckpoint `
            -Games $GamesPerSeat `
            -PlayerIndex $playerIndex `
            -Output (Join-Path $evaluationDirectory "v1_board_columns_generation0_197800_epoch001_${GamesPerSeat}_same_seed_p${playerIndex}.json")
    }

    $milestones = @()
    foreach ($size in $sizes) {
        $milestones += [ordered]@{
            training_games = $size
            checkpoint = "models\win_value_v1_board_columns_generation0_${size}_epoch001.pt"
            seat0_milestone = Read-Evaluation (Join-Path $evaluationDirectory "v1_board_columns_generation0_${size}_epoch001_${MilestoneGames}_same_seed_p0.json")
        }
    }
    $seats = @()
    foreach ($playerIndex in 0..3) {
        $seats += Read-Evaluation (Join-Path $evaluationDirectory "v1_board_columns_generation0_197800_epoch001_${GamesPerSeat}_same_seed_p${playerIndex}.json")
    }
    $summary = [ordered]@{
        status = 'complete'
        model = 'V1 board columns generation0 197800 epoch001'
        canonicalization = 'board_columns_v1_history_none'
        source = $source
        data = $data
        training_seed = $TrainingSeed
        evaluation_seed = $EvaluationSeed
        milestones = $milestones
        final_all_seats = [ordered]@{
            games = ($seats | Measure-Object -Property games -Sum).Sum
            wins = ($seats | Measure-Object -Property wins -Sum).Sum
            win_rate = (($seats | Measure-Object -Property wins -Sum).Sum / ($seats | Measure-Object -Property games -Sum).Sum)
            seats = $seats
        }
        timings = $timings
    }
    Write-AtomicJson -Payload $summary -Path $summaryPath
    Write-Status -Step 'complete' -State 'complete'
}
catch {
    Write-Status -Step 'failed' -State 'failed' -Message $_.Exception.Message
    throw
}
