param(
    [int]$GamesPerSeat = 1000,
    [int]$EvaluationSeed = 20260725,
    [int]$TrainingSeed = 20260727,
    [string]$PythonExe = 'python'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'

$data = 'data\v1_board_centered_new_88966'
$manifest = Join-Path $data 'conversion_manifest.json'
$checkpoint = 'models\win_value_v1_board_centered_new_88966_epoch001.pt'
$progressCheckpoint = 'models\win_value_v1_board_centered_new_88966_epoch001.progress.pt'
$evaluationDirectory = 'results\evaluations'
$statusPath = Join-Path $evaluationDirectory 'v1_board_centered_new_88966_all_seats.status.json'
$timingsPath = Join-Path $evaluationDirectory 'v1_board_centered_new_88966_all_seats.timings.json'
$summaryPath = Join-Path $evaluationDirectory "v1_board_centered_new_88966_epoch001_${GamesPerSeat}_same_seed_all_seats.json"
$markdownPath = Join-Path $evaluationDirectory "v1_board_centered_new_88966_epoch001_${GamesPerSeat}_same_seed_all_seats.md"
$pidPath = 'logs\v1_board_centered_new_88966_all_seats.pid'
$stdoutPath = 'logs\v1_board_centered_new_88966_all_seats.stdout.log'
$stderrPath = 'logs\v1_board_centered_new_88966_all_seats.stderr.log'
$lastCompleted = ''
$timings = [ordered]@{}

New-Item -ItemType Directory -Force -Path 'models', $evaluationDirectory, 'logs' | Out-Null
$PID | Set-Content -Encoding ascii -LiteralPath $pidPath

function Write-Utf8NoBom {
    param([string]$Path, [string]$Text)
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText((Join-Path (Get-Location) $Path), $Text, $encoding)
}

function Write-AtomicJson {
    param([object]$Payload, [string]$Path, [int]$Depth = 12)
    $temporary = "$Path.$PID.tmp"
    Write-Utf8NoBom -Path $temporary -Text (($Payload | ConvertTo-Json -Depth $Depth) + "`n")
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
        data = $data
        checkpoint = $checkpoint
        games_per_seat = $GamesPerSeat
        evaluation_seed = $EvaluationSeed
        training_seed = $TrainingSeed
        summary = $summaryPath
        stdout = $stdoutPath
        stderr = $stderrPath
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
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    $watch.Stop()
    $timings[$Name] = $watch.Elapsed.TotalSeconds
    Write-AtomicJson -Path $timingsPath -Payload $timings
    Complete-Step $Name
}

try {
    if (-not (Test-Path -LiteralPath $manifest)) {
        throw "missing conversion manifest: $manifest"
    }

    if (Test-Path -LiteralPath $timingsPath) {
        $existingTimings = Get-Content -Raw -Encoding utf8 $timingsPath | ConvertFrom-Json
        foreach ($property in $existingTimings.PSObject.Properties) {
            $timings[$property.Name] = [double]$property.Value
        }
    }

    if (-not (Test-Path -LiteralPath $checkpoint)) {
        Invoke-PythonStep -Name 'train_epoch001' -Arguments @(
            '-m', 'yellowstone.train_value',
            '--data', $data,
            '--checkpoint', $checkpoint,
            '--epochs', '1',
            '--batch-size', '256',
            '--learning-rate', '1e-3',
            '--seed', "$TrainingSeed",
            '--split-game-count', '88966',
            '--train-game-id-limit', '88966',
            '--input-canonicalization', 'board_centered_v1',
            '--value-schema', 'yellowstone.value.v1',
            '--history-semantics', 'rolling_last_two_placements',
            '--training-games', '88966',
            '--progress-checkpoint', $progressCheckpoint
        )
    }
    else {
        Complete-Step 'train_epoch001'
    }

    $seatResults = @()
    foreach ($playerIndex in 0..3) {
        $stepName = "evaluate_p${playerIndex}"
        $output = Join-Path $evaluationDirectory "v1_board_centered_new_88966_epoch001_${GamesPerSeat}_same_seed_p${playerIndex}.json"
        if (-not (Test-Path -LiteralPath $output)) {
            Invoke-PythonStep -Name $stepName -Arguments @(
                '-m', 'yellowstone.evaluate_value',
                '--checkpoint', $checkpoint,
                '--games', "$GamesPerSeat",
                '--seed', "$EvaluationSeed",
                '--player-index', "$playerIndex",
                '--adaptive-pq-pruning',
                '--approximate-new-color-neighbors',
                '--output', $output
            )
        }
        else {
            Complete-Step $stepName
        }
        $payload = Get-Content -Raw -Encoding utf8 $output | ConvertFrom-Json
        if ([int]$payload.games -ne $GamesPerSeat) {
            throw "unexpected evaluation payload: $output"
        }
        $seatResults += [ordered]@{
            player_index = $playerIndex
            games = [int]$payload.games
            wins = [double]$payload.wins
            fractional_wins = [double]$payload.fractional_wins
            win_rate = [double]$payload.win_rate
            one_card_turns = [int]$payload.evaluated_player_one_card_turns
            two_card_turns = [int]$payload.evaluated_player_two_card_turns
            one_card_turn_rate = [double]$payload.evaluated_player_one_card_turn_rate
            evaluation_path = $output
        }
    }

    Write-Status -Step 'summarize' -State 'running'
    $allGames = 0
    $allWins = 0.0
    $allOne = 0
    $allTwo = 0
    foreach ($seat in $seatResults) {
        $allGames += [int]$seat['games']
        $allWins += [double]$seat['fractional_wins']
        $allOne += [int]$seat['one_card_turns']
        $allTwo += [int]$seat['two_card_turns']
    }
    $summary = [ordered]@{
        schema = 'yellowstone.value.v1.board_centered.all_seat_screen'
        official_four_seat_evaluation = $true
        model = 'Original V1 board-centered'
        checkpoint = $checkpoint
        data = $data
        conversion_manifest = $manifest
        games_per_seat = $GamesPerSeat
        evaluation_seed = $EvaluationSeed
        training_seed = $TrainingSeed
        value_schema = 'yellowstone.value.v1'
        input_canonicalization = 'board_centered_v1'
        history_semantics = 'rolling_last_two_placements'
        pruning = 'adaptive_pq'
        approximate_new_color_neighbors = $true
        seats = @($seatResults)
        all_seats_games = $allGames
        all_seats_fractional_wins = $allWins
        all_seats_win_rate = $allWins / $allGames
        all_seats_one_card_turns = $allOne
        all_seats_two_card_turns = $allTwo
        all_seats_one_card_turn_rate = $allOne / ($allOne + $allTwo)
        timings = $timings
        generated_at = (Get-Date).ToString('o')
    }
    Write-AtomicJson -Path $summaryPath -Payload $summary

    $lines = @(
        '# Board-centered V1 all-seat screen',
        '',
        "- checkpoint: ``$checkpoint``",
        "- data: ``$data``",
        "- seed: $EvaluationSeed",
        "- games per seat: $GamesPerSeat",
        "- all-seat win rate: $('{0:P3}' -f $summary.all_seats_win_rate)",
        "- one-card turn rate: $('{0:P3}' -f $summary.all_seats_one_card_turn_rate)",
        '',
        '| seat | win rate | fractional wins | games | one-card rate | one | two |',
        '|---:|---:|---:|---:|---:|---:|---:|'
    )
    foreach ($seat in $seatResults) {
        $lines += "| $($seat.player_index) | $('{0:P3}' -f $seat.win_rate) | $('{0:N3}' -f $seat.fractional_wins) | $($seat.games) | $('{0:P3}' -f $seat.one_card_turn_rate) | $($seat.one_card_turns) | $($seat.two_card_turns) |"
    }
    $lines += "| all | $('{0:P3}' -f $summary.all_seats_win_rate) | $('{0:N3}' -f $summary.all_seats_fractional_wins) | $allGames | $('{0:P3}' -f $summary.all_seats_one_card_turn_rate) | $allOne | $allTwo |"
    Write-Utf8NoBom -Path $markdownPath -Text (($lines -join "`n") + "`n")

    $lastCompleted = 'summarize'
    Write-Status -Step 'done' -State 'complete'
}
catch {
    Write-Status -Step 'failed' -State 'failed' -Message $_.Exception.Message
    throw
}
