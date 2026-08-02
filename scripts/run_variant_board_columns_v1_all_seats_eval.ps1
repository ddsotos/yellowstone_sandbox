param(
    [int]$GamesPerSeat = 1000,
    [int]$EvaluationSeed = 20260725,
    [string]$PythonExe = 'python'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'

$name = 'v2_variant_board5_hand6_oneoff_tiered_board_columns_v1_all_seats_eval'
$evaluationDirectory = 'results\evaluations'
$statusPath = Join-Path $evaluationDirectory "$name.status.json"
$timingsPath = Join-Path $evaluationDirectory "$name.timings.json"
$summaryPath = Join-Path $evaluationDirectory "${name}_${GamesPerSeat}_seed${EvaluationSeed}.json"
$markdownPath = Join-Path $evaluationDirectory "${name}_${GamesPerSeat}_seed${EvaluationSeed}.md"
$pidPath = "logs\$name.pid"
$stdoutPath = "logs\$name.stdout.log"
$stderrPath = "logs\$name.stderr.log"
$lastCompleted = ''
$timings = [ordered]@{}

$models = @(
    [ordered]@{
        key = 'finetune_from_6h'
        label = 'Variant board_columns_v1 finetune from 6h'
        checkpoint = 'models\v2_variant_board5_hand6_oneoff_tiered_board_columns_v1_training_finetune_from_6h_epoch001_pct100.pt'
    },
    [ordered]@{
        key = 'scratch'
        label = 'Variant board_columns_v1 scratch'
        checkpoint = 'models\v2_variant_board5_hand6_oneoff_tiered_board_columns_v1_training_scratch_epoch001_pct100.pt'
    }
)

New-Item -ItemType Directory -Force -Path $evaluationDirectory, 'logs' | Out-Null
$PID | Set-Content -Encoding ascii -LiteralPath $pidPath

function Write-Utf8NoBom {
    param([string]$Path, [string]$Text)
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText((Join-Path (Get-Location) $Path), $Text, $encoding)
}

function Write-AtomicJson {
    param([object]$Payload, [string]$Path, [int]$Depth = 16)
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
        games_per_seat = $GamesPerSeat
        evaluation_seed = $EvaluationSeed
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

function Read-Evaluation {
    param([string]$Path, [int]$PlayerIndex)
    $payload = Get-Content -Raw -Encoding utf8 $Path | ConvertFrom-Json
    if ([int]$payload.games -ne $GamesPerSeat) {
        throw "unexpected game count in $Path"
    }
    return [ordered]@{
        player_index = $PlayerIndex
        games = [int]$payload.games
        wins = [double]$payload.wins
        fractional_wins = [double]$payload.fractional_wins
        win_rate = [double]$payload.win_rate
        one_card_turns = [int]$payload.evaluated_player_one_card_turns
        two_card_turns = [int]$payload.evaluated_player_two_card_turns
        one_card_turn_rate = [double]$payload.evaluated_player_one_card_turn_rate
        elapsed_seconds = [double]$payload.elapsed_seconds
        evaluation_path = $Path
    }
}

try {
    Write-Status -Step 'start' -State 'running'
    $modelSummaries = @()
    foreach ($model in $models) {
        $checkpoint = [string]$model.checkpoint
        if (-not (Test-Path -LiteralPath $checkpoint)) {
            throw "missing checkpoint: $checkpoint"
        }
        $seatResults = @()
        foreach ($playerIndex in 0..3) {
            $stepName = "$($model.key)_p${playerIndex}"
            $output = Join-Path $evaluationDirectory "${name}_$($model.key)_${GamesPerSeat}_seed${EvaluationSeed}_p${playerIndex}.json"
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
            $seatResults += Read-Evaluation -Path $output -PlayerIndex $playerIndex
        }

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
        $modelSummaries += [ordered]@{
            key = [string]$model.key
            label = [string]$model.label
            checkpoint = $checkpoint
            seats = @($seatResults)
            all_seats_games = $allGames
            all_seats_fractional_wins = $allWins
            all_seats_win_rate = $allWins / $allGames
            all_seats_one_card_turns = $allOne
            all_seats_two_card_turns = $allTwo
            all_seats_one_card_turn_rate = $allOne / ($allOne + $allTwo)
        }
    }

    Write-Status -Step 'summarize' -State 'running'
    $summary = [ordered]@{
        schema = 'yellowstone.value.v1.board_columns.variant_heuristic.all_seats'
        official_four_seat_evaluation = $true
        games_per_seat = $GamesPerSeat
        evaluation_seed = $EvaluationSeed
        value_schema = 'yellowstone.value.v1'
        input_canonicalization = 'board_columns_v1_history_none'
        history_semantics = 'none'
        pruning = 'adaptive_pq'
        approximate_new_color_neighbors = $true
        models = @($modelSummaries)
        timings = $timings
        generated_at = (Get-Date).ToString('o')
    }
    Write-AtomicJson -Path $summaryPath -Payload $summary

    $lines = @(
        '# Variant board_columns_v1 all-seat evaluation',
        '',
        "- seed: $EvaluationSeed",
        "- games per seat: $GamesPerSeat",
        "- pruning: adaptive_pq",
        "- approximate new color neighbors: true",
        '',
        '| model | all-seat win rate | fractional wins | games | one-card rate | one | two |',
        '|---|---:|---:|---:|---:|---:|---:|'
    )
    foreach ($model in $modelSummaries) {
        $lines += "| $($model.key) | $('{0:P3}' -f $model.all_seats_win_rate) | $('{0:N3}' -f $model.all_seats_fractional_wins) | $($model.all_seats_games) | $('{0:P3}' -f $model.all_seats_one_card_turn_rate) | $($model.all_seats_one_card_turns) | $($model.all_seats_two_card_turns) |"
    }
    foreach ($model in $modelSummaries) {
        $lines += ''
        $lines += "## $($model.key)"
        $lines += ''
        $lines += '| seat | win rate | fractional wins | games | one-card rate | one | two | seconds |'
        $lines += '|---:|---:|---:|---:|---:|---:|---:|---:|'
        foreach ($seat in $model.seats) {
            $lines += "| $($seat.player_index) | $('{0:P3}' -f $seat.win_rate) | $('{0:N3}' -f $seat.fractional_wins) | $($seat.games) | $('{0:P3}' -f $seat.one_card_turn_rate) | $($seat.one_card_turns) | $($seat.two_card_turns) | $('{0:N1}' -f $seat.elapsed_seconds) |"
        }
    }
    Write-Utf8NoBom -Path $markdownPath -Text (($lines -join "`n") + "`n")

    $lastCompleted = 'summarize'
    Write-Status -Step 'complete' -State 'complete'
}
catch {
    Write-Status -Step 'failed' -State 'failed' -Message $_.Exception.Message
    throw
}
