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

$source = 'data\v2_exploratory_diverse_v1_20260730'
$sourceGames = 76919
$sourceGameIdMin = 1100912
$sourceGameIdMax = 1177830
$evaluationDirectory = 'results\evaluations'
$statusPath = Join-Path $evaluationDirectory 'v1_board_centered_explore_history_variants.status.json'
$timingsPath = Join-Path $evaluationDirectory 'v1_board_centered_explore_history_variants.timings.json'
$summaryPath = Join-Path $evaluationDirectory "v1_board_centered_explore_history_variants_${GamesPerSeat}_same_seed_all_seats.json"
$markdownPath = Join-Path $evaluationDirectory "v1_board_centered_explore_history_variants_${GamesPerSeat}_same_seed_all_seats.md"
$pidPath = 'logs\v1_board_centered_explore_history_variants.pid'
$stdoutPath = 'logs\v1_board_centered_explore_history_variants.stdout.log'
$stderrPath = 'logs\v1_board_centered_explore_history_variants.stderr.log'
$lastCompleted = ''
$timings = [ordered]@{}

$variants = @(
    [ordered]@{ key='none'; canonicalization='board_centered_v1_history_none'; history='none' },
    [ordered]@{ key='own_frame_delta_2cycle'; canonicalization='board_centered_v1_history_own_frame_delta_2cycle'; history='own_frame_delta_2cycle' },
    [ordered]@{ key='v1'; canonicalization='board_centered_v1_history_v1'; history='rolling_last_two_placements' },
    [ordered]@{ key='turn_local'; canonicalization='board_centered_v1_history_turn_local'; history='evaluated_turn_one_or_two_placements' }
)

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
        source = $source
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
    if (Test-Path -LiteralPath $timingsPath) {
        $existingTimings = Get-Content -Raw -Encoding utf8 $timingsPath | ConvertFrom-Json
        foreach ($property in $existingTimings.PSObject.Properties) {
            $timings[$property.Name] = [double]$property.Value
        }
    }
    $variantResults = @()
    foreach ($variant in $variants) {
        $key = [string]$variant.key
        $canonicalization = [string]$variant.canonicalization
        $history = [string]$variant.history
        $data = "data\v1_board_centered_explore_76919_$key"
        $manifest = Join-Path $data 'conversion_manifest.json'
        $checkpoint = "models\win_value_v1_board_centered_explore_76919_${key}_epoch001.pt"
        $progressCheckpoint = "models\win_value_v1_board_centered_explore_76919_${key}_epoch001.progress.pt"

        if (-not (Test-Path -LiteralPath $manifest)) {
            Invoke-PythonStep -Name "convert_$key" -Arguments @(
                '-m', 'yellowstone.convert_replay_v2_to_v1_board_centered',
                '--source', $source,
                '--output', $data,
                '--expected-games', "$sourceGames",
                '--game-id-rebase', "$sourceGameIdMin",
                '--expected-source-game-id-min', "$sourceGameIdMin",
                '--expected-source-game-id-max', "$sourceGameIdMax",
                '--input-canonicalization', $canonicalization
            )
        }
        else {
            Complete-Step "convert_$key"
        }

        if (-not (Test-Path -LiteralPath $checkpoint)) {
            Invoke-PythonStep -Name "train_$key" -Arguments @(
                '-m', 'yellowstone.train_value',
                '--data', $data,
                '--checkpoint', $checkpoint,
                '--epochs', '1',
                '--batch-size', '256',
                '--learning-rate', '1e-3',
                '--seed', "$TrainingSeed",
                '--split-game-count', "$sourceGames",
                '--train-game-id-limit', "$sourceGames",
                '--input-canonicalization', $canonicalization,
                '--value-schema', 'yellowstone.value.v1',
                '--history-semantics', $history,
                '--training-games', "$sourceGames",
                '--progress-checkpoint', $progressCheckpoint
            )
        }
        else {
            Complete-Step "train_$key"
        }

        $seatResults = @()
        foreach ($playerIndex in 0..3) {
            $output = Join-Path $evaluationDirectory "v1_board_centered_explore_76919_${key}_epoch001_${GamesPerSeat}_same_seed_p${playerIndex}.json"
            if (-not (Test-Path -LiteralPath $output)) {
                Invoke-PythonStep -Name "evaluate_${key}_p${playerIndex}" -Arguments @(
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
                Complete-Step "evaluate_${key}_p${playerIndex}"
            }
            $payload = Get-Content -Raw -Encoding utf8 $output | ConvertFrom-Json
            if ([int]$payload.games -ne $GamesPerSeat) {
                throw "unexpected evaluation payload: $output"
            }
            $seatResults += [ordered]@{
                player_index = $playerIndex
                games = [int]$payload.games
                fractional_wins = [double]$payload.fractional_wins
                win_rate = [double]$payload.win_rate
                one_card_turns = [int]$payload.evaluated_player_one_card_turns
                two_card_turns = [int]$payload.evaluated_player_two_card_turns
                one_card_turn_rate = [double]$payload.evaluated_player_one_card_turn_rate
                evaluation_path = $output
            }
        }
        $allGames = 0
        $allWins = 0.0
        $allOne = 0
        $allTwo = 0
        foreach ($seat in $seatResults) {
            $allGames += [int]$seat.games
            $allWins += [double]$seat.fractional_wins
            $allOne += [int]$seat.one_card_turns
            $allTwo += [int]$seat.two_card_turns
        }
        $variantResults += [ordered]@{
            key = $key
            input_canonicalization = $canonicalization
            history_semantics = $history
            data = $data
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
        schema = 'yellowstone.value.v1.board_centered.explore_history_variants'
        official_four_seat_evaluation = $true
        source = $source
        source_games = $sourceGames
        games_per_seat = $GamesPerSeat
        evaluation_seed = $EvaluationSeed
        training_seed = $TrainingSeed
        variants = @($variantResults)
        timings = $timings
        generated_at = (Get-Date).ToString('o')
    }
    Write-AtomicJson -Path $summaryPath -Payload $summary
    $lines = @(
        '# Board-centered V1 explore history variants',
        '',
        "- source: ``$source``",
        "- source games: $sourceGames",
        "- seed: $EvaluationSeed",
        "- games per seat: $GamesPerSeat",
        '',
        '| variant | all-seat win rate | one-card rate | p0 | p1 | p2 | p3 |',
        '|---|---:|---:|---:|---:|---:|---:|'
    )
    foreach ($row in $variantResults) {
        $seatRates = @($row.seats | ForEach-Object { '{0:P3}' -f $_.win_rate })
        $lines += "| $($row.key) | $('{0:P3}' -f $row.all_seats_win_rate) | $('{0:P3}' -f $row.all_seats_one_card_turn_rate) | $($seatRates -join ' | ') |"
    }
    Write-Utf8NoBom -Path $markdownPath -Text (($lines -join "`n") + "`n")
    $lastCompleted = 'summarize'
    Write-Status -Step 'done' -State 'complete'
}
catch {
    Write-Status -Step 'failed' -State 'failed' -Message $_.Exception.Message
    throw
}
