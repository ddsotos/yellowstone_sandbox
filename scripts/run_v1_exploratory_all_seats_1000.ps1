param(
    [int]$GamesPerSeat = 1000,
    [int]$EvaluationSeed = 20260725,
    [string]$PythonExe = 'python'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUNBUFFERED = '1'
$checkpoint = 'models\win_value_v1_exploratory_59826_epoch001_pct100.pt'
$evaluationDirectory = 'results\evaluations'
$statusPath = Join-Path $evaluationDirectory 'v1_exploratory_pct100_all_seats.status.json'
$summaryPath = Join-Path $evaluationDirectory 'v1_exploratory_59826_pct100_1000_all_seats.json'
$markdownPath = Join-Path $evaluationDirectory 'v1_exploratory_59826_pct100_1000_all_seats.md'
$stdoutPath = 'logs\v1_exploratory_pct100_all_seats.stdout.log'
$stderrPath = 'logs\v1_exploratory_pct100_all_seats.stderr.log'
$lastCompleted = ''

New-Item -ItemType Directory -Force -Path 'models', $evaluationDirectory, 'logs' | Out-Null
$PID | Set-Content -Encoding ascii -LiteralPath 'logs\v1_exploratory_pct100_all_seats.pid'

function Write-AtomicJson {
    param([object]$Payload, [string]$Path)
    $temporary = "$Path.$PID.tmp"
    $Payload | ConvertTo-Json -Depth 12 | Set-Content -Encoding utf8 $temporary
    Move-Item -Force -LiteralPath $temporary -Destination $Path
}
function Write-Status {
    param([string]$Step, [string]$State, [string]$Message = '')
    Write-AtomicJson -Path $statusPath -Payload ([ordered]@{
        state=$State; step=$Step; last_completed_step=$lastCompleted
        message=$Message; updated_at=(Get-Date).ToString('o'); pid=$PID
        checkpoint=$checkpoint; games_per_seat=$GamesPerSeat
        evaluation_seed=$EvaluationSeed; summary=$summaryPath
        stdout=$stdoutPath; stderr=$stderrPath
    })
}
try {
    if (-not (Test-Path -LiteralPath $checkpoint)) { throw "missing checkpoint: $checkpoint" }
    $seats = @()
    foreach ($playerIndex in 0..3) {
        $name = "v1_exploratory_59826_pct100_${GamesPerSeat}_seed${EvaluationSeed}_p${playerIndex}.json"
        $output = Join-Path $evaluationDirectory $name
        Write-Status -Step "evaluate_p${playerIndex}" -State 'running'
        if (-not (Test-Path -LiteralPath $output)) {
            & $PythonExe -m yellowstone.evaluate_value `
                --checkpoint $checkpoint --games "$GamesPerSeat" `
                --seed "$EvaluationSeed" --player-index "$playerIndex" `
                --adaptive-pq-pruning --approximate-new-color-neighbors `
                --output $output
            if ($LASTEXITCODE -ne 0) { throw "seat $playerIndex evaluation failed" }
        }
        $payload = Get-Content -Raw -Encoding utf8 $output | ConvertFrom-Json
        $seats += [ordered]@{
            player_index=$playerIndex; games=[int]$payload.games
            wins=[double]$payload.wins; fractional_wins=[double]$payload.fractional_wins
            win_rate=[double]$payload.win_rate
            one_card_turns=[int]$payload.evaluated_player_one_card_turns
            two_card_turns=[int]$payload.evaluated_player_two_card_turns
            one_card_turn_rate=[double]$payload.evaluated_player_one_card_turn_rate
            evaluation_path=$output
        }
        $lastCompleted = "evaluate_p${playerIndex}"
    }
    $allWins = 0.0
    $allGames = 0
    foreach ($seat in $seats) {
        $allWins += [double]$seat['fractional_wins']
        $allGames += [int]$seat['games']
    }
    $comparison = [ordered]@{
        schema='yellowstone.value.v1.all_seat_screen'
        official_four_seat_evaluation=$true
        checkpoint=$checkpoint; games_per_seat=$GamesPerSeat
        evaluation_seed=$EvaluationSeed; seats=$seats
        total_games=[int]$allGames; total_fractional_wins=[double]$allWins
        combined_win_rate=([double]$allWins / [double]$allGames)
        note='Original V1 checkpoint, four seats, same seed and pruning conditions.'
    }
    Write-AtomicJson -Path $summaryPath -Payload $comparison
    $lines = @('# Original V1 100% all-seat screen','',
        "Checkpoint: $checkpoint; $GamesPerSeat games per seat; seed $EvaluationSeed.",'',
        '| Seat | Win rate | One-card rate | Games |','|---:|---:|---:|---:|')
    foreach ($seat in $seats) {
        $lines += "| $($seat.player_index) | $('{0:P3}' -f $seat.win_rate) | $('{0:P3}' -f $seat.one_card_turn_rate) | $($seat.games) |"
    }
    $lines += "| All | $('{0:P3}' -f $comparison.combined_win_rate) | — | $allGames |"
    $lines -join "`n" | Set-Content -Encoding utf8 $markdownPath
    $lastCompleted='summarize'; Write-Status -Step 'done' -State 'complete'
} catch {
    Write-Status -Step 'failed' -State 'failed' -Message $_.Exception.Message
    throw
}
