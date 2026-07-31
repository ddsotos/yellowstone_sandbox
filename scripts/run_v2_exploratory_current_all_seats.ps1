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

$checkpoint = 'models\win_value_v2_exploratory_current_epoch001.pt'
$evaluationDirectory = 'results\evaluations'
$summary = Join-Path $evaluationDirectory "v2_exploratory_current_${GamesPerSeat}_seed${EvaluationSeed}_all_seats.json"
$markdown = Join-Path $evaluationDirectory "v2_exploratory_current_${GamesPerSeat}_seed${EvaluationSeed}_all_seats.md"
$statusPath = Join-Path $evaluationDirectory 'v2_exploratory_current_all_seats.status.json'
$pidPath = 'logs\v2_exploratory_current_all_seats.pid'
$stdoutPath = 'logs\v2_exploratory_current_all_seats.stdout.log'
$stderrPath = 'logs\v2_exploratory_current_all_seats.stderr.log'

New-Item -ItemType Directory -Force -Path $evaluationDirectory, 'logs' | Out-Null
$PID | Set-Content -Encoding ascii $pidPath

function Write-Status([string]$State, [string]$Step, [string]$Message = '') {
    $payload = [ordered]@{
        state = $State; step = $Step; message = $Message
        updated_at = (Get-Date).ToString('o'); pid = $PID
        checkpoint = $checkpoint; games_per_seat = $GamesPerSeat
        evaluation_seed = $EvaluationSeed; summary = $summary
    }
    $tmp = "$statusPath.$PID.tmp"
    $payload | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 $tmp
    Move-Item -Force $tmp $statusPath
}

try {
    if (-not (Test-Path -LiteralPath $checkpoint)) { throw "checkpoint not found: $checkpoint" }
    Write-Status 'running' 'validate_checkpoint'
    $seatResults = @()
    foreach ($playerIndex in 0..3) {
        $output = Join-Path $evaluationDirectory "v2_exploratory_current_${GamesPerSeat}_seed${EvaluationSeed}_p${playerIndex}.json"
        if (Test-Path -LiteralPath $output) {
            $existing = Get-Content -Raw -Encoding utf8 $output | ConvertFrom-Json
            if ([int]$existing.games -eq $GamesPerSeat -and [int]$existing.player_index -eq $playerIndex) {
                $seatResults += $existing
                continue
            }
        }
        Write-Status 'running' "evaluate_p${playerIndex}"
        & $PythonExe -m yellowstone.evaluate_value_v2_exploratory --checkpoint $checkpoint --games $GamesPerSeat --seed $EvaluationSeed --player-index $playerIndex --output $output *>> $stdoutPath 2>> $stderrPath
        if ($LASTEXITCODE -ne 0) { throw "evaluate_p${playerIndex} failed with exit code $LASTEXITCODE" }
        $seatResults += (Get-Content -Raw -Encoding utf8 $output | ConvertFrom-Json)
    }
    Write-Status 'running' 'summarize'
    $allGames = ($seatResults | Measure-Object -Property games -Sum).Sum
    $allWins = ($seatResults | Measure-Object -Property wins -Sum).Sum
    $allTurns = ($seatResults | Measure-Object -Property turns -Sum).Sum
    $allOne = ($seatResults | Measure-Object -Property one_card_turns -Sum).Sum
    $payload = [ordered]@{
        model = 'V2 exploratory current'
        checkpoint = $checkpoint
        games_per_seat = $GamesPerSeat
        evaluation_seed = $EvaluationSeed
        value_schema = 'yellowstone.value.v2-exploratory-refill.v1'
        input_canonicalization = 'strict_residual_v2_uniform_negative_ratios_refill_risk_v1'
        history_semantics = 'rolling_last_three_completed_turns_v2'
        opponent_private_inputs = $false
        seats = @($seatResults)
        all_seats_games = $allGames
        all_seats_fractional_wins = $allWins
        all_seats_win_rate = $allWins / $allGames
        all_seats_turns = $allTurns
        all_seats_one_card_turns = $allOne
        all_seats_two_card_turns = $allTurns - $allOne
        all_seats_one_card_turn_rate = $allOne / $allTurns
        generated_at = (Get-Date).ToString('o')
    }
    $payload | ConvertTo-Json -Depth 12 | Set-Content -Encoding utf8 $summary
    @(
        "# V2 exploratory current・全席評価"
        ""
        "- checkpoint: ``$checkpoint``"
        "- seed: $EvaluationSeed"
        "- games per seat: $GamesPerSeat"
        "- all-seat win rate: {0:P3}" -f $payload.all_seats_win_rate
        "- one-card turn rate: {0:P3}" -f $payload.all_seats_one_card_turn_rate
        ""
        "| seat | win rate | fractional wins | games | one-card rate |"
        "|---:|---:|---:|---:|---:|"
        ($seatResults | ForEach-Object { "| $($_.player_index) | {0:P3} | {1:N3} | $($_.games) | {2:P3} |" -f $_.win_rate, $_.wins, ($_.one_card_turns / $_.turns) })
    ) | Set-Content -Encoding utf8 $markdown
    Write-Status 'complete' 'done'
} catch {
    Write-Status 'failed' 'failed' $_.Exception.Message
    throw
}
