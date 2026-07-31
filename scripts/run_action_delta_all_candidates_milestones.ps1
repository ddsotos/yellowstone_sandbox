param(
    [int]$Games = 1000,
    [int]$PlayerIndex = 0,
    [int]$EvaluationSeed = 20260725,
    [string]$PythonExe = 'python'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'

$percentages = @(30, 50, 100)
$prefix = 'models\action_delta_continuous_epoch001'
$summaryPath = (
    'results\evaluations\action_delta_all_candidates_' +
    'pct030_pct050_pct100_seat0_1000.json'
)
$markdownPath = [IO.Path]::ChangeExtension($summaryPath, '.md')
$statusPath = (
    'results\evaluations\action_delta_all_candidates_milestones.status.json'
)
$timingsPath = (
    'results\evaluations\action_delta_all_candidates_milestones.timings.json'
)
$stdoutPath = 'logs\action_delta_all_candidates_milestones.stdout.log'
$stderrPath = 'logs\action_delta_all_candidates_milestones.stderr.log'
$pidPath = 'logs\action_delta_all_candidates_milestones.pid'
$lastCompleted = ''
$timings = [ordered]@{}

New-Item -ItemType Directory -Force -Path 'results\evaluations','logs' |
    Out-Null
$PID | Set-Content -Encoding ascii -LiteralPath $pidPath

function Write-AtomicJson {
    param([object]$Payload, [string]$Path)
    $temporary = "$Path.$PID.tmp"
    $Payload | ConvertTo-Json -Depth 10 |
        Set-Content -Encoding utf8 -LiteralPath $temporary
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
        percentages = $percentages
        games = $Games
        player_index = $PlayerIndex
        evaluation_seed = $EvaluationSeed
        candidate_source = 'all_retained_turn_end_candidates'
        adaptive_pq_pruning = $true
        approximate_new_color_neighbor_limit = $true
        summary = $summaryPath
        stdout = $stdoutPath
        stderr = $stderrPath
    })
}

function Invoke-Step {
    param([string]$Name, [string[]]$Arguments)
    Write-Status -Step $Name -State 'running'
    $watch = [Diagnostics.Stopwatch]::StartNew()
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    $watch.Stop()
    $timings[$Name] = $watch.Elapsed.TotalSeconds
    Write-AtomicJson -Path $timingsPath -Payload $timings
    $script:lastCompleted = $Name
    Write-Status -Step $Name -State 'complete'
}

try {
    foreach ($percent in $percentages) {
        $tag = '{0:d3}' -f $percent
        $checkpoint = "${prefix}_pct${tag}.pt"
        $evaluation = (
            "results\evaluations\action_delta_all_candidates_pct${tag}_" +
            "${Games}_seed${EvaluationSeed}_p${PlayerIndex}.json"
        )
        $complete = $false
        if (Test-Path -LiteralPath $evaluation) {
            $existing = Get-Content -Raw -Encoding utf8 $evaluation |
                ConvertFrom-Json
            $complete = (
                [int]$existing.games -eq $Games -and
                [int]$existing.seed -eq $EvaluationSeed -and
                [int]$existing.player_index -eq $PlayerIndex -and
                [string]$existing.candidate_source -eq
                    'all_retained_turn_end_candidates'
            )
        }
        $name = "evaluate_pct${tag}"
        if (-not $complete) {
            Invoke-Step $name @(
                '-m', 'yellowstone.evaluate_action_delta',
                '--delta-checkpoint', $checkpoint,
                '--games', "$Games",
                '--seed', "$EvaluationSeed",
                '--player-index', "$PlayerIndex",
                '--output', $evaluation
            )
        } else {
            $lastCompleted = $name
        }
    }

    $rows = @(
        foreach ($percent in $percentages) {
            $tag = '{0:d3}' -f $percent
            $evaluation = (
                "results\evaluations\action_delta_all_candidates_pct${tag}_" +
                "${Games}_seed${EvaluationSeed}_p${PlayerIndex}.json"
            )
            $result = Get-Content -Raw -Encoding utf8 $evaluation |
                ConvertFrom-Json
            [ordered]@{
                percent = $percent
                checkpoint = "${prefix}_pct${tag}.pt"
                evaluation_path = $evaluation
                games = [int]$result.games
                fractional_wins = [double]$result.fractional_wins
                win_rate = [double]$result.win_rate
                evaluated_player_one_card_turns = (
                    [int]$result.evaluated_player_one_card_turns
                )
                evaluated_player_two_card_turns = (
                    [int]$result.evaluated_player_two_card_turns
                )
                evaluated_player_one_card_turn_rate = (
                    [double]$result.evaluated_player_one_card_turn_rate
                )
                mean_candidate_count = (
                    [double]$result.mean_candidate_count
                )
                mean_predicted_delta = (
                    [double]$result.mean_predicted_delta
                )
                elapsed_seconds = [double]$result.elapsed_seconds
            }
        }
    )
    $best = $rows | Sort-Object -Property {
        -[double]$_.win_rate
    }, {
        [int]$_.percent
    } | Select-Object -First 1
    $summary = [ordered]@{
        status = 'complete'
        experiment = 'action_delta_all_retained_candidates_milestones'
        official_four_seat_evaluation = $false
        screen_player_index = $PlayerIndex
        games_per_checkpoint = $Games
        evaluation_seed = $EvaluationSeed
        candidate_source = 'all_retained_turn_end_candidates'
        adaptive_pq_pruning = $true
        approximate_new_color_neighbor_limit = $true
        milestones = $rows
        best_percent = [int]$best.percent
        best_win_rate = [double]$best.win_rate
    }
    Write-AtomicJson -Path $summaryPath -Payload $summary

    $lines = @(
        '# Action delta all-candidate seat-0 comparison',
        '',
        "- seed: ``$EvaluationSeed``",
        "- candidate source: ``all_retained_turn_end_candidates``",
        '- Seat-0 policy screen; not an official four-seat evaluation.',
        '',
        '| Training fraction | Win rate | Fractional wins | One-card rate | One-card | Two-card | Mean candidates | Seconds |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|'
    )
    foreach ($row in $rows) {
        $lines += (
            '| {0}% | {1:P3} | {2:N3} | {3:P3} | {4} | {5} | {6:N2} | {7:N1} |' -f
            $row.percent,
            $row.win_rate,
            $row.fractional_wins,
            $row.evaluated_player_one_card_turn_rate,
            $row.evaluated_player_one_card_turns,
            $row.evaluated_player_two_card_turns,
            $row.mean_candidate_count,
            $row.elapsed_seconds
        )
    }
    $lines += ''
    $lines += (
        "Best win rate: **{0}% ({1:P3})**" -f
        $best.percent,
        $best.win_rate
    )
    $lines | Set-Content -Encoding utf8 -LiteralPath $markdownPath

    $lastCompleted = 'summarize'
    Write-Status -Step 'done' -State 'complete'
} catch {
    Write-Status -Step 'failed' -State 'failed' -Message $_.Exception.Message
    throw
}
