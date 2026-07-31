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

$trainingSummary = 'results\evaluations\action_delta_milestones_seat0_1000.json'
$proposerSelection = 'results\evaluations\action_delta_proposer.selection.json'
$output = 'results\evaluations\action_delta_milestones_pct030_pct100_all_seats.json'
$statusPath = 'results\evaluations\action_delta_selected_all_seats.status.json'
$timingsPath = 'results\evaluations\action_delta_selected_all_seats.timings.json'
$stdoutPath = 'logs\action_delta_selected_all_seats.stdout.log'
$stderrPath = 'logs\action_delta_selected_all_seats.stderr.log'
$lastCompleted = ''
$timings = [ordered]@{}

New-Item -ItemType Directory -Force -Path 'results\evaluations','logs' |
    Out-Null
$PID | Set-Content -Encoding ascii -LiteralPath (
    'logs\action_delta_selected_all_seats.pid'
)

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
        percentages = @(30, 100)
        games_per_seat = $GamesPerSeat
        evaluation_seed = $EvaluationSeed
        output = $output
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
    $proposer = [string]((
        Get-Content -Raw -Encoding utf8 $proposerSelection |
            ConvertFrom-Json
    ).checkpoint)
    foreach ($percent in @(30, 100)) {
        $tag = '{0:d3}' -f $percent
        $checkpoint = "models\action_delta_continuous_epoch001_pct${tag}.pt"
        foreach ($playerIndex in 0..3) {
            $result = (
                "results\evaluations\action_delta_milestone_pct${tag}_" +
                "${GamesPerSeat}_seed${EvaluationSeed}_p${playerIndex}.json"
            )
            $complete = $false
            if (Test-Path -LiteralPath $result) {
                $existing = Get-Content -Raw -Encoding utf8 $result |
                    ConvertFrom-Json
                $complete = (
                    [int]$existing.games -eq $GamesPerSeat -and
                    [int]$existing.seed -eq $EvaluationSeed -and
                    [int]$existing.player_index -eq $playerIndex
                )
            }
            $name = "evaluate_pct${tag}_p${playerIndex}"
            if (-not $complete) {
                Invoke-Step $name @(
                    '-m', 'yellowstone.evaluate_action_delta',
                    '--proposer-checkpoint', $proposer,
                    '--delta-checkpoint', $checkpoint,
                    '--games', "$GamesPerSeat", '--seed', "$EvaluationSeed",
                    '--player-index', "$playerIndex", '--output', $result
                )
            } else {
                $lastCompleted = $name
            }
        }
    }
    Invoke-Step 'summarize' @(
        '-m', 'yellowstone.summarize_action_delta_selected_all_seats',
        '--training-summary', $trainingSummary,
        '--evaluation-directory', 'results\evaluations',
        '--output', $output, '--percentages', '30,100',
        '--games-per-seat', "$GamesPerSeat", '--seed', "$EvaluationSeed"
    )
    Write-Status -Step 'done' -State 'complete'
} catch {
    Write-Status -Step 'failed' -State 'failed' -Message $_.Exception.Message
    throw
}
