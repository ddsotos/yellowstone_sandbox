param(
    [int]$Games = 1000,
    [int]$EvaluationSeed = 20260725,
    [string]$PythonExe = 'python'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$root\src;$env:PYTHONPATH" } else { "$root\src" }

$name = 'explore_policy_v2_min_loss_refill_deck_vs_heuristic_index0_eval'
$evaluationDirectory = 'results\evaluations'
$statusPath = Join-Path $evaluationDirectory "$name.status.json"
$timingsPath = Join-Path $evaluationDirectory "$name.timings.json"
$outputPath = Join-Path $evaluationDirectory "explore_policy_v2_min_loss_refill_deck_vs_heuristic_${Games}_seed${EvaluationSeed}_p0.json"
$pidPath = "logs\$name.pid"
$stdoutPath = "logs\$name.stdout.log"
$stderrPath = "logs\$name.stderr.log"
$checkpoint = 'models\win_value_v1_original_generation0_197800_epoch002.pt'
$lastCompleted = ''
$timings = [ordered]@{}

New-Item -ItemType Directory -Force -Path $evaluationDirectory, 'logs' | Out-Null
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
        games = $Games
        player_index = 0
        evaluation_seed = $EvaluationSeed
        checkpoint = $checkpoint
        output = $outputPath
        stdout = $stdoutPath
        stderr = $stderrPath
    })
}

try {
    if (-not (Test-Path -LiteralPath $checkpoint)) {
        throw "missing checkpoint: $checkpoint"
    }
    Write-Status -Step 'evaluate_explore_vs_heuristic' -State 'running'
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    & $PythonExe @(
        '-m', 'yellowstone.evaluate_explore_vs_heuristic',
        '--checkpoint', $checkpoint,
        '--games', "$Games",
        '--seed', "$EvaluationSeed",
        '--player-index', '0',
        '--output', $outputPath
    )
    if ($LASTEXITCODE -ne 0) {
        throw "evaluate_explore_vs_heuristic failed with exit code $LASTEXITCODE"
    }
    $watch.Stop()
    $timings['evaluate_explore_vs_heuristic'] = $watch.Elapsed.TotalSeconds
    Write-AtomicJson -Path $timingsPath -Payload $timings
    $lastCompleted = 'evaluate_explore_vs_heuristic'
    Write-Status -Step 'done' -State 'complete'
}
catch {
    Write-Status -Step 'failed' -State 'failed' -Message $_.Exception.Message
    throw
}
