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

$evaluationDirectory = 'results\evaluations'
$name = 'v1_explore_59826_plus_mixed_50000_vs_explore3_index0_eval'
$statusPath = Join-Path $evaluationDirectory "$name.status.json"
$timingsPath = Join-Path $evaluationDirectory "$name.timings.json"
$outputPath = Join-Path $evaluationDirectory "v1_explore_59826_plus_mixed_50000_epoch002_vs_explore3_${Games}_seed${EvaluationSeed}_p0.json"
$pidPath = "logs\$name.pid"
$stdoutPath = "logs\$name.stdout.log"
$stderrPath = "logs\$name.stderr.log"
$lastCompleted = ''
$timings = [ordered]@{}
$checkpoint = 'models\win_value_v1_explore_59826_plus_mixed_50000_epoch002.pt'
$exploreCheckpoint = 'models\win_value_v1_original_generation0_197800_epoch002.pt'

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
        opponent_policy = 'exploratory_value_npc'
        opponent_checkpoint = $exploreCheckpoint
        output = $outputPath
        stdout = $stdoutPath
        stderr = $stderrPath
    })
}

try {
    foreach ($path in @($checkpoint, $exploreCheckpoint)) {
        if (-not (Test-Path -LiteralPath $path)) {
            throw "missing checkpoint: $path"
        }
    }
    Write-Status -Step 'evaluate_vs_explore3' -State 'running'
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    & $PythonExe @(
        '-m', 'yellowstone.evaluate_value_vs_explore',
        '--checkpoint', $checkpoint,
        '--explore-checkpoint', $exploreCheckpoint,
        '--games', "$Games",
        '--seed', "$EvaluationSeed",
        '--player-index', '0',
        '--adaptive-pq-pruning',
        '--approximate-new-color-neighbors',
        '--output', $outputPath
    )
    if ($LASTEXITCODE -ne 0) {
        throw "evaluate_vs_explore3 failed with exit code $LASTEXITCODE"
    }
    $watch.Stop()
    $timings['evaluate_vs_explore3'] = $watch.Elapsed.TotalSeconds
    Write-AtomicJson -Path $timingsPath -Payload $timings
    $lastCompleted = 'evaluate_vs_explore3'
    Write-Status -Step 'done' -State 'complete'
}
catch {
    Write-Status -Step 'failed' -State 'failed' -Message $_.Exception.Message
    throw
}
