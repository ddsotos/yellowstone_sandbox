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

$source = 'data\v2_heuristic_one_vs_two_v1_10h_20260729'
$data = 'data\v2_lite_action_new_88966_tensors'
$checkpoint = 'models\win_value_v2_lite_action_new_88966_epoch001.pt'
$progress = 'models\win_value_v2_lite_action_new_88966_epoch001.progress.pt'
$trainingOutput = 'results\evaluations\v2_lite_action_new_88966.training.json'
$summary = 'results\evaluations\v2_lite_action_new_88966_epoch001.json'
$statusPath = 'results\evaluations\v2_lite_action_new_88966.status.json'
$timingsPath = 'results\evaluations\v2_lite_action_new_88966.timings.json'
$reference = 'results\evaluations\v1_original_new_50000_vs_88966.json'
$waitStatus = 'results\evaluations\action_delta_milestones.status.json'
$stdoutPath = 'logs\v2_lite_action_new_88966.stdout.log'
$stderrPath = 'logs\v2_lite_action_new_88966.stderr.log'
$lastCompleted = ''
$timings = [ordered]@{}

New-Item -ItemType Directory -Force -Path (
    'models', 'results\evaluations', 'logs', $data
) | Out-Null
$PID | Set-Content -Encoding ascii -LiteralPath (
    'logs\v2_lite_action_new_88966.pid'
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
        source = $source
        data = $data
        checkpoint = $checkpoint
        progress_checkpoint = $progress
        summary = $summary
        games_per_seat = $GamesPerSeat
        evaluation_seed = $EvaluationSeed
        training_seed = $TrainingSeed
        wait_status = $waitStatus
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
    $watch = [Diagnostics.Stopwatch]::StartNew()
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
    while (Test-Path -LiteralPath $waitStatus) {
        $waiting = Get-Content -Raw -Encoding utf8 $waitStatus |
            ConvertFrom-Json
        $waitProcess = Get-Process -Id ([int]$waiting.pid) `
            -ErrorAction SilentlyContinue
        if (
            $waiting.state -notin @('running', 'waiting') -or
            -not $waitProcess
        ) {
            break
        }
        Write-Status -Step 'waiting_for_action_delta' -State 'waiting'
        Start-Sleep -Seconds 30
    }
    $lastCompleted = 'wait_for_action_delta'

    $manifestPath = Join-Path $data 'manifest.json'
    $conversionComplete = $false
    if (Test-Path -LiteralPath $manifestPath) {
        $manifest = Get-Content -Raw -Encoding utf8 $manifestPath |
            ConvertFrom-Json
        $conversionComplete = (
            $manifest.status -eq 'complete' -and
            [int]$manifest.games -eq 88966 -and
            [int]$manifest.records -eq 6152268
        )
    }
    if (-not $conversionComplete) {
        Invoke-PythonStep 'convert' @(
            '-m', 'yellowstone.convert_replay_v2_lite_action',
            '--source', $source, '--output', $data,
            '--game-id-rebase', '954346',
            '--expected-games', '88966',
            '--expected-source-game-id-min', '954346',
            '--expected-source-game-id-max', '1043311'
        )
    } else {
        Complete-Step 'convert'
    }

    if (-not (Test-Path -LiteralPath $checkpoint)) {
        Invoke-PythonStep 'train_epoch001' @(
            '-m', 'yellowstone.train_value_v2_lite_action',
            '--data', $data, '--checkpoint', $checkpoint,
            '--progress-checkpoint', $progress,
            '--game-count', '88966', '--epochs', '1',
            '--batch-size', '256', '--learning-rate', '1e-3',
            '--seed', "$TrainingSeed", '--output', $trainingOutput
        )
    } else {
        Complete-Step 'train_epoch001'
    }

    foreach ($playerIndex in 0..3) {
        $name = "evaluate_p${playerIndex}"
        $output = (
            'results\evaluations\v2_lite_action_new_88966_epoch001_' +
            "${GamesPerSeat}_seed${EvaluationSeed}_p${playerIndex}.json"
        )
        $complete = $false
        if (Test-Path -LiteralPath $output) {
            $existing = Get-Content -Raw -Encoding utf8 $output |
                ConvertFrom-Json
            $complete = (
                [int]$existing.games -eq $GamesPerSeat -and
                $null -ne $existing.evaluated_player_one_card_turn_rate
            )
        }
        if (-not $complete) {
            Invoke-PythonStep $name @(
                '-m', 'yellowstone.evaluate_value_v2_lite_action',
                '--checkpoint', $checkpoint,
                '--games', "$GamesPerSeat", '--seed', "$EvaluationSeed",
                '--player-index', "$playerIndex", '--output', $output
            )
        } else {
            Complete-Step $name
        }
    }

    Invoke-PythonStep 'summarize' @(
        '-m', 'yellowstone.summarize_v2_lite_action',
        '--checkpoint', $checkpoint,
        '--evaluation-directory', 'results\evaluations',
        '--output', $summary, '--games-per-seat', "$GamesPerSeat",
        '--seed', "$EvaluationSeed", '--timings', $timingsPath,
        '--reference', $reference
    )
    Write-Status -Step 'done' -State 'complete'
} catch {
    Write-Status -Step 'failed' -State 'failed' -Message $_.Exception.Message
    throw
}
