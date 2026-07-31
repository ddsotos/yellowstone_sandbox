param(
    [int]$GamesPerSeat = 1000,
    [int]$EvaluationSeed = 20260725,
    [int]$TrainingSeed = 20260727
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUNBUFFERED = '1'

$backup = (
    'D:\codex-backup\yellow_3_derived_training_data_2026-07-29\' +
    'v1_original_generation0_197800_canonical'
)
$data = 'data\v1_original_generation0_197800_canonical'
$checkpoint = 'models\win_value_v1_original_conv3_generation0_197800_epoch002.pt'
$progressCheckpoint = (
    'models\win_value_v1_original_conv3_generation0_197800_epoch002.progress.pt'
)
$evaluationDirectory = 'results\evaluations'
$statusPath = Join-Path $evaluationDirectory 'v1_original_conv3.status.json'
$timingsPath = Join-Path $evaluationDirectory 'v1_original_conv3.timings.json'
$summaryPath = Join-Path $evaluationDirectory (
    'v1_original_conv3_generation0_197800_epoch002.json'
)
$restoreMarker = Join-Path $data '.conv3_restore_complete.json'
$v2StatusPath = Join-Path $evaluationDirectory 'v2_lite_transition.status.json'
$timings = [ordered]@{}
$lastCompleted = ''

New-Item -ItemType Directory -Force -Path (
    'models', $evaluationDirectory, 'results\smoke'
) | Out-Null
if (Test-Path -LiteralPath $timingsPath) {
    $existing = Get-Content -Raw -Encoding utf8 $timingsPath |
        ConvertFrom-Json
    foreach ($property in $existing.PSObject.Properties) {
        $timings[$property.Name] = [double]$property.Value
    }
}

function Write-AtomicJson {
    param([object]$Payload, [string]$Path)
    $temporary = "$Path.tmp"
    $Payload | ConvertTo-Json -Depth 8 |
        Set-Content -Encoding utf8 $temporary
    Move-Item -Force -LiteralPath $temporary -Destination $Path
}

function Write-Status {
    param([string]$Step, [string]$State, [string]$Message = '')
    Write-AtomicJson -Path $statusPath -Payload ([ordered]@{
        step = $Step
        state = $State
        last_completed_step = $lastCompleted
        message = $Message
        updated_at = (Get-Date).ToString('o')
        pid = $PID
        backup = $backup
        data = $data
        checkpoint = $checkpoint
        progress_checkpoint = $progressCheckpoint
        summary = $summaryPath
        games_per_seat = $GamesPerSeat
        evaluation_seed = $EvaluationSeed
        training_seed = $TrainingSeed
        waiting_for = $v2StatusPath
        stdout = 'logs\v1_original_conv3.stdout.log'
        stderr = 'logs\v1_original_conv3.stderr.log'
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
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    $watch.Stop()
    $timings[$Name] = $watch.Elapsed.TotalSeconds
    Write-AtomicJson -Payload $timings -Path $timingsPath
    Complete-Step $Name
}

try {
    while ($true) {
        if (-not (Test-Path -LiteralPath $v2StatusPath)) {
            Write-Status -Step 'waiting_for_v2_lite' -State 'waiting'
            Start-Sleep -Seconds 30
            continue
        }
        $v2Status = Get-Content -Raw -Encoding utf8 $v2StatusPath |
            ConvertFrom-Json
        if ($v2Status.state -eq 'complete') {
            Complete-Step 'wait_for_v2_lite'
            break
        }
        if ($v2Status.state -eq 'failed') {
            throw "V2-lite pipeline failed: $($v2Status.message)"
        }
        $v2Process = Get-Process -Id ([int]$v2Status.pid) `
            -ErrorAction SilentlyContinue
        if (-not $v2Process) {
            throw 'V2-lite status is non-terminal but its PID is dead'
        }
        Write-Status -Step 'waiting_for_v2_lite' -State 'waiting'
        Start-Sleep -Seconds 30
    }

    $restored = $false
    if (
        (Test-Path -LiteralPath $restoreMarker) -and
        (Test-Path -LiteralPath (Join-Path $data 'conversion_manifest.json'))
    ) {
        $marker = Get-Content -Raw -Encoding utf8 $restoreMarker |
            ConvertFrom-Json
        $restored = (
            [int]$marker.games -eq 197800 -and
            [int]$marker.parts -eq 1978
        )
    }
    if (-not $restored) {
        Invoke-PythonStep -Name 'restore_training_tensors' -Arguments @(
            '-m', 'yellowstone.restore_v1_dataset',
            '--source', $backup,
            '--destination', $data,
            '--expected-games', '197800',
            '--expected-schema', 'yellowstone.value.v1',
            '--expected-history-semantics',
            'rolling_last_two_placements',
            '--expected-canonicalization', 'fast_lr_ud_color_v1'
        )
        Write-AtomicJson -Path $restoreMarker -Payload ([ordered]@{
            games = 197800
            parts = 1978
            source = $backup
            restored_at = (Get-Date).ToString('o')
        })
    }
    else {
        Complete-Step 'restore_training_tensors'
    }

    if (-not (Test-Path -LiteralPath $checkpoint)) {
        Invoke-PythonStep -Name 'train_conv3_epoch002' -Arguments @(
            '-m', 'yellowstone.train_value',
            '--data', $data,
            '--checkpoint', $checkpoint,
            '--progress-checkpoint', $progressCheckpoint,
            '--epochs', '2',
            '--batch-size', '256',
            '--learning-rate', '1e-3',
            '--seed', "$TrainingSeed",
            '--convolution-layers', '3',
            '--input-canonicalization', 'fast_lr_ud_color_v1',
            '--value-schema', 'yellowstone.value.v1',
            '--history-semantics', 'rolling_last_two_placements',
            '--training-games', '197800',
            '--split-game-count', '197800'
        )
    }
    else {
        Complete-Step 'train_conv3_epoch002'
    }

    $smoke = 'results\smoke\v1_original_conv3_epoch002_3.json'
    if (-not (Test-Path -LiteralPath $smoke)) {
        Invoke-PythonStep -Name 'smoke_conv3' -Arguments @(
            '-m', 'yellowstone.evaluate_value',
            '--checkpoint', $checkpoint,
            '--games', '3',
            '--seed', "$EvaluationSeed",
            '--player-index', '0',
            '--adaptive-pq-pruning',
            '--approximate-new-color-neighbors',
            '--output', $smoke
        )
    }
    else {
        Complete-Step 'smoke_conv3'
    }

    foreach ($playerIndex in 0..3) {
        $name = "evaluate_p${playerIndex}"
        $output = Join-Path $evaluationDirectory (
            'v1_original_conv3_generation0_197800_epoch002_' +
            "${GamesPerSeat}_seed${EvaluationSeed}_p${playerIndex}.json"
        )
        $complete = $false
        if (Test-Path -LiteralPath $output) {
            $existing = Get-Content -Raw -Encoding utf8 $output |
                ConvertFrom-Json
            $complete = (
                [int]$existing.games -eq $GamesPerSeat -and
                $null -ne $existing.evaluated_player_one_card_turns -and
                $null -ne $existing.evaluated_player_two_card_turns
            )
        }
        if (-not $complete) {
            Invoke-PythonStep -Name $name -Arguments @(
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
            Complete-Step $name
        }
    }

    Invoke-PythonStep -Name 'summarize' -Arguments @(
        '-m', 'yellowstone.summarize_v1_conv3',
        '--checkpoint', $checkpoint,
        '--baseline',
        'results\evaluations\v1_history_matched_evaluation.json',
        '--baseline-comparison',
        'results\evaluations\v2_v1_original_history_comparison.json',
        '--evaluation-directory', $evaluationDirectory,
        '--timings', $timingsPath,
        '--output', $summaryPath,
        '--games-per-seat', "$GamesPerSeat",
        '--seed', "$EvaluationSeed"
    )
    Write-Status -Step 'complete' -State 'complete'
}
catch {
    Write-Status -Step 'failed' -State 'failed' `
        -Message $_.Exception.Message
    throw
}
