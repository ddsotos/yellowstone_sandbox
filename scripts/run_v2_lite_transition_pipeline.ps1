param(
    [int]$GamesPerSeat = 1000,
    [int]$EvaluationSeed = 20260725,
    [int]$TrainingSeed = 20260726
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUNBUFFERED = '1'

$source = 'data\v2_generation0_200k_frame_features'
$data = 'data\v2_lite_transition_generation0_197800_tensors'
$checkpoint = 'models\win_value_v2_lite_transition_generation0_197800_epoch001.pt'
$evaluationDirectory = 'results\evaluations'
$statusPath = Join-Path $evaluationDirectory 'v2_lite_transition.status.json'
$timingsPath = Join-Path $evaluationDirectory 'v2_lite_transition.timings.json'
$summaryPath = Join-Path $evaluationDirectory 'v2_lite_transition_generation0_197800_epoch001.json'
$waitPidPath = 'logs\v1_history_cross_2x2.pid'
$timings = [ordered]@{}
$lastCompleted = ''

New-Item -ItemType Directory -Force -Path $evaluationDirectory | Out-Null
if (Test-Path -LiteralPath $timingsPath) {
    $existing = Get-Content -Raw -Encoding utf8 $timingsPath | ConvertFrom-Json
    foreach ($property in $existing.PSObject.Properties) {
        $timings[$property.Name] = [double]$property.Value
    }
}

function Write-AtomicJson {
    param([object]$Payload, [string]$Path)
    $temporary = "$Path.tmp"
    $Payload | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 $temporary
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
        source = $source
        data = $data
        checkpoint = $checkpoint
        summary = $summaryPath
        games_per_seat = $GamesPerSeat
        evaluation_seed = $EvaluationSeed
        training_seed = $TrainingSeed
        stdout = 'logs\v2_lite_transition.stdout.log'
        stderr = 'logs\v2_lite_transition.stderr.log'
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
    if (Test-Path -LiteralPath $waitPidPath) {
        $waitPid = [int](Get-Content -Raw $waitPidPath).Trim()
        while (Get-Process -Id $waitPid -ErrorAction SilentlyContinue) {
            Write-Status -Step 'waiting_for_v1_history_cross' -State 'waiting'
            Start-Sleep -Seconds 30
        }
    }

    $conversionComplete = $false
    $manifestPath = Join-Path $data 'manifest.json'
    if (Test-Path -LiteralPath $manifestPath) {
        $manifest = Get-Content -Raw -Encoding utf8 $manifestPath |
            ConvertFrom-Json
        $conversionComplete = (
            $manifest.status -eq 'complete' -and
            [int]$manifest.games -eq 197800
        )
    }
    if (-not $conversionComplete) {
        Invoke-PythonStep -Name 'convert' -Arguments @(
            '-m', 'yellowstone.convert_replay_v2_lite',
            '--source', $source,
            '--output', $data
        )
    }
    else {
        Complete-Step 'convert'
    }

    if (-not (Test-Path -LiteralPath $checkpoint)) {
        Invoke-PythonStep -Name 'train_epoch001' -Arguments @(
            '-m', 'yellowstone.train_value_v2_lite',
            '--data', $data,
            '--checkpoint', $checkpoint,
            '--epochs', '1',
            '--batch-size', '256',
            '--learning-rate', '1e-3',
            '--seed', "$TrainingSeed"
        )
    }
    else {
        Complete-Step 'train_epoch001'
    }

    foreach ($playerIndex in 0..3) {
        $name = "evaluate_p${playerIndex}"
        $output = Join-Path $evaluationDirectory (
            'v2_lite_transition_generation0_197800_epoch001_' +
            "${GamesPerSeat}_seed${EvaluationSeed}_p${playerIndex}.json"
        )
        $complete = $false
        if (Test-Path -LiteralPath $output) {
            $existing = Get-Content -Raw -Encoding utf8 $output |
                ConvertFrom-Json
            $complete = (
                [int]$existing.games -eq $GamesPerSeat -and
                $null -ne $existing.one_card_turn_rate
            )
        }
        if (-not $complete) {
            Invoke-PythonStep -Name $name -Arguments @(
                '-m', 'yellowstone.value_evaluation_v2_lite',
                '--checkpoint', $checkpoint,
                '--games', "$GamesPerSeat",
                '--seed', "$EvaluationSeed",
                '--player-index', "$playerIndex",
                '--output', $output
            )
        }
        else {
            Complete-Step $name
        }
    }

    Invoke-PythonStep -Name 'summarize' -Arguments @(
        '-m', 'yellowstone.summarize_v2_lite',
        '--checkpoint', $checkpoint,
        '--evaluation-directory', $evaluationDirectory,
        '--output', $summaryPath,
        '--games-per-seat', "$GamesPerSeat",
        '--seed', "$EvaluationSeed",
        '--timings', $timingsPath
    )
    Write-Status -Step 'complete' -State 'complete'
}
catch {
    Write-Status -Step 'failed' -State 'failed' -Message $_.Exception.Message
    throw
}

