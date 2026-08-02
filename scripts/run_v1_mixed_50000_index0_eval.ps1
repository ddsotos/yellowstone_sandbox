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
$name = 'v1_mixed_50000_index0_eval'
$statusPath = Join-Path $evaluationDirectory "$name.status.json"
$timingsPath = Join-Path $evaluationDirectory "$name.timings.json"
$summaryPath = Join-Path $evaluationDirectory "v1_mixed_50000_index0_${Games}_seed${EvaluationSeed}.json"
$pidPath = "logs\$name.pid"
$stdoutPath = "logs\$name.stdout.log"
$stderrPath = "logs\$name.stderr.log"
$lastCompleted = ''
$timings = [ordered]@{}

$models = @(
    [ordered]@{
        name = 'v1_mixed_50000_epoch001'
        checkpoint = 'models\win_value_v1_original_explore_safe_counts_rank_color_50000_epoch001.pt'
        output = Join-Path $evaluationDirectory "v1_mixed_50000_epoch001_${Games}_seed${EvaluationSeed}_p0.json"
    },
    [ordered]@{
        name = 'v1_explore_59826_plus_mixed_50000_epoch002'
        checkpoint = 'models\win_value_v1_explore_59826_plus_mixed_50000_epoch002.pt'
        output = Join-Path $evaluationDirectory "v1_explore_59826_plus_mixed_50000_epoch002_${Games}_seed${EvaluationSeed}_p0.json"
    }
)

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
        summary = $summaryPath
        models = $models
        stdout = $stdoutPath
        stderr = $stderrPath
    })
}

function Complete-Step {
    param([string]$StepName)
    $script:lastCompleted = $StepName
    Write-Status -Step $StepName -State 'complete'
}

function Invoke-PythonStep {
    param([string]$StepName, [string[]]$Arguments)
    Write-Status -Step $StepName -State 'running'
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE"
    }
    $watch.Stop()
    $timings[$StepName] = $watch.Elapsed.TotalSeconds
    Write-AtomicJson -Path $timingsPath -Payload $timings
    Complete-Step $StepName
}

try {
    if (Test-Path -LiteralPath $timingsPath) {
        $existingTimings = Get-Content -Raw -Encoding utf8 $timingsPath | ConvertFrom-Json
        foreach ($property in $existingTimings.PSObject.Properties) {
            $timings[$property.Name] = [double]$property.Value
        }
    }

    foreach ($model in $models) {
        if (-not (Test-Path -LiteralPath $model.checkpoint)) {
            throw "missing checkpoint: $($model.checkpoint)"
        }
        $step = "evaluate_$($model.name)"
        if (-not (Test-Path -LiteralPath $model.output)) {
            Invoke-PythonStep -StepName $step -Arguments @(
                '-m', 'yellowstone.evaluate_value',
                '--checkpoint', $model.checkpoint,
                '--games', "$Games",
                '--seed', "$EvaluationSeed",
                '--player-index', '0',
                '--adaptive-pq-pruning',
                '--approximate-new-color-neighbors',
                '--output', $model.output
            )
        }
        else {
            Complete-Step $step
        }
    }

    $rows = @()
    foreach ($model in $models) {
        $payload = Get-Content -Raw -Encoding utf8 $model.output | ConvertFrom-Json
        if ([int]$payload.games -ne $Games) {
            throw "unexpected games in $($model.output)"
        }
        $rows += [ordered]@{
            name = $model.name
            checkpoint = $model.checkpoint
            output = $model.output
            games = [int]$payload.games
            wins = [double]$payload.wins
            fractional_wins = [double]$payload.fractional_wins
            win_rate = [double]$payload.win_rate
            evaluated_player_one_card_turns = [int]$payload.evaluated_player_one_card_turns
            evaluated_player_two_card_turns = [int]$payload.evaluated_player_two_card_turns
            evaluated_player_one_card_turn_rate = [double]$payload.evaluated_player_one_card_turn_rate
            checkpoint_contract = $payload.checkpoint_contract
        }
    }
    Write-AtomicJson -Path $summaryPath -Payload ([ordered]@{
        schema = 'yellowstone.v1_mixed_50000_index0_eval.v1'
        generated_at = (Get-Date).ToString('o')
        games = $Games
        player_index = 0
        evaluation_seed = $EvaluationSeed
        rows = $rows
        timings = $timings
    })
    $lastCompleted = 'all'
    Write-Status -Step 'done' -State 'complete'
}
catch {
    Write-Status -Step 'failed' -State 'failed' -Message $_.Exception.Message
    throw
}
