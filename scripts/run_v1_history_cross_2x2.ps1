param(
    [int]$GamesPerSeat = 1000,
    [int]$Seed = 20260725
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUNBUFFERED = '1'

$evaluationDirectory = 'results\evaluations'
$statusPath = Join-Path $evaluationDirectory 'v1_history_cross_2x2.status.json'
$timingsPath = Join-Path $evaluationDirectory 'v1_history_cross_2x2.timings.json'
$summaryPath = Join-Path $evaluationDirectory 'v1_history_cross_2x2.json'
$lastCompleted = ''
$timings = [ordered]@{}
$conditions = @(
    [ordered]@{
        model = 'original'
        history = 'rolling'
        checkpoint = 'models\win_value_v1_original_generation0_197800_epoch002.pt'
        current_turn_only = $false
    },
    [ordered]@{
        model = 'original'
        history = 'turn_local'
        checkpoint = 'models\win_value_v1_original_generation0_197800_epoch002.pt'
        current_turn_only = $true
    },
    [ordered]@{
        model = 'historyfix'
        history = 'rolling'
        checkpoint = 'models\win_value_v1_historyfix_generation0_197800_epoch002.pt'
        current_turn_only = $false
    },
    [ordered]@{
        model = 'historyfix'
        history = 'turn_local'
        checkpoint = 'models\win_value_v1_historyfix_generation0_197800_epoch002.pt'
        current_turn_only = $true
    }
)

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
        games_per_seat = $GamesPerSeat
        seed = $Seed
        summary = $summaryPath
        stdout = 'logs\v1_history_cross_2x2.stdout.log'
        stderr = 'logs\v1_history_cross_2x2.stderr.log'
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
    foreach ($condition in $conditions) {
        foreach ($playerIndex in 0..3) {
            $name = (
                "evaluate_$($condition.model)_$($condition.history)_" +
                "p${playerIndex}"
            )
            $output = Join-Path $evaluationDirectory (
                "v1_history_cross_$($condition.model)_" +
                "$($condition.history)_epoch002_" +
                "${GamesPerSeat}_seed${Seed}_p${playerIndex}.json"
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
            if ($complete) {
                Complete-Step $name
                continue
            }
            $arguments = @(
                '-m', 'yellowstone.evaluate_value',
                '--checkpoint', $condition.checkpoint,
                '--games', "$GamesPerSeat",
                '--seed', "$Seed",
                '--player-index', "$playerIndex",
                '--adaptive-pq-pruning',
                '--approximate-new-color-neighbors',
                '--output', $output
            )
            if ($condition.current_turn_only) {
                $arguments += '--current-turn-history-only'
            }
            Invoke-PythonStep -Name $name -Arguments $arguments
        }
    }
    Invoke-PythonStep -Name 'summarize' -Arguments @(
        '-m', 'yellowstone.summarize_v1_history_cross',
        '--evaluation-directory', $evaluationDirectory,
        '--output', $summaryPath,
        '--games-per-seat', "$GamesPerSeat",
        '--seed', "$Seed"
    )
    Write-Status -Step 'complete' -State 'complete'
}
catch {
    Write-Status -Step 'failed' -State 'failed' -Message $_.Exception.Message
    throw
}
