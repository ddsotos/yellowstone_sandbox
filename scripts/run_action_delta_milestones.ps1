param(
    [int]$Games = 1000,
    [int]$PlayerIndex = 0,
    [int]$EvaluationSeed = 20260725,
    [int]$TrainingSeed = 20260727,
    [string]$PythonExe = 'python'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'

$data = 'data\action_delta_next_turn_continuation'
$snapshot = Join-Path $data 'training_snapshot_20260730.json'
$prefix = 'models\action_delta_continuous_epoch001'
$progress = "$prefix.progress.pt"
$trainingSummary = 'results\evaluations\action_delta_milestones.training.json'
$summary = 'results\evaluations\action_delta_milestones_seat0_1000.json'
$statusPath = 'results\evaluations\action_delta_milestones.status.json'
$timingsPath = 'results\evaluations\action_delta_milestones.timings.json'
$proposerSelection = 'results\evaluations\action_delta_proposer.selection.json'
$stdoutPath = 'logs\action_delta_milestones.stdout.log'
$stderrPath = 'logs\action_delta_milestones.stderr.log'
$lastCompleted = ''
$timings = [ordered]@{}

New-Item -ItemType Directory -Force -Path (
    'models', 'results\evaluations', 'logs'
) | Out-Null
$PID | Set-Content -Encoding ascii -LiteralPath (
    'logs\action_delta_milestones.pid'
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
        data = $data
        snapshot = $snapshot
        checkpoint_prefix = $prefix
        progress_checkpoint = $progress
        training_summary = $trainingSummary
        summary = $summary
        games = $Games
        player_index = $PlayerIndex
        evaluation_seed = $EvaluationSeed
        training_seed = $TrainingSeed
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
    if (-not (Test-Path -LiteralPath $snapshot)) {
        Invoke-Step 'freeze_snapshot' @(
            '-m', 'yellowstone.action_delta_snapshot',
            '--data', $data, '--output', $snapshot
        )
    } else {
        $lastCompleted = 'freeze_snapshot'
    }

    if (-not (Test-Path -LiteralPath $trainingSummary)) {
        Invoke-Step 'train_milestones' @(
            '-m', 'yellowstone.train_action_delta',
            '--snapshot', $snapshot,
            '--checkpoint-prefix', $prefix,
            '--progress-checkpoint', $progress,
            '--milestones', '10,30,50,100',
            '--epochs', '1', '--batch-size', '256',
            '--learning-rate', '1e-3', '--seed', "$TrainingSeed",
            '--output', $trainingSummary
        )
    } else {
        $lastCompleted = 'train_milestones'
    }

    $proposer = [string]((
        Get-Content -Raw -Encoding utf8 $proposerSelection |
            ConvertFrom-Json
    ).checkpoint)
    foreach ($percent in @(10, 30, 50, 100)) {
        $tag = '{0:d3}' -f $percent
        $checkpoint = "${prefix}_pct${tag}.pt"
        $evaluation = (
            "results\evaluations\action_delta_milestone_pct${tag}_" +
            "${Games}_seed${EvaluationSeed}_p${PlayerIndex}.json"
        )
        if (-not (Test-Path -LiteralPath $evaluation)) {
            Invoke-Step "evaluate_pct${tag}" @(
                '-m', 'yellowstone.evaluate_action_delta',
                '--proposer-checkpoint', $proposer,
                '--delta-checkpoint', $checkpoint,
                '--games', "$Games", '--seed', "$EvaluationSeed",
                '--player-index', "$PlayerIndex", '--output', $evaluation
            )
        } else {
            $lastCompleted = "evaluate_pct${tag}"
        }
    }

    Invoke-Step 'summarize' @(
        '-m', 'yellowstone.summarize_action_delta_milestones',
        '--training-summary', $trainingSummary,
        '--evaluation-directory', 'results\evaluations',
        '--output', $summary, '--games', "$Games",
        '--seed', "$EvaluationSeed", '--player-index', "$PlayerIndex"
    )
    Write-Status -Step 'done' -State 'complete'
} catch {
    Write-Status -Step 'failed' -State 'failed' -Message $_.Exception.Message
    throw
}
