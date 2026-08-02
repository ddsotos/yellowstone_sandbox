param(
    [string]$PythonExe = 'python',
    [int]$TrainingGames = 50000,
    [int]$TrainingSeed = 20260727,
    [int]$PollSeconds = 60
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$root\src;$env:PYTHONPATH" } else { "$root\src" }

$name = 'explore_safe_counts_rank_color_50000_v1_preplay_training'
$sourceOneoff = 'data\v2_exploratory_safe_counts_oneoff_20260731'
$sourceCurrent = 'data\v2_exploratory_safe_counts_rank_color_20260801'
$sourceManifest = Join-Path $sourceCurrent 'collection_manifest.json'
$sourceCurrentGames = 6857
$sourceGameIdMin = 0
$sourceGameIdMax = $sourceGameIdMin + $TrainingGames - 1
$snapshot = "data\v2_exploratory_safe_counts_mixed_${TrainingGames}_snapshot"
$snapshotManifest = Join-Path $snapshot 'collection_manifest.json'

$v1Data = "data\v1_original_explore_safe_counts_rank_color_${TrainingGames}_canonical"
$v1Checkpoint = "models\win_value_v1_original_explore_safe_counts_rank_color_${TrainingGames}_epoch001.pt"
$v1Progress = "models\win_value_v1_original_explore_safe_counts_rank_color_${TrainingGames}_epoch001.progress.pt"

$preplayData = "data\privileged_state_explore_safe_counts_rank_color_${TrainingGames}_tensors"
$preplayPrefix = "models\preplay_v1_explore_safe_counts_rank_color_${TrainingGames}"
$preplaySelection = "results\evaluations\preplay_v1_explore_safe_counts_rank_color_${TrainingGames}.selection.json"

$statusPath = "results\evaluations\$name.status.json"
$timingsPath = "results\evaluations\$name.timings.json"
$pidPath = "logs\$name.pid"
$stdoutPath = "logs\$name.stdout.log"
$stderrPath = "logs\$name.stderr.log"
$lastCompleted = ''
$timings = [ordered]@{}

New-Item -ItemType Directory -Force -Path 'models', 'logs', 'results\evaluations', 'data' | Out-Null
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
    $availableGames = 0
    $availableShards = 0
    if (Test-Path -LiteralPath $sourceManifest) {
        try {
            $manifest = Get-Content -Raw -Encoding utf8 $sourceManifest | ConvertFrom-Json
            $availableGames = [int]$manifest.games
            $availableShards = [int]$manifest.completed_shards
        }
        catch {}
    }
    Write-AtomicJson -Path $statusPath -Payload ([ordered]@{
        state = $State
        step = $Step
        last_completed_step = $lastCompleted
        message = $Message
        updated_at = (Get-Date).ToString('o')
        pid = $PID
        training_games = $TrainingGames
        available_games = $availableGames
        available_shards = $availableShards
        source_oneoff = $sourceOneoff
        source_current = $sourceCurrent
        source_current_games = $sourceCurrentGames
        snapshot = $snapshot
        v1_data = $v1Data
        v1_checkpoint = $v1Checkpoint
        preplay_data = $preplayData
        preplay_prefix = $preplayPrefix
        preplay_selection = $preplaySelection
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
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
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
    if (Test-Path -LiteralPath $timingsPath) {
        $existingTimings = Get-Content -Raw -Encoding utf8 $timingsPath | ConvertFrom-Json
        foreach ($property in $existingTimings.PSObject.Properties) {
            $timings[$property.Name] = [double]$property.Value
        }
    }

    if (-not (Test-Path -LiteralPath $snapshotManifest)) {
        Invoke-PythonStep -Name 'snapshot_50000_games' -Arguments @(
            '-m', 'yellowstone.make_replay_snapshot',
            '--source', "$sourceOneoff",
            '--source', "${sourceCurrent}=${sourceCurrentGames}",
            '--output', $snapshot,
            '--games', "$TrainingGames",
            '--shard-games', '100'
        )
        Complete-Step 'snapshot_50000_games'
    }
    else {
        Complete-Step 'snapshot_50000_games'
    }

    if (-not (Test-Path -LiteralPath (Join-Path $v1Data 'conversion_manifest.json'))) {
        Invoke-PythonStep -Name 'convert_v1_original' -Arguments @(
            '-m', 'yellowstone.convert_replay_v2_to_v1_original',
            '--source', $snapshot,
            '--output', $v1Data,
            '--expected-games', "$TrainingGames",
            '--game-id-rebase', "$sourceGameIdMin",
            '--expected-source-game-id-min', "$sourceGameIdMin",
            '--expected-source-game-id-max', "$sourceGameIdMax",
            '--input-canonicalization', 'fast_lr_ud_color_v1'
        )
    }
    else {
        Complete-Step 'convert_v1_original'
    }

    if (-not (Test-Path -LiteralPath $v1Checkpoint)) {
        Invoke-PythonStep -Name 'train_v1_original_epoch001' -Arguments @(
            '-m', 'yellowstone.train_value',
            '--data', $v1Data,
            '--checkpoint', $v1Checkpoint,
            '--epochs', '1',
            '--batch-size', '256',
            '--learning-rate', '1e-3',
            '--seed', "$TrainingSeed",
            '--split-game-count', "$TrainingGames",
            '--input-canonicalization', 'fast_lr_ud_color_v1',
            '--value-schema', 'yellowstone.value.v1',
            '--history-semantics', 'rolling_last_two_placements',
            '--training-games', "$TrainingGames",
            '--progress-checkpoint', $v1Progress
        )
    }
    else {
        Complete-Step 'train_v1_original_epoch001'
    }

    if (-not (Test-Path -LiteralPath (Join-Path $preplayData 'manifest.json'))) {
        Invoke-PythonStep -Name 'convert_preplay_privileged' -Arguments @(
            '-m', 'yellowstone.convert_privileged_state',
            '--source', $snapshot,
            '--output', $preplayData
        )
    }
    else {
        Complete-Step 'convert_preplay_privileged'
    }

    if (-not (Test-Path -LiteralPath $preplaySelection)) {
        Invoke-PythonStep -Name 'train_preplay_privileged_epochs002' -Arguments @(
            '-m', 'yellowstone.train_privileged_state',
            '--data', $preplayData,
            '--checkpoint-prefix', $preplayPrefix,
            '--epochs', '2',
            '--batch-size', '256',
            '--learning-rate', '1e-3',
            '--seed', "$TrainingSeed",
            '--selection-output', $preplaySelection
        )
    }
    else {
        Complete-Step 'train_preplay_privileged_epochs002'
    }

    $lastCompleted = 'all'
    Write-Status -Step 'done' -State 'complete'
}
catch {
    Write-Status -Step 'failed' -State 'failed' -Message $_.Exception.Message
    throw
}
