param(
    [int]$TrainingSeed = 20260727,
    [string]$PythonExe = 'python'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'

$source = 'data\v2_exploratory_diverse_v1_20260730'
$sourceGameIdMin = 1100912
$refillCountGames = 59826
$refillCountSourceGameIdMax = $sourceGameIdMin + $refillCountGames - 1
$refillCountCanonicalization = 'fast_lr_ud_color_v1_refill_count_v1'
$refillCountData = 'data\v1_original_exploratory_59826_refill_count'
$refillCountCheckpoint = 'models\win_value_v1_exploratory_59826_refill_count_epoch001.pt'
$refillCountProgress = 'models\win_value_v1_exploratory_59826_refill_count_epoch001.progress.pt'

$bcenterData = 'data\v1_board_centered_explore_76919_none'
$bcenterEpoch1 = 'models\win_value_v1_board_centered_explore_76919_none_epoch001.pt'
$bcenterEpoch1Progress = 'models\win_value_v1_board_centered_explore_76919_none_epoch001.progress.pt'
$bcenterEpoch2 = 'models\win_value_v1_board_centered_explore_76919_none_epoch002.pt'
$bcenterEpoch2Progress = 'models\win_value_v1_board_centered_explore_76919_none_epoch002.progress.pt'

$statusPath = 'results\evaluations\v1_refill_count_and_bcenter_epoch2.status.json'
$timingsPath = 'results\evaluations\v1_refill_count_and_bcenter_epoch2.timings.json'
$pidPath = 'logs\v1_refill_count_and_bcenter_epoch2.pid'
$stdoutPath = 'logs\v1_refill_count_and_bcenter_epoch2.stdout.log'
$stderrPath = 'logs\v1_refill_count_and_bcenter_epoch2.stderr.log'
$lastCompleted = ''
$timings = [ordered]@{}

New-Item -ItemType Directory -Force -Path 'models', 'logs', 'results\evaluations' | Out-Null
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
        source = $source
        refill_count_data = $refillCountData
        refill_count_checkpoint = $refillCountCheckpoint
        bcenter_data = $bcenterData
        bcenter_epoch2_checkpoint = $bcenterEpoch2
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

    if (-not (Test-Path -LiteralPath (Join-Path $refillCountData 'conversion_manifest.json'))) {
        Invoke-PythonStep -Name 'convert_refill_count' -Arguments @(
            '-m', 'yellowstone.convert_replay_v2_to_v1_original',
            '--source', $source,
            '--output', $refillCountData,
            '--expected-games', "$refillCountGames",
            '--game-id-rebase', "$sourceGameIdMin",
            '--expected-source-game-id-min', "$sourceGameIdMin",
            '--expected-source-game-id-max', "$refillCountSourceGameIdMax",
            '--input-canonicalization', $refillCountCanonicalization
        )
    }
    else {
        Complete-Step 'convert_refill_count'
    }

    if (-not (Test-Path -LiteralPath $refillCountCheckpoint)) {
        Invoke-PythonStep -Name 'train_refill_count_epoch001' -Arguments @(
            '-m', 'yellowstone.train_value',
            '--data', $refillCountData,
            '--checkpoint', $refillCountCheckpoint,
            '--epochs', '1',
            '--batch-size', '256',
            '--learning-rate', '1e-3',
            '--seed', "$TrainingSeed",
            '--split-game-count', "$refillCountGames",
            '--input-canonicalization', $refillCountCanonicalization,
            '--value-schema', 'yellowstone.value.v1',
            '--history-semantics', 'rolling_last_two_placements',
            '--training-games', "$refillCountGames",
            '--progress-checkpoint', $refillCountProgress
        )
    }
    else {
        Complete-Step 'train_refill_count_epoch001'
    }

    if (-not (Test-Path -LiteralPath $bcenterEpoch2)) {
        if (-not (Test-Path -LiteralPath $bcenterEpoch1)) {
            throw "missing b-center epoch001 checkpoint: $bcenterEpoch1"
        }
        if (-not (Test-Path -LiteralPath $bcenterEpoch2Progress)) {
            if (-not (Test-Path -LiteralPath $bcenterEpoch1Progress)) {
                throw "missing b-center epoch001 progress checkpoint: $bcenterEpoch1Progress"
            }
            Copy-Item -LiteralPath $bcenterEpoch1Progress -Destination $bcenterEpoch2Progress
        }
        Invoke-PythonStep -Name 'train_bcenter_none_epoch002' -Arguments @(
            '-m', 'yellowstone.train_value',
            '--data', $bcenterData,
            '--checkpoint', $bcenterEpoch2,
            '--epochs', '2',
            '--batch-size', '256',
            '--learning-rate', '1e-3',
            '--seed', "$TrainingSeed",
            '--split-game-count', '76919',
            '--train-game-id-limit', '76919',
            '--input-canonicalization', 'board_centered_v1_history_none',
            '--value-schema', 'yellowstone.value.v1',
            '--history-semantics', 'none',
            '--training-games', '76919',
            '--progress-checkpoint', $bcenterEpoch2Progress
        )
    }
    else {
        Complete-Step 'train_bcenter_none_epoch002'
    }

    $lastCompleted = 'all'
    Write-Status -Step 'done' -State 'complete'
}
catch {
    Write-Status -Step 'failed' -State 'failed' -Message $_.Exception.Message
    throw
}
