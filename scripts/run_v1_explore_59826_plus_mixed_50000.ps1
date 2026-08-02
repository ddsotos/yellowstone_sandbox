param(
    [string]$PythonExe = 'python',
    [int]$TrainingSeed = 20260727,
    [int]$PollSeconds = 60
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$root\src;$env:PYTHONPATH" } else { "$root\src" }

$name = 'v1_explore_59826_plus_mixed_50000'
$data = 'data\v1_original_explore_safe_counts_rank_color_50000_canonical'
$dataManifest = Join-Path $data 'conversion_manifest.json'
$resume = 'models\win_value_v1_exploratory_59826_epoch001_pct100.pt'
$checkpoint = 'models\win_value_v1_explore_59826_plus_mixed_50000_epoch002.pt'
$progress = 'models\win_value_v1_explore_59826_plus_mixed_50000_epoch002.progress.pt'
$statusPath = "results\evaluations\$name.status.json"
$timingsPath = "results\evaluations\$name.timings.json"
$pidPath = "logs\$name.pid"
$stdoutPath = "logs\$name.stdout.log"
$stderrPath = "logs\$name.stderr.log"
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
        data = $data
        resume = $resume
        checkpoint = $checkpoint
        progress = $progress
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
    if (-not (Test-Path -LiteralPath $resume)) {
        throw "missing resume checkpoint: $resume"
    }

    while (-not (Test-Path -LiteralPath $dataManifest)) {
        Write-Status -Step 'wait_for_mixed_50000_v1_tensors' -State 'running'
        Start-Sleep -Seconds $PollSeconds
    }
    Complete-Step 'wait_for_mixed_50000_v1_tensors'

    if (-not (Test-Path -LiteralPath $checkpoint)) {
        Invoke-PythonStep -Name 'train_resume_epoch002' -Arguments @(
            '-m', 'yellowstone.train_value',
            '--data', $data,
            '--checkpoint', $checkpoint,
            '--epochs', '1',
            '--batch-size', '256',
            '--learning-rate', '1e-3',
            '--seed', "$TrainingSeed",
            '--resume', $resume,
            '--split-game-count', '50000',
            '--input-canonicalization', 'fast_lr_ud_color_v1',
            '--value-schema', 'yellowstone.value.v1',
            '--history-semantics', 'rolling_last_two_placements',
            '--training-games', '109826',
            '--progress-checkpoint', $progress
        )
    }
    else {
        Complete-Step 'train_resume_epoch002'
    }

    $lastCompleted = 'all'
    Write-Status -Step 'done' -State 'complete'
}
catch {
    Write-Status -Step 'failed' -State 'failed' -Message $_.Exception.Message
    throw
}
