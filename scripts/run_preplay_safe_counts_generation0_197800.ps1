param(
    [int]$TrainingSeed = 20260727,
    [string]$PythonExe = 'python'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$root\src;$env:PYTHONPATH" } else { "$root\src" }

$name = 'preplay_safe_counts_generation0_197800'
$statusPath = "results\evaluations\$name.status.json"
$timingsPath = "results\evaluations\$name.timings.json"
$source = 'data\v2_generation0_200k_frame_features'
$data = 'data\privileged_state_generation0_197800_safe_counts_tensors'
$prefix = 'models\preplay_safe_counts_generation0_197800'
$selection = "results\evaluations\$name.selection.json"
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
        source = $source
        data = $data
        checkpoint_prefix = $prefix
        epoch001_checkpoint = "${prefix}_epoch001.pt"
        epoch002_checkpoint = "${prefix}_epoch002.pt"
        selection = $selection
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

    $manifestPath = Join-Path $data 'manifest.json'
    if (-not (Test-Path -LiteralPath $manifestPath)) {
        Invoke-PythonStep -StepName 'convert_preplay_safe_counts' -Arguments @(
            '-m', 'yellowstone.convert_privileged_state',
            '--source', $source,
            '--output', $data
        )
    }
    else {
        Complete-Step 'convert_preplay_safe_counts'
    }

    if (-not (Test-Path -LiteralPath $selection)) {
        Invoke-PythonStep -StepName 'train_preplay_safe_counts_epochs002' -Arguments @(
            '-m', 'yellowstone.train_privileged_state',
            '--data', $data,
            '--checkpoint-prefix', $prefix,
            '--epochs', '2',
            '--batch-size', '256',
            '--learning-rate', '1e-3',
            '--seed', "$TrainingSeed",
            '--selection-output', $selection
        )
    }
    else {
        Complete-Step 'train_preplay_safe_counts_epochs002'
    }

    $lastCompleted = 'all'
    Write-Status -Step 'done' -State 'complete'
}
catch {
    Write-Status -Step 'failed' -State 'failed' -Message $_.Exception.Message
    throw
}
