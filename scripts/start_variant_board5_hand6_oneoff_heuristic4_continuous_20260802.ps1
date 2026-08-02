$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force -Path 'logs','results\collections','data' | Out-Null

$name = 'v2_variant_board5_hand6_oneoff_tiered_heuristic4_continuous_20260802'
$stdoutPath = Join-Path $root "logs\$name.stdout.log"
$stderrPath = Join-Path $root "logs\$name.stderr.log"
$launcherPidPath = Join-Path $root "logs\$name.launcher.pid"
$statusPath = Join-Path $root "results\collections\$name.status.json"

function Write-AtomicJson {
    param([object]$Payload, [string]$Path)
    $temporary = "$Path.tmp"
    $Payload | ConvertTo-Json -Depth 12 | Set-Content -Encoding utf8 $temporary
    Move-Item -Force -LiteralPath $temporary -Destination $Path
}

function Write-LaunchStatus {
    param([string]$State, [string]$Message = '')
    Write-AtomicJson -Path $statusPath -Payload ([ordered]@{
        state = $State
        step = 'launch'
        last_completed_step = ''
        message = $Message
        updated_at = (Get-Date).ToString('o')
        output = "data\$name"
        stop_file = "results\collections\$name.stop"
        stdout = "logs\$name.stdout.log"
        stderr = "logs\$name.stderr.log"
    })
}

$arguments = @(
    '-NoProfile',
    '-ExecutionPolicy', 'Bypass',
    '-File',
    (Join-Path $root 'scripts\run_variant_board5_hand6_oneoff_heuristic4_continuous_20260802.ps1')
)

try {
    Write-LaunchStatus -State 'launching'
    $process = Start-Process `
        -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
        -ArgumentList $arguments `
        -WorkingDirectory $root `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -WindowStyle Hidden `
        -PassThru
    Set-Content -Encoding ascii -LiteralPath $launcherPidPath -Value $process.Id
    Write-LaunchStatus -State 'running' -Message "launcher_pid=$($process.Id)"
    Write-Output "started $name launcher_pid=$($process.Id)"
}
catch {
    Write-LaunchStatus -State 'failed' -Message $_.Exception.Message
    throw
}
