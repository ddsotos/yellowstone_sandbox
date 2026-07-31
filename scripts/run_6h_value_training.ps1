param(
    [int]$Hours = 6,
    [int]$EpochsPerRound = 1,
    [string]$ResumeCheckpoint = '',
    [string]$CheckpointPrefix = 'win_value_6h_epoch',
    [string]$LogName = 'training_6h.log'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
$deadline = (Get-Date).AddHours($Hours)
$logDirectory = Join-Path $root 'logs'
$dataDirectory = Join-Path $root 'data\heuristic_value_data'
$modelDirectory = Join-Path $root 'models'
$log = Join-Path $logDirectory $LogName

function Resolve-ResumeCheckpoint {
    param([string]$Requested)
    if ($Requested) {
        $requestedPaths = if ([System.IO.Path]::IsPathRooted($Requested)) {
            @($Requested)
        } else {
            @(
                (Join-Path $root $Requested),
                (Join-Path $modelDirectory $Requested)
            )
        }
        foreach ($requestedPath in $requestedPaths) {
            if (Test-Path $requestedPath) {
                return $requestedPath
            }
        }
    }
    foreach ($candidate in @(
        'win_value_latest.pt',
        'win_value_660k_1epoch.pt',
        'win_value_40.pt',
        'win_value.pt'
    )) {
        $candidatePath = Join-Path $modelDirectory $candidate
        if (Test-Path $candidatePath) {
            return $candidatePath
        }
    }
    throw 'no resume checkpoint found'
}

function Resolve-NextRound {
    param([string]$Prefix)
    $matches = Get-ChildItem -LiteralPath $modelDirectory -Filter "$Prefix`_*.pt" -ErrorAction SilentlyContinue
    if (-not $matches) {
        return 1
    }
    $maxRound = 0
    foreach ($match in $matches) {
        if ($match.BaseName -match '^.+_(\d+)$') {
            $round = [int]$Matches[1]
            if ($round -gt $maxRound) {
                $maxRound = $round
            }
        }
    }
    return $maxRound + 1
}

try {
    New-Item -ItemType Directory -Force -Path $logDirectory, $modelDirectory | Out-Null
    $checkpoint = Resolve-ResumeCheckpoint -Requested $ResumeCheckpoint
    $round = Resolve-NextRound -Prefix $CheckpointPrefix
    while ((Get-Date) -lt $deadline) {
        $next = Join-Path $modelDirectory ("{0}_{1:D3}.pt" -f $CheckpointPrefix, $round)
        "$(Get-Date -Format o) round=$round resume=$checkpoint next=$next" | Add-Content $log
        & python -m yellowstone.train_value `
            --data $dataDirectory `
            --checkpoint $next `
            --resume $checkpoint `
            --epochs $EpochsPerRound `
            --batch-size 256 2>&1 | Tee-Object -FilePath $log -Append
        if ($LASTEXITCODE -ne 0) {
            throw "training failed with exit code $LASTEXITCODE"
        }
        Copy-Item -LiteralPath $next -Destination (Join-Path $modelDirectory 'win_value_latest.pt') -Force
        $checkpoint = $next
        $round++
    }
    "$(Get-Date -Format o) completed rounds=$($round - 1)" | Add-Content $log
} catch {
    "$(Get-Date -Format o) failed: $_" | Add-Content $log
    exit 1
}
