param(
    [int]$BatchSize = 256,
    [string]$LogName = 'canonical_training_pipeline.log'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
$logDirectory = Join-Path $root 'logs'
$resultDirectory = Join-Path $root 'results\audits\canonicalization_fast_v1'
$log = Join-Path $logDirectory $LogName
$canonicalization = 'fast_lr_ud_color_v1'

function Invoke-LoggedPython {
    param([string[]]$Arguments)
    & python @Arguments 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) {
        throw "python failed with exit code $LASTEXITCODE`: $($Arguments -join ' ')"
    }
}

function Convert-Range {
    param(
        [string]$Output,
        [int]$StartPart,
        [int]$EndPart
    )
    "$(Get-Date -Format o) convert output=$Output range=$StartPart..$EndPart" |
        Add-Content $log
    Invoke-LoggedPython @(
        '-m', 'yellowstone.canonicalize_value_data',
        '--source', 'data\heuristic_value_data',
        '--output', $Output,
        '--start-part', "$StartPart",
        '--end-part', "$EndPart"
    )
}

function Train-Stage {
    param(
        [string]$Data,
        [string]$Checkpoint,
        [string]$Resume = ''
    )
    if (Test-Path $Checkpoint) {
        "$(Get-Date -Format o) skip existing checkpoint=$Checkpoint" | Add-Content $log
        return
    }
    "$(Get-Date -Format o) train data=$Data checkpoint=$Checkpoint resume=$Resume" |
        Add-Content $log
    $arguments = @(
        '-m', 'yellowstone.train_value',
        '--data', $Data,
        '--checkpoint', $Checkpoint,
        '--epochs', '1',
        '--batch-size', "$BatchSize",
        '--input-canonicalization', $canonicalization
    )
    if ($Resume) {
        $arguments += @('--resume', $Resume)
    }
    Invoke-LoggedPython $arguments
}

try {
    New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
    New-Item -ItemType Directory -Force -Path $resultDirectory | Out-Null
    "$(Get-Date -Format o) started canonicalization=$canonicalization" | Add-Content $log

    Invoke-LoggedPython @(
        '-m', 'yellowstone.audit_value_canonicalization',
        '--games', '2',
        '--max-records', '100',
        '--output', 'results\audits\canonicalization_fast_v1\orbit_audit_100.json'
    )

    Convert-Range 'data\heuristic_value_data_canonical_old' 0 660000
    Convert-Range 'data\heuristic_value_data_canonical_6h_only' 660100 960000
    Convert-Range 'data\heuristic_value_data_canonical_8h_only' 966400 1466300

    Train-Stage `
        'data\heuristic_value_data_canonical_old' `
        'models\win_value_canonical_old_001.pt'
    Train-Stage `
        'data\heuristic_value_data_canonical_6h_only' `
        'models\win_value_canonical_old_plus_6h_001.pt' `
        'models\win_value_canonical_old_001.pt'
    Train-Stage `
        'data\heuristic_value_data_canonical_8h_only' `
        'models\win_value_canonical_old_plus_6h_plus_8h_001.pt' `
        'models\win_value_canonical_old_plus_6h_001.pt'

    "$(Get-Date -Format o) completed" | Add-Content $log
}
catch {
    "$(Get-Date -Format o) failed: $_" | Add-Content $log
    exit 1
}
