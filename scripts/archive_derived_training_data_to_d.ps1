param(
    [string]$BackupBase = 'D:\codex-backup',
    [string]$TargetName = 'yellow_3_derived_training_data_2026-07-29'
)

$ErrorActionPreference = 'Stop'

$projectRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot '..')
)
$sourceRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $projectRoot 'data')
)
$backupRoot = [System.IO.Path]::GetFullPath($BackupBase)
$targetRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $backupRoot $TargetName)
)
$sourcePrefix = $sourceRoot.TrimEnd(
    [System.IO.Path]::DirectorySeparatorChar
) + [System.IO.Path]::DirectorySeparatorChar
$backupPrefix = $backupRoot.TrimEnd(
    [System.IO.Path]::DirectorySeparatorChar
) + [System.IO.Path]::DirectorySeparatorChar
$names = @(
    'v2_generation0_197800_tensors',
    'v1_historyfix_generation0_197800_canonical',
    'v1_original_generation0_197800_canonical',
    'v1_historyfix_legacy_200k_canonical',
    'v1_history3_legacy_100k_canonical',
    'v2_preflight_dev_tensors',
    'v2_preflight_10_tensors',
    'v2_preflight_10_frame_features_tensors'
)

if (-not $targetRoot.StartsWith(
    $backupPrefix,
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw "Unsafe backup target: $targetRoot"
}

$activeStatusPath = Join-Path $projectRoot (
    'results\evaluations\v1_original_new_50000_vs_88966.status.json'
)
if (Test-Path -LiteralPath $activeStatusPath) {
    $activeStatus = Get-Content -Raw -Encoding utf8 $activeStatusPath |
        ConvertFrom-Json
    foreach ($activePath in @($activeStatus.source, $activeStatus.data)) {
        if ([string]::IsNullOrWhiteSpace([string]$activePath)) {
            continue
        }
        $activeName = Split-Path -Leaf ([string]$activePath)
        if ($names -contains $activeName) {
            throw "Refusing to archive active pipeline data: $activePath"
        }
    }
}

$requiredBytes = 0L
foreach ($name in $names) {
    $source = [System.IO.Path]::GetFullPath((Join-Path $sourceRoot $name))
    $destination = Join-Path $targetRoot $name
    if (-not $source.StartsWith(
        $sourcePrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Unsafe source: $source"
    }
    if (
        (Test-Path -LiteralPath $source) -and
        (Test-Path -LiteralPath $destination)
    ) {
        throw "Source and destination both exist: $name"
    }
    if (
        -not (Test-Path -LiteralPath $source) -and
        -not (Test-Path -LiteralPath $destination)
    ) {
        throw "Source and destination both missing: $name"
    }
    if (Test-Path -LiteralPath $source) {
        $requiredBytes += (
            Get-ChildItem -LiteralPath $source -File -Recurse |
                Measure-Object Length -Sum
        ).Sum
    }
}

$drive = [System.IO.DriveInfo]::new(
    [System.IO.Path]::GetPathRoot($targetRoot)
)
if (-not $drive.IsReady) {
    throw "Backup drive is not ready: $($drive.Name)"
}
if ($drive.AvailableFreeSpace -lt $requiredBytes) {
    throw (
        "Insufficient backup space: required=$requiredBytes " +
        "available=$($drive.AvailableFreeSpace)"
    )
}

New-Item -ItemType Directory -Path $targetRoot -Force | Out-Null
foreach ($name in $names) {
    $source = Join-Path $sourceRoot $name
    $destination = Join-Path $targetRoot $name
    if (Test-Path -LiteralPath $destination) {
        Write-Output "$(Get-Date -Format o) already archived $name"
        continue
    }
    Write-Output "$(Get-Date -Format o) moving $source -> $destination"
    Move-Item -LiteralPath $source -Destination $targetRoot
    if (
        (Test-Path -LiteralPath $source) -or
        -not (Test-Path -LiteralPath $destination)
    ) {
        throw "Move did not complete cleanly: $name"
    }
    Write-Output "$(Get-Date -Format o) completed $name"
}

Write-Output "$(Get-Date -Format o) archive complete: $targetRoot"
