param(
    [string]$BackupBase = 'D:\codex-backup',
    [string]$TargetName = 'yellow_3_legacy_training_data_2026-07-27'
)

$ErrorActionPreference = 'Stop'

$sourceRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot '..\data')
)
$backupRoot = [System.IO.Path]::GetFullPath($BackupBase)
$targetRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $backupRoot $TargetName)
)
$backupPrefix = $backupRoot.TrimEnd(
    [System.IO.Path]::DirectorySeparatorChar
) + [System.IO.Path]::DirectorySeparatorChar
$sourcePrefix = $sourceRoot.TrimEnd(
    [System.IO.Path]::DirectorySeparatorChar
) + [System.IO.Path]::DirectorySeparatorChar

if (-not $targetRoot.StartsWith(
    $backupPrefix,
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw "Unsafe backup target: $targetRoot"
}

$names = @(
    'heuristic_value_data',
    'heuristic_value_data_6h_only',
    'heuristic_value_data_8h_only',
    'heuristic_value_data_canonical_6h_only',
    'heuristic_value_data_canonical_8h_only',
    'heuristic_value_data_canonical_old'
)

New-Item -ItemType Directory -Path $targetRoot -Force | Out-Null

foreach ($name in $names) {
    $source = [System.IO.Path]::GetFullPath((Join-Path $sourceRoot $name))
    $destination = Join-Path $targetRoot $name

    if (-not $source.StartsWith(
        $sourcePrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Unsafe source: $source"
    }
    if (-not (Test-Path -LiteralPath $source -PathType Container)) {
        throw "Missing source: $source"
    }
    if (Test-Path -LiteralPath $destination) {
        throw "Destination already exists: $destination"
    }

    Write-Output "$(Get-Date -Format o) moving $source -> $destination"
    Move-Item -LiteralPath $source -Destination $targetRoot
    Write-Output "$(Get-Date -Format o) completed $name"
}
