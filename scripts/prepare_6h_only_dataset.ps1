param(
    [string]$SourceDir = 'data\heuristic_value_data',
    [string]$DestDir = 'data\heuristic_value_data_6h_only',
    [int]$StartPart = 660100,
    [int]$EndPart = 960000,
    [string]$ManifestPath = ''
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
$sourcePath = Join-Path $root $SourceDir
$destPath = Join-Path $root $DestDir

if (-not (Test-Path $sourcePath)) {
    throw "source directory not found: $sourcePath"
}

New-Item -ItemType Directory -Force -Path $destPath | Out-Null

$selected = Get-ChildItem -LiteralPath $sourcePath -Filter 'part_*.npz' | Where-Object {
    if ($_.BaseName -match '^part_(\d+)$') {
        $part = [int]$Matches[1]
        $part -ge $StartPart -and $part -le $EndPart
    }
    else {
        $false
    }
} | Sort-Object Name

foreach ($item in $selected) {
    $target = Join-Path $destPath $item.Name
    if (-not (Test-Path $target)) {
        New-Item -ItemType HardLink -Path $target -Value $item.FullName | Out-Null
    }
}

$manifest = if ($ManifestPath) {
    Join-Path $root $ManifestPath
}
else {
    "${destPath}_manifest.txt"
}
$selected.FullName | Set-Content -Path $manifest
Write-Output ("selected_files={0} manifest={1}" -f $selected.Count, $manifest)
