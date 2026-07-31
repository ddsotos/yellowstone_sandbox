param(
    [string]$Destination = '..\online_bundle'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$destinationPath = [System.IO.Path]::GetFullPath((Join-Path $root $Destination))

function Copy-Tree {
    param(
        [string]$Source,
        [string]$Target
    )
    $sourcePath = [System.IO.Path]::GetFullPath($Source)
    New-Item -ItemType Directory -Force -Path $Target | Out-Null
    Get-ChildItem -LiteralPath $sourcePath -Recurse -File |
        Where-Object { $_.FullName -notmatch '[\\/](?:__pycache__|\.pytest_cache)[\\/]' } |
        ForEach-Object {
            $relative = $_.FullName.Substring($sourcePath.Length).TrimStart('\', '/')
            $targetFile = Join-Path $Target $relative
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $targetFile) |
                Out-Null
            Copy-Item -LiteralPath $_.FullName -Destination $targetFile -Force
        }
}

New-Item -ItemType Directory -Force -Path $destinationPath | Out-Null
Copy-Tree (Join-Path $root 'src\yellowstone') (Join-Path $destinationPath 'src\yellowstone')
Copy-Tree (Join-Path $root 'tests') (Join-Path $destinationPath 'tests')
Copy-Tree (Join-Path $root 'docs') (Join-Path $destinationPath 'docs')
Copy-Item -LiteralPath (Join-Path $root 'pyproject.toml') -Destination $destinationPath -Force
Copy-Item -LiteralPath (Join-Path $root 'docs\online-bundle-readme.md') `
    -Destination (Join-Path $destinationPath 'README.md') -Force

$modelCandidates = @(
    @{
        Path = 'models\win_value_canonical_old_plus_6h_plus_8h_001.pt'
        Canonical = $true
        Lineage = 'canonical old + 6h + 8h'
    },
    @{
        Path = 'models\win_value_canonical_old_plus_6h_001.pt'
        Canonical = $true
        Lineage = 'canonical old + 6h'
    },
    @{
        Path = 'models\win_value_canonical_old_001.pt'
        Canonical = $true
        Lineage = 'canonical old'
    },
    @{
        Path = 'models\win_value_6h_plus_8h_001.pt'
        Canonical = $false
        Lineage = 'legacy old + 6h + 8h'
    }
)
$selectedModel = $modelCandidates |
    Where-Object { Test-Path (Join-Path $root $_.Path) } |
    Select-Object -First 1
if ($null -eq $selectedModel) {
    throw 'no exportable win-value checkpoint found'
}

$modelDirectory = Join-Path $destinationPath 'models'
New-Item -ItemType Directory -Force -Path $modelDirectory | Out-Null
$sourceModel = Join-Path $root $selectedModel.Path
$targetModel = Join-Path $modelDirectory 'win_value.pt'
Copy-Item -LiteralPath $sourceModel -Destination $targetModel -Force

$metadata = [ordered]@{
    exported_at = (Get-Date -Format o)
    source_root = $root
    source_checkpoint = $selectedModel.Path
    bundled_checkpoint = 'models/win_value.pt'
    lineage = $selectedModel.Lineage
    canonical_input_model = $selectedModel.Canonical
    canonicalization = if ($selectedModel.Canonical) {
        'fast_lr_ud_color_v1'
    }
    else {
        $null
    }
    note = if ($selectedModel.Canonical) {
        'TorchWinValueEstimator canonicalizes candidate inputs automatically.'
    }
    else {
        'Temporary legacy model; rerun export_online_bundle.ps1 after canonical training completes.'
    }
}
$metadata | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath (Join-Path $destinationPath 'MODEL_STATUS.json') -Encoding utf8

$staleManifest = Join-Path $destinationPath 'BUNDLE_MANIFEST.sha256'
if (Test-Path $staleManifest) {
    Remove-Item -LiteralPath $staleManifest
}
$manifest = Join-Path $destinationPath 'MANIFEST.sha256'
Get-ChildItem -LiteralPath $destinationPath -Recurse -File |
    Where-Object {
        $_.FullName -ne $manifest `
        -and $_.Extension -ne '.pt' `
        -and $_.FullName -notmatch '[\\/]models[\\/]' `
        -and $_.FullName -notmatch '[\\/](?:__pycache__|\.pytest_cache)[\\/]'
    } |
    Sort-Object FullName |
    ForEach-Object {
        $relative = $_.FullName.Substring($destinationPath.Length).TrimStart('\', '/')
        $relative = $relative.Replace('\', '/')
        $hash = (
            Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256 |
                Select-Object -ExpandProperty Hash
        ).ToLowerInvariant()
        "$hash  $relative"
    } | Set-Content -LiteralPath $manifest -Encoding ascii

Write-Output "destination=$destinationPath"
Write-Output "model=$($selectedModel.Path)"
Write-Output "canonical=$($selectedModel.Canonical)"
