param(
    [string]$Destination = '..\online_bundle_v2_preview'
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
        Where-Object {
            $_.FullName -notmatch '[\\/](?:__pycache__|\.pytest_cache)[\\/]'
        } |
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
Copy-Item -LiteralPath (Join-Path $root 'docs\online-bundle-v2-readme.md') `
    -Destination (Join-Path $destinationPath 'README.md') -Force
Copy-Item -LiteralPath (Join-Path $root 'docs\online-bundle-v2-diff.md') `
    -Destination (Join-Path $destinationPath 'CHANGES_FROM_PREVIOUS_BUNDLE.md') -Force

$evaluationDirectory = Join-Path $destinationPath 'evaluations'
New-Item -ItemType Directory -Force -Path $evaluationDirectory | Out-Null
$evaluationRelative = 'results\evaluations\v2_generation0_197800_1000_same_seed_p0_corrected.json'
$evaluationSource = Join-Path $root $evaluationRelative
$evaluationPresent = Test-Path -LiteralPath $evaluationSource
if ($evaluationPresent) {
    Copy-Item -LiteralPath $evaluationSource `
        -Destination (Join-Path $evaluationDirectory 'v2_generation0_197800_1000_same_seed_p0.json') `
        -Force
}

$modelDirectory = Join-Path $destinationPath 'models'
New-Item -ItemType Directory -Force -Path $modelDirectory | Out-Null
$v2Relative = 'models\win_value_v2_generation0_197800_epoch001.pt'
$v2Source = Join-Path $root $v2Relative
$v2Present = Test-Path -LiteralPath $v2Source
if ($v2Present) {
    Copy-Item -LiteralPath $v2Source `
        -Destination (Join-Path $modelDirectory 'win_value_v2.pt') -Force
}

$legacyRelative = 'models\win_value_canonical_old_plus_6h_001.pt'
$legacySource = Join-Path $root $legacyRelative
if (Test-Path -LiteralPath $legacySource) {
    Copy-Item -LiteralPath $legacySource `
        -Destination (Join-Path $modelDirectory 'legacy_canonical_old_plus_6h.pt') `
        -Force
    Copy-Item -LiteralPath $legacySource `
        -Destination (Join-Path $modelDirectory 'win_value.pt') -Force
}

$metadata = [ordered]@{
    exported_at = (Get-Date -Format o)
    bundle_kind = 'yellowstone-v2-developer-preview'
    replaces_previous_bundle = $false
    v2_checkpoint_present = $v2Present
    v2_checkpoint_source = if ($v2Present) { $v2Relative } else { $null }
    v2_checkpoint_status = if ($v2Present) {
        'evaluated_below_legacy_baseline'
    } else {
        'training'
    }
    v2_value_schema = 'yellowstone.value.v2'
    v2_context_size = 300
    v2_canonicalization = 'strict_residual_v2'
    runtime_rules_version = 'yellowstone-python-2026-07-27-empty-deck-settlement'
    training_data_rules_version = 'yellowstone-python-2026-07-26'
    v2_training_games = 197800
    drop_in_compatible_with_v1 = $false
    practical_evaluation_complete = $evaluationPresent
    practical_evaluation = if ($evaluationPresent) {
        [ordered]@{
            games = 1000
            seed = 20260725
            player_index = 0
            v2_win_rate = 0.2603333333333333
            canonical_old_win_rate = 0.2823333333333333
            canonical_old_plus_6h_win_rate = 0.2908333333333334
            source = $evaluationRelative
        }
    } else {
        $null
    }
    legacy_comparison_checkpoint = $legacyRelative
    legacy_compatibility_alias = 'models/win_value.pt'
    legacy_alias_is_v2 = $false
    note = if ($v2Present) {
        'V2 checkpoint is included, but its corrected seat-0 evaluation underperformed the legacy baseline. Keep it as a preview.'
    }
    else {
        'V2 checkpoint is still training. This export contains code, tests, design, and a legacy comparison model.'
    }
}
$metadata | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath (Join-Path $destinationPath 'MODEL_STATUS.json') -Encoding utf8

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
Write-Output "v2_checkpoint_present=$v2Present"
