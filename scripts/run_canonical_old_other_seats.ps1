param(
    [int]$Games = 1000,
    [int]$Seed = 20260725
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
$checkpoint = 'models\win_value_canonical_old_001.pt'
$resultDirectory = Join-Path $root 'results\evaluations'
$log = Join-Path $root 'logs\canonical_old_other_seats.log'

try {
    New-Item -ItemType Directory -Force -Path $resultDirectory | Out-Null
    foreach ($playerIndex in 1..3) {
        $output = Join-Path $resultDirectory (
            "canonical_old_${Games}_same_seed_p${playerIndex}.json"
        )
        if (Test-Path $output) {
            "$(Get-Date -Format o) skip existing player_index=$playerIndex output=$output" |
                Add-Content $log
            continue
        }
        "$(Get-Date -Format o) start player_index=$playerIndex games=$Games seed=$Seed" |
            Add-Content $log
        & python -m yellowstone.evaluate_value `
            --checkpoint $checkpoint `
            --games $Games `
            --seed $Seed `
            --player-index $playerIndex `
            --adaptive-pq-pruning `
            --approximate-new-color-neighbors `
            --output $output 2>&1 |
            Tee-Object -FilePath $log -Append | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "evaluation failed for player_index=$playerIndex with exit code $LASTEXITCODE"
        }
        $result = Get-Content $output -Raw | ConvertFrom-Json
        "$(Get-Date -Format o) completed player_index=$playerIndex wins=$($result.wins) win_rate=$($result.win_rate)" |
            Add-Content $log
    }

    $summary = [ordered]@{
        checkpoint = $checkpoint
        games_per_seat = $Games
        seed = $Seed
        seats = [ordered]@{}
    }
    foreach ($playerIndex in 0..3) {
        $output = Join-Path $resultDirectory (
            "canonical_old_${Games}_same_seed_p${playerIndex}.json"
        )
        if (Test-Path $output) {
            $result = Get-Content $output -Raw | ConvertFrom-Json
            $summary.seats["$playerIndex"] = [ordered]@{
                wins = $result.wins
                win_rate = $result.win_rate
            }
        }
    }
    $summary | ConvertTo-Json -Depth 5 |
        Set-Content (Join-Path $resultDirectory 'canonical_old_1000_same_seed_all_seats.json') `
            -Encoding utf8
    "$(Get-Date -Format o) all completed" | Add-Content $log
}
catch {
    "$(Get-Date -Format o) failed: $_" | Add-Content $log
    exit 1
}
