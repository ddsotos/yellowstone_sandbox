param(
    [string]$Checkpoint = 'models\win_value_v1_historyfix_generation0_197800_epoch001.pt',
    [int]$Games = 1000,
    [int]$Seed = 20260725
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
$checkpointPath = Join-Path $root $Checkpoint
$resultDirectory = Join-Path $root 'results\evaluations'
New-Item -ItemType Directory -Force -Path $resultDirectory | Out-Null

while (-not (Test-Path -LiteralPath $checkpointPath)) {
    Start-Sleep -Seconds 30
}

# Avoid opening a checkpoint while the producer is still closing the file.
Start-Sleep -Seconds 3

$results = @()
foreach ($playerIndex in 0..3) {
    $name = "v1_historyfix_generation0_197800_1000_same_seed_p${playerIndex}.json"
    $output = Join-Path $resultDirectory $name
    if (-not (Test-Path -LiteralPath $output)) {
        & python -m yellowstone.evaluate_value `
            --checkpoint $checkpointPath `
            --games $Games `
            --seed $Seed `
            --player-index $playerIndex `
            --adaptive-pq-pruning `
            --approximate-new-color-neighbors `
            --current-turn-history-only `
            --output $output
        if ($LASTEXITCODE -ne 0) {
            throw "seat evaluation failed: player_index=$playerIndex"
        }
    }
    $payload = Get-Content -Raw -Encoding utf8 $output | ConvertFrom-Json
    $results += [ordered]@{
        player_index = $playerIndex
        games = [int]$payload.games
        wins = [double]$payload.wins
        win_rate = [double]$payload.win_rate
        result = "results/evaluations/$name"
    }
}

$summary = [ordered]@{
    checkpoint = $Checkpoint
    value_schema = 'yellowstone.value.v1_historyfix'
    history_semantics = 'evaluated_turn_only_one_card_zero_padded'
    games_per_seat = $Games
    seed = $Seed
    adaptive_pq_pruning = $true
    approximate_new_color_neighbors = $true
    seats = $results
}
$summaryPath = Join-Path $resultDirectory `
    'v1_historyfix_generation0_197800_1000_same_seed_all_seats.json'
$summary | ConvertTo-Json -Depth 5 |
    Set-Content -LiteralPath $summaryPath -Encoding utf8
Write-Output ($summary | ConvertTo-Json -Depth 5)
