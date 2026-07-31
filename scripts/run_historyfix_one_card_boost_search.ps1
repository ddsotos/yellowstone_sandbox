param(
    [int]$Games = 100,
    [int]$Seed = 20260725
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUNBUFFERED = '1'

python -m yellowstone.search_one_card_boost `
    --checkpoint models\win_value_v1_historyfix_generation0_197800_epoch002.pt `
    --games $Games `
    --seed $Seed `
    --player-index 0 `
    --output-directory results\evaluations\historyfix_v1_one_card_boost_screen `
    --comparison results\evaluations\historyfix_v1_one_card_boost_screen.json `
    --markdown results\evaluations\historyfix_v1_one_card_boost_screen.md `
    --status results\evaluations\historyfix_v1_one_card_boost_screen.status.json `
    --mode maximize-range `
    --range-min 10 `
    --range-max 60 `
    --coarse-step 10 `
    --resolution 2

if ($LASTEXITCODE -ne 0) {
    throw "Historyfix one-card boost search failed with exit code $LASTEXITCODE"
}
