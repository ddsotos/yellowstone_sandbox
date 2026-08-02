$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = 'src'
$name = 'v2_variant_board5_hand6_oneoff_tiered_heuristic4_20260802'
$statusPath = "results\collections\$name.status.json"
$stopFile = "results\collections\$name.stop"
$output = "data\$name"
$maxGames = if ($env:VARIANT_HEURISTIC4_MAX_GAMES) { [int]$env:VARIANT_HEURISTIC4_MAX_GAMES } else { 100000 }
New-Item -ItemType Directory -Force -Path 'logs','results\collections' | Out-Null
Set-Content -Encoding ascii -LiteralPath "logs\$name.pid" -Value $PID
if (Test-Path -LiteralPath $stopFile) {
    Remove-Item -LiteralPath $stopFile -Force
}
$arguments = @(
    '-m','yellowstone.collect_heuristic_safe_counts_v2',
    '--policy','fixed_frame_hand_six_one_off_min_loss_one_card',
    '--seed','20260802',
    '--game-id-offset','3000000',
    '--output',$output,
    '--stop-file',$stopFile,
    '--status-file',$statusPath,
    '--shard-games','100',
    '--max-games',[string]$maxGames
)
& python @arguments
