$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = 'src'
$name = 'v2_heuristic_safe_counts_rank_color_20260801'
$statusPath = "results\collections\$name.status.json"
$stopFile = "results\collections\$name.stop"
$output = "data\$name"
$maxGames = if ($env:HEURISTIC_SAFE_COUNTS_MAX_GAMES) { [int]$env:HEURISTIC_SAFE_COUNTS_MAX_GAMES } else { 0 }
New-Item -ItemType Directory -Force -Path 'logs','results\collections' | Out-Null
Set-Content -Encoding ascii -LiteralPath "logs\$name.pid" -Value $PID
if (Test-Path -LiteralPath $stopFile) {
    Remove-Item -LiteralPath $stopFile -Force
}
$arguments = @(
    '-m','yellowstone.collect_heuristic_safe_counts_v2',
    '--seed','20260730',
    '--game-id-offset','1500000',
    '--output',$output,
    '--stop-file',$stopFile,
    '--status-file',$statusPath,
    '--shard-games','100'
)
if ($maxGames -gt 0) {
    $arguments += @('--max-games', [string]$maxGames)
}
& python @arguments
