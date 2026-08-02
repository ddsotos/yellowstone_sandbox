$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$name = 'v2_exploratory_safe_counts_oneoff_20260731'
$statusPath = "results\collections\$name.status.json"
$stopFile = "results\collections\$name.stop"
$output = "data\$name"
$checkpoint = 'models\win_value_v1_original_generation0_197800_epoch002.pt'
$maxGames = if ($env:EXPLORATORY_MAX_GAMES) { [int]$env:EXPLORATORY_MAX_GAMES } else { 0 }
New-Item -ItemType Directory -Force -Path 'logs','results\collections' | Out-Null
Set-Content -Encoding ascii -LiteralPath "logs\$name.pid" -Value $PID
$arguments = @('-m','yellowstone.exploratory_collection','--checkpoint',$checkpoint,'--seed','20260730','--game-id-offset','1301600','--output',$output,'--stop-file',$stopFile,'--status-file',$statusPath,'--shard-games','100','--lazy-single-pass')
if ($maxGames -gt 0) { $arguments += @('--max-games', [string]$maxGames) }
& python @arguments
