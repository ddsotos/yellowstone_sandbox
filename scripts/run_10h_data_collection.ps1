param(
    [int]$Hours = 10,
    [int]$GamesPerRound = 50000,
    [int]$ChunkGames = 100
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
$dataDirectory = Join-Path $root 'data\heuristic_value_data'
$logDirectory = Join-Path $root 'logs'
$log = Join-Path $logDirectory 'data_collection_10h.log'
$deadline = (Get-Date).AddHours($Hours)

try {
    New-Item -ItemType Directory -Force -Path $dataDirectory | Out-Null
    New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
    $parts = Get-ChildItem $dataDirectory -Filter 'part_*.npz' -ErrorAction SilentlyContinue
    $offset = if ($parts) {
        (($parts | ForEach-Object {
            [int]([System.IO.Path]::GetFileNameWithoutExtension($_.Name).Replace('part_', ''))
        } | Measure-Object -Maximum).Maximum + $ChunkGames)
    } else {
        0
    }
    $round = 1
    while ((Get-Date) -lt $deadline) {
        "$(Get-Date -Format o) round=$round game_offset=$offset games=$GamesPerRound" |
            Add-Content $log
        & python -m yellowstone.value_learning `
            --games $GamesPerRound `
            --chunk-games $ChunkGames `
            --seed 0 `
            --game-id-offset $offset `
            --output $dataDirectory 2>&1 | Tee-Object -FilePath $log -Append
        if ($LASTEXITCODE -ne 0) {
            throw "collection failed with exit code $LASTEXITCODE"
        }
        $offset += $GamesPerRound
        $round++
    }
    "$(Get-Date -Format o) completed rounds=$($round - 1)" | Add-Content $log
} catch {
    "$(Get-Date -Format o) failed: $_" | Add-Content $log
    exit 1
}
