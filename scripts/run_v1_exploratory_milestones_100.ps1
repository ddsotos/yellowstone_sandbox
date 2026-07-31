param(
    [int]$Games = 100,
    [int]$PlayerIndex = 0,
    [int]$EvaluationSeed = 20260725,
    [int]$TrainingSeed = 20260727,
    [string]$PythonExe = 'python'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'

$data = 'data\v1_original_exploratory_diverse_v1_20260730_canonical'
$conversionManifest = Join-Path $data 'conversion_manifest.json'
$snapshotPath = (
    'results\evaluations\v1_exploratory_current_milestones.snapshot.json'
)
$statusPath = (
    'results\evaluations\v1_exploratory_current_milestones.status.json'
)
$timingsPath = (
    'results\evaluations\v1_exploratory_current_milestones.timings.json'
)
$stdoutPath = 'logs\v1_exploratory_current_milestones.stdout.log'
$stderrPath = 'logs\v1_exploratory_current_milestones.stderr.log'
$pidPath = 'logs\v1_exploratory_current_milestones.pid'
$lastCompleted = ''
$timings = [ordered]@{}

New-Item -ItemType Directory -Force -Path (
    'models', 'results\evaluations', 'logs'
) | Out-Null
$PID | Set-Content -Encoding ascii -LiteralPath $pidPath

function Write-AtomicJson {
    param([object]$Payload, [string]$Path)
    $temporary = "$Path.$PID.tmp"
    $Payload | ConvertTo-Json -Depth 12 |
        Set-Content -Encoding utf8 -LiteralPath $temporary
    Move-Item -Force -LiteralPath $temporary -Destination $Path
}

function Write-Status {
    param(
        [string]$Step,
        [string]$State,
        [string]$Message = ''
    )
    Write-AtomicJson -Path $statusPath -Payload ([ordered]@{
        state = $State
        step = $Step
        last_completed_step = $lastCompleted
        message = $Message
        updated_at = (Get-Date).ToString('o')
        pid = $PID
        data = $data
        snapshot = $snapshotPath
        checkpoint_prefix = $script:checkpointPrefix
        training_summary = $script:trainingSummary
        comparison = $script:comparisonPath
        games = $Games
        player_index = $PlayerIndex
        evaluation_seed = $EvaluationSeed
        training_seed = $TrainingSeed
        stdout = $stdoutPath
        stderr = $stderrPath
    })
}

function Invoke-Step {
    param([string]$Name, [string[]]$Arguments)
    Write-Status -Step $Name -State 'running'
    $watch = [Diagnostics.Stopwatch]::StartNew()
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    $watch.Stop()
    $timings[$Name] = $watch.Elapsed.TotalSeconds
    Write-AtomicJson -Path $timingsPath -Payload $timings
    $script:lastCompleted = $Name
    Write-Status -Step $Name -State 'running'
}

try {
    if (-not (Test-Path -LiteralPath $snapshotPath)) {
        $manifest = (
            Get-Content -Raw -Encoding utf8 $conversionManifest |
                ConvertFrom-Json
        )
        $parts = @(
            Get-ChildItem -LiteralPath $data -Filter 'part_*.npz' |
                ForEach-Object {
                    [pscustomobject]@{
                        number = [int](
                            $_.BaseName.Substring('part_'.Length)
                        )
                        name = $_.Name
                    }
                } |
                Sort-Object number
        )
        if (-not $parts.Count) {
            throw 'no V1 tensor shards are available'
        }
        if ([int]$manifest.converted_shards -ne $parts.Count) {
            throw 'conversion manifest and tensor shard count differ'
        }
        Write-AtomicJson -Path $snapshotPath -Payload ([ordered]@{
            schema = 'yellowstone.value.v1.fixed_training_snapshot'
            created_at = (Get-Date).ToString('o')
            data = $data
            games = [int]$manifest.games
            records = [int64]$manifest.records
            shards = $parts.Count
            start_part = [int]$parts[0].number
            end_part = [int]$parts[-1].number
            source_game_id_min = [int]$manifest.source_game_id_min
            source_game_id_max = [int]$manifest.source_game_id_max
            value_schema = [string]$manifest.schema
            history_semantics = [string]$manifest.history_semantics
            input_canonicalization = [string]$manifest.canonicalization
        })
    }
    $snapshot = (
        Get-Content -Raw -Encoding utf8 $snapshotPath |
            ConvertFrom-Json
    )
    $snapshotGames = [int]$snapshot.games
    $script:checkpointPrefix = (
        "models\win_value_v1_exploratory_${snapshotGames}_epoch001"
    )
    $progressCheckpoint = "$checkpointPrefix.progress.pt"
    $script:trainingSummary = (
        "results\evaluations\v1_exploratory_${snapshotGames}_" +
        'milestones.training.json'
    )
    $script:comparisonPath = (
        "results\evaluations\v1_exploratory_${snapshotGames}_" +
        "milestones_${Games}_seed${EvaluationSeed}_p${PlayerIndex}.json"
    )
    $markdownPath = [IO.Path]::ChangeExtension($comparisonPath, '.md')
    $lastCompleted = 'freeze_snapshot'
    Write-Status -Step 'freeze_snapshot' -State 'running'

    if (-not (Test-Path -LiteralPath $trainingSummary)) {
        Invoke-Step 'train_milestones' @(
            '-m', 'yellowstone.train_value_milestones',
            '--data', $data,
            '--checkpoint-prefix', $checkpointPrefix,
            '--progress-checkpoint', $progressCheckpoint,
            '--split-game-count', "$snapshotGames",
            '--start-part', "$($snapshot.start_part)",
            '--end-part', "$($snapshot.end_part)",
            '--milestones', '10,30,50,100',
            '--batch-size', '256',
            '--learning-rate', '1e-3',
            '--seed', "$TrainingSeed",
            '--output', $trainingSummary
        )
    } else {
        $lastCompleted = 'train_milestones'
    }

    foreach ($percent in @(10, 30, 50, 100)) {
        $tag = '{0:d3}' -f $percent
        $checkpoint = "${checkpointPrefix}_pct${tag}.pt"
        $evaluation = (
            "results\evaluations\v1_exploratory_${snapshotGames}_" +
            "pct${tag}_${Games}_seed${EvaluationSeed}_" +
            "p${PlayerIndex}.json"
        )
        if (-not (Test-Path -LiteralPath $evaluation)) {
            Invoke-Step "evaluate_pct${tag}" @(
                '-m', 'yellowstone.evaluate_value',
                '--checkpoint', $checkpoint,
                '--games', "$Games",
                '--seed', "$EvaluationSeed",
                '--player-index', "$PlayerIndex",
                '--adaptive-pq-pruning',
                '--approximate-new-color-neighbors',
                '--output', $evaluation
            )
        } else {
            $lastCompleted = "evaluate_pct${tag}"
        }
    }

    Write-Status -Step 'summarize' -State 'running'
    $training = (
        Get-Content -Raw -Encoding utf8 $trainingSummary |
            ConvertFrom-Json
    )
    $rows = @(
        foreach ($milestone in $training.milestones) {
            $tag = '{0:d3}' -f [int]$milestone.percent
            $evaluationPath = (
                "results\evaluations\v1_exploratory_${snapshotGames}_" +
                "pct${tag}_${Games}_seed${EvaluationSeed}_" +
                "p${PlayerIndex}.json"
            )
            $evaluation = (
                Get-Content -Raw -Encoding utf8 $evaluationPath |
                    ConvertFrom-Json
            )
            [ordered]@{
                percent = [int]$milestone.percent
                checkpoint = [string]$milestone.checkpoint
                processed_train_records = (
                    [int64]$milestone.processed_train_records
                )
                actual_fraction = [double]$milestone.actual_fraction
                metrics = $milestone.metrics
                evaluation_path = $evaluationPath
                wins = [double]$evaluation.wins
                win_rate = [double]$evaluation.win_rate
                one_card_turns = (
                    [int]$evaluation.evaluated_player_one_card_turns
                )
                two_card_turns = (
                    [int]$evaluation.evaluated_player_two_card_turns
                )
                one_card_turn_rate = (
                    [double]$evaluation.evaluated_player_one_card_turn_rate
                )
                elapsed_seconds = [double]$evaluation.elapsed_seconds
                policy_fingerprint = (
                    [string]$evaluation.policy_fingerprint
                )
            }
        }
    )
    $comparison = [ordered]@{
        schema = 'yellowstone.value.v1.milestone_screen'
        official_four_seat_evaluation = $false
        note = (
            'Seat-0 100-game training-progress screen; ' +
            'not an official four-seat model evaluation.'
        )
        snapshot = $snapshot
        training_seed = $TrainingSeed
        evaluation_seed = $EvaluationSeed
        games = $Games
        player_index = $PlayerIndex
        pruning = 'adaptive_pq'
        approximate_new_color_neighbors = $true
        milestones = $rows
        timings = $timings
    }
    Write-AtomicJson -Path $comparisonPath -Payload $comparison
    $markdown = @(
        '# V1 exploratory continuous-training milestone screen'
        ''
        (
            "Snapshot: $snapshotGames games / $($snapshot.records) " +
            'records. Seat 0, 100 games; not an official four-seat result.'
        )
        ''
        '| Progress | Win rate | One-card rate | Brier (test) | Logloss (test) |'
        '|---:|---:|---:|---:|---:|'
        foreach ($row in $rows) {
            (
                "| $($row.percent)% | " +
                ('{0:P3}' -f $row.win_rate) + ' | ' +
                ('{0:P3}' -f $row.one_card_turn_rate) + ' | ' +
                ('{0:F6}' -f $row.metrics.test_brier) + ' | ' +
                ('{0:F6}' -f $row.metrics.test_log_loss) + ' |'
            )
        }
    )
    $markdown -join "`n" |
        Set-Content -Encoding utf8 -LiteralPath $markdownPath
    $lastCompleted = 'summarize'
    Write-Status -Step 'done' -State 'complete'
} catch {
    Write-Status -Step 'failed' -State 'failed' `
        -Message $_.Exception.Message
    throw
}
