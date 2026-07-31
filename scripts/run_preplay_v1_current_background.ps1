param([string]$PythonExe = 'python')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot; Set-Location $root
$env:PYTHONUNBUFFERED='1'; $env:PYTHONDONTWRITEBYTECODE='1'
$data='data\privileged_state_exploratory_diverse_v1_20260730_tensors'
$manifestPath=Join-Path $data 'manifest.json'
$snapshotPath='results\evaluations\preplay_v1_current.snapshot.json'
$statusPath='results\evaluations\preplay_v1_current_training.status.json'
$selectionPath='results\evaluations\preplay_v1_current.selection.json'
$prefix='models\preplay_v1_current'
$stdoutPath='logs\preplay_v1_current_training.stdout.log'; $stderrPath='logs\preplay_v1_current_training.stderr.log'
$lastCompleted=''; New-Item -ItemType Directory -Force -Path 'models','results\evaluations','logs'|Out-Null
$PID|Set-Content -Encoding ascii 'logs\preplay_v1_current_training.pid'
function Write-AtomicJson { param([object]$Payload,[string]$Path); $tmp="$Path.$PID.tmp"; $Payload|ConvertTo-Json -Depth 12|Set-Content -Encoding utf8 $tmp; Move-Item -Force $tmp $Path }
function Status { param([string]$Step,[string]$State,[string]$Message=''); Write-AtomicJson $([ordered]@{state=$State;step=$Step;last_completed_step=$lastCompleted;message=$Message;updated_at=(Get-Date).ToString('o');pid=$PID;data=$data;snapshot=$snapshotPath;checkpoint_prefix=$prefix;selection=$selectionPath;stdout=$stdoutPath;stderr=$stderrPath}) $statusPath }
try {
    if(-not(Test-Path $snapshotPath)) {
        $manifest=Get-Content -Raw $manifestPath|ConvertFrom-Json
        $parts=@(Get-ChildItem $data -Filter 'part_*.npz'|ForEach-Object{[pscustomobject]@{number=[int]$_.BaseName.Substring(5);name=$_.Name}}|Sort-Object number)
        if($parts.Count -ne [int]$manifest.shards){throw 'preplay manifest and shard count differ'}
        Write-AtomicJson $([ordered]@{schema='yellowstone.value.privileged-state.v1.fixed_snapshot';created_at=(Get-Date).ToString('o');data=$data;games=[int]$manifest.games;records=[int]$manifest.records;shards=$parts.Count;start_part=[int]$parts[0].number;end_part=[int]$parts[-1].number;value_schema=$manifest.schema;history_semantics=$manifest.history_semantics;canonicalization=$manifest.canonicalization}) $snapshotPath
    }
    $snap=Get-Content -Raw $snapshotPath|ConvertFrom-Json; Status 'freeze_snapshot' 'running'
    if(-not(Test-Path $selectionPath)) {
        Status 'train_epochs' 'running'
        & $PythonExe -m yellowstone.train_privileged_state --data $data --checkpoint-prefix $prefix --epochs 2 --batch-size 256 --learning-rate 1e-3 --seed 20260727 --start-part "$($snap.start_part)" --end-part "$($snap.end_part)" --selection-output $selectionPath
        if($LASTEXITCODE -ne 0){throw 'preplay training failed'}
    } else {$lastCompleted='train_epochs'}
    $lastCompleted='train_epochs'; Status 'done' 'complete'
} catch { Status 'failed' 'failed' $_.Exception.Message; throw }
