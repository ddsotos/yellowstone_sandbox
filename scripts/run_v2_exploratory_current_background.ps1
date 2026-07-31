param([string]$PythonExe = 'python')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot; Set-Location $root
$env:PYTHONUNBUFFERED='1'; $env:PYTHONDONTWRITEBYTECODE='1'
$source='data\v2_exploratory_diverse_v1_20260730'
$manifestPath=Join-Path $source 'collection_manifest.json'
$data='data\v2_exploratory_current_tensors'
$snapshotPath='results\evaluations\v2_exploratory_current.snapshot.json'
$statusPath='results\evaluations\v2_exploratory_current_training.status.json'
$timingsPath='results\evaluations\v2_exploratory_current_training.timings.json'
$checkpoint='models\win_value_v2_exploratory_current_epoch001.pt'
$trainingOutput='results\evaluations\v2_exploratory_current.training.json'
$stdoutPath='logs\v2_exploratory_current_training.stdout.log'; $stderrPath='logs\v2_exploratory_current_training.stderr.log'
$lastCompleted=''; New-Item -ItemType Directory -Force -Path 'models','results\evaluations','logs',$data | Out-Null
$PID | Set-Content -Encoding ascii 'logs\v2_exploratory_current_training.pid'
function Write-AtomicJson { param([object]$Payload,[string]$Path); $tmp="$Path.$PID.tmp"; $Payload|ConvertTo-Json -Depth 12|Set-Content -Encoding utf8 $tmp; Move-Item -Force $tmp $Path }
function Status { param([string]$Step,[string]$State,[string]$Message=''); Write-AtomicJson $([ordered]@{state=$State;step=$Step;last_completed_step=$lastCompleted;message=$Message;updated_at=(Get-Date).ToString('o');pid=$PID;source=$source;data=$data;snapshot=$snapshotPath;checkpoint=$checkpoint;training_output=$trainingOutput;stdout=$stdoutPath;stderr=$stderrPath}) $statusPath }
function RunStep { param([string]$Name,[string[]]$Arguments); Status $Name 'running'; & $PythonExe @Arguments; if($LASTEXITCODE -ne 0){throw "$Name failed"}; $script:lastCompleted=$Name; Status $Name 'running' }
try {
    if(-not (Test-Path $snapshotPath)) {
        $cm=Get-Content -Raw $manifestPath|ConvertFrom-Json
        $parts=@(Get-ChildItem $source -Filter 'part_*.jsonl.gz'|ForEach-Object{[pscustomobject]@{number=[int](($_.BaseName -replace '^part_','') -replace '\.jsonl$','');name=$_.Name}}|Sort-Object number)
        if($parts.Count -ne [int]$cm.completed_shards){throw 'raw manifest and shard count differ'}
        Write-AtomicJson $([ordered]@{schema='yellowstone.value.v2-exploratory.fixed_snapshot';created_at=(Get-Date).ToString('o');source=$source;games=[int]$cm.games;shards=$parts.Count;start_part=[int]$parts[0].number;end_part=[int]$parts[-1].number;raw_updated=$cm.updated_at}) $snapshotPath
    }
    $snap=Get-Content -Raw $snapshotPath|ConvertFrom-Json; Status 'freeze_snapshot' 'running'
    if(-not (Test-Path (Join-Path $data 'manifest.json'))) { RunStep -Name 'convert' -Arguments @('-m','yellowstone.convert_replay_v2_exploratory','--source',$source,'--output',$data,'--start-part',"$($snap.start_part)",'--end-part',"$($snap.end_part)") } else {$lastCompleted='convert'}
    if(-not (Test-Path $checkpoint)) { RunStep -Name 'train_epoch001' -Arguments @('-m','yellowstone.train_value_v2_exploratory','--data',$data,'--checkpoint',$checkpoint,'--epochs','1','--batch-size','256','--learning-rate','1e-3','--seed','20260727') } else {$lastCompleted='train_epoch001'}
    $manifest=Get-Content -Raw (Join-Path $data 'manifest.json')|ConvertFrom-Json
    Write-AtomicJson $([ordered]@{schema='yellowstone.value.v2-exploratory.background_training';snapshot=$snap;tensor_manifest=$manifest;checkpoint=$checkpoint;training_summary=$trainingOutput;training_seed=20260727}) $trainingOutput
    $lastCompleted='summarize'; Status 'done' 'complete'
} catch { Status 'failed' 'failed' $_.Exception.Message; throw }
