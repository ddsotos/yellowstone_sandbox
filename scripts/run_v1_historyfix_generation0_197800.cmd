@echo off
setlocal
python -m yellowstone.convert_replay_v2_to_v1_historyfix ^
  --source data\v2_generation0_200k_frame_features ^
  --output data\v1_historyfix_generation0_197800_canonical ^
  --expected-games 197800
if errorlevel 1 exit /b %errorlevel%

python -m yellowstone.train_value_historyfix ^
  --data data\v1_historyfix_generation0_197800_canonical ^
  --checkpoint models\win_value_v1_historyfix_generation0_197800_epoch001.pt ^
  --epochs 1 ^
  --batch-size 256 ^
  --seed 20260727 ^
  --training-games 197800
exit /b %errorlevel%
