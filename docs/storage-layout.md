# Yellowstone実験データの配置ルール

この文書を、`rl_bundle`のデータ・モデル・結果・ログ配置の正本とする。

## Cドライブに置くもの

- `data/`
  - 現在収集中、変換中、学習中のデータ
  - 近い実験で再利用するraw replay
  - 現行モデル系統のtensor
- `models/`
  - 学習済みcheckpoint。容量が小さいため、旧比較モデルも原則として残す
- `results/`
  - 評価JSON、比較表、監査ケース、benchmark、smoke成果物
- `logs/`
  - stdout、stderr、PID、実行時間、再開用status

実行中プロセスのstatusに記載された`source`と`data`は移動しない。
移動前にはPIDの生存と最新ログを確認する。

## Dドライブに置くもの

`D:\codex-backup`は、再利用頻度が低い学習データの保管場所とする。

- `yellow_3_legacy_training_data_2026-07-27/`
  - 旧heuristic学習データとcanonical化済み増分
- `yellow_3_derived_training_data_2026-07-29/`
  - 旧generation0、historyfix、history3の変換済みtensor
  - 開発時のpreflight tensor

raw replayは再変換の根拠になるため、当面はCに残す。変換済みtensorは、
対応モデル・評価結果・変換manifestが揃い、直近の比較に使わない場合にDへ移す。

## 現行データ

- raw replay:
  - `data/v2_heuristic_one_vs_two_v1_10h_20260729/`
  - 88,966戦、source game ID `954346..1043311`
- Original V1 tensor:
  - `data/v1_original_new_88966_canonical/`
  - game IDを`0..88965`へrebase
  - `fast_lr_ud_color_v1`
  - rolling直近2配置

## 退避と復元

退避はプロジェクトルートから次を実行する。

```powershell
.\scripts\archive_derived_training_data_to_d.ps1
```

このスクリプトは固定リストだけを通常の`Move-Item`で移動する。SHA-256照合は
行わない。移動元と移動先が同時に存在する場合は停止し、Dにだけ存在する項目は
完了済みとしてスキップする。

旧実験を再実行する場合は、対象ディレクトリをDのarchiveから`data/`へ戻す。
歴史的スクリプトのC側固定パスを無理にDへ向けず、復元後に実行する。

## 新しい実験の命名

- 収集データ: `v2_<policy>_<duration>_<yyyymmdd>`
- 変換済みデータ: `v1_<semantics>_<population>_canonical`
- checkpoint: `win_value_<schema>_<population>_epochNNN.pt`
- 評価: checkpoint名に評価戦数、seed共有、席番号を加える
- 長時間処理: 同じstemで`.pid`、`.stdout.log`、`.stderr.log`、status JSONを残す

日付や戦数を名前に含め、`latest`だけを監査・再開の根拠にしない。
