# Safe-count Teacher → Action Delta/Q V2（保留）

Status: deferred at user request on 2026-07-30. No implementation or pipeline
was started from this plan.

## Summary

- generation0 197,800戦で、各プレイヤーの「マイナスカード増加0で
  1枚出しできる手札枚数」を追加したprivileged teacherを再学習する。
- 改善したteacherを使い、現行近似後に残る全候補と全合法補充方法について、
  行動・補充直後の`Q(s,a)`と`delta = Q - S`を同時収集する。
- 山札・マイナス札補充は候補間で共通の4乱数系列を使い、4結果の平均を
  1レコードへ保存する。
- 初回は10時間収集し、deltaを優先学習する。QラベルとQ学習経路も残す。

## Teacher V2

- safe cardは「その物理手札カードについて、厳密合法配置のどれかで
  1枚出し終了したとき、マイナスカード枚数が増えない」と定義する。
- 4プレイヤーのsafe-card countを現在プレイヤー相対順でcontextへ追加し、
  `count / 6`で正規化する。
- schema/canonicalizationをV2へ更新し、旧checkpointとの混用をhard failする。
- generation0の同一game-ID split、最終勝敗・同着按分ラベル、2epoch、
  batch 256、LR `1e-3`を維持する。
- 全体とsafe count 0のsliceでBrier/logloss/calibrationを比較する。

## Action Delta/Q data

- 現行近似列制限と条件付きadaptive p/q後に残る全1枚・2枚候補を扱い、
  V1上位3＋5選抜は使わない。
- 合法な`deck`、`none`、`negative_cards`を別候補として扱い、補充種別を
  Action入力へ追加する。サンプルされたカード自体は入力しない。
- 評価地点は行動・補充直後とし、相手heuristicの一巡は行わない。
- 各候補に`before_value`、`q_target`、`delta_target`、Q標本標準偏差、
  標本数、game/turn ID、1/2枚、補充種別、候補数を保存する。
- 新88,966戦と同方策のcontinuation replayを重複監査後に統合し、
  seed `20260727`、10時間、再開可能なbackground収集とする。

## Training and verification

- trainerへ`--target-kind delta|q`を追加する。初回の長時間学習はdelta、
  Qはsmokeのみ行う。
- 各ターンの全候補合計weightを1にし、候補数の多い局面の過剰加重を防ぐ。
- 10/30/50/100% checkpointを保存し、MAE/RMSE、turn内top-1一致率、
  pairwise順位一致率を記録する。
- metadata、補充4標本、共通乱数、Q/delta関係、同着按分、game-ID split、
  restart/resumeをテストしてからbackground pipelineを開始する。

この計画は、新しい探索方策によるreplay収集を先に検討・実行するため保留する。
