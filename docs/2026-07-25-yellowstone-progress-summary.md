# 2026-07-25 Yellowstone progress summary

> この文書は2026-07-25時点の履歴。完了後のgeneration0、履歴比較、学習量曲線、
> 全席評価、新88,966戦実験の現状は
> [`2026-07-29-results-summary.md`](2026-07-29-results-summary.md)を参照。

## 目的

- `yellowstone` で、公開情報 + 自分視点情報から勝率を推定するモデルを使い、
  各ターンで勝率最大の手を選ぶ
- まずは heuristic bot 対戦から学習データを集め、実戦勝率を上げる

## ここまでの実装

### 学習データ

- heuristic 同士の対戦から value 学習データを収集
- 全プレイヤー視点のデータを使用
- プレイヤーごとの入力は公開情報 + 自分の手札
- 直近の公開行動履歴も特徴量に追加
- データ保存先:
  - `data/heuristic_value_data`

### モデル

- CNN + context ベースの勝率推定モデルを実装
- 学習/推論コードを追加
- 候補手を列挙して、勝率最大の手を選ぶ value policy を実装

### 評価

- value player vs heuristic 3人 の評価系を実装
- seed 固定で比較できる CLI を実装
- 4番手評価用に `--player-index` を追加

## データ収集状況

- 10時間データ収集を完了
- 6600 chunk
- 約 660,000 ゲーム分

## 学習状況

### 旧ベース

- `models/win_value.pt`
- `models/win_value_40.pt`
- `models/win_value_660k_1epoch.pt`

### 2026-07-25 の追加学習

6時間・1epochごと保存で再学習:

- `models/win_value_6h_epoch_001.pt`
- `models/win_value_6h_epoch_002.pt`
- `models/win_value_6h_epoch_003.pt`

指標推移:

- epoch 001
  - validation_brier `0.128690`
  - test_brier `0.128571`
  - test_logloss `0.414036`
- epoch 002
  - validation_brier `0.128285`
  - test_brier `0.128141`
  - test_logloss `0.412732`
- epoch 003
  - validation_brier `0.128133`
  - test_brier `0.127976`
  - test_logloss `0.412176`

結論:

- 再学習で少しずつ改善

## 実戦評価の主な結果

### player0, same seed 1000戦

- heuristic: `22.95%`
- old (`models/win_value_660k_1epoch.pt`): `25.9833%`
- new (`models/win_value_6h_epoch_002.pt`): `27.00%`

差分:

- old - heuristic: `+3.0333pt`
- new - heuristic: `+4.05pt`
- new - old: `+1.0167pt`

### 4番手 heuristic, same seed 1000戦

- heuristic (`player_index=3`): `26.0333%`

### 4番手 old/new

- `player_index=3` の old/new 1000戦比較ジョブをバックグラウンドで実行中
- 出力予定:
  - `results/evaluations/compare_old_1000_same_seed_p3.json`
  - `results/evaluations/compare_new_1000_same_seed_p3.json`

## 枝刈りと高速化

### 失点ベース枝刈り

- 0失点2枚手が存在する時だけ有効化
- adaptive p/q:
  - `negative + loss >= 10` → `increase <= 4`
  - それ未満 → `increase <= 8`

### 列縛り近似

- 未出色だけ近傍列制限
- 2枚目は1枚目後の盤面で再計算（B方式）
- 既定は OFF

### exact 高速化

- `legal_actions` 依存を減らし、value 探索専用の exact 列挙を追加
- board の列情報・色列情報を再利用
- 候補集合が `legal_actions` と一致するテストを追加

## 高速化の測定結果

### 枝刈りなし vs 枝刈りあり

5戦:

- 枝刈りなし: `33.478s`
- 枝刈りあり: `24.370s`
- speedup: `1.374x`

### 列縛りだけ vs 列縛り + 枝刈り

5戦:

- 列縛りだけ: `18.122s`
- 列縛り + 枝刈り: `14.773s`
- speedup: `1.227x`

### adaptive p/q exact vs 近似B

5戦:

- exact: `25.655s`
- approx B: `14.794s`
- speedup: `1.734x`

その後の current 実装でも:

- exact: `24.734s`
- approx B: `14.711s`
- speedup: `1.681x`

## 勝率差の確認

### 近似Bの影響

20戦:

- exact adaptive p/q: `30.0%`
- approx B adaptive p/q: `30.0%`
- 差: `0.0pt`

### 列縛りだけ vs 列縛り + 枝刈り

20戦:

- 列縛りだけ: `30.0%`
- 列縛り + 枝刈り: `30.0%`
- 差: `0.0pt`

注意:

- 戦数が少ないので断定は不可
- ただし短期検証では大きな悪化は見えていない

## 監査・保存ファイル

重要:

- `results/audits/pruning_audit_heuristic_100/top10_cases.md`
  - 重要ケースの参照元

その他:

- `results/audits/adaptive_pq_1h/adaptive_pq_1h.json`
- `results/audits/adaptive_pq_1h/adaptive_pq_1h_q_top10.md`
- `docs/value-search-speedup-candidates.md`
- `../AGENTS.md`

## 現在の方針

- 補充時の「失点から引くか」の学習は後回し
- 今は従来どおりの行動空間で勝率向上に集中
- 次の有力施策:
  - 学習データをさらに追加
  - 現在の実装（列縛り + 枝刈り）で継続評価

## 直近の次アクション

1. 4番手 old/new 1000戦の完了確認
2. heuristic データ追加収集
3. 必要なら追加学習
4. 新旧モデル比較を継続
