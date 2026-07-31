# 2026-07-25 Yellowstone 勝率比較メモ

> この表は初期モデル系統の履歴。Original V1 generation0、legacy履歴比較、
> 全席合算を含む現行サマリーは
> [`2026-07-29-results-summary.md`](2026-07-29-results-summary.md)を参照。

このメモの「勝率」は、指定した `player_index` の 1 体だけを対象にした 1000 戦評価の値です。
対戦形式は `player_index` の bot vs 残り 3 体の heuristic bot です。
seed はすべて `20260725` です。

## モデル定義

| 表記 | 正確な意味 |
| --- | --- |
| heuristic | ルールベースの heuristic bot。学習モデルではない。 |
| old | `models/win_value_660k_1epoch.pt`。heuristic self-play の約 66 万ゲーム相当のデータで 1 epoch 学習した value model。 |
| new | 6 時間分の追加収集データで再学習した value model 系。過去の比較ログでは `models/win_value_6h_epoch_002.pt` と `models/win_value_6h_epoch_003.pt` が混在しているため、この表では「6h 追加学習系モデル」として扱う。 |
| 6h_only | `models/win_value_6h_only_001.pt`。`models/win_value_660k_1epoch.pt` を `--resume` し、`data/heuristic_value_data_6h_only` のみで 1 epoch 追加学習した checkpoint。ゼロから学習したモデルではない。 |

## 勝率比較表

| player_index | heuristic | old | new | 6h_only |
| --- | ---: | ---: | ---: | ---: |
| 0 | 22.95% | 25.98% | 27.00% | 25.50% |
| 1 | 24.93% | 26.03% | 25.53% | — |
| 2 | 26.08% | 26.00% | 24.95% | 25.85% |
| 3 | 26.03% | 27.67% | 26.07% | 26.45% |

## 補足

- `new` は過去ログで checkpoint 名が揺れているため、このメモでは「6h 追加学習系の value model」という意味で統一している。
- `6h_only` は「6時間分のデータだけでゼロから学習」ではなく、「`old` から継続学習したモデル」。
- `player_index = 1` の `6h_only` はこの時点では未評価。
