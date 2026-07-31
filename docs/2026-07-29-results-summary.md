# 2026-07-29 Yellowstone学習・勝率評価サマリー

この文書は、2026-07-29までに保存された評価JSONから主要結果を再集計した
現行サマリーである。推定勝率ではなく、特記がない限り「学習bot 1体 vs
heuristic bot 3体」の実戦評価勝率を記載する。同着はwinner数で均等按分する。

## 共通評価条件

- evaluation seed: `20260725`
- 通常は各席1,000戦
- adaptive p/q枝刈り: 有効
- 新色近傍列の近似制限: 有効
- 補充: 従来方式
- 席別結果は同じseedでも独立した1,000戦評価であり、席による差がある

## generation0でのモデル・履歴比較

学習母集団は197,800戦、batch 256、learning rate `1e-3`。表の勝率は席0の
1,000戦評価。

| モデル | epoch | test Brier | test logloss | 席0勝率 | 学習時間 |
|---|---:|---:|---:|---:|---:|
| V2 | 1 | 0.130029 | 0.418697 | 26.033% | 39.0分 |
| V2 | 2 | 0.129081 | 0.416270 | 24.283% | 56.4分 |
| V1履歴修正版 | 1 | 0.129943 | 0.417332 | 18.100% | 31.8分 |
| V1履歴修正版 | 2 | 0.129953 | 0.417150 | 16.233% | 65.6分 |
| Original V1 | 1 | 0.129868 | 0.417366 | 24.550% | 40.4分 |
| Original V1 | 2 | 0.129829 | 0.416873 | **27.667%** | 57.8分 |

確定した読み取り:

- 席0ではOriginal V1 2epochが最良。
- V2 2epochはこの比較でtest Brier/loglossが最良だが、実戦勝率はV2
  1epochより低い。calibration指標だけでは採用モデルを決められない。
- evaluated turnだけを残すV1履歴修正版は、Original V1より大幅に低い。
  rolling直近2配置を維持する。

出典:

- `results/evaluations/v2_v1_original_history_comparison.json`
- `results/evaluations/v2_v1_original_history_comparison.md`

## Original V1 generation0 2epochの全席評価

| 席 | 勝率 |
|---:|---:|
| 0 | 27.667% |
| 1 | 23.283% |
| 2 | 22.750% |
| 3 | 24.883% |
| 全席合算 | **24.646%** |

席0だけでは27.667%だが、全席合算は24.646%。今後のモデル比較は席0だけで
結論を出さず、4席合算を主要指標にする。

出典:

- `results/evaluations/v1_original_generation0_197800_epoch002_1000_same_seed_all_seats.json`

### 履歴意味を一致させた全席比較

generation0 197,800戦、2epoch、seed `20260725`、各席1,000戦。学習tensorと
推論時の履歴意味が一致する組合せだけを正規結果として集計した。

| checkpoint | 推論履歴 | 席0 | 席1 | 席2 | 席3 | 全席 | 1枚出し率 |
|---|---|---:|---:|---:|---:|---:|---:|
| Original V1 | rolling | 27.667% | 23.283% | 22.750% | 24.883% | 24.646% | 36.125% |
| Historyfix V1 | turn-local | 16.233% | 12.450% | 14.383% | 14.483% | 14.388% | 28.107% |

Original×turn-localとHistoryfix×rollingは、学習時と推論時の履歴意味が異なるため
モデル成績には含めない。これらは
`results/evaluations/v1_history_mismatch_input_audit.json`へ入力契約監査として
隔離した。

## legacyデータでの履歴比較

各モデルは先頭200,000戦からランダム初期化で1epoch学習。

| モデル | 席0 | 席1 | 席2 | 席3 | 全席合算 |
|---|---:|---:|---:|---:|---:|
| Original V1 200k | 27.933% | 28.850% | 25.033% | 27.050% | **27.217%** |
| V1履歴修正版 200k | 23.817% | 24.383% | 22.533% | 20.733% | **22.867%** |
| canonical old 660,001 | 28.233% | 26.983% | 27.033% | 27.450% | **27.425%** |

別実装・別母集団でもOriginal V1が履歴修正版を上回った。history3
（直近3完了ターン、100k、席0）の勝率は22.250%で、rolling直近2配置を
置き換える根拠にはならなかった。

出典:

- `results/evaluations/v1_legacy_200k_comparison.md`
- `results/evaluations/v1_history3_legacy_100k_eval1000_same_seed_p0.json`

## Original V1の学習量曲線

legacy母集団、各条件はランダム初期化・1epoch。勝率は席0の1,000戦評価。
実際の最適化対象は80%のtrain split。

| source戦数 | 実train戦数 | 勝率 |
|---:|---:|---:|
| 1,000 | 800 | 13.900% |
| 5,000 | 4,000 | 21.333% |
| 10,000 | 8,000 | 18.733% |
| 20,000 | 16,000 | 25.800% |
| 50,000 | 40,000 | 26.350% |
| 100,000 | 80,000 | 26.900% |
| 150,000 | 120,000 | **28.133%** |

学習量増加は全体として有効だが、10kが5kを下回るなど単調ではない。
1,000戦評価の揺れと学習母集団差を分離するため、共通validation/testと
4席評価が必要。

出典:

- `results/evaluations/v1_learning_curve_p0_summary.json`

## 高速NPC収集方針の評価

Original V1 generation0 2epochを使い、簡易補充を含む高速NPC条件で評価した。
この条件は上の従来評価と対戦方針が異なるため、数値を直接混ぜない。

| 方針 | 席 | 勝率 | 秒/戦 |
|---|---:|---:|---:|
| heuristic代表・1枚対2枚 | 0 | 21.400% | 0.205 |
| heuristic代表・1枚対2枚 | 3 | 25.900% | 0.169 |
| random最小失点・1枚対2枚 | 3 | 21.017% | 0.192 |

席3ではheuristic代表方針がrandom代表方針を約4.88ポイント上回った。この
heuristic代表方針で88,966戦を新規収集済み。

出典:

- `results/evaluations/v1_original_epoch002_heuristic_minloss_one_vs_two_p0_1000_seed20260725.json`
- `results/evaluations/v1_original_epoch002_heuristic_minloss_one_vs_two_p3_1000_seed20260725.json`
- `results/evaluations/v1_original_epoch002_random_minloss_one_vs_two_p3_1000_seed20260725.json`

## 枝刈り・速度評価

- adaptive p/qの1時間評価: 852戦、238.833勝、勝率
  **28.032%**。
- 過去の5戦benchmarkではexact adaptive p/qから近似列制限を加えることで
  約1.68倍から1.73倍高速化。
- 20戦の短期比較では近似あり／なしとも30.0%だったが、戦数が少ないため
  勝率同等の証明には使わない。

出典:

- `results/audits/adaptive_pq_1h/adaptive_pq_1h.json`
- `results/audits/adaptive_pq_1h/adaptive_pq_1h_q_top10.md`
- `docs/value-search-speedup-candidates.md`

## 現在進行中の新データ比較

新規収集した88,966戦だけを使い、Original V1の50,000戦条件と88,966戦条件を
比較するパイプラインを実行中。

- 両モデルともランダム初期化、1epoch、batch 256、learning rate `1e-3`
- training seed `20260727`
- 50kは共通train split中のgame ID `<50000`のみ使用
- validation/testは共通の88,966戦母集団
- 各モデルを4席各1,000戦で評価

変換と両モデルの学習は完了した。共通testでの中間確定値:

| 条件 | 実train戦数 | validation Brier | test Brier | test logloss | 学習時間 |
|---|---:|---:|---:|---:|---:|
| 50,000戦条件 | 40,017 | 0.132414 | 0.131674 | 0.422793 | 10.4分 |
| 88,966戦条件 | 71,172 | 0.130808 | **0.130034** | **0.418239** | 14.2分 |

88,966戦条件は50k条件よりtest Brierが`0.001640`、test loglossが
`0.004555`低い。少なくとも共通test calibrationでは学習量増加が改善した。

50kモデルで完了済みの各1,000戦評価:

| 席 | 勝率 |
|---:|---:|
| 0 | 24.183% |
| 1 | 27.250% |

席2以降と88,966戦モデルの全席評価は実行中。部分2席の平均を全席合算として
扱わない。最新状態と完了後の比較値は次を正本とする。

- `results/evaluations/v1_original_new_50000_vs_88966.status.json`
- `results/evaluations/v1_original_new_50000_vs_88966.json`
- `results/evaluations/v1_original_new_50000_vs_88966.md`

## 現時点の結論

1. 現行履歴はOriginal V1のrolling直近2配置を維持する。
2. calibrationと実戦勝率は順位が一致しないため、両方を記録する。
3. 席依存が大きいため、採用判断は4席合算を中心にする。
4. 学習量は有望だが単調改善ではない。新50k対88,966の共通split比較で
   新収集データ内の純粋な学習量差を確認する。
5. adaptive p/qと近似列制限は速度上の利点があり、引き続き現行評価条件とする。
6. 新データの共通testでは88,966戦条件が50k条件を上回った。最終採用は
   実行中の4席評価完了後に決める。
