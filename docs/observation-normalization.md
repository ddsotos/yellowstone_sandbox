# Observation Normalization

強化学習では、既存の `state_to_observation` が返す固定長整数 tuple をそのまま中核APIとして残し、Gymnasium / Stable-Baselines3 に渡す直前で `float32` の 0..1 正規化配列に変換する。

## Policy

- Core API: `tuple[int, ...]`
- Gymnasium API: `np.ndarray` with `dtype=np.float32`
- Scale: feature ごとの上限値で割り、上限を超えた値は `1.0` に丸める
- Action mask: 観測とは分けて `action_masks()` と `info["action_mask"]` で渡す

この方針により、ルール実装・JSON変換・テストでは整数表現を維持し、学習ライブラリ側ではニューラルネットワークに渡しやすい値域にする。

## Bounds

上限値は `src/yellowstone/observation_normalization.py` に集約する。

- 盤面アンカー: `BOARD_SIZE - 1`
- 盤面の3x3セルごとのカード総数: 全カード枚数
- 手札の相対色one-hot: `1`
- 手札枚数: `HAND_SIZE`
- 山札段階: `3`
- negative cards: 全カード枚数
- loss score: baselineでは `64` でclip
- settlement count: baselineでは `64` でclip

loss score と settlement count は実ゲームで増えうるため、学習ログを見て必要なら上限を調整する。
