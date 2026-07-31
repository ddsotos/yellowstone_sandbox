# Training Library Selection

この文書は、Yellowstone Strategy で実際に強化学習を回すための初期ライブラリ選定を記録する。

## 結論

初期学習は以下を採用する。

- Environment API: Gymnasium
- Algorithm framework: Stable-Baselines3
- Masked algorithm: sb3-contrib `MaskablePPO`

## 理由

### Gymnasium

Gymnasium は `reset()` と `step()`、`action_space`、`observation_space` を中心にした標準的な環境APIを提供する。既存の `YellowstoneEnv` はすでに `reset(seed)` と `step(action_index)` に近い形なので、薄い wrapper で接続しやすい。

### Stable-Baselines3

最初から自前の PyTorch 学習ループを書くより、実績のある実装で学習が成立するかを先に確認する方がよい。モデル保存、評価、ログ、標準的なPPO/DQN系の実験がしやすい。

### sb3-contrib MaskablePPO

Yellowstone の低レベル行動空間は固定長 `1054` だが、多くの状態で合法手はその一部だけになる。違法手を大量に含む通常の離散行動空間として学習すると効率が悪い。そのため、合法手 mask を使える `MaskablePPO` を最初の候補にする。

初期学習ではさらに行動空間を簡略化し、1枚プレイ6通りと2枚プレイ30通りの turn-level action `36` 通りを使う。

## Dependency

強化学習用の依存は通常の開発依存から分ける。

```bash
python -m pip install -e ".[rl]"
```

通常のルール実装やテストでは、Gymnasium、Stable-Baselines3、sb3-contrib は必須にしない。

## First Training Target

最初の学習対象は以下にする。

- 学習対象: player 0
- 相手: heuristic bot 3人
- 観測: `state_to_observation`
- 行動: turn-level の36通り discrete action
- mask: `legal_turn_action_mask`
- アルゴリズム: `MaskablePPO("MlpPolicy", env)`

低レベルの `ACTION_SPACE_SIZE=1054` は互換APIとして残すが、初期学習では使わない。
配置先、3x3枠、補充元は heuristic に任せ、学習対象は手札のどのカードを使うかに集中させる。

## Notes

- 学習前に、Gymnasium wrapper が返す observation dtype と shape を固定する。
- MaskablePPO 用に `action_masks()` を実装する。
- 報酬は `docs/reward-design.md` の baseline から開始し、学習ログを見て調整する。
