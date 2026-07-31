# Reward Design

この文書は、初期の強化学習環境で使う reward を定義する。

目的は、まず学習ループを安定して回せる単純な reward を用意すること。勝敗だけの疎な reward では学習初期の信号が弱すぎるため、失点チャートとマイナスカードの変化を補助信号として加える。

## Reward Formula

学習対象プレイヤー `p` について、1回の `step()` の前後の状態から以下を計算する。

```text
reward =
  loss_score_delta
  + 0.1 * negative_card_delta
  + terminal_reward
```

各項目:

- `loss_score_delta = before.loss_score - after.loss_score`
- `negative_card_delta = before.negative_count - after.negative_count`
- `terminal_reward = +1.0` if game over and `p` is a winner
- `terminal_reward = -1.0` if game over and `p` is not a winner
- `terminal_reward = 0.0` otherwise

## Interpretation

- 失点チャートが下がると正の reward。
- 決算などで失点チャートが上がると負の reward。
- マイナスカードが減ると小さな正の reward。
- マイナスカードが増えると小さな負の reward。
- 最終勝敗は小さめの終端 bonus/penalty として加える。

## Scope

この reward は初期学習用の baseline とする。

今後、学習挙動を見て以下を調整する可能性がある。

- 終端 reward の重み
- マイナスカード補助 reward の重み
- NPCとの相対失点差
- ターン数ペナルティ
- 決算直前のリスク評価
