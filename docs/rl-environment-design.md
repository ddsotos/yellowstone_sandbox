# Reinforcement Learning Environment Design

この文書は、Yellowstone Park を強化学習環境として扱うための最小APIを定義する。

現時点では Gymnasium などの外部インターフェースへ直接合わせない。まずプロジェクト内部の薄い環境APIを作り、その上に必要なら Gymnasium wrapper を足す。

## 目的

- 学習対象プレイヤー1人を外部 agent として扱う。
- 残り3人は deterministic heuristic bot で自動進行する。
- ルール実装は既存の `GameState -> legal_actions -> apply_action -> GameState` を使う。
- 状態、行動、合法手、結果を JSON 変換しやすい形に保つ。
- 学習ループではテキスト描画やログ生成を使わない。

## 基本API

想定する環境クラスは `YellowstoneEnv` とする。

```python
class YellowstoneEnv:
    def reset(self, seed: int | None = None) -> EnvReset: ...
    def step(self, action_index: int) -> EnvStep: ...
```

`reset(seed)` は新しいゲームを開始し、学習対象プレイヤーの手番まで NPC を自動で進めた状態を返す。

`step(action_index)` は学習対象プレイヤーの行動を1つ適用する。学習対象プレイヤーの手番が終わった場合は、次に学習対象プレイヤーの手番が来るまで NPC を自動で進める。

## 戻り値

`reset` は以下を返す。

```python
@dataclass(frozen=True, slots=True)
class EnvReset:
    observation: Observation
    legal_action_mask: tuple[bool, ...]
    info: dict[str, object]
```

`step` は以下を返す。

```python
@dataclass(frozen=True, slots=True)
class EnvStep:
    observation: Observation
    reward: float
    done: bool
    legal_action_mask: tuple[bool, ...]
    info: dict[str, object]
```

## observation

観測表現は `docs/observation-action-space.md` と `src/yellowstone/observation.py` で定義する。

最低限含める情報:

- 7x7 盤面
- 各セルのカード stack
- 学習対象プレイヤーの手札
- 各プレイヤーのマイナスカード枚数
- 各プレイヤーの失点チャート位置
- 現在プレイヤー
- フェーズ
- ターン中に置いたカード枚数
- 山札枚数
- 決算回数

初期実装では、外部依存を増やさず `tuple[int, ...]` の固定長数値表現にする。NumPy や PyTorch の tensor 変換は wrapper 側で行う。

## action_index

行動空間は `docs/observation-action-space.md` と `src/yellowstone/action_space.py` で定義する。

必要な行動種別:

- 配置: `hand_index`, `position`, `frame`
- ターン終了: 1枚置いた後の終了
- 補充: `deck`, `negative_cards`, `none`

`legal_action_mask` は固定長の `tuple[bool, ...]` とし、合法な `action_index` だけ `True` にする。

## reward

reward は別TODOで設計する。初期案では以下を組み合わせる。

- 最終勝敗報酬
- 自分の失点チャート増減
- 自分のマイナスカード枚数増減
- NPCとの相対失点差

reward 設計は学習挙動に強く影響するため、実装前に文書化してからテストを追加する。

## done

`done` は `GameState.phase == Phase.GAME_OVER` のとき `True` にする。

ゲーム終了前に合法手がなくなった場合は、初期実装では `done=True` とし、`info["stopped_reason"] = "no_legal_action"` を返す。

## info

`info` にはデバッグと評価に必要な補助情報を入れる。

想定キー:

- `state`: JSON 変換済み `GameState`
- `legal_actions`: JSON 変換済み合法手一覧
- `current_player_index`
- `learning_player_index`
- `winners`
- `settlement_count`
- `stopped_reason`

## NPC自動進行

学習対象以外のプレイヤーの手番では `HeuristicBot` を使う。

自動進行は以下で止める。

- 学習対象プレイヤーの手番になった
- ゲーム終了した
- 合法手がなくなった

NPCの行動ログは学習ループでは生成しない。必要な場合だけ別のデバッグ runner で生成する。

## 乱数

- `reset(seed)` で初期化 seed を受け取る。
- 環境は `random.Random` を保持し、補充や山札再生成で使う。
- テストでは seed を固定する。

## 依存関係

初期実装では追加依存を増やさない。Gymnasium、NumPy、PyTorch への接続は、環境APIが固まってから wrapper として追加する。
