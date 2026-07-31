# Observation And Action Space

この文書は、強化学習用の固定長観測表現と固定行動空間を定義する。

初期実装では NumPy や Gymnasium に依存せず、Python 標準の `tuple[int, ...]` と `tuple[bool, ...]` で扱う。必要になったら、この表現を wrapper 側で tensor へ変換する。

## Observation

実装は `src/yellowstone/observation.py` に置く。

入口:

```python
state_to_observation(state: GameState) -> tuple[int, ...]
observation_metadata() -> dict[str, int]
```

観測長は `OBSERVATION_SIZE` で固定する。2026-06-26 時点では `71`。

内訳:

1. 盤面: `7 + 1 + 3 = 11`
2. 現在プレイヤーの手札: `6 * 6 = 36`
3. プレイヤー概要: `4 * 3 = 12`
4. 相手3人の直前ターンのプレイ枚数: `3`
5. 1枚案・2枚案の即時ボーナス点: `2`
6. 1枚案・2枚案の即時マイナス増加量: `2`
7. 1枚案・2枚案で出す数字: `3`
8. スカラー: `2`

### 盤面

盤面は以下の11値で表す。

```text
rank_counts[7]
left_column_index
left_center_right_column_counts[3]
```

`rank_counts` は数字ごとの盤面カード総数、`left_column_index` は一番左の列番号、最後の3値は左・中央・右列ごとのカード総数。盤面が空の場合はすべて0にする。

盤面には赤・青・緑・黄という絶対色を入れない。色は手札側で相対色one-hotとして表す。

### 手札

現在プレイヤーの手札だけを、最大6スロットで表す。手札は状態更新時に
`rank_index, color` の順で整列し、観測時には並べ替えない。

各スロットは以下の6値:

```text
present, relative_color_0, relative_color_1, relative_color_2, relative_color_3, rank_index
```

相対色IDは、盤面3列の色から決める。3x3範囲の右列にある色を `relative_color_0`、中央列にある色を `relative_color_1`、左列にある色を `relative_color_2` とする。盤面にまだ出ていない色は、現在プレイヤーの手札スロット順で空いている相対色IDに割り当てる。これにより、赤・青・緑・黄という絶対色は学習入力から消しつつ、手札内で同じ色かどうかは保持する。

空スロットはすべて0にする。

### プレイヤー概要

学習環境は4人対戦を前提にし、4人ぶんを固定長で持つ。

各プレイヤーは以下の3値:

```text
hand_count, negative_card_count, loss_score
```

`active` は4人固定では常に1になるため入れない。

### その他

- 相手の直前ターン枚数: 現在プレイヤーから順に次の3人
- heuristic即時ボーナス: `one_card_bonus`, `two_card_bonus`
- heuristic即時マイナス増加: `one_card_negative_delta`, `two_card_negative_delta`
- heuristicで出す数字: `one_card_rank`, `two_card_low_rank`, `two_card_high_rank`
- スカラー: `deck_bucket`, `settlement_count`

2枚案の数字は実際のプレイ順ではなく昇順で入れる。これにより、同じ2枚を
使う判断が順序違いの別特徴として扱われることを避ける。

`deck_bucket` は山札枚数を以下の4段階に丸める。

```text
0: 山札0枚
1: 1..6枚
2: 7..18枚
3: 19枚以上
```

turn-level 学習では、学習対象の行動選択時点が現在プレイヤーのプレイフェーズ開始時に揃う。そのため `current_player`、`phase`、`cards_played_this_turn`、`player_count` は入力に入れない。

## Turn-Level Action Space

現在の学習では、配置座標や3x3枠を直接学習対象にしない。
学習対象は「手札のどのスロットを使うか」だけを選び、実際の配置と補充は heuristic に従って決める。

実装は `src/yellowstone/turn_action_space.py` と `src/yellowstone/turn_env.py` に置く。

入口:

```python
turn_action_to_index(action: TurnAction) -> int
turn_action_from_index(index: int) -> TurnAction
resolve_turn_action(state: GameState, index: int) -> tuple[Action, ...]
legal_turn_action_indices(state: GameState) -> tuple[int, ...]
legal_turn_action_mask(state: GameState) -> tuple[bool, ...]
```

手札が6枚ある標準状態では、turn-level action は36通り。

1. 1枚プレイ: `6` 通り。手札スロット `0..5` のどれを1枚だけ使うか。
2. 2枚プレイ: `6 * 5 = 30` 通り。1枚目と2枚目に使う元の手札スロットの順序付きペア。

2枚プレイでは、1枚目を置いた後に手札indexが詰まる。
そのため2枚目は、手番開始時点の元スロットを基準に指定し、実行時に現在の手札indexへ変換する。

選ばれたカードの置き場所、3x3枠、補充元は、既存の heuristic 評価で決める。
これにより、初期学習では「何を出すか」に集中し、「どう置くか」はその場の失点を抑える既存ルールに寄せる。

turn-level の合法手 mask は、配置を実際に解決して検証しない。
ターン開始時の盤面は3x3枠内に収まっており、失点を許す限り手札カードは配置可能というルール前提に基づき、存在する手札スロットだけを `True` にする。
2枚プレイも、手番開始時点で存在する異なる2スロットの順序付きペアを合法とする。

## Low-Level Action Space

低レベル action space は、配置座標や3x3枠も含めて直接学習するための互換APIとして残す。

実装は `src/yellowstone/action_space.py` に置く。

入口:

```python
action_to_index(action: Action) -> int
action_from_index(index: int, state: GameState) -> Action
legal_action_indices(state: GameState) -> tuple[int, ...]
legal_action_mask(state: GameState) -> tuple[bool, ...]
action_space_metadata() -> dict[str, int]
```

行動空間の長さは `ACTION_SPACE_SIZE` で固定する。2026-06-13 時点では `1054`。

内訳:

1. 配置: `6 hand slots * 7 x positions * 25 frames = 1050`
2. ターン終了: `1`
3. 補充: `3`

### 配置 action

配置 action の index は以下で決める。

```text
hand_index, position.x, frame.y, frame.x
```

`position.y` は、現在状態の `hand[hand_index].rank_index` から復元する。これにより、rank と行の対応を保ったまま行動空間を小さくする。

### ターン終了 action

`EndTurnAction` は配置 action 群の直後の1 index に割り当てる。

### 補充 action

補充 action は以下の順に割り当てる。

1. `deck`
2. `negative_cards`
3. `none`

### legal_action_mask

`legal_action_mask(state)` は `ACTION_SPACE_SIZE` と同じ長さの `tuple[bool, ...]` を返す。

`legal_actions(state)` に含まれる行動だけ `True` にし、それ以外は `False` にする。
