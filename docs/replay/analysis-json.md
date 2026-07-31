# 検証データJSONの読み方

AI分析画面の「検証データをダウンロード」を押すと、次のような名前のJSONファイルが保存されます。

```text
yellowstone-analysis-2026-07-26T08-32-35-123Z.json
```

このファイルには、AI分析を行った手番開始時の状態、自分が選んだ手、AI上位3択、
AIが評価した全候補が入っています。表示された勝率や候補の選ばれ方に疑問がある場合に、
分析結果を手元で確認・比較するためのデータです。

## 最初に見る場所

通常は、次の3項目を順に確認します。

1. `playerSelection.evaluation`
   - 自分が選んだ手と、その推定勝率です。
2. `aiTop3`
   - 画面に表示されたAI 1位から3位です。
3. `allAiCandidates`
   - AIが比較した全候補です。上位3択の選出過程を詳しく調べるときに使います。

勝率は `probability` に0から1の数値で保存されます。たとえば `0.327` は32.7%です。
画面では整数に丸めますが、順位付けにはJSON内の丸め前の数値を使います。

## ルート項目

| 項目 | 内容 |
| --- | --- |
| `schemaVersion` | この検証JSONの形式バージョンです。現在は `1` です。 |
| `exportedAt` | ダウンロードした日時です。UTCのISO 8601形式です。 |
| `application` | アプリ名とバージョンです。 |
| `model` | 使用したAIランタイム、モデルの場所、モデル付属情報です。 |
| `settings` | NPC難易度と分析表示モードです。 |
| `turnStartState` | AI比較を始めた手番の完全なゲーム状態です。 |
| `recentHistory` | AI入力に使う直近の配置履歴です。 |
| `playerSelection` | プレイヤーが選んだ補充方法、手順、勝率、適用後状態です。 |
| `aiTop3` | カード構成で重複を除いたAI上位3択です。 |
| `allAiCandidates` | AIが評価した全候補です。 |

`model.metadata` は `public/models/win_value.json` の内容です。ファイル取得に失敗した場合は
`null` になりますが、ゲーム状態や候補評価には影響しません。

## カードの表現

カードは次の形式です。

```json
{
  "color": "blue",
  "rankIndex": 4
}
```

- `color`: `red`、`blue`、`green`、`yellow` のいずれかです。
- `rankIndex`: 0始まりの数字です。画面に表示する数字は `rankIndex + 1` です。

上の例は「青5」です。

## ゲーム状態

`turnStartState` は、分析開始時点を再現する基礎データです。

| 項目 | 内容 |
| --- | --- |
| `players` | 各プレイヤーの手札、マイナスカード、現在の失点です。0番が人間です。 |
| `board` | 盤面です。キーは `"x,y"`、値はそのマスに重なったカードの配列です。 |
| `deck` | 山札です。配列の先頭が次に引くカードです。 |
| `currentPlayerIndex` | 現在のプレイヤー番号です。人間の分析時は `0` です。 |
| `phase` | `play`、`refill`、`game_over` のいずれかです。 |
| `cardsPlayedThisTurn` | この手番ですでにプレイした枚数です。 |
| `settlementCount` | 決算が起きた回数です。 |
| `lastTurnPlayCounts` | 各プレイヤーが直前の手番でプレイした枚数です。 |
| `randomState` | シャッフルを再現するための乱数状態です。 |

盤面座標の `x` と `y` はどちらも0から6です。`y` はカードの `rankIndex` と一致します。
画面表示では、座標を人間向けの1から7へ変換しています。

このJSONには対戦相手の手札と山札順も含まれます。AI入力の確認とゲームの再現を優先した
検証用データです。

## 手順の表現

`actions` には、1手番で行う処理が実行順に並びます。

### カードを置く

```json
{
  "type": "place",
  "handIndex": 2,
  "position": { "x": 4, "y": 1 },
  "frame": { "x": 2, "y": 0 }
}
```

- `handIndex`: その処理の直前にある手札配列の0始まりの位置です。
- `position`: カードを置く盤面座標です。
- `frame`: 残す3×3枠の基準座標です。`x` から `x + 2`、`y` から `y + 2` が枠内です。

1枚目を置くと手札配列が変わるため、2枚目の `handIndex` は
`turnStartState.players[0].hand` の位置と直接一致しない場合があります。
正確に読むには、`actions` を先頭から順番に適用してください。

### 1枚で手番を終える

```json
{ "type": "end_turn" }
```

### 補充する・補充しない

```json
{ "type": "refill", "source": "deck" }
```

`source` の意味は次のとおりです。

| 値 | 内容 |
| --- | --- |
| `deck` | 山札から手札上限まで補充します。 |
| `negative_cards` | マイナスカードから手札を作ります。 |
| `none` | 補充しません。 |

## 評価データ

プレイヤー選択とAI候補は、主に次の形で保存されます。

```json
{
  "probability": 0.327,
  "playedCardsSignature": "blue:1|yellow:5",
  "actions": [],
  "historyAfter": [],
  "resultingState": {}
}
```

| 項目 | 内容 |
| --- | --- |
| `probability` | モデルが推定した勝率です。0から1です。 |
| `playedCardsSignature` | プレイするカード構成を比較するための内部表現です。 |
| `actions` | 配置、手番終了、補充を含む手順です。 |
| `historyAfter` | この候補を適用した後にAI入力へ渡される直近履歴です。 |
| `resultingState` | 候補を最後まで適用した状態です。自分の手とAI上位3択に入ります。 |

`allAiCandidates` はデータ量を抑えるため `resultingState` を持ちません。
`turnStartState` に `actions` を順番に適用すれば結果を再構築できます。

## AI上位3択の選び方

`playedCardsSignature` は、プレイしたカードを `色:rankIndex` で表し、
順番を無視するために並べ替えて `|` で連結したものです。

```text
blue:1|yellow:5
```

これは「青2と黄6をプレイする」カード構成を表します。次の違いは同じグループとして扱います。

- 置く順番
- 置く場所や残す3×3枠
- 補充方法

同じ署名の候補から `probability` が最も高いものだけを代表として残し、
代表を勝率順に並べた先頭3件が `aiTop3` です。
プレイヤー選択と同じ署名でも除外しないため、画面では「同じカード」と表示される場合があります。

上位3択を検算する場合は、`allAiCandidates` を `playedCardsSignature` ごとにまとめ、
各グループの最大 `probability` を降順に並べてください。

## PowerShellでの確認例

ファイルを読み込みます。

```powershell
$data = Get-Content -Raw -Encoding utf8 ".\yellowstone-analysis-....json" |
  ConvertFrom-Json
```

自分の推定勝率をパーセントで表示します。

```powershell
[math]::Round($data.playerSelection.evaluation.probability * 100, 2)
```

AI上位3択を表示します。

```powershell
$data.aiTop3 |
  Select-Object playedCardsSignature,
    @{Name="winRatePercent"; Expression={
      [math]::Round($_.probability * 100, 2)
    }},
    actions
```

全候補から、カード構成ごとの最高勝率を再計算します。

```powershell
$data.allAiCandidates |
  Group-Object playedCardsSignature |
  ForEach-Object {
    $_.Group | Sort-Object probability -Descending | Select-Object -First 1
  } |
  Sort-Object probability -Descending |
  Select-Object -First 3 playedCardsSignature,
    @{Name="winRatePercent"; Expression={
      [math]::Round($_.probability * 100, 2)
    }}
```

## 値を確認するときの注意点

- `probability` は厳密な勝率計算ではなく、学習モデルによる推定値です。
- 画面の整数表示とJSONの値がわずかに違う場合は、画面側の丸めによるものです。
- 補充方法が違っても上位3択の別グループにはなりません。同じカード構成内で最も高い候補が残ります。
- 同じカードを違う順番で出す場合も別グループにはなりません。
- 候補の再評価には、JSON記載のモデルと同じモデルファイル、入力変換処理が必要です。
