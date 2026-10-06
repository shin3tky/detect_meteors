# NPF Ruleについて

[English](NPF_RULE.md)

**NPF Rule**（Frédéric Michaud氏が開発）は、地球の自転によって星が流れて写り始めるまでの最大露光時間を求める科学的な方法です。従来の「500ルール」「600ルール」と異なり、現代のカメラセンサーの画素ピッチを考慮します。

```
NPF Exposure (seconds) = (35 × F-number + 30 × Pixel Pitch) / Focal Length
```

各変数の意味：

- **F-number**：絞り値（例：f/2.8 → 2.8）。
- **Pixel Pitch**：画素の物理的な大きさ（μm）。
- **Focal Length**：35mm換算焦点距離（mm）。

本ソフトウェアでは、この式を次の目的で使用します。

- 露光設定が適切かを評価。
- 実際の露光時間での星の光跡長を推定。
- NPF適合度に応じて検出パラメータを調整。

## センサー幅の指定

精度を高めるには、カメラのセンサー幅を指定してください。

```bash
# Micro Four Thirds (17.3mm)
uv run python detect_meteors_cli.py --auto-params --sensor-width 17.3

# APS-C (23.5mm)
uv run python detect_meteors_cli.py --auto-params --sensor-width 23.5

# Full Frame (36.0mm)
uv run python detect_meteors_cli.py --auto-params --sensor-width 36.0
```

`--sensor-width` を指定しない場合、既定の画素ピッチ4.0μmを使用します。

## 焦点距離の扱い

焦点距離はEXIFメタデータから自動抽出します。取得できない場合や値が正しくない場合は、次のように指定できます。

```bash
# Specify 35mm equivalent focal length directly
uv run python detect_meteors_cli.py --auto-params --sensor-width 17.3 --focal-length 24

# Or use crop factor (converts actual focal length to 35mm equivalent)
uv run python detect_meteors_cli.py --auto-params --sensor-width 17.3 --focal-factor 2.0
```

## NPF適合度の解析

`--show-npf` で、検出処理をせずに詳しいNPF解析を表示できます。

```bash
uv run python detect_meteors_cli.py --show-npf --sensor-width 17.3
```

出力例：

```
NPF Rule Analysis
============================================================
  Pixel pitch:      3.30μm (sensor: 17.3mm)
  NPF recommended:  8.2s
  Actual exposure:  5.0s ✓ OK
  Star trail est.:  ~1.5 pixels
  Impact:           LOW
============================================================
```

次の点を確認できます。

- 露光時間が適切か。
- 画像で予想される星の光跡長。
- 流星検出の品質への影響の程度。

## 代表的なセンサー幅

| カメラシステム | センサー幅（mm） | クロップ係数 | 代表的な画素ピッチ（μm） |
|----------------|------------------|--------------|--------------------------|
| 1型 | 13.2 | 2.7 | 2.4-2.9 |
| マイクロフォーサーズ | 17.3 | 2.0 | 3.3-3.7 |
| APS-C（Canon） | 22.3 | 1.6 | 4.1-4.5 |
| APS-C（Sony、Nikon、Fuji） | 23.5 | 1.5 | 3.9-4.3 |
| APS-H（Canon） | 27.9 | 1.3 | 5.0-6.4 |
| フルサイズ | 36.0 | 1.0 | 4.3-8.4 |
| 中判44×33 | 43.8 | 0.79 | 3.3-5.3 |
| 中判54×40 | 53.4 | 0.64 | 4.0-4.6 |

## 魚眼レンズの補正

魚眼レンズでは射影の幾何学的な性質により、有効焦点距離が画像内の位置で変化します。中央は公称の焦点距離ですが、周辺ほど有効焦点距離が短くなります。

### 魚眼レンズに特別な処理が必要な理由

最も一般的な魚眼の方式である **等立体角射影** では、次の関係になります。

- **式**：r = 2f × sin(θ/2)。
- **中央（θ=0°）**：有効焦点距離は公称焦点距離と同じ。
- **周辺（θ=90°）**：有効焦点距離は公称値の約0.707倍（cos(45°)）。

このため、次のような違いが生じます。

- 画像周辺の星は、単位時間あたりにより多くの画素を移動します。
- 隅の星の光跡は、中央の約 **1.414倍** になります。
- NPF Ruleでは、周辺の焦点距離を使って余裕を持った推奨値を求めます。

### 魚眼補正の使い方

`--fisheye` を追加すると、等立体角射影の補正を有効にします。

```bash
# MFT camera with 8mm fisheye (16mm equiv.)
uv run python detect_meteors_cli.py --auto-params --sensor-type MFT --focal-length 16 --fisheye

# Full Frame with 8mm fisheye
uv run python detect_meteors_cli.py --auto-params --sensor-type FF --focal-length 8 --fisheye

# Check NPF analysis with fisheye correction
uv run python detect_meteors_cli.py --show-npf --sensor-type MFT --focal-length 16 --fisheye
```

### 例：魚眼補正あり・なしのNPF解析

**`--fisheye` なし**（MFTの8mm F1.8、35mm換算16mm）：

```
NPF Rule Analysis
============================================================
  Pixel pitch:      3.70μm (sensor: 17.3mm)
  NPF recommended:  10.9s
  Star trail est.:  ~1.4 pixels
============================================================
```

**`--fisheye` あり**：

```
Fisheye Correction
============================================================
  Projection model:   Equisolid Angle Projection
  Nominal focal:      16.0mm (center)
  Effective focal:    11.3mm (edge)
  Trail length ratio: 1.41× (edge vs center)
  NPF calculation:    Based on edge (worst case)
============================================================

NPF Rule Analysis
============================================================
  Pixel pitch:      3.70μm (sensor: 17.3mm)
  NPF recommended:  15.4s
  Star trail est.:  ~1.9 pixels
============================================================
```

### --fisheyeを使う条件

次の場合に `--fisheye` を使います。

- 専用の魚眼レンズ（円周魚眼・対角魚眼）を使用する場合。
- 強い樽型歪みを持つ、超広角の直線投影レンズを使用する場合。
- 対角画角が180°以上のレンズを使用する場合。

### 対応する射影モデル

現在は **等立体角射影** のみを実装しています。英語版では、次のような一般的な魚眼レンズを対象として挙げています。

- OM SYSTEM M.ZUIKO DIGITAL ED 8mm F1.8 Fisheye PRO
- Canon EF 8-15mm F4L Fisheye USM
- Nikon AF-S Fisheye NIKKOR 8-15mm f/3.5-4.5E ED
- Sigma 15mm F1.4 DG DN DIAGONAL FISHEYE ART
- TTArtisan 11mm f/2.8 Fisheye
- TTArtisan 7.5mm f/2 C Fisheye
- Samyang 7.5 mm f/3.5 Fish-Eye

将来のバージョンでは、等距離射影や立体射影などへの対応を追加する可能性があります。
