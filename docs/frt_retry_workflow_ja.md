# 正の差分ハフで棄却したペアのFRT再検査

FRT再検査は設定で有効にする実験です。組み込みの `hough` で
`difference_mode: positive` と `frt_retry: {}` を指定すると、
棄却した画像ペアを Fast Radon Transform（FRT）で再検査する。元のハフで採用したペアは
そのまま保存する。FRTで採用したペアも、通常の候補RAWとデバッグ画像の保存経路を通る。

## 有効にする

航空機サンプルのRAWに対して実行する設定例を用意した。独自の撮影系列ではROIとハフ閾値を
撮影条件に合わせて指定する。 [設定例](../config_examples/hough_frt_retry_sample.yaml)

```bash
uv run detect-meteors --config config_examples/hough_frt_retry_sample.yaml --debug-image
```

同じ入力とROIで戻し先を選べる。`hough_positive_sample.yaml` は正の差分ハフまでを実行し、
`hough_legacy_sample.yaml` は従来の絶対差分ハフを実行する。候補と進捗の保存先は設定ごとに分かれている。
[正の差分ハフ](../config_examples/hough_positive_sample.yaml) · [従来のハフ](../config_examples/hough_legacy_sample.yaml)

既存の設定では `detector_config` に以下を指定し、`hooks` から `star_registration` を外す。
RAWローダーは `normalize: false` で使う。有効にすると再検査の分だけ処理時間が増える。

```yaml
detector_name: hough
detector_config:
  difference_mode: positive
  frt_retry: {}
```

`frt_retry` は省略または `null` で無効になる。空のマッピングで有効化し、
次の項目を指定すると個別の閾値を上書きできる。

| 項目 | 既定値 | 条件 |
| --- | ---: | --- |
| `bilateral_noise_min` | 0.3 | 中心帯と左右それぞれの比較で必要な局所コントラストの下限（雑音単位） |
| `max_gap_samples` | 3 | 支持区間の中で許す途切れ（0〜3標本）。ROI外をまたがず、支持標本が区間の75%以上を占めること |
| `min_support_px` | 32 | 支持区間の最小長（ビニング後のpx） |
| `maximum_prior_texture_noise` | 0.8 | 直前RAWの局所凹凸の90パーセンタイルの上限（雑音単位） |
| `frt_min_snr` | 10 | FRT得点の下限（設定可能な最小値は8） |

FRTで採用した場合、結果の得点は採用区間の最高FRT得点になる。Pythonの検出結果では
`extras["frt_retry"]` にハフ得点と採用区間の座標・長さ・得点・局所凹凸が入る。
進捗ファイルの `hough_frt_retry` に有効な閾値を記録する。再検査の有効化、解除、閾値変更時は
再開指定でもペアが再処理される。比較する場合は出力フォルダーと進捗ファイルを分ける。
`--analyze-image` は最初のハフ検出を測定する。

## 同じ121ペアの系列除外評価

各回で評価対象の系列を除いて閾値を選び、得られた設定を既存ワークフローに渡した。
各回で選ばれた閾値は上記の既定値と一致した。2ワーカーでRAWから再処理し、
ハフの採用3枚を保持して棄却した118ペアをFRTで再検査した。

| 評価系列 | ペア | ハフ採用 | FRT追加 | 軸が一致した流星 | 偽陽性 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 同梱の航空機サンプル | 11 | 2 | 0 | 2/2 | 0/9 |
| 2021GEMINI | 35 | 1 | 2 | 3/3 | 0/32 |
| 2024GEMINI後半 | 75 | 0 | 1 | 1/1 | 0/74 |
| 合計 | 121 | 3 | 3 | **6/6** | **0/115** |

FRTが追加したのは `_C150319.ORF`、`_C150326.ORF`、`_C140521.ORF`。6枚すべてで
前回の実験と同じ注釈軸被覆を再現し、候補RAWとデバッグ画像の保存を確認した。
`_C140521.ORF` の被覆は50.0536%で、判定基準の50%に近い。緩和の方法はこの121ペアの
診断から設計した。 [各系列の設定・候補・軸被覆](../experiments/region_comparison/results/frt_c140521/frt_retry_workflow.json) · [短い途切れの診断と閾値選択](frt_c140521_gap_verification_ja.md)

## 2025LEOでの結果と処理負荷

121ペアに含まれない2025LEOでは、FRT閾値を固定して連続95ペアを比較した。目視で確認できた
流星の追加回収は0枚で、連続する露光で移動する動体の光跡が追加され、偽陽性は2枚から20枚に増えた。
撮影間隔の空いた4ペアも含めた99ペアの実測では、2ワーカーの処理時間は約30秒から約107秒に増えた。
ビニング後2620×1956 pxの代表1ペアを単一プロセスで再検査した際のピーク常駐メモリは約2.2〜2.5 GiBだった。
サンプル設定ではワーカー数を2に指定している。

## 再現

リポジトリのルートで実行する。評価スクリプトは記録済みの系列別の設定を使い、入力ハッシュ、
ROIマスク、ペアの順序、候補判定と注釈軸被覆を照合する。

```bash
run_dir=$(mktemp -d)
raw_root=../detect_meteors_assets/rawfiles
PYTHONPATH=. uv run --locked python experiments/region_comparison/benchmark_frt_retry_workflow.py \
  "$run_dir/workflow" --raw-root "$raw_root" --workers 2
```

閾値も再選択する場合は、[前回の再現手順](frt_c140521_gap_verification_ja.md#再現)で生成した
`foreground/background_verification.json` のパスを `--calibration` に渡す。
