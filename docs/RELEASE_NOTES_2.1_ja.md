# バージョン2.1 リリースノート

[English](RELEASE_NOTES_2.1.md)

<a id="version-211"></a>

## バージョン2.1.1（2026-11-21）📺 世界テレビ・デー

2.1.1では、既存の検出器と出力を使いながら、検出直前に画像ペアの前処理を差し込めるようにします。
既定は、位置合わせやFRT再検査を使わない従来の絶対差分ハフとファイル出力です。

### 主な変更

- **画像ペアの前処理フック。** 現在画像と直前画像の読み込み後、検出器を呼ぶ前に、設定順に
  `on_before_detection(context)` を実行します。組み込みの `star_registration` は、ROIの恒星を基準に
  直前画像を回転・平行移動してから検出する例です。[位置合わせの仕組みと使い方](star_registration_workflow_ja.md)
- **正の差分ハフ。** `detector_config.difference_mode: positive` で、現在画像が明るくなった画素だけを
  ハフ検出に渡します。直前画像の流星が消えたときの負の残差を除外できます。
  [差分方式の比較](hough_registration_thresholds_ja.md#正の差分だけを渡した場合)
- **実験用のFRT再検査。** 位置合わせなしの正の差分ハフで棄却したペアにだけ適用します。
  `detector_config.frt_retry: {}` で有効になり、短い途切れを含む光跡を再検査します。
  時間とワーカーあたりのメモリ使用量が増えます。[閾値、処理負荷と制約](frt_retry_workflow_ja.md)

### 設定の切り替えと移行

既存のYAML／JSONは引き続き使用できます。パイプライン設定ファイル形式は `schema_version: 1` です。
同じ設定で続ける場合は従来の進捗から再開できます。画像ペア前処理フック、差分方式、FRT再検査の
設定を変更すると、進捗との照合で再処理されます。設定比較には別々の出力先と進捗ファイルを使います。

航空機サンプルを用意したソースのルートで実行します。
[サンプルRAWの取得](INSTALL_ja.md#サンプル-raw-画像)

```bash
uv run detect-meteors --config config_examples/hough_legacy_sample.yaml --debug-image
```

`--config` に渡すファイルを変えると、同じ入力・ROI・ハフ閾値で比較できます。候補、
デバッグ画像、進捗の保存先は設定ごとに分かれています。

| 処理 | 設定ファイル |
| --- | --- |
| 従来の絶対差分ハフ | [hough_legacy_sample.yaml](../config_examples/hough_legacy_sample.yaml) |
| 恒星位置合わせを前処理に追加 | [hough_registered_sample.yaml](../config_examples/hough_registered_sample.yaml) |
| 位置合わせなしの正の差分ハフ | [hough_positive_sample.yaml](../config_examples/hough_positive_sample.yaml) |
| 正の差分ハフ＋FRT再検査の実験 | [hough_frt_retry_sample.yaml](../config_examples/hough_frt_retry_sample.yaml) |

従来の検出に戻すには `hough_legacy_sample.yaml` を選びます。手元の設定でフックだけを外すには
`hooks: []` または `--hooks ""` を指定します。FRTだけを外すには `frt_retry` を省略するか `null` にします。
[インストールガイドの設定切り替え手順](INSTALL_ja.md#検出設定の切り替え) · [Wikiの設定切り替え手順](https://github.com/shin3tky/detect_meteors/wiki/Home#検出設定の切り替え)

### 検証結果

- **従来設定の互換性。** 固定した従来の設定で実RAW 121ペアを処理し、変更前のローカル `main` と
  候補、線分、スコア、デバッグ画像が一致することを確認しました。同じ設定で元の進捗からも再開できました。
  4つのサンプル設定の切り替えは航空機サンプル11ペアで確認しました。
- **121ペアの系列除外評価。** 位置合わせなしの正の差分ハフでは流星3/6枚、偽陽性0/115枚。
  FRT再検査を追加すると流星6/6枚、偽陽性0/115枚でした。各回の閾値選択には評価系列を使わず、
  再検査の方式はこの121ペアの診断をもとに設計しました。[系列別の結果と再現手順](frt_retry_workflow_ja.md#同じ121ペアの系列除外評価)
- **別系列の2025LEO。** 同じFRT閾値を使った連続95ペアでは、目視で確認できた流星の追加回収は0枚で、
  偽陽性はハフ単体の2枚から20枚に増えました。追加候補は複数の露光にわたって移動する動体の光跡でした。
  この系列で流星を目視確認できなかったため、回収率は測れませんでした。[別系列の結果と処理負荷](frt_retry_workflow_ja.md#2025leoでの結果と処理負荷)

### プラグイン開発

`on_before_detection()` は `DetectionContext` を受け取り、同じ型を返します。逐次時はメインプロセス、
並列時はワーカーのバッチ内で実行します。返したコンテキストは検出器と `on_detection_complete()` に渡されます。
座標とROIの契約、`hook_error_mode`、フック・プラグインの追加と取り外しは開発ガイドにまとめています。
[前処理フックの契約](PLUGIN_AUTHOR_GUIDE_ja.md#frame-pair-hook) · [設定の切り替え](PLUGIN_AUTHOR_GUIDE_ja.md#enable-disable-plugins)

### 動作環境と配布

公開した2.1.1のwheelとソース配布物は、どちらも Python 3.12、3.13、3.14 で uv を使って
インストールできます。導入前に `SHA256SUMS` と照合してください。
[GitHub Release 2.1.1](https://github.com/shin3tky/detect_meteors/releases/tag/2.1.1) · [インストールガイド](INSTALL_ja.md) · [ダウンロードした配布物の検証](INSTALL_ja.md#verify-release-downloads)

1.6.13からの依存関係の更新は、
実行時のPydantic 2.14.4とPyYAML 6.0.4、開発用のRuff 0.17.6、ty 0.0.107、coverage 7.17.4です。
設定例とガイドはソース配布物で参照できます。サンプルRAW、目視ラベル、チェックサムは
`detect_meteors_assets` から取得します。実験スクリプトを再実行するときはGitのチェックアウトを使います。
[公開前の確認手順](INSTALL_DEV_ja.md#release-211-checklist)
