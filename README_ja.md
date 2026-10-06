# Detect Meteors CLI

[English](README.md)

![social_preview](social_preview.jpg)

[![tests](https://github.com/shin3tky/detect_meteors/actions/workflows/python-test.yml/badge.svg)](https://github.com/shin3tky/detect_meteors/actions/workflows/python-test.yml)

連続撮影した星空のRAW画像から、フレーム間差分を使って流星候補を自動抽出します。流星かどうかは候補画像を目視で確認してください。

## 開発の背景

流星群の撮影では、数千枚のRAW画像を手作業で確認して流星を探すのに多くの時間がかかります。このツールは最初の候補抽出を自動化し、詳しく確認する画像を素早く絞り込めるようにします。

![workflow](workflow.png)

📅 **流星撮影を計画していますか？** 今後の流星群の日程や観測のヒントは、[流星群カレンダー](https://github.com/shin3tky/detect_meteors/wiki/Meteor-Showers-Calendar)を参照してください。

> [!TIP]
> 🌠 **2026年10月は、2つの流星群を撮影しよう！**
>
> - **りゅう座流星群（Draconids）：2026年10月8〜9日**。[りゅう座流星群の詳細](https://github.com/shin3tky/detect_meteors/wiki/Meteor-Showers-2026#draconids)で、極大夜の観測条件や撮影のヒントを確認してください。
> - **オリオン座流星群（Orionids）：2026年10月21〜22日**。[オリオン座流星群の詳細](https://github.com/shin3tky/detect_meteors/wiki/Meteor-Showers-2026#orionids)で、極大夜の観測条件や撮影のヒントを確認してください。

## 特徴

- **自動調整**：NPF Ruleに基づきEXIFメタデータを解析し、検出パラメータを科学的に調整します。
- **実画像での検証**：プロジェクトの実画像テストデータ（OM Digital OM-1、1,000枚以上のRAW画像）では検出率100%を報告しています。結果は撮影条件やパラメータによって変わります。
- **RAW形式への対応**：[`rawpy`](https://github.com/letmaik/rawpy)が対応する形式を利用できます。
- **画像処理と再開機能**：ROIによる領域の絞り込み、Hough変換による線分検出、中断後に再開できるバッチ処理に対応します。
- **高速処理**：マルチコア並列処理で、1画像あたり約0.18秒の処理実績があります。

## 動作要件

- Python 3.12、3.13
- macOS、Windows、Linux
- 依存ライブラリ：`numpy`、`opencv-python`、`rawpy`、`psutil`、`pillow`、`pydantic`、`pyyaml`

## インストール

詳しい手順は[INSTALL_ja.md](docs/INSTALL_ja.md)を参照してください。

## クイックスタート

### 手順1：EXIFメタデータを確認する

```bash
uv run python detect_meteors_cli.py --show-exif
```

焦点距離が取得できていることを確認します。取得できない場合は、`--focal-length` で指定してください。

### 手順2：検出を実行する

```bash
# Micro Four Thirds camera
uv run python detect_meteors_cli.py --auto-params --sensor-type MFT

# APS-C camera (Sony/Nikon/Fuji)
uv run python detect_meteors_cli.py --auto-params --sensor-type APS-C

# Full Frame camera
uv run python detect_meteors_cli.py --auto-params --sensor-type FF

# With fisheye lens
uv run python detect_meteors_cli.py --auto-params --sensor-type MFT --focal-length 16 --fisheye
```

> [!IMPORTANT]
> 検出にはフレーム間差分を使用します。  
> **N枚**のRAWファイルから、連続する **N−1組**を解析します。**最初の画像は基準フレームとして使い、検出の評価対象にはしません。**  
> このため「入力100枚 → 処理99組」は正常です。最初の画像は手動で確認してください。

### 手順3：候補を確認する

`candidates/` フォルダに保存された流星候補画像を確認してください。

## 対応するセンサータイプ

| センサータイプ | 説明 |
|----------------|------|
| `1INCH` | 1型センサー |
| `MFT` | マイクロフォーサーズ |
| `APS-C` | APS-C（Sony/Nikon/Fuji） |
| `APS-C_CANON` | APS-C（Canon） |
| `APS-H` | APS-H |
| `FF` | 35mmフルサイズ |
| `MF44X33` | 中判44×33mm |
| `MF54X40` | 中判54×40mm |

プリセット一覧：`uv run python detect_meteors_cli.py --list-sensor-types`

## 入力と出力

- **入力**：RAW画像のディレクトリ（既定：`rawfiles/`）
  - EXIFの撮影日時ではなく、ファイル名順に並べます。撮影順を維持できるファイル名を使ってください。
  - 組み込みRAWローダーは、センサー画素の2×2ブロックを平均して1つの `uint16` 画素にします。対応する設定は `binning: 2` のみです。
- **出力**：
  - `candidates/` 内の候補画像（`-o` で別の保存先も指定可能）
  - `--debug-image` と `--debug-dir` による任意のデバッグマスク
  - 処理の再開に使う `progress.json`

既定の `hough` 検出器は、隣接フレームの絶対差分を求め、しきい値処理、ROIの適用、モルフォロジーのオープニング処理を行います。その後、輪郭の面積・縦横比と、Hough変換で得た線分の長さの合計を調べます。これらは候補を選ぶためのヒューリスティックです。機械学習による分類はロードマップに含まれています。ROI座標、検出する線分の長さや領域の面積は、ビニング後の画像を基準にします。

## 設定ファイル（YAML/JSON）

CLIは設定ファイルからパイプライン設定を読み込めます。ファイルは `PipelineConfig` のフィールドに対応するキーを持つ、JSONまたはYAMLのオブジェクトにしてください。
CLIと `load_pipeline_config()` は一部の設定だけを指定したファイルにも対応し、省略したフィールドには組み込みの既定値を使います。相対パスは設定ファイルのあるディレクトリではなく、現在の作業ディレクトリを基準に解決します。

**最上位のキー**

- `target_folder`、`output_folder`、`debug_folder`（既定：`rawfiles`、`candidates`、`debug_masks`）
- `params`（検出パラメータ）
- `num_workers`、`batch_size`、`auto_batch_size`、`enable_parallel`
- `progress_file`、`output_overwrite`
- `input_loader_name`、`input_loader_config`
- `detector_name`、`detector_config`
- `output_handler_name`、`output_handler_config`
- `hooks`（実行順に並べたフック名・設定の一覧。既定ではフックなし）
- `hook_error_mode`（`raise` または `warn`。既定：`raise`）

**YAMLの例**

```yaml
target_folder: ./rawfiles
output_folder: ./candidates
debug_folder: ./debug_masks
params:
  diff_threshold: 8
  min_area: 10
  min_aspect_ratio: 3.0
input_loader_name: raw
input_loader_config:
  binning: 2
  normalize: true
detector_name: hough
output_handler_name: file
```

RAWローダーの既定値は `normalize: false` です。`true` にすると画素を [0, 1] の範囲の `float32` として返し、パイプラインはそれに合わせて `diff_threshold` を調整します。組み込みHough検出器のしきい値は `params` で指定し、`detector_config` には空の設定を渡せます。ファイル出力ハンドラーの上書き制御には、`overwrite` ではなく `output_overwrite` を使います。

`output_handler_name: file` を明示的に選ぶ場合は、保存先と上書き動作を `output_handler_config` で指定してください。省略したフィールドにはハンドラー自身の既定値を使います。最上位の `output_folder`、`debug_folder`、`output_overwrite` を引き継ぐには、`output_handler_name` を省略し、既定のファイルハンドラーを使ってください。

**設定例ファイル**：[`config_examples/pipeline.yaml`](config_examples/pipeline.yaml)

**CLIでの使用例**

```bash
uv run python detect_meteors_cli.py --config config_examples/pipeline.yaml
```

`--input-loader`、`--detector`、`--output-handler` などのCLIオプションで、使用するプラグインを上書きできます。プラグイン設定はJSON/YAML文字列またはファイルパスで渡せます。従来のパラメータオプション（`--diff-threshold` など）は引き続き `PipelineConfig.params` に反映されますが、今後は設定ファイルへの移行に伴い非推奨になる予定です。

**Pythonでの使用例**

```python
from meteor_core import MeteorDetectionPipeline, load_pipeline_config

config = load_pipeline_config("config_examples/pipeline.yaml")
pipeline = MeteorDetectionPipeline(config)
pipeline.run()
```

## 中断と再開

- Ctrl-Cでいつでも中断できます。
- 同じコマンドを再実行すると再開できます。
- 最初から処理する場合は `--no-resume` を使います。

### 飛行機の光跡解析（任意）

組み込みフックを有効にすると、候補に飛行機の光跡らしさを示す情報を付加します。

```bash
uv run python detect_meteors_cli.py --hooks aircraft_trail --no-roi
```

このフックは処理完了後にフレーム順で線分の形状を追跡し、`progress.json` の `detected_details` にある候補レコードへ `aircraft` 情報を書き込みます。候補判定、スコア、コピー済みRAWファイルは変えません。likelihoodはヒューリスティックなスコアであり、校正された確率ではありません。
再開時に解析するのは、その実行で新たに処理したフレームのみです。前回の進捗から追跡状態は復元しません。設定と制約は[実装説明](docs/aircraft_light_trails_hook_design.md)を参照してください。

再現可能なサンプルとして、[`config_examples/aircraft_trail_sample.yaml`](config_examples/aircraft_trail_sample.yaml)を利用できます。

```bash
uv run python detect_meteors_cli.py \
  --config config_examples/aircraft_trail_sample.yaml \
  --no-roi --no-resume --debug-image
```

Gitで管理している12枚のサンプルには、すべて飛行機が映っています。所有者が流星を確認しているのは `_C140338.ORF` と `_C140344.ORF` です。解析する11組はすべて候補として残ります。このフックは飛行機を除外するフィルターではありません。後者の流星画像はサンプル設定で飛行機のlikelihoodも高くなるため、両方が映った画像も目視確認してください。
RAWファイルとチェックサムは、リポジトリの [`rawfiles/2024GEMINI_AIRCRAFT`](rawfiles/2024GEMINI_AIRCRAFT/README.md) にあります。RAW画像はPythonの配布ファイルには含めません。[使い方と結果の読み方](docs/aircraft_light_trails_hook_design.md#read-the-results)、[サンプルの検証結果](docs/aircraft_sample_validation.md)も参照してください。

## ドキュメント

| 文書 | 内容 |
|------|------|
| [COMMAND_OPTIONS.md](docs/COMMAND_OPTIONS.md) | CLIオプションの全リファレンス |
| [NPF_RULE_ja.md](docs/NPF_RULE_ja.md) | NPF Ruleと焦点距離の扱い |
| [INSTALL_ja.md](docs/INSTALL_ja.md) | インストール手順 |
| [INSTALL_DEV_ja.md](docs/INSTALL_DEV_ja.md) | 開発環境の構築 |
| [PLUGIN_AUTHOR_GUIDE_ja.md](docs/PLUGIN_AUTHOR_GUIDE_ja.md) | プラグインの開発 |
| [飛行機の光跡ガイド](docs/aircraft_light_trails_hook_design.md) | フックの有効化、設定、メタデータの確認 |
| [飛行機サンプルの検証](docs/aircraft_sample_validation.md) | 12枚のサンプルでの結果と制約 |
| [Wiki](https://github.com/shin3tky/detect_meteors/wiki) | 技術的な詳細 |

## v1.6.10の新機能

**リリース日：2026年10月6日（2026-10-06）— 🌠 夢をかなえる日。** 1.6.9はスキップし、v1.6.8の次のリリースをv1.6.10とします。

- **ソート済み検出フック**：時系列順の検出解析に使う新しいパイプラインフック
  - `on_batch_results_sorted`：フレーム順を保証するバッチ単位のフック
  - `on_all_detections_sorted`：フレームをまたいで解析する、パイプライン処理後のフック
- **SortedDetectionデータクラス**：ソート済みフックに使う、メモリ効率のよい軽量コンテナ
- **AircraftTrailHookの改善**：エラー処理、ログ出力、360°の角度正規化により堅牢性を向上

詳しい移行情報は[RELEASE_NOTES_1.6_ja.md](docs/RELEASE_NOTES_1.6_ja.md)を参照してください。

### 以前のリリース

| バージョン | 主な変更 | 詳細 |
|------------|----------|------|
| v1.6.x | スキーマのバージョン管理、MLに備えた設計、uv/Ruff開発環境 | [RELEASE_NOTES_1.6_ja.md](docs/RELEASE_NOTES_1.6_ja.md) |
| v1.5.x | プラグイン構成、センサープリセット、魚眼対応 | [RELEASE_NOTES_1.5.md](docs/RELEASE_NOTES_1.5.md) |
| v1.4.x | NPF Ruleによる最適化、EXIF抽出 | [RELEASE_NOTES_1.4.md](docs/RELEASE_NOTES_1.4.md) |
| v1.3.x | パラメータの自動推定 | [RELEASE_NOTES_1.3.md](docs/RELEASE_NOTES_1.3.md) |
| v1.2.x | しきい値推定の改善 | [RELEASE_NOTES_1.2.md](docs/RELEASE_NOTES_1.2.md) |

全リリース履歴は[CHANGELOG_ja.md](docs/CHANGELOG_ja.md)を参照してください。

## ロードマップ

今後の機能は[ROADMAP.md](docs/ROADMAP.md)を参照してください。

## 作者

Detect Meteors CLIの作者はShinichi Morita（shin3tky）です。

NPF Ruleの実装は、Société Astronomique du Havre（SAH）のFrédéric Michaud氏が開発した数式に基づきます。帰属情報の詳細は[NOTICE](NOTICE)を参照してください。

## 貢献

IssueやPull Requestを歓迎します。大きな変更は、先にIssueを作成して相談してください。

開発環境の構築は[INSTALL_DEV_ja.md](docs/INSTALL_DEV_ja.md)を参照してください。

## ライセンス

このプロジェクトは[Apache License 2.0](LICENSE)で公開しています。
