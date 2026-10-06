# 変更履歴

[English](CHANGELOG.md)

## v1.6.10 - 2026-02-04

- **リリース用パッケージ**：setuptoolsのビルドバックエンドを明示し、インストール可能な `detect-meteors` コマンドとCLIモジュールをwheelに収録。ソース配布版にはドキュメント、設定例、テストを収録。
- **飛行機の光跡の使い方と検証**：設定例、結果の読み方、12枚の `2024GEMINI_AIRCRAFT` サンプルによる検証を追加。飛行機メタデータは、飛行機と流星が両方映った画像を除外しません。
- **ソート済み検出フック**：時系列順の検出解析に使うパイプラインフックを追加。
  - `on_batch_results_sorted(detections) -> List[SortedDetection]`：フレーム順を保証するバッチ単位のフック。
  - `on_all_detections_sorted(detections) -> List[SortedDetection]`：フレームをまたいで解析する、パイプライン処理後のフック。
  - パイプラインが `SortedDetection` を蓄積し、完了後に `frame_index` でソート。
- **SortedDetectionデータクラス**：ソート済みフック用の軽量な検出コンテナ。
  - `debug_image`、`current_image`、`previous_image`、`roi_mask` を含めず、メモリを節約。
  - フィールド：`frame_index`、`prev_frame_index`、`filename`、`filepath`、`is_candidate`、`score`、`aspect_ratio`、`lines`、`extras`。
  - `SortedDetection.from_detection_result()` で変換可能。
  - フックのメタデータを追加できる、変更可能な `extras` 辞書。
- **AircraftTrailHookの改善**：堅牢性と追跡精度を向上。
  - 検出ごとのtry-exceptとログにより、エラー時も可能な範囲で処理を継続。
  - 初期化・完了をDEBUG、エラー報告をWARNINGで記録。
  - 360°の角度正規化と一貫した端点順により、方向を考慮した追跡。
  - フレーム間追跡のため、`on_all_detections_sorted` を使う構成へ変更。
- **プラグイン開発ガイドを更新**：ソート済み検出フック、`SortedDetection`、フックの使い分けを詳しく説明。

## v1.6.8 - 2026-01-07 - 🌿 七草の節句

- **tyによる静的型検査**：Astralの `ty` を導入し、プラグイン開発の安定性を向上。
  - `pyproject.toml` に段階的なルール適用を設定。
  - pre-commitに自動型検査を統合。
  - エラールール：`invalid-return-type`、`invalid-method-override`。
  - 警告ルール：`invalid-assignment`、`call-non-callable`、`too-many-positional-arguments`、`invalid-argument-type`。
- **プラグインの型安全性を改善**：すべてのプラグインレジストリの型検査上の問題を解消。
  - 検出器、フック、入力、出力レジストリで型付きファクトリーキャストを使用。
  - 検証用callableによりメソッドのオーバーライド競合を回避。
  - `OutputWriter.save_candidate` の戻り値型を `BaseOutputHandler` に統一。
  - ROI選択でnumpy画像を使用することを保証。
  - i18nで省略可能なロケールと正規化に対応。
- **ワーカー上限設定**：`MAX_NUM_WORKERS` を追加し、`meteor_core` から公開。
  - パイプライン設定で最大ワーカー数を検証。
  - CLIの `--workers` ヘルプに上限を表示。
- **CIの改善**：GitHub Actionsを `ubuntu-latest` から `ubuntu-slim` に変更し、ビルドを高速化。
- **依存ライブラリを更新**：`pillow` と `ty` を更新。

## v1.6.7 - 2025-12-29 - 🎂 shin3tkyの誕生日

- **ロードマップを具体化**：v2.0/v3.0のマイルストーンを分野別に整理。
  - v2.x「設計と拡張性」：パイプラインのモジュール化、プラグイン環境の拡充、連携と相互運用。
  - v3.x「知能と学習」：MLによる検出、高度な後処理、性能と配備。
- **Pythonのバージョンを明確化**：対応するバージョンを3.12と3.13に明示。
- **ドキュメント**：ROADMAP.mdで今後のバージョンを詳しく説明。

## v1.6.6 - 2025-12-27 - 🧱 レゴ組み立ての日

- **パイプラインフック**：処理段階へ介入できる拡張可能なフック構成を導入。
  - `on_file_found(filepath) -> bool`：読み込み前にファイルを選別（`False` で除外）。
  - `on_image_loaded(context) -> InputContext`：読み込み後の画像変換やメタデータ追加。
  - `on_detection_complete(result, context) -> DetectionResult`：スコアやフラグなどの検出結果を調整。
  - `on_output_saved(result) -> None`：出力保存後に通知。
- **HookRegistry**：フックの検出と管理を集約（`meteor_core.hooks.HookRegistry`）。
  - エントリーポイント：`detect_meteors.hook`。
  - プラグインディレクトリ：`~/.detect_meteors/hook_plugins/`。
  - 実行時登録：テストや単一プロセスでは `HookRegistry.register(MyHook)` を利用可能。
- **フックの基底クラス**：型付き設定に対応する `BaseHook`、`DataclassHook`、`PydanticHook`。
- **CLIのフック指定**：コマンドラインからフックと設定を指定可能。
  - `--hooks`：実行順に並べた、カンマ区切りのフック名。
  - `--hook-config`：JSON/YAML文字列またはファイルパス。
- **PipelineConfigでのフック指定**：Pythonから設定可能。
  - `hooks`：名前と任意の設定辞書を持つ `HookConfig` の順序付き一覧。
  - `hook_error_mode`：エラー時の動作（`"raise"` または `"warn"`）。
- **マルチプロセス対応**：エントリーポイントやプラグインディレクトリ経由のフックをワーカープロセスでも利用可能。
- **プラグイン開発ガイドを更新**：フックのライフサイクル図、登録方法、例を詳しく説明。

## v1.6.5 - 2025-12-26 - 📓 お礼の手紙の日

- **パイプライン設定ファイル**：`--config` でYAML/JSONから設定を読み込み可能。
  - パス、検出パラメータ、プラグイン、ワーカーなど、`PipelineConfig` の全フィールドに対応。
  - Python用の `load_pipeline_config()` を追加。
  - 設定例：[`config_examples/pipeline.yaml`](../config_examples/pipeline.yaml)。
- **CLIのプラグイン設定**：引数から設定可能。
  - 選択：`--input-loader`、`--detector`、`--output-handler`。
  - 設定：`--input-loader-config`、`--detector-config`、`--output-handler-config`。
  - JSON文字列、YAML文字列、ファイルパスに対応。
- **DetectionContextの正規化API**：変換関数の登録と正規化を公開。
  - `register_detection_context_converter()`：バージョン別の変換関数を登録。
  - `normalize_detection_context()`：現行スキーマへ正規化。
  - `InputContext`、`DetectionResult`、`OutputResult` と一貫したAPI。
- **パイプラインへの統合**：CLIの処理を `MeteorDetectionPipeline` に統一。
- **プラグイン開発ガイドを更新**：スキーマのバージョン管理と正規化APIを説明。
- **後方互換性**：`--diff-threshold` などの従来オプションは引き続き動作しますが、設定ファイルへの移行に伴い非推奨。

## v1.6.4 - 2025-12-25 - 🎄 クリスマス

- **出力ハンドラーの `on_detection_result` フック**：シリアライズしたコンテキストとともに、検出ごとに呼ぶコールバックを追加。
  - 線分・extrasを持つ `DetectionResult` と、実行パラメータ、ファイル情報、時刻などのコンテキスト辞書を受け取ります。
  - 呼び出し順を文書化：`on_detection_result` → `on_candidate_image` → `on_debug_image`。
- **DetectionResultの受け渡し**：`process_image_batch()` の結果タプルに `DetectionResult` を追加。
  - 出力ハンドラーから検出器の `lines`、`extras`、`metrics` にアクセス可能。
  - CLIとパイプラインを拡張したタプルに対応。
- **フレーム番号**：パイプライン全体に `frame_index` と `prev_frame_index` を追加。
  - 検出コンテキストのメタデータにも両フィールドを追加。
  - `progress.json` の `detected_details` にも追加。
  - バッチ結果用の `_extract_frame_indices()` を追加。
  - 進捗表示で検出フレームを表示（例：フレーム42、108、215）。
- **デバッグ画像の最適化**：候補の場合のみ画像を生成。
  - 非候補の `DetectionResult.debug_image` をクリア。
  - 大規模バッチ処理のメモリ使用量を削減。
- **性能の改善**：`_build_runtime_params()` をループ外へ移動。
- **バッチ進捗の記録**：後処理用にフレーム番号を `progress.json` へ記録。
- **プラグイン開発ガイドを更新**：フック、コンテキストのシリアライズ、呼び出し順を説明。

## v1.6.3 - 2025-12-24 - 🎅 クリスマスイブ

- **RuntimeParams契約**：`RuntimeParams` として実行時パラメータの受け渡しを正式なデータクラスで定義。
  - バージョン定数 `RUNTIME_PARAMS_SCHEMA_VERSION = 1`。
  - `schema_version`、`global_params`、検出器ごとの上書きを持つ `detector` を収録。
  - シリアライズ用の `to_dict()` を追加。
  - `BaseDetector` に `split_runtime_params()`、`build_runtime_params()`、`detect_legacy()` を追加。
- **DetectionContextのシリアライズ**：ログ・デバッグ用の `to_dict()` を追加（画像とマスクは除外）。
- **パイプライン境界での正規化**：`InputContext`、`DetectionResult`、`OutputResult` を自動正規化。
- **従来の真偽値への対応**：出力ハンドラーの `bool` を `OutputResult` へ自動変換し、非推奨の警告を表示。
- **プラグイン開発ガイドを更新**：RuntimeParamsとパイプライン正規化を詳しく説明。
- **後方互換性**：v1.6.2と完全互換。従来の `bool` 戻り値も警告付きで動作。

## v1.6.2 - 2025-12-23 - 🇯🇵 上皇陛下の誕生日

- **入出力コンテキスト契約**：ローダー・ハンドラーの戻り値を標準化する `InputContext` と `OutputResult` を追加。
  - バージョン定数：`INPUT_CONTEXT_SCHEMA_VERSION = 1`、`OUTPUT_RESULT_SCHEMA_VERSION = 1`。
  - `InputContext`：`image_data`、`filepath`、`metadata`、`loader_info`、`schema_version`。
  - `OutputResult`：`saved`、`output_path`、`debug_path`、`handler_info`、`metrics`、`schema_version`。
  - 両クラスにシリアライズ用の `to_dict()` を追加。
  - 既存プラグインを壊さず、ローダー・ハンドラーを段階的に移行できる仕組み。
- **プラグイン契約の網羅**：入力・検出・出力の3種類すべてをバージョン付き契約に統一。
- **プラグイン開発ガイドを更新**：入出力契約を詳しく説明。
- **後方互換性**：プラグイン構成は実験的です。出力ハンドラーは `OutputResult` を返す必要があり、`bool` を返す従来の `OutputWriter` やハンドラーは、変更せずにはv1.6.2のパイプラインと互換になりません。

## v1.6.1 - 2025-12-22 - 🌃 冬至

- **プラグイン契約のバージョン管理**：将来の移行に備え、`DetectionContext` と `DetectionResult` に `schema_version` を追加。
  - `DETECTION_CONTEXT_SCHEMA_VERSION = 1`、`DETECTION_RESULT_SCHEMA_VERSION = 1`。
  - 既存プラグインを壊さず段階的に移行できる仕組み。
- **複数フレームワークの画像対応**：`numpy.ndarray`、`torch.Tensor`、`PIL.Image.Image` に対応する `ImageLike` 型エイリアス。
  - PyTorch、TensorFlowなどのML検出器に備えた設計。
  - 安全な画像変換用に `meteor_core.utils` の `ensure_numpy()` を追加。
- **DetectionResultの診断情報**：標準的な診断を格納する `metrics` を追加。
  - 推奨キー：`duration_ms`、`num_contours`、`mask_area`、`hough_votes`。
  - シリアライズ用の `to_dict()` を追加。
- **プラグイン開発ガイドを更新**：新しいスキーマ契約とImageLikeの扱いを説明。
- **後方互換性**：v1.6.0と完全互換。既存検出器は変更不要。

## v1.6.0 - 2025-12-21 - 💌 遠距離恋愛の日

- **開発ツールを刷新**：pip/black/flake8からuv/ruffへ移行し、作業を高速化・統一。
  - **uv**：Rust製のPythonパッケージ管理ツール（pipの10〜100倍高速）。
  - **Ruff**：blackとflake8を置き換えるRust製の静的解析・整形ツール。
  - 設定を `pyproject.toml` に統一（`.flake8` を削除）。
  - `.pre-commit-config.yaml` をローカルRuffフックへ変更。
  - `uv sync --all-extras` で開発環境を一括構築。
- **開発者向け文書を更新**：INSTALL_DEV.mdをuv/Ruffの手順に改訂。
- **後方互換性**：CLI、実行時の動作、検出アルゴリズムは変更なし。

## v1.5.13 - 2025-12-19 - ⛄️ 雪だるまを作る日

- **国際化（i18n）**：CLIの利用者向けメッセージを多言語化。
  - 表示言語を選ぶ `--locale`（既定：`en`）。
  - 既定ロケール用の環境変数 `DETECT_METEORS_LOCALE`。
  - 複数形に対応したICU形式テンプレート。
  - `meteor_core/locales/` のYAML翻訳カタログ。
  - 英語（`en`）と日本語（`ja`）に対応。
  - UI/UXを翻訳し、システム・デバッグメッセージは英語を維持。
- **進捗ファイルの正規化**：`normalize_progress_data()` で `progress.json` を整え、読み込み前に合計を再計算。
- **検証の改善**：`validate_and_apply_sensor_preset()` がエラー文を返す代わりに `MeteorValidationError` を直接送出。
- **依存ライブラリを追加**：翻訳カタログ用の `PyYAML`。
- **テスト**：i18n・進捗正規化用に `test_i18n.py` と `test_infrastructure_v1x.py` を追加。
- **シェル補完を更新**：bash/zshの両方に `--locale` を追加。

## v1.5.12 - 2025-12-18 - 🔮 ファイナルファンタジーの日

- **独自の例外階層**：エラー処理と診断のため、構造化された例外クラスを導入。
  - `MeteorError`：診断情報に対応する基底例外。
  - `MeteorLoadError`：画像の読み込み失敗（破損、I/Oエラー）。
  - `MeteorUnsupportedFormatError`：未対応ファイル形式。
  - `MeteorOutputError`：出力操作失敗の基底例外。
  - `MeteorWriteError`：コピー、画像保存、ディレクトリ作成の失敗。
  - `MeteorProgressError`：進捗の読み書き、解析、シリアライズ、検証の失敗。
  - `MeteorValidationError`：パラメータ・入力の検証エラー。
  - `MeteorConfigError`：設定・プラグイン構築のエラー。
- **診断レポート**：システム情報、依存バージョン、コンテキストを持つ `DiagnosticInfo` を追加し、GitHub Issueで報告可能。
- **問題調査用のCLIオプション**：
  - `--verbose`：詳細なエラー診断と、入力処理モジュールのDEBUGログ。
  - `--save-diagnostic [FILE]`：エラー時に報告用の診断ファイルを保存。
- **構造化ログ**：`meteor_core` の `inputs`、`detectors`、`outputs`、`pipeline` などに標準Python loggingを導入。`--verbose` でレベルを設定。
- **わかりやすいエラー表示**：`format_error_for_user()` で対処しやすいメッセージと任意の詳細診断を表示。
- **テスト**：例外とログ用の `test_exceptions_v1x.py`、`test_inputs_logging_v1x.py` を追加。

## v1.5.11 - 2025-12-15 - 📜 権利章典の日

- **パッケージ間の一貫性**：`inputs` と `detectors` のレジストリ動作を統一。
  - `LoaderRegistry` と `DetectorRegistry` で大文字・小文字を区別しない検索に対応（`get("raw")`、`get("RAW")`、`get("Raw")` は同じクラス）。
  - ディレクトリ名を `~/.detect_meteors/input_plugins/`、`~/.detect_meteors/detector_plugins/`、`~/.detect_meteors/output_plugins/` に統一。
  - 両パッケージで検索対象から除くクラスに `Generic`、`DataclassDetector`、`PydanticDetector` などを追加。
- **BaseInputLoaderを拡張**：`name`、`version`、`get_info()` を追加し、`BaseDetector` と同じ方法でメタデータにアクセス可能。
- **設定検証を改善**：`LoaderRegistry._coerce_config()` が失敗時に `None` を返す代わりに `TypeError` または `ValueError` を送出し、早期にわかりやすく検出。
- **関数名を変更**：`discover_input_loaders()` を `discover_detectors()` と揃えて `discover_loaders()` に変更し、旧名を削除。`discover_loaders()` も非推奨のため、`LoaderRegistry.discover()` を推奨。

## v1.5.10 - 2025-12-11 - ⛰️ 国際山岳デー

- **プラグイン構成を移行**：ProtocolからABC（抽象基底クラス）へ変更。
  - `InputLoader` → `BaseInputLoader`（ABC）。
  - `MetadataExtractor` → `BaseMetadataExtractor`（ABC）。
  - `OutputHandler` → `BaseOutputHandler`（ABC）。
- **即時のエラー検出**：実行時に検出するProtocolに対し、ABCは抽象メソッドの実装漏れをインスタンス生成時に検出。
- **IDE支援を改善**：補完と未実装メソッドの警告に対応。
- **明確な継承関係**：プラグイン契約を見つけやすい明示的なクラス階層。
- **開発者向け文書**：`INSTALL_DEV.md` に入力・検出・出力の独自プラグイン例を含む説明を追加。
- **補足**：プラグイン構成は実験的であり、v2.0の安定版までに変更される可能性があります。

## v1.5.9 - 2025-12-10 - 👤 人権デー

- **PEP 621対応**：名前、説明、作者、キーワード、分類などのメタデータを `pyproject.toml` へ移行。
- **ツール設定の統一**：flake8-pyprojectを使い、`.flake8` の設定を `pyproject.toml` に統合。
- **依存管理**：実行時依存と任意の開発依存を `pyproject.toml` に定義。
- **テスト環境**：テストとカバレッジの設定を `pyproject.toml` に追加。
- **プロジェクトURL**：ホームページ、リポジトリ、ドキュメント、Issue、変更履歴へのリンクをメタデータに追加。

## v1.5.8 - 2025-12-09 - ♿️ 障害者の日

- **品質管理ツール**：Blackを補うflake8と、プロジェクト用の `.flake8` 設定を追加。
- **静的解析の基準**：Black互換の除外ルール（E203、W503、E501、E226）、複雑度上限70、除外パターンを設定。
- **開発手順**：コミット前の `flake8 .` などの手動静的解析を組み込み、品質を統一。

## v1.5.7 - 2025-12-08 - 🪡 針供養

- **進捗メタデータを拡充**：`progress.json` にCLIパラメータ（`params`）、ROI（`roi`）、確定した処理パラメータ（`processing_params`）を記録し、確認・再実行時に参照可能。
- **パイプラインの一貫性**：CLIと `PipelineConfig` 経由の実行で同じメタデータを保存。

## v1.5.6 - 2025-12-07 - ☃️ 大雪

- **入力ローダーのプラグイン化**：`InputLoader`/`MetadataExtractor` Protocol、dataclass/Pydanticの基底ヘルパー、組み込み `RawImageLoader` を追加。`detect_meteors.input` エントリーポイントとローカルの `~/.detect_meteors/plugins` で、決定的な順序のローダー検出に対応。
- **パイプライン設定オブジェクト**：`PipelineConfig` で実行設定を集約し、`DetectionPipeline` Protocolとローダー・メタデータ抽出を解決するヘルパーを提供。
- **出力の拡張性**：v2.0に向け、候補・デバッグ画像の保存を定義する `OutputHandler` Protocolを導入。

## v1.5.5 - 2025-12-05 - 👖 ブルージーンズの日

- **コード構成のリファクタリング**：v2.xのプラグイン構成に備え、モジュールへ分割。
  - `detect_meteors_cli.py`：引数解析と利用者との対話を行うCLI。
  - `meteor_core/`：中核ロジック。
    - `schema.py`：型定義とデータ構造。
    - `pipeline.py`：処理パイプラインの制御。
    - `image_io.py`：RAW読み込みとEXIF抽出。
    - `roi_selector.py`：ROI選択。
    - `utils.py`：ユーティリティ。
    - `detectors/`：検出アルゴリズム。
      - `base.py`：検出器の抽象基底クラス。
      - `hough_default.py`：標準のHough変換検出器。
    - `outputs/`：出力処理。
      - `writer.py`：結果ファイルの書き込み。
- **型安全性を改善**：TypedDictで構造化データの型ヒントを強化。
- **後方互換性**：CLIは変更せず、既存コマンドはそのまま動作。

## v1.5.4 - 2025-12-03 - 👩 主婦の日

- **ROI選択表示を改善**：暗い画像でも見やすいように表示を明るく調整。
- **NOTICE**：第三者ライセンスの帰属・謝辞を記すNOTICEを追加。

## v1.5.3 - 2025-12-02 - 🪒 安全カミソリの日

- **魚眼レンズ補正**：等立体角射影の補正に使う `--fisheye` を追加。
  - 画像位置によって変化する有効焦点距離を考慮。
  - 中央は公称焦点距離を維持。
  - 周辺では有効焦点距離が低下（対角画角180°でcos(45°) ≈ 0.707倍）。
  - 周辺ほど星の光跡が長くなり、隅では最大約1.414倍。
- **魚眼用のNPF計算**：周辺の最悪条件の有効焦点距離を使用。
  - より余裕のあるNPF推奨露光時間。
  - 広角魚眼撮影での適合度評価を改善。
- **魚眼関数を追加**：
  - `calculate_fisheye_effective_focal_length()`：位置に応じた焦点距離。
  - `calculate_fisheye_edge_focal_length()`：NPF用の周辺焦点距離。
  - `calculate_fisheye_trail_length_ratio()`：画像内の光跡長の変化。
  - `get_fisheye_max_trail_ratio()`：周辺の最大光跡長比。
  - `display_fisheye_info()`：補正パラメータの表示。
- **射影モデルの基盤**：等距離射影や立体射影などの将来の拡張に対応。
  - 現在は等立体角射影のみを実装。
  - `FISHEYE_PROJECTION_MODELS` 辞書で拡張可能。
- **シェル補完を更新**：bash/zshに `--fisheye` を追加。
- **テスト**：27件の `test_fisheye_v1x.py` を追加。

## v1.5.2 - 2025-12-01 - ♥️️ いのちの日

- **センサー設定の上書き検証**：`--sensor-type` の値を `--sensor-width` や `--pixel-pitch` で上書きしたときに自動検証。
  - `--sensor-width` がプリセットから±30%を超えると警告。
  - `--pixel-pitch` がプリセットから±50%を超えると警告。
  - 警告は情報提供のみで、通常どおり処理を継続。
  - 柔軟性を保ちながら設定ミスを発見しやすく改善。
- **apply_sensor_preset()を拡張**：検証用に、パラメータとセンサープリセット辞書を返します。
- **テスト**：23件の `test_sensor_validation_v1x.py` を追加。
- **既存テストを更新**：`apply_sensor_preset()` の5要素タプルに合わせて `test_sensor_presets_v1x.py` を修正。

## v1.5.1 - 2025-11-30 - 📷 オートフォーカスカメラの日

- **中判センサーに対応**：35mmフルサイズより大きいセンサーを追加。
  - `MF44X33`：Fujifilm GFX、Pentax 645Z、Hasselblad X2D/X1D（43.8×32.9mm、クロップ係数0.79）。
  - `MF54X40`：Hasselblad H6D-100c（53.4×40mm、クロップ係数0.64）。
- **センサー順を整理**：`--sensor-type` を小さい順に変更：1INCH → MFT → APSC → APSC_CANON → APSH → FF → MF44X33 → MF54X40。
- **シェル補完を更新**：bash/zshに中判タイプを追加。

## v1.5.0 - 2025-11-29 - 🍖 いい肉の日

- **センサータイプのプリセット**：`DEFAULT_SENSOR_WIDTHS` と `CROP_FACTORS` を、代表的な画素ピッチも持つ `SENSOR_PRESETS` に統合。
- **--sensor-typeを追加**：MFT、APS-C、APS-C_CANON、APS-H、FF、1INCHに応じ、`--focal-factor`、`--sensor-width`、`--pixel-pitch` を一括設定。
- **`--list-sensor-types` を追加**：焦点距離係数、センサー幅、画素ピッチを含む一覧を表示して終了。
- **上書きの優先順**：`--focal-factor`、`--sensor-width`、`--focal-length`、`--pixel-pitch` の個別指定をプリセットより優先。
- **ヘルパー関数**：Pythonから設定できる `get_sensor_preset()` と `apply_sensor_preset()`。
- **シェル補完を更新**：bash/zshに新しい両オプションを追加。
- **後方互換性**：v1.4.xと完全互換。従来オプションと、旧コード用の `CROP_FACTORS`、`DEFAULT_SENSOR_WIDTHS` 辞書を維持。

## v1.4.2 - 2025-11-25 - 🙏 先生ありがとうの日

- **出力ファイルを保護**：既存の保存先ファイルを上書きせず、スキップする動作へ変更。
- **`--output-overwrite` を追加**：明示的に指定した場合のみ上書き可能。
- **安全性の確認**：`--target` と `--output` が同じディレクトリなら警告して終了し、意図しないデータ消失を防止。

## v1.4.1 - 2025-11-24 - 🧍 進化の日

- **NPF Ruleによる科学的な最適化**：露光時間の検証とパラメータ最適化を実装し、物理に基づく検出の基盤を構築。
- **EXIFの統合**：埋め込みサムネイル → PIL → rawpyの複数手法で、ISO、露光時間、絞り、焦点距離、解像度を自動抽出。
- **センサーの特性計算**：センサー幅・画像解像度から画素ピッチを計算し、直接指定やセンサータイプ（MFT、APS-C、FF）の指定にも対応。
- **星の光跡の物理推定**：視野と赤緯を考慮し、自転の恒星時速度15°/時で星の移動を推定。
- **撮影品質の評価**：NPF適合度60%、ISO25%、焦点距離15%の重みで0.0〜1.0を算出。EXCELLENT/GOOD/FAIR/POORに分類。
- **パラメータ最適化を強化**：
  - `diff_threshold`：ISO800から倍増するたびに2倍、NPF超過率1.5倍を超えた後は1倍分の増加ごとに1.5倍で調整。
  - `min_area`：星の光跡長、焦点距離（広角0.7倍、望遠1.3倍）、NPFで調整。
  - `min_line_score`：星の3倍という流星速度の仮定を使い、焦点距離と露光時間で調整。
- **詳しい解析表示**：NPF適合度、画素ピッチ、星の光跡推定、品質スコアとパラメータ調整の根拠を表示。
- **CLIオプションを追加**：
  - `--sensor-width`：センサーの物理幅（mm）。
  - `--pixel-pitch`：画素ピッチ（μm）の直接指定。
  - `--show-npf`：NPF解析を表示して終了。
  - `--show-exif`：EXIFを表示して終了。
- **実画像で検証**：OM Digital OM-1（MFT、24mm、ISO1600、5秒）で、確認済み流星2枚を含む候補9枚、検出率100%、品質1.00（EXCELLENT）を達成。
- **後方互換性**：v1.3.1と完全互換。EXIFがない場合は画像ベースの推定へフォールバック。

## v1.3.1 - 2025-11-23 - 🙏 勤労感謝の日

- **全パラメータの自動推定**：`--auto-params` を `diff_threshold`（v1.2.1）、新しい `min_area`、新しい `min_line_score` の3つに拡張。
- **星の大きさの分布解析**：サンプル画像で98パーセンタイルの輝度しきい値を用いて星を検出し、75パーセンタイル×2.0の頑健な式で `min_area` を推定。
- **画像形状に基づくスコア**：対角線の2.5%を基準に `min_line_score` を自動推定し、レンズに応じた焦点距離補正も可能。
- **焦点距離対応**：広角14mm、標準24mm、望遠50mm以上に合わせて光跡長を最適化する `--focal-length` を追加。
- **進捗管理を復元**：v1.1.0の `progress.json` による再開処理を再統合。安全なCtrl-C中断と、ハッシュによるパラメータの自動検証に対応。
- **未公開v1.3.0からの重要な修正**：
  - 焦点距離補正の逆転を修正（除算すべき箇所で乗算していた）。
  - 実流星データに基づき、基本係数を4%から2.5%へ変更。
  - 星の検出しきい値を95から98パーセンタイルへ変更し、2〜100画素²の大きさで選別。
- **実画像で検証**：OM Digital OM-1（24mm、ISO1600、5秒）で、自動パラメータによる流星検出率100%（2/2）を達成。

## v1.2.1 - 2025-11-22 - 🍣 回転寿司の日

- **自動推定を改善**：夜空の偏った輝度分布に対応するため、`diff_threshold` を3σからパーセンタイル方式へ変更。
- **実画像で検証**：通常の推定しきい値を25から15へ下げ、流星の検出感度を向上。
- **統計表示を拡張**：98・99パーセンタイルと、98パーセンタイル、平均+1.5σ、中央値×3の3方式の内訳を表示。
- **しきい値範囲を最適化**：実画像の結果に基づき、制限範囲を4〜25から3〜18へ変更。

## v1.2.0 - 2025-11-22（未公開）- 😱 考えなしの月曜日

- **新機能：パラメータ自動推定**：ROI統計から最適な `diff_threshold` を推定する `--auto-params` を追加。
- 最初のサンプル5枚で3σ方式による初期の自動調整を実装。
- 推定時に平均、標準偏差、中央値、パーセンタイルを表示。
- `--auto-params` と明示的な値を併用した場合は、手動指定を優先。

## v1.1.0 - 2025-11-22 - 👩‍❤️‍👨 いい夫婦の日

- 処理・検出件数をJSONに保存し、パラメータが一致した場合のみ再開する進捗管理を追加。
- `--progress-file`、`--no-resume`、`--remove-progress` で進捗ファイルを管理・初期化。
- `Ctrl-C` で進捗を失わず中断し、長い処理を後から再開可能。

## v1.0.3 - 2025-11-21 - 📺 世界テレビ・デー

- 任意のRAW事前検証と進捗表示を追加し、破損した入力を処理前に除外。
- 空きメモリに基づくバッチサイズ自動調整（`psutil` が必要）と、所要時間の計測オプションを追加。
- 既定の入力フォルダを `rawfiles` にし、READMEのCLIリファレンスを拡充。

## v1.0.2 - 2025-11-20 - 🍕 ピザの日

- Hough変換パラメータと最小線分スコアに、既定値を含むヘルプ説明を追加。

## v1.0.1 - 2025-11-20 - 🧒 世界子どもの日

- 初回リリース。
