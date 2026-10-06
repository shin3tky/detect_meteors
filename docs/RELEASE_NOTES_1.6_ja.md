# バージョン1.6 リリースノート

[English](RELEASE_NOTES_1.6.md)

## バージョン1.6.10（2026-10-06）- 🌠 夢をかなえる日

1.6.9はスキップし、v1.6.8の次のリリースをv1.6.10とします。

### ソート済み検出フックと飛行機の光跡メタデータ

v1.6.10ではフレーム順の解析フックと、画像データを含まない軽量な `SortedDetection` 契約を追加しました。

- `on_batch_results_sorted(detections)` は、バッチの出力・進捗を記録した後にメインプロセスで実行します。バッチ内を `frame_index` 順に並べますが、並列処理のバッチは完了順に届きます。逐次処理では1組ずつ渡します。
- `on_all_detections_sorted(detections)` は `OutputHandler.on_pipeline_complete()` の後にメインプロセスで実行し、その実行で処理した結果をバッチ横断で並べます。非候補も含みますが、フレーム番号のない失敗結果は除外します。再開時に過去の結果は復元せず、Ctrl-Cではこの最終解析を行いません。
- `SortedDetection` はフレーム番号、パス、候補フラグ、スコア、縦横比、線分、extras、スキーマバージョンを保持し、画像やROIの配列は含みません。
- 組み込み `aircraft_trail` は最終ソート済みフックで形状を追跡し、フレームごとのエラー処理と角度正規化を行います。候補判定を変えず、likelihood、追跡ID、根拠を追加します。
- 最終解析後、進捗管理が `extras["aircraft"]` を `progress.json` の `detected_details` の既存候補へ反映します。ほかのextrasは組み込みライターでは自動保存しません。

### 使い方

```bash
uv run python detect_meteors_cli.py --hooks aircraft_trail --no-roi
```

フックは任意です。フックを設定していない既存設定では実行しません。独自のソート済みフックは検出一覧を返し、注釈には `extras` を使ってください。この時点では出力ファイルと候補件数を記録済みです。飛行機フックの `likelihood_threshold` は現在未使用で、指定しても除外は行いません。

設定と呼び出し順は[プラグイン開発ガイド](PLUGIN_AUTHOR_GUIDE_ja.md#110-batch-results-sorted-hook-pipeline)、[飛行機フックの実装説明](aircraft_light_trails_hook_design.md)を参照してください。

### 飛行機サンプルと配布ファイル

Gitで管理する `2024GEMINI_AIRCRAFT` の12枚にはすべて飛行機が映り、所有者が流星を確認しているのは `_C140338.ORF` と `_C140344.ORF` のみです。[設定例](../config_examples/aircraft_trail_sample.yaml)で追跡の許容値を調整できます。両流星画像は候補に残り、`_C140344.ORF` の飛行機likelihoodも高くなります。両方が映った画像を、このメタデータだけで丸ごと除外しないでください。[検証結果](aircraft_sample_validation.md)を参照してください。

v1.6.10のwheelにはCLIモジュールを含め、`detect-meteors` をインストールします。ソース配布版には文書、設定例、テストを含めます。Python配布版にはRAW画像や生成済み出力は含めず、サンプルRAWはリポジトリにあります。[インストール](INSTALL_ja.md)、[リリースのビルド手順](INSTALL_DEV_ja.md#building-a-release)を参照してください。

## バージョン1.6.8（2026-01-07）🌿

### 🔍 tyによる静的型検査

v1.6.8ではAstralの `ty` を開発ツールに導入し、安定したプラグイン開発の基盤を整えました。すべてのプラグインレジストリの型検査上の問題を解消し、外部プラグインの開発を支援します。

### 主な変更

- **tyを導入**：Ruffとuvの開発元であるAstralの、Rust製静的型検査ツール。
- **プラグインの型安全性**：検出器、フック、入力、出力のレジストリを修正。
- **ワーカー上限**：`MAX_NUM_WORKERS` と、パイプライン・CLIでの検証。
- **CIの最適化**：GitHub Actionsを `ubuntu-slim` へ変更して高速化。

### 変更の理由

v2.0に向けてプラグイン環境が成熟するにつれ、型安全性が重要になります。

| 項目 | 変更前（v1.6.7） | 変更後（v1.6.8） |
|------|------------------|------------------|
| **型検査** | 実行時のみ | tyによる静的解析 |
| **レジストリ** | ジェネリック型の問題 | 型付きファクトリーキャスト |
| **メソッドのオーバーライド** | 競合の可能性 | 検証用callable |
| **pre-commit** | Ruffのみ | Ruff + ty |

利点：

1. **早期のエラー検出**：実行前に型エラーを発見。
2. **開発時の安心感**：独自プラグインの型契約を明確化。
3. **IDE支援**：補完とエラー表示を改善。
4. **ドキュメント**：型そのものが説明になります。

### tyの設定

`pyproject.toml` でルールを段階的に適用します。

**エラールール**（修正必須）：

| ルール | 説明 |
|--------|------|
| `invalid-return-type` | 戻り値の型が宣言と不一致 |
| `invalid-method-override` | 基底クラスと互換性のないオーバーライド |

**警告ルール**（確認を推奨）：

| ルール | 説明 |
|--------|------|
| `invalid-assignment` | 代入値が宣言した型と不一致 |
| `call-non-callable` | 呼び出し可能でないオブジェクトを呼び出した |
| `too-many-positional-arguments` | 仮引数より多い位置引数 |
| `invalid-argument-type` | 引数の型が仮引数と不一致 |

### プラグインレジストリの改善

すべてのレジストリで型付きファクトリーキャストと検証用callableを使います。

```python
# Before: Generic typing issues in registries
class DetectorRegistry(PluginRegistryBase):
    def _validate_plugin(self, plugin_class):
        # Potential type conflicts with base class
        ...

# After: Typed factory pattern
class DetectorRegistry(PluginRegistryBase[Type[BaseDetector]]):
    def __init__(self):
        super().__init__(validator=self._validate_detector)
    
    def _validate_detector(self, plugin_class: type) -> bool:
        # Clean validation without override conflicts
        return issubclass(plugin_class, BaseDetector)
```

**各レジストリの変更**：

| レジストリ | 変更 |
|------------|------|
| `DetectorRegistry` | 型付きキャスト、検証用callable |
| `LoaderRegistry` | 型付きキャスト、検証用callable |
| `OutputHandlerRegistry` | 型付きキャスト、検証用callable |
| `HookRegistry` | 型付きキャスト、フック検証の型付け |

### OutputWriterの整合

`OutputWriter.save_candidate` は引数を明示し、`BaseOutputHandler` に合わせて `OutputResult` を返します。

```python
# Before: Implicit return type
def save_candidate(self, source_path, filename, ...) -> OutputResult:
    # ... save logic ...
    return OutputResult(saved=True, output_path=dest_path)

# After: Explicit parameter alignment with base class
def save_candidate(
    self,
    source_path: str,
    filename: str,
    output_folder: str,
    score: float,
    aspect_ratio: float,
    debug_image: Optional[np.ndarray] = None,
    debug_folder: Optional[str] = None,
) -> OutputResult:
    # Full signature matches BaseOutputHandler
    ...
```

### ROI選択の型安全性

ROI選択でnumpy画像を使うことを保証します。

```python
# Pipeline ensures numpy array before ROI selection
from meteor_core.utils import ensure_numpy

image_data = ensure_numpy(input_context.image_data)
roi_mask = select_roi(image_data)  # Guaranteed np.ndarray
```

### i18nのロケール処理

省略可能なロケールと正規化に対応します。

```python
# Before: Strict locale matching
def get_message(key: str, locale: str) -> str:
    ...

# After: Optional locale with fallback normalization
def get_message(key: str, locale: Optional[str] = None) -> str:
    normalized_locale = normalize_locale(locale)  # "en_US" -> "en"
    ...
```

### MAX_NUM_WORKERSの設定

`meteor_core` から新しい定数を公開します。

```python
from meteor_core import MAX_NUM_WORKERS

# Default: 16 (reasonable limit for most systems)
# Pipeline config validation enforces this limit
# CLI --workers help text displays the limit
```

**PipelineConfigでの検証**：

```python
@dataclass
class PipelineConfig:
    num_workers: int = 4
    
    def __post_init__(self):
        if self.num_workers > MAX_NUM_WORKERS:
            raise MeteorConfigError(
                f"num_workers ({self.num_workers}) exceeds MAX_NUM_WORKERS ({MAX_NUM_WORKERS})"
            )
```

### 開発手順

#### tyの実行

```bash
# Run ty type checker
uv run ty check

# Run via pre-commit
uv run pre-commit run ty-check --all-files
```

#### pre-commitの設定

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      # ... ruff hooks ...
      - id: ty-check
        name: ty type check
        entry: .venv/bin/ty check
        language: system
        types: [python]
        pass_filenames: false
```

### 変更ファイル

| ファイル | 変更 |
|----------|------|
| `pyproject.toml` | `[tool.ty]` 設定、依存の更新 |
| `.pre-commit-config.yaml` | ty-checkフック |
| `meteor_core/detectors/registry.py` | 型付きキャスト、検証用callable |
| `meteor_core/inputs/registry.py` | 型付きキャスト、検証用callable |
| `meteor_core/outputs/registry.py` | 型付きキャスト、検証用callable |
| `meteor_core/hooks/registry.py` | 型付きキャスト、フック検証の型付け |
| `meteor_core/outputs/writer.py` | `save_candidate` の戻り値を統一 |
| `meteor_core/pipeline.py` | ROIでnumpyを保証、MAX_NUM_WORKERS検証 |
| `meteor_core/i18n.py` | 省略可能なロケールと正規化 |
| `meteor_core/__init__.py` | `MAX_NUM_WORKERS` を公開 |
| `.github/workflows/python-test.yml` | `ubuntu-slim` へ変更 |
| `CHANGELOG.md` | v1.6.8を追加 |
| `README.md` | 新機能の説明を更新 |

### 後方互換性

✅ v1.6.7以前と **完全な後方互換性** があります。

- **CLI**：変更なし。
- **実行時**：検出の動作は変更なし。
- **API**：既存コードはそのまま動作。
- **プラグイン**：そのまま動作し、型の改善は内部のみ。

### プラグイン作者向け移行ガイド

**移行作業は不要です。** 公開APIを変えず、内部の型安全性を改善しています。

**推奨事項**：

1. **プラグインにtyを実行** し、型エラーを早期に発見します。

   ```bash
   uv add ty --dev
   uv run ty check your_plugin/
   ```

2. **型ヒントを明示** します。

   ```python
   from meteor_core.schema import DetectionContext, DetectionResult
   
   def detect(self, context: DetectionContext) -> DetectionResult:
       # ty will validate your implementation
       ...
   ```

3. オーバーライド時は **基底クラスのシグネチャと正確に一致** させ、`invalid-method-override` を防ぎます。

---

## バージョン1.6.7（2025-12-29）🎂

### 📋 ロードマップの具体化とPythonバージョンの明確化

v1.6.7ではv2.0/v3.0のロードマップを分野別に詳しく整理し、対応するPythonバージョンを明確にしました。

### 主な変更

- **ロードマップ**：v2.0/v3.0のマイルストーンを分野別に整理。
- **Pythonバージョン**：3.12と3.13を明示。

### 変更の理由

v1.6.xの完了とv2.0の開発に向け、利用者と貢献者が方針を理解できるようにします。

| バージョン | 主題 | 分野 |
|------------|------|------|
| **v2.x** | 設計と拡張性 | モジュール化、プラグイン、連携と相互運用 |
| **v3.x** | 知能と学習 | ML検出、高度な後処理、性能と配備 |

### v2.xロードマップ：設計と拡張性（2026年第1四半期〜）

パイプライン全体のモジュール化とプラグイン環境の拡充を進めます。

**パイプラインのモジュール化**

- 検出器の連結・フォールバック順を含む、差し替え可能な検出器群。
- ノイズ除去、マスク、ROI変換などの前処理・後処理プラグイン。
- 名前付き設定と上書きによるプリセット・プロファイル。
- バージョン付きパイプラインスキーマと移行ヘルパー。

**プラグイン環境の拡充**

- 外部プラグイン用のSDKテンプレートと検証ツール。
- プラグイン契約のバージョン互換表。
- 対応機能・要件の宣言による機能検出。
- 配布ガイドとプラグイン例の一覧。

**連携と相互運用**

- COCO、YOLO、CSV/Parquetなどの注釈形式への出力。
- S3/GCS/Azureなどのリモートストレージ連携フック。
- マルチプロセス・キューベースのバッチ実行ヘルパー。

### v3.xロードマップ：知能と学習（2026年第2四半期〜）

機械学習で検出精度を高めます。

**MLによる検出**

- 基本ML検出器の統合（任意で利用し、既定にはしない）。
- ラベル付きデータの取り込みと注釈ツール。
- 再現可能な設定を使う学習・評価CLI。
- モデルレジストリとバージョン付きモデル選択。

**高度な後処理**

- 流星とノイズを区別する高度なパターン認識。
- 利用者のフィードバックによる適応学習と誤検出の抑制。
- 流星、飛行機、人工衛星などの複数対象分類。

**性能と配備**

- ONNX、GPUなどによる推論の高速化。
- ストリーミング・準リアルタイム検出。
- エッジ環境向けの軽量モデル。

### 対応するPythonバージョン

次の対応範囲を明示しました。

| バージョン | 対応 |
|------------|------|
| 3.11以前 | ❌ 未対応 |
| 3.12 | ✅ 対応 |
| 3.13 | ✅ 対応 |
| 3.14以降 | ❌ 現時点では未対応 |

`pyproject.toml` は `requires-python = ">=3.12,<3.14"` と指定しています。

### 変更ファイル

| ファイル | 変更 |
|----------|------|
| `ROADMAP.md` | v2.x/v3.xのマイルストーンを詳しく整理 |
| `CHANGELOG.md` | v1.6.7を追加 |
| `README.md` | 新機能の説明を更新 |

### 後方互換性

✅ v1.6.6以前と **完全な後方互換性** があります。

- **CLI**：変更なし。
- **実行時**：検出動作は変更なし。
- **API**：変更なし。
- **プラグイン**：変更なし。

ドキュメントのみのリリースで、コードの変更はありません。

---

## バージョン1.6.6（2025-12-27）🧱

### 🪝 パイプラインフック

v1.6.6では処理の主要な段階へ介入するフックを導入しました。中核パイプラインを変更せずに、ファイル選別、画像の前処理、検出結果の調整、保存後の通知ができます。

### 主な変更

- **パイプラインフック**：検出のライフサイクルをカバーする4つのフック。
- **HookRegistry**：フックプラグインの検出と管理。
- **マルチプロセス対応**：エントリーポイント・ディレクトリ経由のフックをワーカーでも利用可能。
- **エラー処理**：`hook_error_mode` で停止・警告を設定。
- **CLIとの統合**：`--hooks` と `--hook-config` で実行時に設定。

### 変更の理由

入力・検出・出力のどれか1つに収まりにくい補助処理の拡張口を提供します。

| 用途 | フック | 説明 |
|------|--------|------|
| **ファイル選別** | `on_file_found` | パターン、拡張子、メタデータで除外 |
| **画像前処理** | `on_image_loaded` | 補正、ノイズ除去、形式変換 |
| **結果の調整** | `on_detection_complete` | スコア調整、誤検出除外、メタデータ追加 |
| **通知** | `on_output_saved` | 通知、計測値の記録、後続処理の起動 |

利点：

1. **役割の分離**：補助ロジックを中核プラグインから分離。
2. **組み合わせ**：複数フックを順に連結。
3. **再利用**：異なるパイプライン設定で同じフックを使用。
4. **テスト**：フックを独立して検証可能。

### フックのライフサイクル

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                 Detection Pipeline (with Hook insertion points)              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. Collect Files ──▶ Hook: on_file_found ──▶ Filter files                   │
│                                                                              │
│  2. Load Image ──────▶ Hook: on_image_loaded ──▶ Transform/enrich            │
│                                                                              │
│  3. Detect ──────────▶ Hook: on_detection_complete ──▶ Adjust results        │
│                                                                              │
│  4. Save Output ─────▶ Hook: on_output_saved ──▶ Notify/log                  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### フックAPIリファレンス

| フック | シグネチャ | 戻り値 | 説明 |
|--------|------------|--------|------|
| `on_file_found` | `(filepath: str)` | `bool` | `True` で保持、`False` で除外 |
| `on_image_loaded` | `(context: InputContext)` | `InputContext` | 更新したコンテキスト |
| `on_detection_complete` | `(result: DetectionResult, context: DetectionContext)` | `DetectionResult` | 更新した結果 |
| `on_output_saved` | `(result: OutputResult)` | `None` | 読み取り専用の通知 |

### フックを作成する

```python
from dataclasses import dataclass
from meteor_core.hooks import DataclassHook, HookRegistry
from meteor_core.schema import InputContext, DetectionResult, DetectionContext

@dataclass
class MyHookConfig:
    min_score_threshold: float = 50.0

class ScoreFilterHook(DataclassHook[MyHookConfig]):
    plugin_name = "score_filter"
    name = "Score Filter Hook"
    version = "1.0.0"
    ConfigType = MyHookConfig

    def on_detection_complete(
        self,
        result: DetectionResult,
        context: DetectionContext,
    ) -> DetectionResult:
        # Adjust is_candidate based on custom threshold
        if result.score < self.config.min_score_threshold:
            return DetectionResult(
                is_candidate=False,
                score=result.score,
                lines=result.lines,
                aspect_ratio=result.aspect_ratio,
                debug_image=result.debug_image,
                extras={**result.extras, "filtered_by": "score_filter"},
            )
        return result

# Register for runtime use (single-process only)
HookRegistry.register(ScoreFilterHook)
```

### フックの検出

3つの方法で検出します。

| 方法 | 場所 | マルチプロセス |
|------|------|----------------|
| **エントリーポイント** | `pyproject.toml` の `detect_meteors.hook` | ✅ 対応 |
| **プラグインディレクトリ** | `~/.detect_meteors/hook_plugins/*.py` | ✅ 対応 |
| **実行時登録** | `HookRegistry.register(MyHook)` | ❌ 単一プロセスのみ |

**エントリーポイントの例**（`pyproject.toml`）：

```toml
[project.entry-points."detect_meteors.hook"]
my_hook = "my_package.hooks:MyHook"
```

### CLIでの使い方

```bash
# Specify hooks by name (comma-separated, in execution order)
uv run python detect_meteors_cli.py --hooks score_filter,logger_hook

# Provide hook configuration
uv run python detect_meteors_cli.py \
    --hooks score_filter \
    --hook-config '{"score_filter": {"min_score_threshold": 75.0}}'

# Or via file
uv run python detect_meteors_cli.py \
    --hooks score_filter \
    --hook-config hooks_config.yaml
```

### Python API

```python
from meteor_core.schema import PipelineConfig, HookConfig

config = PipelineConfig(
    target_folder="./raw",
    output_folder="./candidates",
    debug_folder="./debug",
    hooks=[
        HookConfig(name="score_filter", config={"min_score_threshold": 75.0}),
        HookConfig(name="logger_hook"),
    ],
    hook_error_mode="warn",  # or "raise" (default)
)
```

### エラー処理

`hook_error_mode` でフックの例外に対する動作を設定します。

| モード | 動作 |
|--------|------|
| `"raise"`（既定） | エラー時にパイプラインを停止 |
| `"warn"` | 警告を記録して継続（実運用で推奨） |

### スキーマ変更

**HookConfig**（新しいデータクラス）：

```python
@dataclass
class HookConfig:
    name: str                                    # Hook plugin name
    config: Optional[Dict[str, Any]] = None      # Hook-specific configuration
```

**PipelineConfigへの追加**：

```python
@dataclass
class PipelineConfig:
    # ... existing fields ...
    hooks: Optional[List[HookConfig]] = None     # Ordered hook list (None = skip hooks)
    hook_error_mode: str = "raise"               # "raise" or "warn"
```

### 変更ファイル

| ファイル | 変更 |
|----------|------|
| `meteor_core/hooks/__init__.py` | フック基盤の新パッケージ |
| `meteor_core/hooks/base.py` | `BaseHook`、`DataclassHook`、`PydanticHook` |
| `meteor_core/hooks/registry.py` | 検出・管理用の `HookRegistry` |
| `meteor_core/hooks/discovery.py` | フックの検出方法 |
| `meteor_core/schema.py` | `HookConfig`、`hooks`、`hook_error_mode` |
| `meteor_core/pipeline.py` | 各段階でのフック呼び出し |
| `detect_meteors_cli.py` | `--hooks`、`--hook-config` |
| `COMMAND_OPTIONS.md` | 新しいオプションの説明 |
| `PLUGIN_AUTHOR_GUIDE.md` | フックの詳しい説明 |

### 後方互換性

✅ v1.6.5以前と **完全な後方互換性** があります。

- **CLI**：既存オプションはそのまま動作。
- **実行時**：検出の動作は変更なし。
- **API**：既存コードは変更不要。フックは明示的に有効化。
- **プラグイン**：変更不要。

`hooks` が既定の `None` の場合、従来どおりの動作となり、フック処理の追加負荷はありません。

---

## バージョン1.6.5（2025-12-26）📓

### 🔧 パイプライン設定ファイルとCLIのプラグイン選択

v1.6.5ではYAML/JSON設定ファイルとCLIでのプラグイン選択を導入しました。プラグインを含むパイプライン全体を外部設定でき、v2.0の構成へ向けた基盤になります。

### 主な変更

- **設定ファイル**：`--config` で全設定を読み込み。一部だけの設定も可能で、省略分は既定値。
- **プラグイン選択**：`--input-loader`、`--detector`、`--output-handler`。
- **プラグイン設定**：`--input-loader-config`、`--detector-config`、`--output-handler-config`。
- **性能オプション**：`--auto-batch-size`/`--no-auto-batch-size`、`--parallel`/`--no-parallel` で設定ファイルを明示的に上書き。
- **DetectionContext正規化API**：`register_detection_context_converter()`、`normalize_detection_context()` を公開。
- **処理の統一**：CLIを `MeteorDetectionPipeline` へ統合。
- **非推奨への移行**：従来のパラメータ指定は動作しますが、設定ファイルを推奨。

### 変更の理由

設定を外部化し、v1.xとv2.0の間をつなぎます。

| 機能 | 変更前（v1.6.4） | 変更後（v1.6.5） |
|------|------------------|------------------|
| **プラグイン選択** | 組み込みの既定値 | `--config`、`--detector hough` |
| **プラグイン設定** | なし | `--detector-config '{...}'` |
| **検出パラメータ** | CLIのみ | 設定ファイルまたはCLI |
| **パイプライン実行** | 独立した関数 | `MeteorDetectionPipeline` |
| **コンテキスト正規化** | 内部のみ | 公開API |

利点：

1. **再現可能な実行**：パイプライン全体の設定を保存・共有。
2. **プラグインの試行**：コード変更なしで切り替え。
3. **CI/CDとの連携**：設定ファイルで実行を管理。
4. **段階的な移行**：従来オプションも新しい設定と併用可能。

### 設定ファイル形式

`PipelineConfig` のフィールドを持つYAML/JSONを作成します。不要なフィールドは省略でき、既定値を使います。

```yaml
# pipeline.yaml
target_folder: ./rawfiles
output_folder: ./candidates
debug_folder: ./debug_masks

# Detection parameters
params:
  diff_threshold: 8
  min_area: 10
  min_aspect_ratio: 3.0
  min_line_score: 30.0

# Worker settings
num_workers: 4
batch_size: 50
enable_parallel: true

# Plugin selection and configuration
input_loader_name: raw
input_loader_config:
  binning: 2
  normalize: true

detector_name: hough
detector_config: {}

output_handler_name: file
output_handler_config:
  output_folder: ./candidates
  debug_folder: ./debug_masks
  output_overwrite: false
```

**使い方**：

```bash
# Load entire configuration from file
uv run python detect_meteors_cli.py --config pipeline.yaml

# Override specific settings via CLI
uv run python detect_meteors_cli.py --config pipeline.yaml --detector threshold
```

### CLIでのプラグイン選択

コマンドラインから選択・設定できます。

```bash
# Select plugins by name
uv run python detect_meteors_cli.py \
    --input-loader raw \
    --detector hough \
    --output-handler file

# Provide plugin configs as JSON strings
uv run python detect_meteors_cli.py \
    --input-loader raw \
    --input-loader-config '{"binning": 2, "normalize": true}'

# Or as YAML strings
uv run python detect_meteors_cli.py \
    --output-handler slack \
    --output-handler-config "webhook_url: https://hooks.slack.com/..."

# Or as file paths
uv run python detect_meteors_cli.py \
    --detector-config my_detector_settings.yaml
```

**プラグイン用のオプション**：

| オプション | 説明 |
|------------|------|
| `--config FILE` | YAML/JSONからパイプライン設定を読み込む |
| `--input-loader NAME` | 入力ローダーを名前で選択 |
| `--input-loader-config VALUE` | ローダー設定のJSON/YAML文字列またはパス |
| `--detector NAME` | 検出器を名前で選択 |
| `--detector-config VALUE` | 検出器設定のJSON/YAML文字列またはパス |
| `--output-handler NAME` | 出力ハンドラーを名前で選択 |
| `--output-handler-config VALUE` | ハンドラー設定のJSON/YAML文字列またはパス |

### 設定の優先順

複数の指定がある場合は、次の順に優先します。

1. **CLI引数**（最優先）。
2. **設定ファイル**（`--config`）。
3. **組み込みの既定値**（最低優先）。

基本設定を読み込んで一部を上書きできます。

```bash
# Load base config, override detector
uv run python detect_meteors_cli.py --config base.yaml --detector threshold

# Load base config, override detection threshold
uv run python detect_meteors_cli.py --config base.yaml --diff-threshold 12
```

### Python API

Pythonから設定を読み込み、利用できます。

```python
from meteor_core import MeteorDetectionPipeline, load_pipeline_config

# Load configuration from file
config = load_pipeline_config("pipeline.yaml")

# Create and run pipeline
pipeline = MeteorDetectionPipeline(config)
pipeline.run()
```

### DetectionContext正規化API

`InputContext`、`DetectionResult`、`OutputResult` と同様に、`DetectionContext` の正規化を公開しました。

```python
from meteor_core.schema import (
    DetectionContext,
    register_detection_context_converter,
    normalize_detection_context,
)

# Register a converter for older schema versions
def upgrade_v0_to_v1(context: DetectionContext) -> DetectionContext:
    # Migration logic
    return DetectionContext(
        current_image=context.current_image,
        previous_image=context.previous_image,
        roi_mask=context.roi_mask,
        runtime_params=context.runtime_params,
        metadata=context.metadata,
        schema_version=1,
    )

register_detection_context_converter(0, upgrade_v0_to_v1)

# Normalize a context (applies converters if needed)
normalized = normalize_detection_context(context)
```

**正規化関数の一覧**：

| データクラス | 正規化関数 | 変換関数の登録 |
|--------------|------------|----------------|
| `InputContext` | `normalize_input_context()` | `register_input_context_converter()` |
| `DetectionContext` | `normalize_detection_context()` | `register_detection_context_converter()` |
| `DetectionResult` | `normalize_detection_result()` | `register_detection_result_converter()` |
| `OutputResult` | `normalize_output_result()` | `register_output_result_converter()` |

### 移行ガイド

**CLI利用者向け**：

直ちに移行する必要はありません。従来のオプションはすべて動作します。

```bash
# These still work (but are deprecated)
uv run python detect_meteors_cli.py --diff-threshold 8 --min-area 10

# Recommended: Use config files for complex setups
uv run python detect_meteors_cli.py --config pipeline.yaml
```

**プラグイン作者向け**：

1. 設定ファイル利用者向けに **ConfigTypeのフィールドを文書化** してください。

   ```python
   @dataclass
   class MyDetectorConfig:
       """Configuration for MyDetector.
       
       YAML example:
           detector_name: my_detector
           detector_config:
             sensitivity: 0.8
             use_gpu: true
       """
       sensitivity: float = 0.5
       use_gpu: bool = False
   ```

2. 任意の設定を省略できるよう、**適切な既定値を用意** してください。
3. YAML/JSONのキーとしてわかりやすい **フィールド名** を使ってください。

**Python API利用者向け**：

設定ファイルには `load_pipeline_config()` を使います。

```python
# Before: Manual PipelineConfig construction
config = PipelineConfig(
    target_folder="./rawfiles",
    output_folder="./candidates",
    params={"diff_threshold": 8},
)

# After: Load from file
config = load_pipeline_config("pipeline.yaml")
```

### パイプライン実行の統一

CLIを `MeteorDetectionPipeline` に統合し、Python APIと動作を揃えました。

```
┌─────────────────────────────────────────────────────────────────┐
│  CLI (detect_meteors_cli.py)                                    │
│                                                                 │
│  1. Parse CLI arguments                                         │
│  2. Load --config file (if provided)                            │
│  3. Apply CLI overrides                                         │
│  4. Build PipelineConfig                                        │
│  5. Create MeteorDetectionPipeline(config)                      │
│  6. pipeline.run()                                              │
└─────────────────────────────────────────────────────────────────┘
```

これにより、次の処理を統一します。

- CLIとAPIのプラグイン解決。
- 正規化するタイミング。
- ライフサイクルフックの呼び出し。

### 変更ファイル

| ファイル | 変更 |
|----------|------|
| `detect_meteors_cli.py` | 設定・プラグイン用引数、パイプラインへの統合 |
| `meteor_core/pipeline.py` | `load_pipeline_config()`、プラグイン設定の処理 |
| `meteor_core/schema.py` | コンテキストの正規化と変換関数登録を公開 |
| `meteor_core/completions/bash_completion.sh` | 新しいオプション |
| `meteor_core/completions/zsh_completion.sh` | 新しいオプション |
| `config_examples/pipeline.yaml` | 新しい設定例 |
| `PLUGIN_AUTHOR_GUIDE.md` | 設定ファイルとCLIの説明 |
| `CHANGELOG.md` | v1.6.5を追加 |
| `README.md` | 設定ファイルと新機能の説明 |
| `ROADMAP.md` | v1.6.5、設定ファイル対応を完了として記録 |

### 後方互換性

✅ v1.6.4、v1.6.3、v1.6.2、v1.6.1、v1.6.0と **完全な後方互換性** があります。

- **CLI**：既存オプションはそのまま動作。
- **実行時**：検出動作は変更なし。
- **API**：既存コードはそのまま動作。
- **プラグイン**：変更不要。

**非推奨について**：`--diff-threshold`、`--min-area` などの従来オプションは将来のリリースで非推奨になる予定です。`--config` や設定内の `params:` へ移行してください。

---

## バージョン1.6.4（2025-12-25）🎄

### 🔧 出力ハンドラーのフックとフレーム追跡

v1.6.4では `on_detection_result` を追加し、`DetectionResult` をパイプラインへ受け渡します。フレーム番号で進捗報告と後処理を改善しました。

### 主な変更

- **on_detection_result**：シリアライズしたコンテキストを受け取る検出ごとのコールバック。
- **DetectionResultの受け渡し**：`process_image_batch()` から `lines`、`extras`、`metrics` にアクセス可能。
- **フレーム番号**：検出コンテキストと `progress.json` に `frame_index`、`prev_frame_index`。
- **デバッグ画像の最適化**：候補だけ画像を生成し、メモリ使用量を削減。
- **性能の改善**：`_build_runtime_params()` をループ外へ移動。
- **進捗の記録**：後処理用にフレーム番号を保存。

### 変更の理由

処理の観察と後処理の能力を高めます。

| 機能 | 変更前（v1.6.3） | 変更後（v1.6.4） |
|------|------------------|------------------|
| **検出ごとのフック** | なし | `on_detection_result(context, result, filepath)` |
| **DetectionResultの参照** | 受け渡しなし | 出力ハンドラーで利用可能 |
| **フレーム追跡** | ファイル名のみ | `frame_index`、`prev_frame_index` |
| **デバッグ画像** | 全結果で生成 | 候補のみ |
| **進捗の詳細** | ファイル件数 | フレーム番号（42、108、215など） |

利点：

1. **実行中の確認**：保存前に出力ハンドラーが結果を確認。
2. **詳しい診断**：線分、extras、計測値を参照。
3. **後処理**：フレーム番号で外部データと対応付け。
4. **メモリ効率**：非候補のデバッグ画像を解放。
5. **進捗表示**：検出したフレームを具体的に表示。

### 出力ハンドラーのライフサイクル（v1.6.4）

フレームごとに次の順で呼び出します。

```
┌─────────────────────────────────────────────────────────────────┐
│  For each detection result:                                     │
│                                                                 │
│  1. on_detection_result(context, result, filepath)              │
│     └── Inspect result.lines, result.extras, result. metrics    │
│     └── context contains runtime_params, metadata               │
│                                                                 │
│  2. save_candidate() [only if result.is_candidate]              │
│     └── Save the candidate image                                │
│                                                                 │
│  3. on_candidate_detected(filename, saved, score, aspect_ratio) │
│     └── Send notifications, update counters                     │
└─────────────────────────────────────────────────────────────────┘
```

### 検出コンテキストのフレーム番号

メタデータにフレーム番号を追加しました。

```python
context.metadata = {
    "current": {
        "frame_index": 42,      # 0-based index of current frame
        # ... other metadata
    },
    "previous": {
        "frame_index": 41,      # 0-based index of previous frame
        # ... other metadata
    },
}
```

### 進捗ファイルの変更（progress.json）

`detected_details` にフレーム番号を追加しました。

```json
{
  "processed": 150,
  "detected": 3,
  "detected_details": [
    {
      "filename": "IMG_0042.CR2",
      "score": 85.5,
      "aspect_ratio": 3.2,
      "frame_index": 42,
      "prev_frame_index": 41
    },
    {
      "filename": "IMG_0108.CR2",
      "score": 92.1,
      "aspect_ratio": 4.1,
      "frame_index": 108,
      "prev_frame_index": 107
    }
  ]
}
```

次の用途に使えます。

- 外部の時刻ログとの対応付け。
- フレーム列での検出パターン解析。
- 動画・タイムラプスのメタデータとの連携。

### プラグイン作者向け移行ガイド

**直ちに移行する必要はありません。** 既存プラグインは動作します。

**出力ハンドラー作者への推奨事項**：

1. 検出ごとの処理には **on_detection_resultを実装** してください。

   ```python
   def on_detection_result(
       self,
       context: Dict[str, Any],
       result: DetectionResult,
       filepath: str,
   ) -> None:
       """Called immediately after each detection result.
   
       Args:
           context: Serialized DetectionContext (no image data)
           result: The DetectionResult from the detector
           filepath: Path to the current image file
       """
       # Inspect detector outputs
       if result.is_candidate:
           logger.info(f"Detected {len(result.lines)} lines in {filepath}")
           logger.debug(f"Metrics: {result.metrics}")
   
       # Access runtime params from context
       runtime_params = context.get("runtime_params", {})
       logger.debug(f"Used params: {runtime_params}")
   ```

2. コンテキストから **フレーム番号を取得** してください。

   ```python
   def on_detection_result(self, context, result, filepath):
       metadata = context.get("metadata", {})
       current_frame = metadata.get("current", {}).get("frame_index")
       prev_frame = metadata.get("previous", {}).get("frame_index")
       logger.info(f"Processing frames {prev_frame} → {current_frame}")
   ```

### デバッグ画像の最適化

大規模バッチ処理のメモリ使用量を減らします。

- 非候補の結果を返す前に `DetectionResult.debug_image` をクリア。
- 候補だけでデバッグ画像を生成・保持。
- 数千枚の処理時のメモリ負荷を大幅に削減。

### 変更ファイル

| ファイル | 変更 |
|----------|------|
| `meteor_core/pipeline.py` | フック呼び出し、フレーム番号、デバッグ画像最適化 |
| `meteor_core/schema.py` | メタデータ内のフレーム番号 |
| `detect_meteors_cli.py` | 結果タプル、フレーム番号を含む進捗表示 |
| `PLUGIN_AUTHOR_GUIDE.md` | ライフサイクルフックの説明 |
| `CHANGELOG.md` | v1.6.4を追加 |
| `README.md` | 新機能の説明を更新 |

### 後方互換性

✅ v1.6.3、v1.6.2、v1.6.1、v1.6.0と **完全な後方互換性** があります。

- **CLI**：変更なし。
- **実行時**：検出動作は変更なし。
- **API**：既存プラグインはそのまま動作。新フックは任意。
- **progress.json**：新フィールドを追加し、既存フィールドは維持。

---

## バージョン1.6.3（2025-12-24）🎅

### 🔧 RuntimeParams契約とパイプライン正規化

v1.6.3ではバージョン付き `RuntimeParams` とパイプラインの自動正規化を導入し、スキーマのバージョン管理を完成させました。

### 主な変更

- **RuntimeParams**：`schema_version`、名前空間付きパラメータ、`to_dict()` による正式な受け渡し。
- **DetectionContext.to_dict()**：画像・マスクを除外するログ・デバッグ用のシリアライズ。
- **正規化**：パイプライン境界で `InputContext`、`DetectionResult`、`OutputResult` を自動処理。
- **従来のboolとの互換性**：`OutputResult` に変換し、非推奨の警告を表示。
- **文書の更新**：全契約についてプラグイン開発ガイドを拡充。

### 変更の理由

すべてのプラグインインターフェースの契約を標準化します。

| 契約 | v1.6.1 | v1.6.2 | v1.6.3 |
|------|--------|--------|--------|
| `DetectionContext` | ✅ schema_version | — | ✅ `to_dict()` |
| `DetectionResult` | ✅ schema_version、metrics、`to_dict()` | — | — |
| `InputContext` | — | ✅ schema_version、`to_dict()` | — |
| `OutputResult` | — | ✅ schema_version、metrics、`to_dict()` | — |
| `RuntimeParams` | — | — | ✅ 新規：schema_version、`to_dict()` |
| **パイプライン正規化** | — | — | ✅ 全契約 |

利点：

1. **シリアライズの統一**：全契約でJSON互換の `to_dict()` を提供。
2. **パラメータの名前空間**：全体設定と検出器固有の上書きを分離。
3. **自動検証**：プラグイン出力を正規化し、スキーマ不一致を早期発見。
4. **段階的な移行**：従来の `bool` も警告付きで動作。

### スキーマ変更（v1.6.3）

**RuntimeParams**（v1.6.3で追加）：

```python
@dataclass
class RuntimeParams:
    """Runtime parameters passed into detector execution."""

    schema_version: int = 1                             # RUNTIME_PARAMS_SCHEMA_VERSION
    global_params: Dict[str, Any] = field(default_factory=dict)
    detector: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self, include_schema_version: bool = True) -> Dict[str, Any]:
        """Serialize to dict for JSON/logging."""
        ...
```

`to_dict()` によるシリアライズ形式：

```python
{
    "schema_version": 1,
    "global": {"diff_threshold": 8, "min_area": 10, ...},
    "detector": {"hough": {"hough_threshold": 50}, ...},
}
```

**DetectionContext.to_dict()**（v1.6.3で追加）：

```python
@dataclass
class DetectionContext:
    # ... existing fields ...

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context for JSON/logging.

        Excludes current_image, previous_image, and roi_mask
        to avoid large binary data in logs.
        """
        ...
```

**追加した定数**：

```python
RUNTIME_PARAMS_SCHEMA_VERSION = 1
```

### パイプライン正規化

次の時点でプラグイン出力を正規化します。

| 時点 | 関数 | 処理 |
|------|------|------|
| `loader.load()` の後 | `normalize_input_context()` | スキーマ検証、変換関数の適用 |
| `detector.detect()` の後 | `normalize_detection_result()` | スキーマ検証、変換関数の適用 |
| `handler.save_candidate()` の後 | `normalize_output_result()` | スキーマ検証、従来の `bool` を変換 |

**従来の真偽値との互換性**：

`OutputResult` の代わりに `bool` を返す場合、自動変換します。

```python
# Legacy handler (deprecated but still works)
def save_candidate(self, source_path, filename, ...) -> bool:
    # ... save logic ...
    return True  # or False

# Pipeline automatically converts to:
# OutputResult(saved=True, output_path=None, debug_path=None)
# with a deprecation warning logged
```

### プラグイン作者向け移行ガイド

**直ちに移行する必要はありません。** 既存プラグインは動作します。

**推奨事項**：

1. **検出器作者**：名前空間付き構造から実行パラメータを取得してください。

   ```python
   def detect(self, context: DetectionContext) -> DetectionResult:
       # Use helper methods from BaseDetector
       global_params, detector_params = self.split_runtime_params(
           context.runtime_params
       )
       params = {**global_params, **detector_params}
   ```

2. **出力ハンドラー作者**：`bool` の代わりに `OutputResult` を返してください。

   ```python
   # Before (deprecated)
   def save_candidate(self, ...) -> bool:
       return True
   
   # After (recommended)
   def save_candidate(self, ...) -> OutputResult:
       return OutputResult(saved=True, output_path=dest_path, debug_path=None)
   ```

3. **デバッグ・ログ**：`to_dict()` を使ってください。

   ```python
   logger.debug(f"Detection context: {context.to_dict()}")
   logger.debug(f"Runtime params: {context.runtime_params.to_dict()}")
   ```

### BaseDetectorのヘルパー

`BaseDetector` は `RuntimeParams` を扱う便利なメソッドを提供します。

| メソッド | シグネチャ | 説明 |
|----------|------------|------|
| `split_runtime_params` | `(runtime_params) -> (global_params, detector_params)` | 名前空間からパラメータを抽出 |
| `build_runtime_params` | `(flat_params) -> RuntimeParams` | 平坦な辞書を変換 |
| `detect_legacy` | `(current, previous, roi, params) -> DetectionResult` | 旧シグネチャのアダプター |

### 変更ファイル

| ファイル | 変更 |
|----------|------|
| `meteor_core/schema.py` | `RuntimeParams`、`DetectionContext.to_dict()` |
| `meteor_core/pipeline.py` | 正規化、従来のboolの変換 |
| `PLUGIN_AUTHOR_GUIDE.md` | 全契約の説明 |
| `CHANGELOG.md` | v1.6.3を追加 |

### 後方互換性

✅ v1.6.2、v1.6.1、v1.6.0と **完全な後方互換性** があります。

- **CLI**：変更なし。
- **実行時**：検出動作は変更なし。
- **API**：既存プラグインはそのまま動作。
- **出力ハンドラー**：従来の `bool` は警告付きで動作。
- **検出器**：従来の平坦な辞書にも対応。

---

## バージョン1.6.2（2025-12-23）🇯🇵

### 🔧 入出力コンテキスト契約

v1.6.2では入力ローダーと出力ハンドラーにもスキーマのバージョン管理を拡張し、全プラグイン種別の契約を標準化しました。

### 主な変更

- **入力契約**：`InputContext` に `schema_version`、`loader_info`、`to_dict()`。
- **出力契約**：`OutputResult` に `schema_version`、`handler_info`、`metrics`、`to_dict()`。
- **契約の網羅**：入力・検出・出力の3種をバージョン付き契約に統一。
- **文書の更新**：新しい両契約をプラグイン開発ガイドで説明。

### 変更の理由

v1.6.1で始めたバージョン管理を完成させます。

| 種別 | v1.6.1 | v1.6.2 |
|------|--------|--------|
| **検出器** | ✅ `DetectionContext`、`DetectionResult` | — |
| **入力ローダー** | — | ✅ `InputContext` |
| **出力ハンドラー** | — | ✅ `OutputResult` |

利点：

1. **契約の統一**：schema_version、情報辞書、to_dict()の共通パターン。
2. **将来の移行**：既存プラグインを壊さず段階的に変更可能。
3. **診断の改善**：`OutputResult.metrics` で性能情報を標準化。
4. **シリアライズ**：JSON互換のログ・デバッグ。

### スキーマ変更（v1.6.2）

**InputContext**（v1.6.2で追加）：

```python
@dataclass
class InputContext:
    image_data: ImageLike              # Loaded image (numpy, torch, or PIL)
    filepath: str                      # Original file path
    metadata: Dict[str, Any] = {}      # Loader-extracted metadata (EXIF, etc.)
    loader_info: Dict[str, Any] = {}   # Loader identity (name, version)
    schema_version: int = 1            # INPUT_CONTEXT_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context for JSON/logging (excludes image_data)."""
        ...
```

**OutputResult**（v1.6.2で追加）：

```python
@dataclass
class OutputResult:
    saved: bool                        # Whether file was persisted
    output_path: Optional[str]         # Path to saved candidate
    debug_path: Optional[str]          # Path to saved debug image
    handler_info: Dict[str, Any] = {}  # Handler identity (name, version)
    metrics: Dict[str, Any] = {}       # Performance diagnostics
    schema_version: int = 1            # OUTPUT_RESULT_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result for JSON/logging."""
        ...
```

**追加した定数**：

```python
INPUT_CONTEXT_SCHEMA_VERSION = 1
OUTPUT_RESULT_SCHEMA_VERSION = 1
```

### 契約パターンの一覧

すべての契約は次の共通パターンに従います。

| 契約 | 種別 | 情報フィールド | 計測値 | スキーマバージョン |
|------|------|----------------|--------|--------------------|
| `DetectionContext` | 検出器の入力 | — | — | ✅ |
| `DetectionResult` | 検出器の出力 | — | ✅ | ✅ |
| `InputContext` | ローダーの出力 | `loader_info` | — | ✅ |
| `OutputResult` | ハンドラーの出力 | `handler_info` | ✅ | ✅ |

### プラグイン作者向け移行ガイド

プラグイン構成は **実験的** であり、v1.6.2はパイプラインで使う出力ハンドラーに互換性のない変更を導入します。

**出力ハンドラー作者の必須作業**：

1. `save_candidate` から `OutputResult` を返してください。パイプラインは `.saved`、`.output_path`、`.debug_path` を参照します。
2. 従来の `OutputWriter` ベースのハンドラーや、`bool` を返すラッパーを更新してください。

**入力ローダー作者の任意の改善**：

1. 画像配列の代わりに `InputContext` を返す。
2. `BaseInputLoader` の `self.get_info()` で `loader_info` を設定。
3. ログに `context.to_dict()` を使う。

**出力ハンドラー作者への推奨事項**：

1. `BaseOutputHandler` の `self.get_info()` で `handler_info` を設定。
2. duration_ms、bytes_writtenなどの性能情報に `metrics` を使用。
3. ログに `result.to_dict()` を使う。

### 例：更新したローダー

```python
from meteor_core.inputs import DataclassInputLoader
from meteor_core.schema import InputContext

class MyLoader(DataclassInputLoader[MyConfig]):
    def load(self, filepath: str) -> InputContext:
        image = self._load_image(filepath)
        return InputContext(
            image_data=image,
            filepath=filepath,
            metadata=self.extract_metadata(filepath),
            loader_info=self.get_info(),  # {"name": ..., "version": ...}
        )
```

### 例：更新したハンドラー

```python
from meteor_core.outputs import DataclassOutputHandler
from meteor_core.schema import OutputResult
import time

class MyHandler(DataclassOutputHandler[MyConfig]):
    def save_candidate(self, source_path, filename, ...) -> OutputResult:
        start = time.perf_counter()
        dest_path = self._save_file(source_path, filename)
        duration_ms = (time.perf_counter() - start) * 1000

        return OutputResult(
            saved=True,
            output_path=dest_path,
            debug_path=None,
            handler_info=self.get_info(),
            metrics={"duration_ms": duration_ms},
        )
```

### 変更ファイル

| ファイル | 変更 |
|----------|------|
| `meteor_core/schema.py` | 両契約とスキーマバージョン定数 |
| `PLUGIN_AUTHOR_GUIDE.md` | 新しいローダー・ハンドラー契約 |
| `CHANGELOG.md` | v1.6.2を追加 |
| `README.md` | 新機能の説明を更新 |
| `ROADMAP.md` | v1.6.2のマイルストーン |

### 後方互換性

✅ v1.6.1とv1.6.0と **完全な後方互換性** があります。

- **CLI**：変更なし。
- **実行時**：検出動作は変更なし。
- **API**：既存ローダー・ハンドラープラグインはそのまま動作。
- **設定**：変更なし。

---

## バージョン1.6.1（2025-12-22）🌃

### 🔧 プラグインスキーマのバージョン管理とMLに備えた設計

v1.6.1では検出器契約のバージョン管理と複数フレームワークの画像対応を導入し、v2.xのML検出器に備えました。

### 主な変更

- **スキーマバージョン**：`DetectionContext`、`DetectionResult` に `schema_version`。
- **複数の画像型**：numpy、PyTorch、PILに対応する `ImageLike`。
- **診断の改善**：標準的な性能情報を持つ `DetectionResult.metrics`。
- **シリアライズ**：JSON互換の `DetectionResult.to_dict()`。

### 変更の理由

次の拡張に備えます。

1. **将来の移行**：バージョンを使って段階的に変更可能。
2. **ML検出器**：PyTorchのTensorをそのまま扱える画像型。
3. **標準的な計測値**：異なる検出器でも診断情報を統一。

### スキーマ変更（v1.6.1）

**DetectionContext**（v1.6.1）：

```python
@dataclass
class DetectionContext:
    current_image: ImageLike      # NEW: Union[np.ndarray, torch.Tensor, PIL.Image]
    previous_image: ImageLike     # NEW: Union[np.ndarray, torch.Tensor, PIL.Image]
    roi_mask: Any
    runtime_params: Dict[str, Any]
    metadata: Dict[str, Any]
    schema_version: int = 1       # NEW: For migration support
```

**DetectionResult**（v1.6.1）：

```python
@dataclass
class DetectionResult:
    is_candidate: bool
    score: float
    lines: List[Tuple[int, int, int, int]]
    aspect_ratio: float
    debug_image: Optional[Any]
    extras: Dict[str, Any]
    metrics: Dict[str, Any]       # NEW: Standard diagnostics
    schema_version: int = 1       # NEW: For migration support
    def to_dict(self) -> Dict[str, Any]:  # NEW: Serialization
        ...
```

### 新しいユーティリティ

`meteor_core.utils` の **ensure_numpy()**：

```python
from meteor_core.utils import ensure_numpy

# Convert any ImageLike to numpy array
image = ensure_numpy(context.current_image)  # Works with numpy, torch, PIL
```

### プラグイン作者向け移行ガイド

移行作業は不要で、既存プラグインは変更なしで動作します。

**任意の改善**：

1. 型安全な画像処理に `ensure_numpy()` を使う。
2. `metrics` に標準的な診断情報を設定。
3. ログに `result.to_dict()` を使う。

### 変更ファイル

| ファイル | 変更 |
|----------|------|
| `meteor_core/schema.py` | `ImageLike`、スキーマバージョン、`to_dict()` |
| `meteor_core/utils.py` | `ensure_numpy()` |
| `PLUGIN_AUTHOR_GUIDE.md` | DetectionContext/DetectionResultの説明 |
| `CHANGELOG.md` | v1.6.1を追加 |
| `README.md` | 新機能の説明を更新 |

### 後方互換性

✅ v1.6.0と **完全な後方互換性** があります。

- **CLI**：変更なし。
- **実行時**：検出動作は変更なし。
- **API**：既存検出器はそのまま動作。
- **設定**：変更なし。

---

## バージョン1.6.0（2025-12-21）💌

### ⚡ 開発ツールの刷新

v1.6.0ではpip/black/flake8からuv/Ruffへ移行しました。依存管理の高速化と静的解析・整形の統一で、開発環境を改善します。

### 主な変更

- **uv**：Rust製のPythonパッケージ管理ツール（pipの10〜100倍高速）。
- **Ruff**：blackとflake8を置き換える、Rust製の静的解析・整形ツール。
- **設定の統一**：全ツールの設定を `pyproject.toml` へ集約。
- **簡単な環境構築**：1コマンドですべてインストール。

### 変更の理由

| 項目 | 変更前（v1.5.x） | 変更後（v1.6.0） |
|------|------------------|------------------|
| **パッケージ管理** | pip | uv |
| **整形** | black | Ruff |
| **静的解析** | flake8 | Ruff |
| **設定ファイル** | `pyproject.toml` + `.flake8` | `pyproject.toml` のみ |
| **インストール時間** | 約30〜60秒 | 約2〜5秒 |
| **静的解析と整形** | 別々の2ツール | 1つに統一 |

### 性能の比較

**依存ライブラリのインストール**：

```bash
# Before (pip)
pip install -e ".[dev]"  # ~30-60 seconds

# After (uv)
uv sync --all-extras     # ~2-5 seconds
```

**静的解析と整形**：

```bash
# Before (black + flake8)
black .                  # ~2 seconds
flake8 .                 # ~1 second

# After (Ruff)
ruff check --fix .       # ~0.1 seconds
ruff format .            # ~0.1 seconds
```

### 貢献者向け移行ガイド

#### 新しい開発環境の構築

```bash
# 1. Install uv (if not already installed)
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Clone and setup
git clone https://github.com/shin3tky/detect_meteors.git
cd detect_meteors
uv sync --all-extras

# 3. Install pre-commit hooks
uv run pre-commit install

# 4. Verify setup
uv run pre-commit run --all-files
```

#### 既存の開発環境から移行する

既存環境がある場合は、次の手順に従ってください。

```bash
# Update your local repository
git pull origin main

# Remove old virtual environment (optional but recommended)
rm -rf .venv

# Create new environment with uv
uv sync --all-extras

# Reinstall pre-commit hooks
uv run pre-commit install
```

### 新しい開発手順

#### コマンドの実行

Pythonのコマンドには `uv run` を付けて実行してください。

```bash
# Run tests
uv run python run_tests.py

# Run the CLI
uv run python detect_meteors_cli.py --help

# Run linter
uv run ruff check .

# Run formatter
uv run ruff format .
```

#### pre-commitフック

ローカルRuffフックを使用します。

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: ruff-check
        name: ruff check
        entry: .venv/bin/ruff check --fix
        language: system
        types: [python]
      - id: ruff-format
        name: ruff format
        entry: .venv/bin/ruff format
        language: system
        types: [python]
```

### 設定の変更

#### pyproject.toml

`[tool.ruff]` がblackとflake8の設定を置き換えます。

```toml
[tool.ruff]
line-length = 88
target-version = "py312"

[tool.ruff.lint]
select = ["E", "W", "F", "C90"]
ignore = ["E203", "E501", "E226"]

[tool.ruff.lint.mccabe]
max-complexity = 40

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

### Ruffとblack/flake8の互換性

Ruffはblackとflake8の代替として設計されています。主な対応は次のとおりです。

| 機能 | black/flake8 | Ruff |
|------|--------------|------|
| 行の長さ | `max-line-length = 88` | `line-length = 88` |
| 引用符 | blackの既定（ダブル） | `quote-style = "double"` |
| 複雑度 | `max-complexity = 40` | `[tool.ruff.lint.mccabe] max-complexity = 40` |
| 除外ルール | `.flake8` の一覧 | `[tool.ruff.lint] ignore` |

### 依存ライブラリの更新

**開発用の任意依存**：

```toml
[project.optional-dependencies]
dev = [
    "ruff==0.14.10",      # Replaces black + flake8
    "pre-commit>=4.5.0",
    "coverage>=7.6.0",
]
```

**開発依存から削除**：

- `black>=25.12.0`
- `flake8>=7.3.0`
- `flake8-pyproject>=1.2.4`

### よくある質問

#### Q：uvのインストールは必要ですか？

はい。このプロジェクトでは依存管理にuvを推奨します。単一のバイナリで、依存ライブラリはありません。

#### Q：引き続きpipを使えますか？

基本的なインストール（`pip install -e .`）には使えますが、開発手順はuvに合わせています。pre-commitフックはuvの仮想環境でRuffを利用できることを前提にします。

#### Q：Ruffの整形はblackと異なりますか？

blackとの互換性を意識した設計で、多くの場合は同じ出力になります。特殊なケースでは小さな違いがありますが、見た目の差であり機能には影響しません。

#### Q：静的解析は厳しくなりますか？

同じルールを適用します。flake8と同じE、W、F、C90のエラーコードに対応し、除外一覧も維持しています。

### 変更ファイル

| ファイル | 変更 |
|----------|------|
| `pyproject.toml` | `[tool.ruff]` と開発依存 |
| `.pre-commit-config.yaml` | black/flake8をローカルRuffへ変更 |
| `INSTALL_DEV.md` | uv/Ruffの開発手順へ改訂 |
| `CHANGELOG.md` | v1.6.0を追加 |
| `README.md` | 新機能の説明を更新 |

### 後方互換性

✅ v1.5.xと **完全な後方互換性** があります。

- **CLI**：変更なし。
- **実行時**：検出動作や出力は変更なし。
- **API**：`meteor_core` のインターフェースは変更なし。
- **設定**：検出パラメータやセンサープリセットは変更なし。

開発ツールのみを変更するため、開発依存なしで利用する場合は動作に違いはありません。

### リンク

- [uvドキュメント](https://docs.astral.sh/uv/)
- [Ruffドキュメント](https://docs.astral.sh/ruff/)
- [INSTALL_DEV_ja.md](INSTALL_DEV_ja.md) — 更新した開発環境ガイド

---

## バージョン情報

### v1.6.10 🌠（最新）

- **リリース日**：2026-10-06 - 夢をかなえる日
- **バージョンの順序**：1.6.9をスキップし、v1.6.8の次にリリース。
- **主題**：ソート済み検出フックと飛行機の光跡メタデータ。
- **主な変更**：
  - `SortedDetection` によるバッチ内・実行全体のフレーム順解析。
  - 飛行機likelihoodとサンプルの検証結果。
  - インストール可能なCLI、ソース配布版、wheel。

### v1.6.8 🌿

- **バージョン**：1.6.8
- **リリース日**：2026-01-07
- **主な変更**：
  - AstralのRust製tyによる静的型検査。
  - 全レジストリの型安全性の改善。
  - MAX_NUM_WORKERSとパイプライン・CLIの検証。
  - ubuntu-slimによるCIの最適化。

### v1.6.7 🎂

- **バージョン**：1.6.7
- **リリース日**：2025-12-29
- **主な変更**：
  - v2.0/v3.0のマイルストーンを分野別に具体化。
  - Python 3.12、3.13への対応を明示。
  - ドキュメントのみのリリース。

### v1.6.6 🧱

- **バージョン**：1.6.6
- **リリース日**：2025-12-27
- **主な変更**：
  - `on_file_found`、`on_image_loaded`、`on_detection_complete`、`on_output_saved` の4つのフック。
  - 検出・管理を集約するHookRegistry。
  - `BaseHook`、`DataclassHook`、`PydanticHook`。
  - 設定用の `--hooks`、`--hook-config`。
  - Pythonから制御する `hooks`、`hook_error_mode`。
  - エントリーポイントとディレクトリによるマルチプロセス対応の検出。

### v1.6.5 📓

- **バージョン**：1.6.5
- **リリース日**：2025-12-26
- **主な変更**：
  - `--config` によるYAML/JSON設定。
  - `--input-loader`、`--detector`、`--output-handler` の選択。
  - `--input-loader-config`、`--detector-config`、`--output-handler-config` の設定。
  - DetectionContextの正規化APIを公開。
  - CLIを `MeteorDetectionPipeline` へ統合。
  - 従来パラメータから設定ファイルへの移行。

### v1.6.4 🎄

- **バージョン**：1.6.4
- **リリース日**：2025-12-25
- **主な変更**：
  - 出力ハンドラーの `on_detection_result`。
  - パイプラインでDetectionResultを受け渡し。
  - 検出コンテキスト・進捗の `frame_index`、`prev_frame_index`。
  - メモリ効率のためのデバッグ画像最適化。
  - progress.jsonにフレーム番号を記録。

### v1.6.3

- **バージョン**：1.6.3
- **リリース日**：2025-12-24
- **主な変更**：
  - バージョン付きRuntimeParams。
  - DetectionContext.to_dict()によるシリアライズ。
  - 全契約のパイプライン正規化。
  - 従来の真偽値への対応。
  - プラグイン開発ガイドの拡充。

### v1.6.2

- **バージョン**：1.6.2
- **リリース日**：2025-12-23
- **主な変更**：
  - InputContextによるローダーの戻り値の標準化。
  - OutputResultによるハンドラーの戻り値の標準化。
  - 全プラグイン契約のスキーマバージョン管理。
  - 契約の文書化。

### v1.6.1

- **バージョン**：1.6.1
- **リリース日**：2025-12-22
- **主な変更**：
  - DetectionContext/DetectionResultのバージョン管理。
  - 複数フレームワーク用のImageLike。
  - 診断を標準化するDetectionResult.metrics。
  - 型安全な変換用のensure_numpy()。

### v1.6.0

- **バージョン**：1.6.0
- **リリース日**：2025-12-21
- **主な変更**：
  - パッケージ管理をpipからuvへ。
  - 整形・静的解析をblack + flake8からRuffへ。
  - pyproject.tomlで設定を統一。
  - 開発環境の構築を簡単に。

---

**状態**：実運用可能  
**互換性**：v1.5.xと完全な後方互換性  
**Python**：3.12、3.13  
**推奨**：ファイル選別、前処理、通知などの補助処理にはパイプラインフックを、再現可能な設定には設定ファイル（`--config`）を使ってください。

流星探しをお楽しみください！ 🌠🌿
