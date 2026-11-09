# プラグイン開発ガイド

[English](PLUGIN_AUTHOR_GUIDE.md)

このガイドは1.6.12開発ブランチの入力ローダー、検出器、出力ハンドラー、パイプラインフックを対象とします。現在のパッケージバージョンは `1.6.11`、データ契約のスキーマバージョンは `1` です。プラグインAPIは開発中です。

環境は [開発者向けインストールガイド](INSTALL_DEV_ja.md) の手順で構築します。特に指定がない限り、CLI・Pythonの実行例はリポジトリまたは展開したソース配布版のルートで実行してください。

---

<a id="architecture-overview"></a>

## 設計の概要

最初に全体の構成を確認します。3つの層を **疎結合** にし、それぞれの責務とデータ契約を明確にしています。

<a id="three-layer-architecture"></a>

### 3層構成

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Plugin Architecture                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐          │
│  │   Input Layer     │   │  Detection Layer  │   │   Output Layer    │          │
│  │  (Input Loaders)  │──▶│    (Detectors)    │──▶│ (Output Handlers) │          │
│  └───────────────────┘   └───────────────────┘   └───────────────────┘          │
│          │                        │                        │                    │
│          ▼                        ▼                        ▼                    │
│   ┌─────────────┐         ┌───────────────┐        ┌─────────────┐              │
│   │InputContext │         │DetectionResult│        │OutputResult │              │
│   └─────────────┘         └───────────────┘        └─────────────┘              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

<a id="data-flow"></a>

### データの流れ

現在・前の2フレームを1組として処理し、流星候補を検出します。

```
filepath ──▶ Input Loader ──▶ InputContext
                                    │
                                    ▼
                            ┌────────────────┐
                            │DetectionContext│  (frame pixels + metadata + ROI + runtime params)
                            └────────────────┘
                                    │
                                    ▼
                              Detector
                                    │
                                    ▼
                            ┌────────────────┐
                            │DetectionResult │  (is_candidate, score, lines, debug_image, ...)
                            └────────────────┘
                                    │
                                    ▼
                            Output Handler
                                    │
                                    ▼
                            ┌───────────────┐
                            │ OutputResult  │  (saved, output_path, debug_path, ...)
                            └───────────────┘
```

<a id="layer-responsibilities"></a>

### 各層の責務

| 層 | 基底クラス | 入力 | 出力 | 責務 |
|----|------------|------|------|------|
| **入力** | `BaseInputLoader` | `filepath` | `InputContext` | CR2、ARW、DNG、TIFF、FITSなどの読み込みとメタデータ抽出 |
| **検出** | `BaseDetector` | `DetectionContext` | `DetectionResult` | フレーム差分、Hough変換、MLなどによる候補解析 |
| **出力** | `BaseOutputHandler` | `DetectionResult` | `OutputResult` | ファイル・クラウド・Slack・データベースなどへの保存、レポート、通知 |

<a id="benefits-of-loose-coupling"></a>

### 疎結合の利点

次のように、柔軟に差し替えられます。

- **入力層**：検出器はファイル形式や読み込みライブラリを意識する必要がありません。rawpy、OpenCV、独自FITSリーダーでも、正規化した `InputContext` を介して画像を提供します。
- **検出層**：前処理・解析・スコア算出を検出器内にまとめます。Hough変換を深層学習、モルフォロジー解析、動画検出などに置き換えても、入出力層は維持できます。
- **出力層**：検出ロジックに影響せず、保存先をローカルからS3/GCS、データベース、Slack/Discordなどへ変更できます。

各層は `InputContext`、`DetectionContext`、`DetectionResult`、`OutputResult` の明確なデータクラスで通信し、内部の変更がパイプラインを壊さないようにします。

---

<a id="table-of-contents"></a>

## 目次

0. [設計の概要](#architecture-overview)
1. [アプリケーションのライフサイクル](#1-application-lifecycle)
2. [拡張ポイント](#2-extension-points)
3. [プラグイン構成](#3-plugin-architecture)
4. [データ契約リファレンス](#4-data-contracts-reference)
5. [コード例](#5-sample-code)
6. [実装上の推奨事項](#6-best-practices)
7. [段階的なチュートリアル](#7-step-by-step-tutorial)

---

<a id="1-application-lifecycle"></a>

## 1. アプリケーションのライフサイクル

効果的なプラグイン開発には、検出パイプラインの処理順を理解することが重要です。

<a id="11-pipeline--hook-overview"></a>

### 1.1 パイプラインとフックの全体像

```text
メイン       ファイル収集・選別 → 初回画像・ROI → 進捗の復元
各バッチ     ペアを読み込む → 画像フック → 検出 → 検出結果フック
メイン       結果通知 → 候補の保存 → 進捗記録 → バッチ内ソート済みフック
メイン       出力ハンドラーの完了通知 → 全体ソート済みフック → 進捗へ反映
```

逐次実行では1ペアが1バッチです。並列実行ではワーカーがペアの読み込みから検出結果フックまでを担当し、メインプロセスが完了したバッチを受け取ります。

<a id="12-pipeline-steps"></a>

### 1.2 パイプラインの処理段階

1. 対象ディレクトリ直下で対応RAW拡張子に一致するファイルを集め、パス順に並べます。`on_file_found()` が受け取るのは正規化した絶対パスです。
2. 選別後の最初の画像を読み込み、指定ポリゴン、対話選択、全画面のいずれかでROIを決めます。初回の読み込みは `on_image_loaded()` の対象外です。
3. 進捗を復元する場合は設定とROIのハッシュを照合します。選別後の連続フレームからペアを作り、処理済みの現在フレームを除外します。

<a id="13-processing-flow-detail"></a>

### 1.3 処理の流れの詳細

1. 現在フレーム、前フレームの順に `load()` で読み込み、それぞれの `InputContext` を `on_image_loaded()` に渡します。
2. 画素、ROI、実行時設定、フレームのメタデータを `DetectionContext` にまとめて `detect()` を呼びます。正規化した `DetectionResult` は `on_detection_complete()` で更新できます。
3. メインプロセスが各ペアの結果を出力ハンドラーに通知し、候補を `save_candidate()` に渡します。候補の保存とデバッグ画像の扱いは出力ハンドラーが担当します。フックの呼び出し順は [実行場所と状態の生存期間](#hook-execution-and-state) を参照してください。

<a id="14-hook-overview"></a>

### 1.4 フックの概要

フックには2種類あります。

**出力ハンドラーのフック**（`BaseOutputHandler` で定義）：

フレーム・バッチ単位のイベントを扱います。`on_detection_result` は `DetectionContext.to_dict()` が返すコンテキスト辞書を受け取ります。

| イベント | 現在の状態 | 用途 |
|----------|------------|------|
| `on_detection_result` | 呼び出す | フレームごとの確認、ログ、計測 |
| `on_candidate_detected` | 呼び出す | Slack/Webhookなどの通知 |
| `on_batch_complete` | 呼び出す | 進捗、計測値の収集 |
| `on_pipeline_complete` | 呼び出す | 最終集計、後始末、レポート |

**パイプラインのフック**（`BaseHook` で定義）：

ファイル選別、画像変換、解析などの補助処理を挿入します。

| イベント | 現在の状態 | 用途 |
|----------|------------|------|
| `on_file_found` | 呼び出す | 読み込み前の選別 |
| `on_image_loaded` | 呼び出す | 画像変換、メタデータ追加 |
| `on_detection_complete` | 呼び出す | スコア調整、メタデータ追加 |
| `on_output_saved` | 呼び出す | 計測値の記録 |
| `on_batch_results_sorted` | 呼び出す（v1.6.10） | フレーム順でバッチ内を解析 |
| `on_all_detections_sorted` | 呼び出す（v1.6.10） | 飛行機の光跡などのフレーム間解析 |

**重要**：入力ローダーと検出器にはライフサイクルイベントを送りません。

---

<a id="15-hook-discovery-pipeline"></a>

### 1.5 フックの検出（パイプライン）

標準のプラグイン検出方式で、メイン・ワーカープロセスの双方から利用できるようにします。

> **重要**：`HookRegistry.register()` による実行時登録は **単一プロセスのみ** で、**ワーカーへ伝わりません**。実運用やマルチプロセスではプラグイン検出を使ってください。

**検出方法**：

- **エントリーポイント**：`detect_meteors.hook` にフッククラスを登録。
- **ローカルディレクトリ**：`~/.detect_meteors/hook_plugins` にクラスを定義した `*.py` を置く。

**マルチプロセスでの使用例**：

- エントリーポイント（`detect_meteors.hook`）としてインストールするか、`~/.detect_meteors/hook_plugins` にファイルを置き、各ワーカーが起動時に検出できるようにします。`num_workers > 1` の場合は、`MyHook` をパッケージ化して登録するか、`my_hook.py` をそのディレクトリに保存してください。

**設定による動的な切り替え**：

- フックを切り替える場合は、実行時登録より設定の切り替えを使ってください。`PipelineConfig.hooks` で実行順を指定します。

  ```python
  from meteor_core.schema import PipelineConfig, DetectionParams

  config = PipelineConfig(
      target_folder="./raw",
      output_folder="./candidates",
      debug_folder="./debug",
      params=DetectionParams(),
      hooks=[
          {"name": "my_hook", "config": {"mode": "strict"}},
          "other_hook",
      ],
  )
  ```

**設定の既定値**：

- `PipelineConfig.hooks` が `None` の場合は **すべてのフックを省略** します。
- フック一覧を明示し、`ConfigType` に必須引数がある場合は設定も渡してください。

**設計指針**：

- **引数なしで生成できるConfigType** を推奨します。設定しやすく、利用者の手間を減らせます。

<a id="hook-execution-and-state"></a>

#### 実行場所と状態の生存期間

| コールバック | 実行場所 | フックのインスタンス |
| --- | --- | --- |
| `on_file_found`、`on_all_detections_sorted` | メインプロセス | 同じ実行内で共有 |
| `on_image_loaded`、`on_detection_complete` | 逐次時はメイン、並列時はワーカー | `process_image_batch()` の呼び出しごとに生成し、同じバッチの画像ペア間で共有 |
| `on_output_saved`、`on_batch_results_sorted` | メインプロセス | 同じ実行内で共有。ただしファイル選別・最終解析のインスタンスとは別 |
| 出力ハンドラーのライフサイクルイベント | メインプロセス | パイプラインが保持する出力ハンドラー |

設定の一覧順に呼び出し、返されたコンテキスト・検出結果・ソート済み一覧を次のフックへ渡します。
フックの状態は上表のグループ間や別バッチのワーカー間では共有されません。逐次処理のバッチは1ペアです。
ROI選択・自動推定用の先頭画像やサンプルの読み込みには画像フックを適用せず、検出する画像ペアに適用します。

成功した結果の処理順は次のとおりです。

1. `OutputHandler.on_detection_result()`。画像配列を含まないコンテキスト辞書を渡します。
2. 候補なら `save_candidate()` → `on_output_saved()` → `OutputHandler.on_candidate_detected()`。
3. 進捗を記録し、バッチ内をソートして `on_batch_results_sorted()` → `OutputHandler.on_batch_complete()`。
4. 正常完了時は `OutputHandler.on_pipeline_complete()` → `on_all_detections_sorted()` → 対応するextrasを進捗へ反映。

再開時の最終ソート対象は、その実行で新たに処理した結果だけです。Ctrl-Cでは最終解析を実行しません。
`warn` は例外を警告して次のフックへ進みますが、例外前の破壊的変更や外部への副作用を元に戻すものではありません。
フック生成・設定の失敗は `warn` でも `MeteorConfigError` として送出します。

---

<a id="16-file-discovery-hook-pipeline"></a>

### 1.6 ファイル検出フック（パイプライン）

入力ファイルの収集直後に呼び出し、`InputLoader.load()` の **前に** 選別できます。

| フック | シグネチャ | 説明 |
|--------|------------|------|
| `on_file_found` | `(filepath: str) -> bool` | `True` で保持、`False` で除外 |

**補足**：

- `filepath` は **正規化した絶対パス** です。
- 拒否したファイルはパイプラインから除外します。
- 拡張子やパスのパターンによる選別に使えます。

**登録**：

- マルチプロセスではエントリーポイントか `~/.detect_meteors/hook_plugins` で検出可能にしてください。
- `meteor_core.hooks.HookRegistry.register(MyHook)` はテスト・単一プロセス向けです。
- 登録順に呼び出します。
- dataclass/Pydanticベースの `ConfigType` など、ほかのプラグインと同じ設定方式です。

---

<a id="17-image-load-hook-pipeline"></a>

### 1.7 画像読み込みフック（パイプライン）

`InputContext` の正規化直後、検出前に呼び出します。画像を変換したり、検出器や出力用のメタデータを追加できます。

| フック | シグネチャ | 説明 |
|--------|------------|------|
| `on_image_loaded` | `(context: InputContext) -> InputContext` | 画像・メタデータ・ローダー情報を更新したコンテキスト |

**補足**：

- `InputContext.metadata["frame_role"]` は `"current"` または `"previous"` で、処理するフレームの役割を示します。
- 新しい `InputContext` を返して、`image_data` やメタデータを差し替えられます。
- 例外時は `PipelineConfig.hook_error_mode` で、`"warn"` による警告・継続、または `"raise"` による送出を選びます。既定は `"raise"`。実運用では、エラーを報告しつつ処理を継続する `"warn"` を推奨します。

**登録**：

- マルチプロセスではエントリーポイントか `~/.detect_meteors/hook_plugins` で検出可能にしてください。
- `meteor_core.hooks.HookRegistry.register(MyHook)` はテスト・単一プロセス向けです。
- フレームごとに登録順で実行します。

---

<a id="18-detection-result-hook-pipeline"></a>

### 1.8 検出結果フック（パイプライン）

検出器が返した `DetectionResult` の正規化後、デバッグ画像・出力の処理前に呼び出します。スコアや候補フラグを調整し、後続処理用のメタデータを付加できます。

| フック | シグネチャ | 説明 |
|--------|------------|------|
| `on_detection_complete` | `(result: DetectionResult, context: DetectionContext) -> DetectionResult` | 更新した検出結果 |

**補足**：

- 戻り値の `DetectionResult` が後続の `is_candidate`、`score`、`debug_image` を決めます。
- 新しい `DetectionResult` で線分やextrasを上書きできます。
- 例外時は `PipelineConfig.hook_error_mode` で `"warn"`（警告して継続）か `"raise"`（送出）を選びます。既定は `"raise"`。実運用では `"warn"` を推奨します。

**登録**：

- マルチプロセスではエントリーポイントか `~/.detect_meteors/hook_plugins` で検出可能にしてください。
- `meteor_core.hooks.HookRegistry.register(MyHook)` はテスト・単一プロセス向けです。
- フレームごとに登録順で実行します。

---

<a id="19-output-saved-hook-pipeline"></a>

### 1.9 出力保存フック（パイプライン）

出力ハンドラーが正規化した `OutputResult` を返した直後に呼び出します。保存結果の計測や通知に使えます。

| フック | シグネチャ | 説明 |
|--------|------------|------|
| `on_output_saved` | `(result: OutputResult) -> None` | ハンドラーの戻り値に対する読み取り専用の通知 |

**補足**：

- `OutputResult` のスナップショットを受け取ります。`handler_info` と `metrics` は浅いコピーなので、入れ子の値は読み取り専用として扱ってください。保存をスキップした場合も `saved=False` で呼び出します。
- 例外時は `PipelineConfig.hook_error_mode` で `"warn"`（警告して継続）か `"raise"`（送出）を選びます。既定は `"raise"`。実運用では `"warn"` を推奨します。

**登録**：

- マルチプロセスではエントリーポイントか `~/.detect_meteors/hook_plugins` で検出可能にしてください。
- `meteor_core.hooks.HookRegistry.register(MyHook)` はテスト・単一プロセス向けです。
- 候補の保存を試みるごとに、登録順で呼び出します。

---

<a id="110-batch-results-sorted-hook-pipeline"></a>

### 1.10 バッチ内ソート済み結果フック（パイプライン）

バッチ処理が完了するたびに、結果を `frame_index` の昇順に並べて呼び出します。

| フック | シグネチャ | 説明 |
|--------|------------|------|
| `on_batch_results_sorted` | `(detections: List[SortedDetection]) -> List[SortedDetection]` | フレーム番号順のバッチ結果を解析 |

**補足**：

- 画像を含まない軽量な `SortedDetection` の一覧を、`frame_index` 順で受け取ります。
- 候補出力と進捗の記録後、メインプロセスで実行します。並列時はワーカーバッチの完了ごとに呼びますが、バッチ間はフレーム順ではなく完了順です。逐次処理では1組ずつ渡します。
- 実行全体での連続性を必要としない **バッチ内解析** に適しています。
- `SortedDetection.extras` を変更して解析情報を付加できます。
- 変更した、またはそのままの一覧を返してください。
- 候補フラグやスコアを変更しても、保存済みファイルや件数には反映しません。extrasは最終フックにも渡し、組み込みの進捗管理は最終解析後に `aircraft` のみを保存します。

**SortedDetectionデータクラス**：

```python
@dataclass
class SortedDetection:
    """Lightweight detection record for sorted hook processing."""

    frame_index: int                                    # 0-based index of current frame
    prev_frame_index: int                               # 0-based index of previous frame
    filename: str                                       # Base filename
    filepath: str                                       # Full path to image file
    is_candidate: bool                                  # Whether marked as candidate
    score: float                                        # Detection confidence score
    aspect_ratio: float                                 # Max contour aspect ratio
    lines: List[Tuple[int, int, int, int]]              # Detected line segments
    extras: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1                             # SORTED_DETECTION_SCHEMA_VERSION
```

**DetectionResultではなくSortedDetectionを使う理由**：

次のデータを除外してメモリを節約します。

- `debug_image`：解析に不要な大きな画像配列。
- `DetectionContext` の画像データ。

メモリ使用量を抑えて全結果を蓄積し、`on_all_detections_sorted` で数千フレームを解析できます。

**使用例**：

```python
from meteor_core.hooks import DataclassHook
from meteor_core.schema import SortedDetection
from typing import List


class MyBatchAnalyzer(DataclassHook[MyConfig]):
    plugin_name = "my_batch_analyzer"

    def on_batch_results_sorted(
        self,
        detections: List[SortedDetection],
    ) -> List[SortedDetection]:
        """Analyze detections within each batch."""
        for detection in detections:
            # Batch-local analysis (no cross-batch state)
            brightness = self._estimate_brightness(detection)
            detection.extras["brightness_estimate"] = brightness
        return detections
```

**登録**：

- マルチプロセスではエントリーポイントか `~/.detect_meteors/hook_plugins` で検出可能にしてください。
- `meteor_core.hooks.HookRegistry.register(MyHook)` はテスト・単一プロセス向けです。

---

<a id="111-all-detections-sorted-hook-pipeline"></a>

### 1.11 全検出結果のソート済みフック（パイプライン）

**パイプライン完了後**、検出結果全体を `frame_index` の昇順に並べて呼び出します。全ワーカー完了後の **メインプロセス** で実行し、バッチ間でもフレーム順を保証します。

| フック | シグネチャ | 説明 |
|--------|------------|------|
| `on_all_detections_sorted` | `(detections: List[SortedDetection]) -> List[SortedDetection]` | 全検出結果をフレーム番号順で解析 |

**補足**：

- 現在の実行で成功した検出を、非候補も含めて `frame_index` 順に渡します。フレーム番号のない失敗結果は除外します。再開時に過去の結果を `progress.json` から復元しません。
- ワーカーではなく **メインプロセス** で実行するため、インスタンス変数で安全に状態を管理できます。
- 連続するフレームが必要な **フレーム間解析** に適しています。
  - 飛行機の光跡の追跡。
  - 時系列のフィルターや平滑化。
  - 複数フレームのイベント対応付け。
- `SortedDetection.extras` に解析情報を追加できます。
- `OutputHandler.on_pipeline_complete()` の後に実行し、Ctrl-Cでは省略します。出力と件数は記録済みなので、`is_candidate` や `score` を変更しても修正しません。組み込み `ProgressManager` は `extras["aircraft"]` のみを `detected_details` の既存候補へ反映し、ほかのextrasには独自の保存処理が必要です。

**2つのソート済みフックの使い分け**：

| 用途 | 推奨フック |
|------|------------|
| バッチ内の統計 | `on_batch_results_sorted` |
| フレームごとの特徴抽出 | `on_batch_results_sorted` |
| 飛行機などのフレーム間追跡 | `on_all_detections_sorted` |
| フレーム順を必要とする時系列処理 | `on_all_detections_sorted` |
| 実行全体の情報が必要な解析 | `on_all_detections_sorted` |

**メモリ使用量**：

全結果をメモリ上で処理します。`SortedDetection` は画像を含みませんが、線分の数、`extras` の大きさ、Pythonオブジェクトの負荷に左右されます。大規模データではこれらも考慮してください。

**例：飛行機の光跡検出**

組み込み `aircraft_trail` は `on_all_detections_sorted` で追跡します。設定で有効にできます。

```python
from meteor_core import MeteorDetectionPipeline
from meteor_core.schema import HookConfig, PipelineConfig

config = PipelineConfig.with_defaults()
config.hooks = [HookConfig(name="aircraft_trail", config={"min_track_frames": 3})]
pipeline = MeteorDetectionPipeline(config)
pipeline.run(enable_roi_selection=False)
```

候補判定とスコアを維持し、各レコードに `aircraft.likelihood`、`track_id`、形状の根拠を追加します。`progress.json` へ保存するのは候補レコードのみです。`likelihood_threshold` は現在未使用で、候補を除外しません。設定フィールドと追跡の制約は[実装説明](aircraft_light_trails_hook_design.md)を参照してください。

**登録**：

- マルチプロセスではエントリーポイントか `~/.detect_meteors/hook_plugins` で検出可能にしてください。
- `meteor_core.hooks.HookRegistry.register(MyHook)` はテスト・単一プロセス向けです。

---

<a id="2-extension-points"></a>

## 2. 拡張ポイント

3つの拡張ポイントを提供します。

<a id="21-input-loaders"></a>

### 2.1 入力ローダー

**目的**：各種ファイル形式から画像を読み込む。

**作成する場面**：

- TIFF、FITSなどの新しい画像形式への対応。
- デベイヤー、正規化などの読み込み時の前処理。
- 独自メタデータの抽出。

標準CLIと `MeteorDetectionPipeline.run()` は、対象ディレクトリ直下で [`meteor_core.schema.EXTENSIONS`](../meteor_core/schema.py) のRAWパターンに一致するファイルを探します。ローダーの選択で変わるのは、選ばれたファイルの読み込み方法です。TIFFやFITSなどは、ローダーへパスを直接渡すか、独自の処理で `process_image_batch()` にペアを明示して使います。

**必須メソッド**：

| メソッド | シグネチャ | 説明 |
|----------|------------|------|
| `load` | `(filepath: str) -> InputContext` | 画像、メタデータ、ローダー情報の読み込み |

**InputContext**（`load` の戻り値）：

```python
ImageLike = Union[np.ndarray, "torch.Tensor", "PIL.Image.Image"]


@dataclass
class InputContext:
    """Input bundle for loader execution."""

    image_data: ImageLike
    filepath: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    loader_info: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1                             # INPUT_CONTEXT_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context for JSON/logging (excludes image_data)."""
        ...
```

**フィールド**：

- `image_data`：読み込んだ画素。検出器で使用するデータ。
- `filepath`：元の画像パス。
- `metadata`：EXIF、時刻、カメラ情報などのローダーのメタデータ。
- `loader_info`：`BaseInputLoader.get_info()` による識別情報。
- `schema_version`：将来の移行用の契約バージョン（現在：`1`）。

**スキーマのバージョン管理**：既存プラグインを壊さず移行するための `schema_version` を持ちます。必須フィールドの追加などでスキーマが変わるとバージョンを上げ、段階的に対応します。現在は `1` です。

**正規化の時点**：`load` の直後に `meteor_core.schema.normalize_input_context` を呼びます。古い `schema_version` には `meteor_core.schema.register_input_context_converter` で登録した変換を使い、対応できない場合は設定エラーとして拒否します。

**辞書への変換**

`context.to_dict()` は `image_data` を除き、メタデータとローダー情報の辞書を含めます。

**任意の機能**：

- EXIFなどの抽出用に `BaseMetadataExtractor` を実装。
- プラグイン情報の `name`、`version` を定義。

**BaseMetadataExtractorミックスイン**：

EXIF、時刻、カメラ情報を抽出する任意のインターフェースです。

```python
class MyLoader(DataclassInputLoader[MyConfig], BaseMetadataExtractor):
    def extract_metadata(self, filepath: str) -> Dict[str, Any]:
        # Return metadata dictionary
        return {"timestamp": ..., "camera": ..., "exposure": ...}
```

**`extract_metadata` の呼び出し時点**：

- 検出時に使うのは各 `InputContext` に含まれるメタデータです。ペアごとに抽出結果を渡すには、ローダーの `load()` で `self.extract_metadata(filepath)` を呼びます。
- 検出器には `current`、`previous` のメタデータと、`frame_index`、`prev_frame_index` を渡します。
- 個別の `MeteorDetectionPipeline.extract_metadata(filepath)` APIでは、ローダーが `BaseMetadataExtractor` を実装していればそのメソッドを、そうでなければ組み込みのEXIF抽出を使います。
- 抽出に失敗した場合は `{}` を返し、処理を続けられるようにしてください。

<a id="22-detectors"></a>

### 2.2 検出器

**目的**：流星検出アルゴリズムの実装。

**作成する場面**：

- MLやモルフォロジーなど、別の検出方法を使う。
- 明るい流星・火球など、特定条件へ最適化。
- 独自のスコアを追加。

**必須メソッド**：

| メソッド | シグネチャ | 説明 |
|----------|------------|------|
| `detect` | `(context: DetectionContext) -> DetectionResult` | 主な検出処理（下記参照） |
| `compute_line_score` | `(mask, hough_params) -> Tuple[float, List]` | 実装が必要な抽象メソッド。個別の線分スコアがない場合は `(0.0, [])` を返せます。 |

**DetectionContext**（`detect` の入力）：

```python
ImageLike = Union[np.ndarray, "torch.Tensor", "PIL.Image.Image"]


@dataclass
class DetectionContext:
    """Input bundle for detector execution."""

    current_image: ImageLike
    previous_image: ImageLike
    roi_mask: Any                                       # Typically np.ndarray (uint8 mask)
    runtime_params: Union["RuntimeParams", Dict[str, Any]]
    metadata: Dict[str, Any]                            # {"current": {...}, "previous": {...}} in pipeline
    schema_version: int = 1                             # DETECTION_CONTEXT_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context for JSON/logging (excludes current_image/previous_image/roi_mask)."""
        ...
```

**スキーマのバージョン管理**：`schema_version` で将来の移行に備えます。必須フィールドなどを変更するとバージョンを上げ、既存プラグインから段階的に移行します。現在は `1` です。

**正規化の時点**：`detect` へ渡す前に、内部で `meteor_core.schema.normalize_detection_context` を使います。通常、プラグイン側で直接呼ぶ必要はありません。独自ツールで旧コンテキストを処理する場合は `meteor_core.schema.register_detection_context_converter` で変換を登録できます。

`current_image`、`previous_image` は通常 `numpy.ndarray` ですが、ML検出器では `torch.Tensor` や `PIL.Image.Image` も渡せます。配列固有の操作が必要な場合は、検出器の冒頭で正規化してください。`meteor_core.utils.ensure_numpy` は3つの型を `numpy.ndarray` へ変換します。PyTorchでは `meteor_core.utils.ensure_tensor` で `torch.Tensor` に変換できます。

**DetectionResult**（`detect` の戻り値）：

```python
@dataclass
class DetectionResult:
    """Result returned by detectors.

    Standard diagnostics belong in ``metrics`` (e.g. ``duration_ms``,
    ``num_contours``, ``mask_area``, ``hough_votes``). Use ``extras`` for
    detector-specific or auxiliary data that should not be part of the
    normalized comparison surface.
    """

    is_candidate: bool
    score: float
    lines: List[Tuple[int, int, int, int]]
    aspect_ratio: float
    debug_image: Optional[Any]                          # Typically np.ndarray (BGR)
    extras: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1                             # DETECTION_RESULT_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result for JSON/logging (excludes debug_image)."""
        ...
```

**スキーマのバージョン管理**：`DetectionContext` と同様に、後続処理がバージョンを確認して将来の結果にも対応できる仕組みです。

**正規化の時点**：`detect` の直後に `meteor_core.schema.normalize_detection_result` を呼びます。古い結果には `meteor_core.schema.register_detection_result_converter` の変換を適用し、変換がなければ拒否します。

**共通出力と検出器固有の出力**：

`lines` は線分を中心とした共通出力です。線分を生成しない検出器も `lines=[]` を返し、線分以外の検出結果は `extras` に格納してください。

推奨するextrasのキー：

- `bounding_boxes`：矩形の一覧（`[{x1, y1, x2, y2}, ...]` など）。
- `polygons`：多角形の一覧（`[[[x, y], [x, y], ...], ...]` など）。
- `masks`：検出器固有のマスク（numpy配列、参照、パス）。

**標準の診断情報**（`DetectionResult.metrics`）：

検出器間で比較できる診断値は `metrics` に置き、後続の解析・可視化で使います。検出器固有・補助的な情報は `extras` に置いてください。

推奨キー：

- `duration_ms`：呼び出し全体の経過時間。
- `num_contours`：二値マスクで見つかった輪郭数。
- `mask_area`：線分・輪郭解析のマスクの非ゼロ画素数。
- `hough_votes`：Houghの線分の根拠数（検出した線分の数など）。

**辞書への変換**

`result.to_dict()` は `debug_image` を除き、元の `lines`、`metrics`、`extras` を含めます。

**実行時パラメータ**（`context.runtime_params`）：

`meteor_core.schema.RuntimeParams` が名前空間付きの構造を定義します。

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

`to_dict()` の形式：

```python
{
    "schema_version": 1,
    "global": {...},          # Pipeline-wide params
    "detector": {
        "<plugin_name>": {...}    # Detector-specific overrides
    },
}
```

**バージョン方針**：

- 構造が後方互換性を失う変更の場合のみ `schema_version` を上げます。
- 既存キーが有効なままなら、任意キーの追加では上げる必要はありません。

**互換性のルール**：

- `context.runtime_params` は `RuntimeParams` または同じキーを持つ辞書です。
- パイプラインが渡すのは `global` と `detector` を持つ辞書です。有効なパラメータを `global` と使用中の検出器の名前空間の両方に入れ、内側の `schema_version` は省きます。独自の呼び出し側では検出器別の上書きや従来の平坦な辞書も使えます。
- ローダーが正規化を指定しているか、最初の画像を正規化済みと判定した場合、`diff_threshold` が `1` を超えていればパイプラインが `65535` で割ります。進捗ファイルには設定値を記録します。

`BaseDetector` のヘルパー：

- `split_runtime_params(runtime_params)` → `(global_params, detector_params)`。
- `build_runtime_params(flat_params)` → 名前空間付きの辞書。
- `detect_legacy(current_image, previous_image, roi_mask, params)` → 旧シグネチャのアダプター。

| キー | 型 | 既定値 | 説明 |
|------|----|--------|------|
| `diff_threshold` | `int` / `float` | `8` | 入力のdtypeに対応するフレーム差分閾値 |
| `min_area` | `int` | `10` | 輪郭の最小面積（画素） |
| `min_line_score` | `float` | `80.0` | 候補とする最小スコア |
| `min_aspect_ratio` | `float` | `3.0` | 輪郭の最小縦横比 |
| `hough_threshold` | `int` | `10` | Hough変換の投票閾値 |
| `hough_min_line_length` | `int` | `15` | 最小線分長（画素） |
| `hough_max_line_gap` | `int` | `5` | 線分間の最大間隔（画素） |

**補足**：`compute_line_score` は通常 `detect` の内部ヘルパーです。パイプラインが呼ぶのは `detect` のみです。

#### 組み込みHoughの計測と再現

[`HoughDetector.analyze(context)`](../meteor_core/detectors/hough_default.py) は、処理済みマスク、Hough線分、線分長の合計、輪郭の計測値を含む `HoughAnalysis` を返します。`analysis.evaluate()` は設定済みのしきい値で判定し、`CandidateThresholds` を渡すと同じ計測結果に対して面積、縦横比、線分スコアのしきい値を再評価できます。差分しきい値、ROI、Hough設定を変えた場合は再計測します。面積と縦横比は同一輪郭で `>`、フレーム全体の線分スコアは `>=` で比較します。

これは組み込みHoughのAPIです。プラグインの契約は `detect(context)` です。`--analyze-image` と `--tune-params` でも組み込みHoughを使い、入力ローダーは選択できます。フックを有効にした設定を再現に使うときは `--hooks ""` で無効にします。

再現では、設定された整数の `diff_threshold` をそのままHoughに渡します。通常検出で正規化入力向けにしきい値が自動換算された場合は、再現時のマスクが変わる可能性があります。詳細は [分析・調整オプション](COMMAND_OPTIONS.md) を参照してください。

<a id="23-output-handlers"></a>

### 2.3 出力ハンドラー

**目的**：結果の保存と通知。

**作成する場面**：

- S3/GCSなどへアップロード。
- Slack、Discord、メールなどへの通知。
- データベースへの保存。
- 独自レポートの生成。

**必須メソッド**：

| メソッド | シグネチャ | 説明 |
|----------|------------|------|
| `save_candidate` | `(source_path, filename, ...) -> OutputResult` | 流星候補の保存 |
| `save_debug_image` | `(debug_image, filename, ...) -> str` | デバッグ画像の保存 |

パイプラインは `save_candidate()` に任意のデバッグ画像とROIポリゴンを渡します。`save_debug_image()` を呼ぶタイミングはハンドラーが決めます。

**OutputResult**（`save_candidate` の戻り値）：

```python
@dataclass
class OutputResult:
    """Result returned by output handlers."""

    saved: bool
    output_path: Optional[str]
    debug_path: Optional[str]
    handler_info: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1                             # OUTPUT_RESULT_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result for JSON/logging."""
        ...
```

**フィールド**：

- `saved`：候補を正常に保存した場合はTrue。
- `output_path`：保存した候補の場所（ある場合）。
- `debug_path`：デバッグ画像の場所（ある場合）。
- `handler_info`：`BaseOutputHandler.get_info()` による識別情報。
- `metrics`：経過時間、バイト数、アップロード時間などの標準的な診断。
- `schema_version`：将来の移行用のバージョン（現在：`1`）。

**スキーマのバージョン管理**：`schema_version` で互換性を維持した段階的移行に備えます。必須フィールドの追加などで変更する場合はバージョンを上げます。現在は `1` です。

**正規化の時点**：`save_candidate` の直後に `meteor_core.schema.normalize_output_result` を呼びます。古い結果には `meteor_core.schema.register_output_result_converter` の変換を使い、なければ拒否します。

**辞書への変換**

`result.to_dict()` は出力パスと診断値を辞書で返します。

**任意のライフサイクルフック**：

| フック | シグネチャ |
|--------|------------|
| `on_detection_result` | `(context, result, filepath) -> None` |
| `on_candidate_detected` | `(filename, saved, score, aspect_ratio) -> None` |
| `on_batch_complete` | `(processed_count, detected_count, batch_size) -> None` |
| `on_pipeline_complete` | `(total_processed, total_detected, elapsed_seconds) -> None` |

**フレームごとの呼び出し順**：

1. 正常に結果が得られたペアを `on_detection_complete()` の後で `on_detection_result()` に渡します。コンテキスト辞書は画像とROI配列を除きます。候補以外の結果やデバッグ出力を無効にした実行では、結果のデバッグ画像を取り除いてから渡します。
2. `result.is_candidate` が `True` の場合に `save_candidate()` を呼びます。
3. 正規化した出力結果を `on_output_saved()` に渡し、その後 `on_candidate_detected()` に候補と `saved` を渡します。

---

<a id="3-plugin-architecture"></a>

## 3. プラグイン構成

<a id="31-registry-system"></a>

### 3.1 レジストリ

種別ごとにレジストリがあります。

```python
from meteor_core.inputs import LoaderRegistry
from meteor_core.detectors import DetectorRegistry
from meteor_core.outputs import OutputHandlerRegistry
from meteor_core.hooks import HookRegistry
```

**操作**：

```python
# Register a plugin class
LoaderRegistry.register(MyLoader)

# Get plugin class by name (case-insensitive)
loader_cls = LoaderRegistry.get("my_loader")
loader_cls = LoaderRegistry.get("MY_LOADER")  # Same result

# Create configured instance
loader = LoaderRegistry.create("my_loader", {"option": "value"})

# Create default instances (requires ConfigType with zero-arg defaults)
detector = DetectorRegistry.create_default()
handler = OutputHandlerRegistry.create_default(
    output_folder="./candidates",
    debug_folder="./debug_masks",
)

# List available plugins
names = LoaderRegistry.list_available()  # ["raw", "my_loader", ...]

# Trigger discovery (automatic on first use)
LoaderRegistry.discover()
```

<a id="32-base-class-hierarchy"></a>

### 3.2 基底クラスの階層

```text
BaseInputLoader
├── DataclassInputLoader[ConfigType]
│   └── RawImageLoader
└── PydanticInputLoader[ConfigType]

BaseDetector
├── HoughDetector
├── DataclassDetector[ConfigType]
│   └── SimpleThresholdDetector
└── PydanticDetector[ConfigType]

BaseOutputHandler
├── DataclassOutputHandler[ConfigType]
│   └── FileOutputHandler
└── PydanticOutputHandler[ConfigType]

BaseHook
├── DataclassHook[ConfigType]
│   ├── AllowAllFilesFoundHook
│   └── AircraftTrailHook
└── PydanticHook[ConfigType]
```

画像のメタデータを別途抽出するローダーは、`BaseMetadataExtractor` も継承できます。

<a id="33-configuration-management-configtype"></a>

### 3.3 設定管理（ConfigType）

型付き設定として `ConfigType` を定義できます。dataclassとPydanticの使い分けは[6.1 ConfigTypeの選択](#61-choosing-configtype)を参照してください。

**変換ルール**：

| 入力 | ConfigType | 結果 |
|------|------------|------|
| `None` | 定義あり | 既定値の `ConfigType()` |
| `None` | 定義なし | `None` |
| ConfigTypeインスタンス | — | そのまま使用 |
| `dict` | Dataclass | `ConfigType(**dict)` |
| `dict` | Pydantic v2 | `ConfigType.model_validate(dict)` |
| `dict` | Pydantic v1 | `ConfigType.parse_obj(dict)` |
| その他 | — | そのまま渡す |

**エラー処理**：

- `TypeError`：必須フィールド不足、型の誤り。
- `ValueError`：Pydanticの検証失敗。

**既定インスタンスの要件**：

組み込みの入力・検出・出力で使う `create_default()` は、`ConfigType()` が完全な既定設定を生成することを前提にします。生成できない場合は `TypeError` を送り、設定不足のプラグインを黙って作らないようにします。

<a id="34-pipeline-configuration-and-cli-plugin-selection"></a>

### 3.4 パイプライン設定とCLIのプラグイン選択

<a id="configuration-files-yamljson"></a>

#### 設定ファイル（YAML/JSON）

`--config` で `PipelineConfig` に対応するYAML/JSONを読み込みます。

CLIと `load_pipeline_config()` は一部だけの設定にも対応し、省略分は既定値です。相対パスは現在の作業ディレクトリを基準にします。

```yaml
# Minimal pipeline configuration
target_folder: ./rawfiles
output_folder: ./candidates
debug_folder: ./debug_masks

params:
  diff_threshold: 8
  min_area: 10
  min_aspect_ratio: 3.0

# Plugin selection
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

**設定キー**：

| キー | 説明 |
|------|------|
| `input_loader_name` | 名前（`"raw"`、`"tiff"`、`"fits"` など） |
| `input_loader_config` | ローダーの `ConfigType` へ渡す辞書 |
| `detector_name` | 名前（`"hough"`、`"threshold"` など） |
| `detector_config` | 検出器の `ConfigType` へ渡す辞書 |
| `output_handler_name` | 名前（`"file"`、`"slack"` など） |
| `output_handler_config` | ハンドラーの `ConfigType` へ渡す辞書 |
| `hooks` | フック名や `{name, config}` の順序付きリスト、または名前から設定へのマッピング |
| `hook_error_mode` | コールバック例外を送出する `"raise"`（既定）、または警告して続行する `"warn"` |

`*_config` は標準のルールで各 `ConfigType` へ変換します。[3.3 設定管理](#33-configuration-management-configtype)を参照してください。

組み込みRAWローダーは `binning: 2` のみ対応し、正規化の既定は `false` です。Hough検出器の `ConfigType` はフィールドを持たないため、閾値は `params` で設定します。`output_handler_name: file` を明示すると `output_handler_config` とハンドラーの既定値を使い、最上位の保存先・上書き設定は統合しません。最上位の設定を引き継ぐ場合は `output_handler_name` を省略してください。

組み込みは `raw`、`hough`、`simple_threshold`、`file` と、`allow_all_files`、`aircraft_trail` のフックです。このガイドのTIFF/FITS、ML、クラウド・通知は独自拡張の例です。

**Pythonでの読み込み**：

```python
from meteor_core import MeteorDetectionPipeline, load_pipeline_config

config = load_pipeline_config("config_examples/pipeline.yaml")
pipeline = MeteorDetectionPipeline(config)
pipeline.run()
```

<a id="cli-plugin-selection"></a>

#### CLIでの選択

引数でもプラグインを指定できます。

```bash
# Select plugins by name
uv run detect-meteors \
    --input-loader raw \
    --detector hough \
    --output-handler file

# Provide plugin configs as JSON strings
uv run detect-meteors \
    --input-loader raw \
    --input-loader-config '{"binning": 2, "normalize": true}'

# Or as YAML strings (requires a custom slack output plugin)
uv run detect-meteors \
    --output-handler slack \
    --output-handler-config "webhook_url: https://hooks.slack.com/..."

# Or as file paths containing the RAW loader configuration
uv run detect-meteors \
    --input-loader-config raw_loader_settings.yaml

# Select hooks and supply per-hook settings
uv run detect-meteors \
    --hooks allow_all_files,aircraft_trail \
    --hook-config '{"aircraft_trail": {"min_track_frames": 3}}'
```

**オプション**：

| オプション | 説明 |
|------------|------|
| `--input-loader NAME` | 入力ローダーの選択 |
| `--input-loader-config VALUE` | JSON/YAML文字列またはパス |
| `--detector NAME` | 検出器の選択 |
| `--detector-config VALUE` | JSON/YAML文字列またはパス |
| `--output-handler NAME` | 出力ハンドラーの選択 |
| `--output-handler-config VALUE` | JSON/YAML文字列またはパス |
| `--hooks NAMES` | 実行順に並べたカンマ区切りのフック名。`--hooks ""` で設定済みのフックを解除 |
| `--hook-config VALUE` | フックのリストまたは名前と設定のマッピング。JSON/YAML文字列またはパス |

<a id="configuration-precedence"></a>

#### 設定の優先順

次の順に優先します。

1. **CLI引数**（`--detector`、`--detector-config` など）。
2. **設定ファイル**（`--config pipeline.yaml`）。
3. **組み込みの既定値**。

CLIでプラグイン設定を指定すると、対応する設定のマッピングを置き換えます。`--hooks` で実行順を指定し、`--hook-config` で各フックの設定を渡します。`--hook-config` だけを指定した場合は、その項目がフック一覧になります。いずれもパイプライン設定ファイルのフック設定を置き換えます。

<a id="plugin-author-considerations"></a>

#### プラグイン作者への留意点

設定ファイルに対応する場合：

1. **わかりやすいフィールド名**：`ConfigType` がYAML/JSONのキーになります。名前と説明を明確にしてください。
2. **適切な既定値**：任意設定を省略できるようにしてください。
3. **必須フィールドの文書化**：必須値がある場合は明記してください。
4. **早期の検証**：Pydanticや `__post_init__` で読み込み時にエラーを発見してください。

説明付き設定の例：

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

    sensitivity: float = 0.5      # Detection sensitivity (0.0-1.0)
    use_gpu: bool = False         # Enable GPU acceleration
    model_path: str = ""          # Path to ML model (optional)
```

<a id="35-plugin-discovery"></a>

### 3.5 プラグインの検出

各レジストリは次の順に検出します。

1. **組み込み**。
2. **エントリーポイント**（名前のアルファベット順）。
3. **ディレクトリ**（ファイル名のアルファベット順）。

同名のクラスがある場合、入力ローダー・検出器・出力ハンドラーでは先に検出したクラスを、フックでは後から検出したクラスを使います。`Registry.register()` で実行時に登録したクラスは、そのプロセス内の検索で優先されます。

**ディレクトリ**：

| 種別 | ディレクトリ |
|------|--------------|
| 入力 | `~/.detect_meteors/input_plugins/` |
| 検出器 | `~/.detect_meteors/detector_plugins/` |
| 出力 | `~/.detect_meteors/output_plugins/` |
| パイプラインフック | `~/.detect_meteors/hook_plugins/` |

**ディレクトリでの検出の仕組み**：

1. 対応するディレクトリに `.py` を保存します。必要なら作成してください。
2. 初回アクセスで `.py` をアルファベット順に読み込みます。
3. 正しい基底クラスを継承し、`plugin_name` を持つクラスを自動登録します。`Registry.register()` は **不要** です。
4. 特別なファイル名は不要ですが、内容のわかる名前を推奨します。

ファイル構成の例：

```
~/.detect_meteors/
└── input_plugins/
    ├── fits_loader.py      # Defines: class FitsLoader(DataclassInputLoader)
    └── tiff_loader.py      # Defines: class TiffLoader(DataclassInputLoader)
```

**エントリーポイント**（`pyproject.toml`）：

```toml
[project.entry-points."detect_meteors.input"]
my_loader = "my_package.loaders:MyLoader"

[project.entry-points."detect_meteors.detector"]
my_detector = "my_package.detectors:MyDetector"

[project.entry-points."detect_meteors.output"]
my_handler = "my_package.handlers:MyHandler"

[project.entry-points."detect_meteors.hook"]
my_hook = "my_package.hooks:MyHook"
```

Detect Meteorsを実行する環境にプラグインをインストールします。隣のディレクトリにあるローカルパッケージなら、Detect Meteorsのソースルートから次のように検出を確認できます。

```bash
uv run --with-editable ../my-plugin python -c "from meteor_core.inputs import LoaderRegistry; print(LoaderRegistry.list_available())"
```

---

<a id="4-data-contracts-reference"></a>

## 4. データ契約リファレンス

プラグインが使うデータクラスの詳細です。すべて `meteor_core.schema` からインポートします。

```python
from meteor_core.schema import (
    InputContext,
    DetectionContext,
    DetectionResult,
    OutputResult,
    RuntimeParams,
)
```

<a id="41-inputcontext"></a>

### 4.1 InputContext

読み込んだ画像とメタデータを、後続の処理へ渡します。

**インポート**：`from meteor_core.schema import InputContext`

```python
@dataclass
class InputContext:
    """Input bundle for loader execution."""

    image_data: ImageLike                               # Loaded image pixels
    filepath: str                                       # Original file path
    metadata: Dict[str, Any] = field(default_factory=dict)
    loader_info: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = INPUT_CONTEXT_SCHEMA_VERSION  # Currently 1

    def to_dict(self) -> Dict[str, Any]: ...
```

**フィールド**：

| フィールド | 型 | 既定値 | 説明 |
|------------|----|--------|------|
| `image_data` | `ImageLike` | — | 画素。`np.ndarray`、`torch.Tensor`、`PIL.Image.Image` に対応 |
| `filepath` | `str` | — | 元の画像パス |
| `metadata` | `Dict[str, Any]` | `{}` | EXIF、時刻、カメラなどのメタデータ |
| `loader_info` | `Dict[str, Any]` | `{}` | `BaseInputLoader.get_info()` の識別情報 |
| `schema_version` | `int` | `1` | 将来の移行用の契約バージョン |

**使用例**：

```python
from meteor_core.schema import InputContext
from meteor_core.inputs import DataclassInputLoader


class MyLoader(DataclassInputLoader[MyConfig]):
    def load(self, filepath: str) -> InputContext:
        image = self._load_image(filepath)
        return InputContext(
            image_data=image,
            filepath=filepath,
            metadata={"camera": "Canon EOS R5", "iso": 6400},
            loader_info=self.get_info(),
        )
```

**辞書への変換**

`context.to_dict()` は `image_data` を除き、メタデータとローダー情報の辞書を共有します。

<a id="42-detectioncontext"></a>

### 4.2 DetectionContext

検出器に必要な入力をまとめます。

**インポート**：`from meteor_core.schema import DetectionContext`

```python
@dataclass
class DetectionContext:
    """Input bundle for detector execution."""

    current_image: ImageLike                            # Current frame
    previous_image: ImageLike                           # Previous frame for differencing
    roi_mask: Any                                       # ROI mask (typically np.ndarray uint8)
    runtime_params: Union[RuntimeParams, Dict[str, Any]]
    metadata: Dict[str, Any]                                # {"current": {...}, "previous": {...}}
    schema_version: int = DETECTION_CONTEXT_SCHEMA_VERSION  # Currently 1

    def to_dict(self) -> Dict[str, Any]: ...
```

**フィールド**：

| フィールド | 型 | 既定値 | 説明 |
|------------|----|--------|------|
| `current_image` | `ImageLike` | — | 解析する現在のフレーム |
| `previous_image` | `ImageLike` | — | 差分用の前フレーム |
| `roi_mask` | `Any` | — | ROIマスク（通常dtypeが `uint8` の `np.ndarray`） |
| `runtime_params` | `RuntimeParams \| Dict` | — | 実行パラメータ（下記参照） |
| `metadata` | `Dict[str, Any]` | — | パイプラインが `current`、`previous`、`frame_index`、`prev_frame_index` を設定 |
| `schema_version` | `int` | `1` | 将来の移行用の契約バージョン |

**RuntimeParamsの構造**：

```python
@dataclass
class RuntimeParams:
    schema_version: int = 1
    global_params: Dict[str, Any] = field(default_factory=dict)  # Pipeline-wide params
    detector: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )  # Per-detector overrides
```

`to_dict()` の形式：

```python
{
    "schema_version": 1,
    "global": {"diff_threshold": 8, "min_area": 10},
    "detector": {"hough": {"hough_threshold": 50}},
}
```

**使用例**：

```python
from meteor_core.schema import DetectionContext, DetectionResult
from meteor_core.detectors import DataclassDetector
from meteor_core.utils import ensure_numpy


class MyDetector(DataclassDetector[MyConfig]):
    def detect(self, context: DetectionContext) -> DetectionResult:
        # Normalize images to numpy arrays
        current = ensure_numpy(context.current_image)
        previous = ensure_numpy(context.previous_image)
        roi_mask = ensure_numpy(context.roi_mask)

        # Extract runtime params
        global_params, detector_params = self.split_runtime_params(
            context.runtime_params
        )
        params = {**global_params, **detector_params}

        # Access per-frame metadata
        current_meta = context.metadata.get("current", {})
        previous_meta = context.metadata.get("previous", {})

        # Perform detection...
        return DetectionResult(...)
```

**辞書への変換**

`context.to_dict()` は `current_image`、`previous_image`、`roi_mask` を除きます。同じプロセス内では、メタデータと入れ子の実行時設定をコンテキストと共有します。並列実行で返すペイロードと結果はpickle化できる必要があります。JSONのログやレポートに渡すメタデータ、`metrics`、`extras` はJSON互換の値にしてください。

**正規化**：`detect` の前に `meteor_core.schema.normalize_detection_context()` で内部正規化します。通常プラグインが直接呼ぶ必要はありませんが、シリアライズ済みコンテキストを扱う独自ツールで利用でき、`register_detection_context_converter()` で変換も登録できます。

<a id="43-detectionresult"></a>

### 4.3 DetectionResult

検出器の `detect()` の出力をまとめます。

**インポート**：`from meteor_core.schema import DetectionResult`

```python
@dataclass
class DetectionResult:
    """Result returned by detectors.

    Standard diagnostics belong in ``metrics`` (e.g. ``duration_ms``,
    ``num_contours``, ``mask_area``, ``hough_votes``). Use ``extras`` for
    detector-specific or auxiliary data that should not be part of the
    normalized comparison surface.
    """

    is_candidate: bool                                  # Detection decision
    score: float                                        # Detection confidence score
    lines: List[Tuple[int, int, int, int]]              # Detected line segments
    aspect_ratio: float                                 # Max contour aspect ratio
    debug_image: Optional[Any]                          # Debug visualization (BGR)
    extras: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = DETECTION_RESULT_SCHEMA_VERSION  # Currently 1

    def to_dict(self) -> Dict[str, Any]: ...
```

**フィールド**：

| フィールド | 型 | 既定値 | 説明 |
|------------|----|--------|------|
| `is_candidate` | `bool` | — | 流星候補を含む場合は `True` |
| `score` | `float` | — | 検出の信頼度スコア（高いほど信頼度が高い） |
| `lines` | `List[Tuple[int, int, int, int]]` | — | `(x1, y1, x2, y2)` の線分一覧 |
| `aspect_ratio` | `float` | — | 輪郭の最大縦横比 |
| `debug_image` | `Optional[Any]` | — | 可視化画像（通常BGRの `np.ndarray`） |
| `extras` | `Dict[str, Any]` | `{}` | 検出器固有の補助情報 |
| `metrics` | `Dict[str, Any]` | `{}` | 解析ツール用の標準診断 |
| `schema_version` | `int` | `1` | 将来の移行用の契約バージョン |

**metricsとextras**：

| 辞書 | 目的 | 推奨キー |
|------|------|----------|
| `metrics` | 後続解析用の共通診断 | `duration_ms`、`num_contours`、`mask_area`、`hough_votes` |
| `extras` | 検出器固有・補助情報 | `bounding_boxes`、`polygons`、`masks`、独自キー |

**使用例**：

```python
from meteor_core.schema import DetectionResult


def detect(self, context: DetectionContext) -> DetectionResult:
    start_time = time.perf_counter()

    # ... detection logic ...

    duration_ms = (time.perf_counter() - start_time) * 1000

    return DetectionResult(
        is_candidate=score >= threshold,
        score=score,
        lines=[(x1, y1, x2, y2) for line in detected_lines],
        aspect_ratio=max_aspect_ratio,
        debug_image=debug_visualization,
        extras={
            "bounding_boxes": [{"x1": 10, "y1": 20, "x2": 100, "y2": 50}],
            "algorithm_variant": "adaptive",
        },
        metrics={
            "duration_ms": duration_ms,
            "num_contours": len(contours),
            "mask_area": int(np.count_nonzero(binary_mask)),
            "hough_votes": len(hough_lines),
        },
    )
```

**辞書への変換**

`result.to_dict()` は `debug_image` を除き、線分リストと診断用の辞書を結果と共有します。

<a id="44-outputresult"></a>

### 4.4 OutputResult

出力ハンドラーの `save_candidate()` の結果をまとめます。

**インポート**：`from meteor_core.schema import OutputResult`

```python
@dataclass
class OutputResult:
    """Result returned by output handlers."""

    saved: bool                                         # Whether save succeeded
    output_path: Optional[str]                          # Path to saved candidate
    debug_path: Optional[str]                           # Path to saved debug image
    handler_info: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = OUTPUT_RESULT_SCHEMA_VERSION  # Currently 1

    def to_dict(self) -> Dict[str, Any]: ...
```

**フィールド**：

| フィールド | 型 | 既定値 | 説明 |
|------------|----|--------|------|
| `saved` | `bool` | — | 正常に候補を保存した場合は `True` |
| `output_path` | `Optional[str]` | — | 候補の保存先（ある場合） |
| `debug_path` | `Optional[str]` | — | デバッグ画像の保存先（ある場合） |
| `handler_info` | `Dict[str, Any]` | `{}` | `BaseOutputHandler.get_info()` の識別情報 |
| `metrics` | `Dict[str, Any]` | `{}` | 経過時間、バイト数などの診断 |
| `schema_version` | `int` | `1` | 将来の移行用の契約バージョン |

**使用例**：

```python
from meteor_core.schema import OutputResult


def save_candidate(
    self,
    source_path: str,
    filename: str,
    debug_image: Optional[np.ndarray] = None,
    roi_polygon: Optional[List[List[int]]] = None,
) -> OutputResult:
    start_time = time.perf_counter()
    dest_path = os.path.join(self.config.output_folder, filename)

    try:
        shutil.copy2(source_path, dest_path)
        bytes_written = os.path.getsize(dest_path)

        debug_path = None
        if debug_image is not None:
            debug_path = self.save_debug_image(debug_image, filename, roi_polygon)

        duration_ms = (time.perf_counter() - start_time) * 1000

        return OutputResult(
            saved=True,
            output_path=dest_path,
            debug_path=debug_path,
            handler_info=self.get_info(),
            metrics={
                "duration_ms": duration_ms,
                "bytes_written": bytes_written,
            },
        )
    except OSError as e:
        return OutputResult(
            saved=False,
            output_path=dest_path,
            debug_path=None,
            handler_info=self.get_info(),
            metrics={"error": str(e)},
        )
```

**辞書への変換**

`result.to_dict()` は出力パス、ハンドラー情報、`metrics` を含めます。

<a id="45-sorteddetection"></a>

### 4.5 SortedDetection

ソート済みフックで使う軽量データクラスです。フレーム間解析で全結果を収集するときのメモリを節約します。

**インポート**：`from meteor_core.schema import SortedDetection`

```python
@dataclass
class SortedDetection:
    """Lightweight detection record for sorted hook processing."""

    frame_index: int                                    # 0-based index of current frame
    prev_frame_index: int                               # 0-based index of previous frame
    filename: str                                       # Base filename
    filepath: str                                       # Full path to image file
    is_candidate: bool                                  # Whether marked as candidate
    score: float                                        # Detection confidence score
    aspect_ratio: float                                 # Max contour aspect ratio
    lines: List[Tuple[int, int, int, int]]              # Detected line segments
    extras: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = SORTED_DETECTION_SCHEMA_VERSION  # Currently 1

    def to_dict(self) -> Dict[str, Any]: ...

    @classmethod
    def from_detection_result(
        cls,
        result: DetectionResult,
        frame_index: int,
        prev_frame_index: int,
        filename: str,
        filepath: str,
    ) -> "SortedDetection": ...
```

**フィールド**：

| フィールド | 型 | 既定値 | 説明 |
|------------|----|--------|------|
| `frame_index` | `int` | — | 現在のフレームの0始まりの番号 |
| `prev_frame_index` | `int` | — | 差分用の前フレームの0始まりの番号 |
| `filename` | `str` | — | ファイル名（例：`"IMG_0001.CR2"`） |
| `filepath` | `str` | — | 処理した画像の絶対パス |
| `is_candidate` | `bool` | — | 候補の場合は `True` |
| `score` | `float` | — | 検出器の信頼度スコア |
| `aspect_ratio` | `float` | — | 輪郭の最大縦横比 |
| `lines` | `List[Tuple[int, int, int, int]]` | — | `(x1, y1, x2, y2)` の線分一覧 |
| `extras` | `Dict[str, Any]` | `{}` | 検出器固有、またはフックが付加した補助情報 |
| `schema_version` | `int` | `1` | 将来の移行用の契約バージョン |

**SortedDetectionを使う理由**：

数千件を収集するときに大きなメモリを使うデータを除外します。

| 除外するデータ | 理由 |
|----------------|------|
| `debug_image` | 各画像が数MBになるBGR配列 |
| `current_image` / `previous_image` | `DetectionContext` のフレームデータ |
| `roi_mask` | 二値マスク配列 |
| `metrics` | フレーム間解析に不要な診断 |

フレーム番号は `on_file_found()` で選別した後、再開時に処理済みフレームを除外する前の並びを表します。先頭ファイルは0です。そのため、再開時のソート済みフックには飛び番があり得ます。

**ファクトリーメソッド**：

`SortedDetection.from_detection_result()` で `DetectionResult` から生成できます。

```python
from meteor_core.schema import SortedDetection, DetectionResult

# After detection completes
detection_result: DetectionResult = detector.detect(context)

sorted_detection = SortedDetection.from_detection_result(
    result=detection_result,
    frame_index=42,
    prev_frame_index=41,
    filename="IMG_0042.CR2",
    filepath="/path/to/IMG_0042.CR2",
)
```

**フックでの使用例**：

```python
from meteor_core.hooks import DataclassHook
from meteor_core.schema import SortedDetection
from typing import List


class MyAnalyzer(DataclassHook[MyConfig]):
    plugin_name = "my_analyzer"

    def on_all_detections_sorted(
        self,
        detections: List[SortedDetection],
    ) -> List[SortedDetection]:
        for detection in detections:
            # Access detection data
            frame_idx = detection.frame_index
            lines = detection.lines
            score = detection.score

            # Add analysis results to extras
            detection.extras["my_analysis"] = {
                "computed_value": self._analyze(detection),
            }

        return detections
```

**辞書への変換**

`detection.to_dict()` は元の `lines` と `extras` を含めます。

<a id="46-schema-versioning-and-normalization"></a>

### 4.6 スキーマのバージョン管理と正規化

4つのデータクラスに `schema_version`（現在：`1`）を持ち、将来の変更に備えます。

**正規化・変換関数の一覧**：

| データクラス | 正規化関数 | 変換関数の登録 |
|--------------|------------|----------------|
| `InputContext` | `meteor_core.schema.normalize_input_context()` | `register_input_context_converter()` |
| `DetectionContext` | `meteor_core.schema.normalize_detection_context()` | `register_detection_context_converter()` |
| `DetectionResult` | `meteor_core.schema.normalize_detection_result()` | `register_detection_result_converter()` |
| `OutputResult` | `meteor_core.schema.normalize_output_result()` | `register_output_result_converter()` |

> **補足**：`DetectionContext` は内部で正規化しますが、独自ツールや処理でも正規化・変換APIを使えます。

**仕組み**：

1. プラグインは実装した `schema_version` のインスタンスを返します。
2. 戻り値を受け取った直後に `normalize_*()` を呼びます。
3. 現行バージョンならそのまま通します。
4. 古いバージョンなら登録した変換で更新します。
5. 公開の正規化関数は未対応のバージョンに対して `ValueError` を送出します。パイプラインはこのエラーを `MeteorConfigError` に変換し、検出ペアや出力処理の失敗時の扱いに従います。

**後方互換用の変換登録**：

```python
from meteor_core.schema import (
    DetectionContext,
    DetectionResult,
    register_input_context_converter,
    register_detection_context_converter,
    register_detection_result_converter,
    register_output_result_converter,
)


def upgrade_detection_context_v0_to_v1(context: DetectionContext) -> DetectionContext:
    """Convert v0 DetectionContext to v1 format."""
    return DetectionContext(
        current_image=context.current_image,
        previous_image=context.previous_image,
        roi_mask=context.roi_mask,
        runtime_params=context.runtime_params,
        metadata=context.metadata,
        schema_version=1,
    )


def upgrade_detection_result_v0_to_v1(result: DetectionResult) -> DetectionResult:
    """Convert v0 DetectionResult to v1 format."""
    return DetectionResult(
        is_candidate=result.is_candidate,
        score=result.score,
        lines=result.lines,
        aspect_ratio=result.aspect_ratio,
        debug_image=result.debug_image,
        extras=result.extras,
        metrics=result.metrics if hasattr(result, "metrics") else {},
        schema_version=1,
    )


register_detection_context_converter(0, upgrade_detection_context_v0_to_v1)
register_detection_result_converter(0, upgrade_detection_result_v0_to_v1)
```

**バージョン方針**：

- 後方互換性のない構造変更の場合のみ `schema_version` を上げます。
- 任意フィールドの追加では上げる必要はありません。
- 現在はすべて `1` です。

---

<a id="5-sample-code"></a>

## 5. コード例

<a id="51-input-loader-complete-example"></a>

### 5.1 入力ローダー（完全な例）

このローダーはTIFFのパスを直接受け取ります。プラグインパッケージの依存に `tifffile` を追加してください。`normalize=True` の場合は入力のビット深度を `bit_depth` で指定し、画素を `float32` の [0, 1] に正規化します。

```python
"""Custom TIFF image loader with metadata extraction, logging, and exceptions."""

import logging
import os
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np

from meteor_core.inputs import (
    DataclassInputLoader,
    BaseMetadataExtractor,
    LoaderRegistry,
)
from meteor_core.schema import InputContext
from meteor_core.exceptions import (
    MeteorLoadError,
    MeteorUnsupportedFormatError,
)

# Set up logger for this plugin
logger = logging.getLogger("meteor_core.inputs.tiff_loader")


@dataclass
class TiffLoaderConfig:
    """Configuration for TIFF loader."""

    normalize: bool = False
    bit_depth: int = 16


class TiffImageLoader(DataclassInputLoader[TiffLoaderConfig], BaseMetadataExtractor):
    """Load TIFF images with optional normalization."""

    plugin_name = "tiff"           # Required: unique identifier
    name = "TIFF Image Loader"     # Optional: human-readable name
    version = "1.0.0"              # Optional: version string
    ConfigType = TiffLoaderConfig  # Required by DataclassInputLoader

    def __init__(self, config: TiffLoaderConfig):
        super().__init__(config)
        logger.debug(f"TiffImageLoader initialized with config: {self.config}")

    def load(self, filepath: str) -> InputContext:
        """Load a TIFF image file.

        Args:
            filepath: Path to the TIFF file.

        Returns:
            InputContext with the loaded image data.

        Raises:
            MeteorUnsupportedFormatError: If file is not a TIFF.
            MeteorLoadError: If file cannot be loaded.
        """
        logger.debug(f"Loading TIFF file: {filepath}")

        # Validate file extension
        if not filepath.lower().endswith((".tiff", ".tif")):
            logger.warning(f"Unsupported file extension: {filepath}")
            raise MeteorUnsupportedFormatError(
                f"Unsupported format: {filepath}",
                filepath=filepath,
                context={"supported_formats": [".tiff", ".tif"]},
            )

        # Check file existence
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            raise MeteorLoadError(
                f"File not found: {filepath}",
                filepath=filepath,
            )

        try:
            import tifffile

            image = tifffile.imread(filepath)
            logger.debug(f"Raw image shape: {image.shape}, dtype: {image.dtype}")

            # Convert to grayscale if needed
            if len(image.shape) == 3:
                logger.debug("Converting RGB to grayscale")
                image = np.mean(image, axis=2)

            # Normalize if configured
            if self.config.normalize:
                max_val = 2**self.config.bit_depth - 1
                image = image.astype(np.float32) / max_val
                logger.debug(
                    f"Normalized to float32 (bit_depth={self.config.bit_depth})"
                )
            else:
                image = image.astype(np.uint16)

            logger.debug(f"Final image shape: {image.shape}, dtype: {image.dtype}")
            return InputContext(
                image_data=image,
                filepath=filepath,
                metadata=self.extract_metadata(filepath),
                loader_info=self.get_info(),
            )

        except ImportError as e:
            logger.error("tifffile package not installed")
            raise MeteorLoadError(
                "tifffile package is required for TIFF support",
                filepath=filepath,
                original_error=e,
                context={"install_hint": "uv add tifffile"},
            )
        except Exception as e:
            logger.exception(f"Failed to load TIFF file: {filepath}")
            raise MeteorLoadError(
                f"Failed to load TIFF file: {e}",
                filepath=filepath,
                original_error=e,
                context={"loader": self.plugin_name},
            )

    def extract_metadata(self, filepath: str) -> Dict[str, Any]:
        """Extract metadata from TIFF file.

        This method should NOT raise exceptions - return empty dict on failure.

        Args:
            filepath: Path to the TIFF file.

        Returns:
            Dictionary with metadata, or empty dict on failure.
        """
        logger.debug(f"Extracting metadata from: {filepath}")

        try:
            import tifffile

            with tifffile.TiffFile(filepath) as tif:
                page = tif.pages[0]
                metadata = {
                    "width": page.shape[1] if len(page.shape) > 1 else page.shape[0],
                    "height": page.shape[0],
                    "dtype": str(page.dtype),
                    "compression": page.compression.name if page.compression else None,
                }
                logger.debug(f"Extracted metadata: {list(metadata.keys())}")
                return metadata

        except Exception as e:
            # Metadata extraction is optional - don't fail the pipeline
            logger.warning(f"Could not extract metadata from {filepath}: {e}")
            return {}


# Register the plugin
LoaderRegistry.register(TiffImageLoader)
```

<a id="52-detector-complete-example"></a>

### 5.2 検出器（完全な例）

```python
"""Simple threshold-based detector for bright meteors with logging."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np

from meteor_core.detectors import DataclassDetector, DetectorRegistry
from meteor_core.schema import DetectionContext, DetectionResult
from meteor_core.utils import ensure_numpy

# Set up logger for this plugin
logger = logging.getLogger("meteor_core.detectors.threshold")


@dataclass
class ThresholdDetectorConfig:
    """Configuration for threshold detector."""

    brightness_multiplier: float = 1.5
    min_contour_area: int = 50


class ThresholdDetector(DataclassDetector[ThresholdDetectorConfig]):
    """Detect bright meteors using simple thresholding."""

    plugin_name = "threshold"
    name = "Threshold Detector"
    version = "1.0.0"
    ConfigType = ThresholdDetectorConfig

    def __init__(self, config: ThresholdDetectorConfig = None):
        super().__init__(config)
        logger.debug(
            f"ThresholdDetector initialized: "
            f"brightness_multiplier={self.config.brightness_multiplier}, "
            f"min_contour_area={self.config.min_contour_area}"
        )

    def detect(
        self,
        context: DetectionContext,
    ) -> DetectionResult:
        """Detect meteor candidates using threshold-based approach.

        Raise for invalid inputs/configuration or return a failure result for
        recoverable cases. The pipeline treats exceptions as no-detection results.

        Args:
            context: Input bundle containing frames, ROI, and runtime params.

        Returns:
            DetectionResult with the detection outcome.
        """
        current_image = ensure_numpy(context.current_image)
        previous_image = ensure_numpy(context.previous_image)
        roi_mask = ensure_numpy(context.roi_mask)
        if current_image.dtype not in (np.uint16, np.float32):
            raise ValueError("Expected uint16 or normalized float32 input")

        global_params, detector_params = self.split_runtime_params(
            context.runtime_params
        )
        params = {**global_params, **detector_params}
        logger.debug(
            f"Starting detection: image_shape={current_image.shape}, "
            f"diff_threshold={params.get('diff_threshold', 8)}"
        )

        try:
            # Compute absolute difference
            diff = cv2.absdiff(current_image, previous_image)
            logger.debug(
                f"Computed frame difference: max={diff.max()}, mean={diff.mean():.1f}"
            )

            # Apply ROI mask
            diff = cv2.bitwise_and(diff, diff, mask=roi_mask)

            # Get threshold from params
            threshold = params.get("diff_threshold", 8)
            effective_threshold = float(threshold) * self.config.brightness_multiplier
            logger.debug(f"Applying threshold: {effective_threshold}")

            # Apply threshold
            _, binary = cv2.threshold(diff, effective_threshold, 255, cv2.THRESH_BINARY)
            binary = binary.astype(np.uint8)

            # Find contours
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            logger.debug(f"Found {len(contours)} raw contours")

            # Filter by area
            valid_contours = [
                c
                for c in contours
                if cv2.contourArea(c) >= self.config.min_contour_area
            ]
            logger.debug(
                f"After area filter (>={self.config.min_contour_area}): "
                f"{len(valid_contours)} contours"
            )

            if not valid_contours:
                logger.debug("No valid contours found - returning no detection")
                return DetectionResult(
                    is_candidate=False,
                    score=0.0,
                    lines=[],
                    aspect_ratio=0.0,
                    debug_image=None,
                    extras={},
                    metrics={
                        "duration_ms": 0.0,
                        "num_contours": 0,
                        "mask_area": 0,
                        "hough_votes": 0,
                    },
                )

            # Compute metrics
            max_area = max(cv2.contourArea(c) for c in valid_contours)
            score = float(max_area) / 100.0

            # Compute aspect ratios
            aspect_ratios = []
            for c in valid_contours:
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratios.append(max(w, h) / max(min(w, h), 1))
            max_aspect_ratio = max(aspect_ratios) if aspect_ratios else 0.0

            logger.debug(
                f"Metrics: score={score:.1f}, max_aspect_ratio={max_aspect_ratio:.2f}"
            )

            # Create debug image
            if current_image.dtype == np.float32:
                preview = (np.clip(current_image, 0, 1) * 255).astype(np.uint8)
            else:
                preview = (current_image // 256).astype(np.uint8)
            debug_image = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(debug_image, valid_contours, -1, (0, 255, 0), 2)

            bounding_boxes = []
            for contour in valid_contours:
                x, y, width, height = cv2.boundingRect(contour)
                bounding_boxes.append(
                    {"x1": x, "y1": y, "x2": x + width, "y2": y + height}
                )

            min_score = params.get("min_line_score", 30.0)
            is_candidate = score >= min_score

            if is_candidate:
                logger.info(f"Candidate detected: score={score:.1f} >= {min_score}")
            else:
                logger.debug(f"Not a candidate: score={score:.1f} < {min_score}")

            return DetectionResult(
                is_candidate=is_candidate,
                score=score,
                lines=[],
                aspect_ratio=max_aspect_ratio,
                debug_image=debug_image,
                extras={"bounding_boxes": bounding_boxes},
                metrics={
                    "duration_ms": 0.0,
                    "num_contours": len(valid_contours),
                    "mask_area": int(np.count_nonzero(binary)),
                    "hough_votes": 0,
                },
            )

        except Exception as e:
            # The pipeline treats exceptions as a failed detection (no candidate).
            # Raise if you want the pipeline to log the error for this file.
            logger.warning(f"Detection failed with error: {e}")
            raise

    def compute_line_score(
        self,
        mask: np.ndarray,
        hough_params: Dict[str, int],
    ) -> Tuple[float, List[Tuple[int, int, int, int]]]:
        """Compute line score (simplified for threshold detector).

        Args:
            mask: Binary mask of detected regions.
            hough_params: Hough transform parameters (unused here).

        Returns:
            Tuple of (score, line_segments).
        """
        logger.debug(f"Computing line score for mask shape: {mask.shape}")

        try:
            # Find contours and compute score
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                logger.debug("No contours found in mask")
                return 0.0, []

            total_area = sum(cv2.contourArea(c) for c in contours)
            score = float(total_area) / 100.0
            logger.debug(f"Line score: {score:.1f} (from {len(contours)} contours)")
            return score, []

        except Exception as e:
            logger.warning(f"compute_line_score failed: {e}")
            return 0.0, []


# Register the plugin
DetectorRegistry.register(ThresholdDetector)
```

<a id="53-output-handler-with-lifecycle-hooks-secondary-handler-example"></a>

### 5.3 ライフサイクルフックを持つ出力ハンドラー（補助ハンドラーの例）

フレームごとのフックは検出中に呼びます。`DetectionResult.lines` で線分を確認し、`DetectionResult.extras` で矩形、マスク、アルゴリズムのタグなどを参照できます。`context` は画像を含まない `DetectionContext.to_dict()` です。ログや観察用ツールへ渡せるよう、`extras` はJSON互換にしてください。

```python
"""Slack notification handler with full lifecycle support, logging, and exceptions."""

import json
import logging
import os
import shutil
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from meteor_core.i18n import DEFAULT_LOCALE, get_message
from meteor_core.outputs import DataclassOutputHandler, OutputHandlerRegistry
from meteor_core.exceptions import MeteorWriteError
from meteor_core.schema import DetectionResult, OutputResult

# Set up logger for this plugin
logger = logging.getLogger("meteor_core.outputs.slack_handler")


@dataclass
class SlackOutputConfig:
    """Configuration for Slack notification handler."""

    output_folder: str = "./candidates"
    debug_folder: str = "./debug_masks"
    webhook_url: str = ""
    notify_on_detection: bool = True
    notify_on_complete: bool = True
    channel: str = "#meteor-alerts"
    locale: str = DEFAULT_LOCALE


class SlackNotificationHandler(DataclassOutputHandler[SlackOutputConfig]):
    """Output handler with Slack notifications."""

    plugin_name = "slack"
    name = "Slack Notification Handler"
    version = "1.0.0"
    ConfigType = SlackOutputConfig

    def __init__(self, config: SlackOutputConfig):
        super().__init__(config)
        self._detection_count = 0

        logger.debug(
            f"SlackNotificationHandler initialized: "
            f"output={self.config.output_folder}, "
            f"webhook={'configured' if self.config.webhook_url else 'not configured'}"
        )

        # Create output directories
        try:
            os.makedirs(self.config.output_folder, exist_ok=True)
            logger.debug(f"Created output folder: {self.config.output_folder}")
            if self.config.debug_folder:
                os.makedirs(self.config.debug_folder, exist_ok=True)
                logger.debug(f"Created debug folder: {self.config.debug_folder}")
        except OSError as e:
            logger.error(f"Failed to create output directories: {e}")
            # Don't raise - let pipeline continue and fail on actual save

    def save_candidate(
        self,
        source_path: str,
        filename: str,
        debug_image: Optional[np.ndarray] = None,
        roi_polygon: Optional[List[List[int]]] = None,
    ) -> OutputResult:
        """Save a meteor candidate file.

        NOTE: This is a secondary (non-critical) handler example.
        It returns False on failure instead of raising exceptions.
        For primary handlers like FileOutputHandler, raise MeteorWriteError instead.

        Args:
            source_path: Path to source RAW file.
            filename: Output filename.
            debug_image: Optional debug visualization.
            roi_polygon: Optional ROI polygon.

        Returns:
            OutputResult with saved flag and paths.
        """
        logger.debug(f"Saving candidate: {filename}")

        dest_path = os.path.join(self.config.output_folder, filename)

        # Skip if exists
        if os.path.exists(dest_path):
            logger.debug(f"File already exists, skipping: {dest_path}")
            return OutputResult(
                saved=False,
                output_path=dest_path,
                debug_path=None,
            )

        try:
            # Copy the file
            shutil.copy2(source_path, dest_path)
            logger.info(f"Saved candidate: {filename}")

            debug_path = None
            if debug_image is not None and self.config.debug_folder:
                debug_filename = os.path.splitext(filename)[0] + "_debug.png"
                debug_path = self.save_debug_image(
                    debug_image, debug_filename, roi_polygon
                ) or None

            return OutputResult(
                saved=True,
                output_path=dest_path,
                debug_path=debug_path,
            )

        except OSError as e:
            # Create structured error with context for diagnostics
            error = MeteorWriteError(
                f"Failed to copy candidate file: {e}",
                filepath=source_path,
                destination_path=dest_path,
                operation="copy",
                original_error=e,
                context={"error_category": "copy_failed"},
            )
            # Log error but don't fail the pipeline
            logger.error(str(error))
            return OutputResult(
                saved=False,
                output_path=dest_path,
                debug_path=None,
            )
        except Exception as e:
            logger.exception(f"Unexpected error saving candidate {filename}")
            return OutputResult(
                saved=False,
                output_path=dest_path,
                debug_path=None,
            )

    def save_debug_image(
        self,
        debug_image: np.ndarray,
        filename: str,
        roi_polygon: Optional[List[List[int]]] = None,
    ) -> str:
        """Save a debug visualization.

        Args:
            debug_image: Debug image (BGR).
            filename: Output filename.
            roi_polygon: Optional ROI polygon to draw.

        Returns:
            Path to saved debug image, or empty string on failure.
        """
        if not self.config.debug_folder:
            return ""

        logger.debug(f"Saving debug image: {filename}")
        path = os.path.join(self.config.debug_folder, filename)

        try:
            # Draw ROI if provided
            if roi_polygon and len(roi_polygon) >= 3:
                pts = np.array(roi_polygon, dtype=np.int32)
                cv2.polylines(debug_image, [pts], True, (255, 0, 0), 2)

            if not cv2.imwrite(path, debug_image):
                logger.warning("Could not write debug image: %s", path)
                return ""
            logger.debug(f"Saved debug image: {path}")
            return path

        except OSError as e:
            error = MeteorWriteError(
                f"Failed to save debug image: {e}",
                destination_path=path,
                operation="save_image",
                original_error=e,
                context={"error_category": "image_write_failed"},
            )
            logger.warning(str(error))
            return ""
        except Exception as e:
            logger.warning(f"Failed to save debug image {filename}: {e}")
            return ""

    # ========== LIFECYCLE HOOKS ==========
    # This notification handler logs callback failures and continues.

    def on_detection_result(
        self,
        context: Dict[str, Any],
        result: DetectionResult,
        filepath: str,
    ) -> None:
        """Called immediately after each detection result.

        Use this to inspect detector outputs before saving candidates.
        This method must NOT raise exceptions.
        """
        try:
            basename = os.path.basename(filepath)
            runtime_params = context.get("runtime_params", {})
            line_count = len(result.lines)
            extra_keys = ", ".join(sorted(result.extras.keys()))
            logger.debug(
                "on_detection_result: %s is_candidate=%s lines=%d extras=%s params=%s",
                basename,
                result.is_candidate,
                line_count,
                extra_keys or "(none)",
                runtime_params,
            )
        except Exception as e:
            # Log optional notification failures without stopping the run
            logger.warning(f"on_detection_result hook failed: {e}")

    def on_candidate_detected(
        self,
        filename: str,
        saved: bool,
        score: float = 0.0,
        aspect_ratio: float = 0.0,
    ) -> None:
        """Called after each meteor detection.

        Use this for real-time notifications.
        This method must NOT raise exceptions.

        Args:
            filename: Detected file name.
            saved: Whether file was saved (False if skipped).
            score: Detection score.
            aspect_ratio: Contour aspect ratio.
        """
        try:
            self._detection_count += 1
            logger.debug(
                f"on_candidate_detected: {filename}, saved={saved}, "
                f"score={score:.1f}, aspect_ratio={aspect_ratio:.2f}"
            )

            if self.config.notify_on_detection and self.config.webhook_url and saved:
                message = get_message(
                    "ui.notification.meteor_detected",
                    locale=self.config.locale,
                    filename=filename,
                    score=f"{score:.1f}",
                    aspect_ratio=f"{aspect_ratio:.2f}",
                )
                self._send_slack(message)

        except Exception as e:
            # Log optional notification failures without stopping the run
            logger.warning(f"on_candidate_detected hook failed: {e}")

    def on_batch_complete(
        self,
        processed_count: int,
        detected_count: int,
        batch_size: int,
    ) -> None:
        """Called after each batch completes.

        Use for progress tracking.
        This method must NOT raise exceptions.

        Args:
            processed_count: Total processed so far.
            detected_count: Total detected so far.
            batch_size: Files in this batch.
        """
        try:
            logger.debug(
                f"on_batch_complete: processed={processed_count}, "
                f"detected={detected_count}, batch_size={batch_size}"
            )
            # Could implement periodic progress updates here

        except Exception as e:
            # Log optional notification failures without stopping the run
            logger.warning(f"on_batch_complete hook failed: {e}")

    def on_pipeline_complete(
        self,
        total_processed: int,
        total_detected: int,
        elapsed_seconds: float,
    ) -> None:
        """Called when pipeline finishes.

        Use for final summary notifications.
        This method must NOT raise exceptions.

        Args:
            total_processed: Total files processed.
            total_detected: Total candidates detected.
            elapsed_seconds: Total time in seconds.
        """
        try:
            minutes = elapsed_seconds / 60
            rate = total_processed / elapsed_seconds if elapsed_seconds > 0 else 0

            logger.info(
                f"Pipeline complete: {total_processed} processed, "
                f"{total_detected} detected, {minutes:.1f} min, {rate:.2f} img/s"
            )

            if self.config.notify_on_complete and self.config.webhook_url:
                message = get_message(
                    "ui.notification.detection_complete",
                    locale=self.config.locale,
                    processed=total_processed,
                    detected=total_detected,
                    minutes=f"{minutes:.1f}",
                    rate=f"{rate:.2f}",
                )
                self._send_slack(message)

        except Exception as e:
            # Log optional notification failures without stopping the run
            logger.warning(f"on_pipeline_complete hook failed: {e}")

    def _send_slack(self, message: str) -> None:
        """Send a message to Slack webhook.

        Args:
            message: Message text (supports Slack markdown).
        """
        if not self.config.webhook_url:
            logger.debug("No webhook URL configured, skipping Slack notification")
            return

        logger.debug(f"Sending Slack notification to {self.config.channel}")

        payload = {
            "channel": self.config.channel,
            "text": message,
            "mrkdwn": True,
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.config.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
            logger.debug("Slack notification sent successfully")

        except urllib.error.URLError as e:
            logger.warning(f"Failed to send Slack notification (network error): {e}")
        except Exception as e:
            # Don't fail pipeline on notification errors
            logger.warning(f"Failed to send Slack notification: {e}")


# Register the plugin
OutputHandlerRegistry.register(SlackNotificationHandler)
```

---

<a id="6-best-practices"></a>

## 6. 実装上の推奨事項

<a id="61-choosing-configtype"></a>

### 6.1 ConfigTypeの選択

次の分岐図で設定方式を選んでください。

```
Need configuration?
        │
       No ──────────▶ Don't define ConfigType
        │              (accept None in __init__)
       Yes
        │
        ▼
Need validation?
(ranges, patterns,
 custom rules)
        │
       No ──────────▶ Use @dataclass
        │              (simple, built-in, no deps)
       Yes
        │
        ▼
Use Pydantic BaseModel
(rich validation, type coercion)
```

**Dataclass（多くの場合に推奨）**：

```python
from dataclasses import dataclass


@dataclass
class MyConfig:
    output_folder: str = "./output"  # Always provide defaults
    threshold: float = 0.5
    enabled: bool = True
```

**Pydantic（複雑な検証向け）**：

```python
from pydantic import BaseModel, Field, field_validator


class MyConfig(BaseModel):
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    url: str = ""

    @field_validator("url")
    @classmethod
    def validate_url(cls, v):
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v

    model_config = {"extra": "forbid"}  # Reject unknown keys
```

<a id="62-required-attributes"></a>

### 6.2 必須属性

| 属性 | 必須 | 型 | 説明 |
|------|------|----|------|
| `plugin_name` | ✅ 必須 | `str` | 大文字・小文字を区別しない固有名 |
| `name` | ❌ 任意 | `str` | 人が読む名前 |
| `version` | ❌ 任意 | `str` | バージョン文字列 |
| `ConfigType` | ❌ 任意 | `type` | 設定クラス |

<a id="63-exception-hierarchy-and-error-handling"></a>

### 6.3 例外階層とエラー処理

<a id="exception-hierarchy"></a>

#### 例外階層

`meteor_core` は一貫したエラー処理のため、構造化した例外階層を提供します。

```
MeteorError (base)
├── MeteorLoadError (image loading failures)
│   └── MeteorUnsupportedFormatError (unsupported file formats)
├── MeteorOutputError (output operation failures)
│   ├── MeteorWriteError (file write failures)
│   └── MeteorProgressError (progress tracking errors)
├── MeteorValidationError (parameter/input validation)
└── MeteorConfigError (configuration errors)
```

**例外のインポート**：

```python
from meteor_core.exceptions import (
    MeteorError,
    MeteorLoadError,
    MeteorUnsupportedFormatError,
    MeteorOutputError,
    MeteorWriteError,
    MeteorProgressError,
    MeteorValidationError,
    MeteorConfigError,
)
```

**例外の属性**：

デバッグ・問題報告用の詳しいコンテキストを持ちます。

| 属性 | 型 | 説明 |
|------|----|------|
| `message` | `str` | 読みやすいエラー説明 |
| `filepath` | `Optional[str]` | 対象ファイル（ある場合） |
| `original_error` | `Optional[Exception]` | 連鎖した元の例外 |
| `context` | `Dict[str, Any]` | 追加のコンテキスト |

**コンテキスト付き例外の作成**：

```python
from meteor_core.exceptions import MeteorLoadError

raise MeteorLoadError(
    "Failed to decode FITS file",
    filepath="/path/to/image.fits",
    original_error=original_exception,
    context={
        "loader": "fits",
        "bit_depth": 16,
        "compression": "RICE_1",
    },
)
```

**出力固有の例外**：

`MeteorWriteError`、`MeteorProgressError` を使います。

```python
from meteor_core.exceptions import MeteorWriteError, MeteorProgressError

# File write error with destination path
error = MeteorWriteError(
    "Failed to copy candidate file",
    filepath="/source/image.CR2",           # Source path
    destination_path="/output/image.CR2",   # Destination path
    operation="copy",                       # Operation type
    original_error=os_error,
    context={"error_category": "copy_failed"},
)

# Progress tracking error
error = MeteorProgressError(
    "Failed to parse progress file",
    filepath="progress.json",
    operation="parse",                      # "load", "save", "parse", "serialize"
    original_error=json_error,
    context={"error_category": "parse_failed"},
)
```

| 例外 | 用途 | 主な属性 |
|------|------|----------|
| `MeteorWriteError` | コピー、画像保存、ディレクトリ作成 | `destination_path`、`operation` |
| `MeteorProgressError` | 進捗の読み書き、JSON解析 | `operation`（load/save/parse/serialize） |

<a id="exception-policy-by-plugin-type"></a>

#### 種別ごとの例外方針

例外を送るか処理を継続するかは、種別によって異なります。

| 種別 | メソッド | 方針 |
|------|----------|------|
| **入力ローダー** | `load()` | **送出**：画像なしでは処理できない |
| **入力ローダー** | `extract_metadata()` | **空辞書**：メタデータは任意 |
| **検出器** | `detect()` | **送出またはDetectionResult**：失敗を未検出として扱う |
| **出力ハンドラー** | `save_candidate()` | **重要度による**：下記参照 |
| **出力ハンドラー** | ライフサイクルフック | 例外は `hook_error_mode` に従う。任意の通知失敗で停止させたくない場合はハンドラー内で処理する |

**重要・非重要な出力ハンドラーの方針**：

役割で2つに分けます。

| 種類 | 例 | 方針 |
|------|----|------|
| **主ハンドラー（重要）** | FileOutputHandler、S3Handler | **送出**：ディスク・ストレージの失敗は重大 |
| **補助ハンドラー（非重要）** | SlackHandler、WebhookHandler | **OutputResult(saved=False, ...)を返す**：通知の失敗は非重大 |

- **主ハンドラー** はRAW・デバッグ画像を保存します。失敗は容量不足、権限、ネットワークストレージの障害など、後続にも影響する問題を示すことが多いため、例外で即座に対処できるようにします。長時間後に保存されていなかったと気付く事態を避けます。
- **補助ハンドラー** は通知などを行います。検出自体は続けられるため、失敗してもパイプラインを停止しないようにします。

組み込み `FileOutputHandler` は **主ハンドラー** であり、書き込み失敗時に `MeteorWriteError` を送ります。このガイドのSlack例は **補助ハンドラー** です。

**入力ローダーの例外**：

```python
class MyLoader(DataclassInputLoader[MyConfig]):
    def load(self, filepath: str) -> InputContext:
        # Check file existence
        if not os.path.exists(filepath):
            raise MeteorLoadError(
                f"File not found: {filepath}",
                filepath=filepath,
            )

        # Check file format
        if not filepath.lower().endswith((".fits", ".fit")):
            raise MeteorUnsupportedFormatError(
                f"Unsupported format: {filepath}",
                filepath=filepath,
                context={"supported_formats": [".fits", ".fit"]},
            )

        try:
            # Load the image
            image = self._read_fits(filepath)
            return InputContext(
                image_data=image,
                filepath=filepath,
                metadata=self.extract_metadata(filepath),
                loader_info=self.get_info(),
            )
        except Exception as e:
            # Wrap low-level errors with context
            raise MeteorLoadError(
                f"Failed to load FITS file: {e}",
                filepath=filepath,
                original_error=e,
                context={"loader": self.plugin_name},
            )

    def extract_metadata(self, filepath: str) -> Dict[str, Any]:
        """Metadata extraction should not raise exceptions."""
        try:
            return self._read_fits_header(filepath)
        except Exception:
            # Return empty dict, don't fail
            return {}
```

**検出器の動作（送出または結果を返す）**：

```python
class MyDetector(DataclassDetector[MyConfig]):
    def detect(self, context: DetectionContext) -> DetectionResult:
        # Raise when configuration or inputs are invalid so the pipeline can
        # log the error for that file.
        if context.current_image.shape != context.previous_image.shape:
            raise ValueError("current_image and previous_image must match")

        # Or return a "no detection" result for recoverable cases.
        return DetectionResult(
            is_candidate=False,
            score=0.0,
            lines=[],
            aspect_ratio=0.0,
            debug_image=None,
            extras={"reason": "no contours"},
        )
```

**出力ハンドラーのエラー処理**：

重要なファイル・ストレージ操作をする **主ハンドラー** は例外を送出します。
逐次・並列の両経路で、保存処理や `OutputResult` の正規化の例外を呼び出し元へ送出します。
`hook_error_mode: warn` は保存失敗を無視する設定ではありません。失敗した候補は処理済みに記録しません。
一方、`OutputResult(saved=False, ...)` は正常な「保存省略」の戻り値であり、保存後の通知を行い、候補件数にも含めます。

```python
from meteor_core.exceptions import MeteorWriteError


class MyFileHandler(DataclassOutputHandler[MyConfig]):
    """Primary handler - raises exceptions on critical failures."""

    def save_candidate(
        self, source_path, filename, debug_image=None, roi_polygon=None
    ) -> OutputResult:
        dest_path = os.path.join(self.config.output_folder, filename)
        try:
            shutil.copy2(source_path, dest_path)
            return OutputResult(
                saved=True,
                output_path=dest_path,
                debug_path=None,
            )
        except OSError as e:
            # Primary handlers raise to stop pipeline on critical errors
            raise MeteorWriteError(
                f"Failed to copy candidate file: {e}",
                filepath=source_path,
                destination_path=dest_path,
                operation="copy",
                original_error=e,
                context={"error_category": "copy_failed"},
            ) from e
```

通知・Webhookなどの **補助ハンドラー** はログを記録し、`OutputResult(saved=False, ...)` を返します。

```python
class MyNotificationHandler(DataclassOutputHandler[MyConfig]):
    """Secondary handler - logs errors and continues."""

    def save_candidate(
        self, source_path, filename, debug_image=None, roi_polygon=None
    ) -> OutputResult:
        try:
            self._upload_to_cloud(source_path)
            return OutputResult(
                saved=True,
                output_path=source_path,
                debug_path=None,
            )
        except Exception as e:
            # Secondary handlers log but don't fail the pipeline
            logger.warning(f"Cloud upload failed (non-critical): {e}")
            return OutputResult(
                saved=False,
                output_path=source_path,
                debug_path=None,
            )

    def on_candidate_detected(self, filename, saved, score, aspect_ratio):
        try:
            self._send_notification(filename)
        except Exception as e:
            # Log optional notification failures without stopping the run
            logger.warning(f"Notification failed: {e}")
```

<a id="64-logging-guidelines"></a>

### 6.4 ログの指針

<a id="setting-up-logging"></a>

#### ログの設定

標準の `logging` と `meteor_core` のロガー階層を使います。

```python
import logging

# Get a logger for your plugin
logger = logging.getLogger("meteor_core.inputs.my_loader")
# Or for detectors: logging.getLogger("meteor_core.detectors.my_detector")
# Or for outputs: logging.getLogger("meteor_core.outputs.my_handler")
```

<a id="log-level-policy"></a>

#### ログレベルの方針

| レベル | 用途 | 例 |
|--------|------|----|
| `DEBUG` | 調査用の詳細な処理記録 | パス、設定、中間結果 |
| `INFO` | 通常動作の主要イベント | 読み込み、処理開始・完了 |
| `WARNING` | 継続可能な問題 | 任意メタデータの不足、低速、非推奨の使用 |
| `ERROR` | 現在の操作の失敗 | 保存失敗、ネットワークのタイムアウト |
| `CRITICAL` | 即座の対処が必要な重大障害 | プラグインでは通常使わない |

**レベルごとの例**：

```python
import logging
from meteor_core.schema import InputContext

logger = logging.getLogger("meteor_core.inputs.fits_loader")


class FitsLoader(DataclassInputLoader[FitsConfig]):
    def __init__(self, config):
        super().__init__(config)
        logger.debug(f"FitsLoader initialized with config: {config}")

    def load(self, filepath: str) -> InputContext:
        logger.debug(f"Loading FITS file: {filepath}")

        # INFO: Significant events
        logger.info(f"Processing {os.path.basename(filepath)}")

        try:
            image = self._read_fits(filepath)
            logger.debug(f"Loaded image shape: {image.shape}, dtype: {image.dtype}")
            return InputContext(
                image_data=image,
                filepath=filepath,
                metadata=self.extract_metadata(filepath),
                loader_info=self.get_info(),
            )
        except Exception as e:
            # ERROR: Operation failed
            logger.error(f"Failed to load {filepath}: {e}")
            raise MeteorLoadError(str(e), filepath=filepath, original_error=e)

    def extract_metadata(self, filepath: str) -> Dict[str, Any]:
        try:
            metadata = self._read_header(filepath)
            logger.debug(f"Extracted metadata: {list(metadata.keys())}")
            return metadata
        except Exception as e:
            # WARNING: Non-critical failure
            logger.warning(f"Could not extract metadata from {filepath}: {e}")
            return {}
```

<a id="logging-best-practices"></a>

#### ログの推奨事項

**推奨**：

- 適切なレベルを一貫して使用。
- ファイル名や設定などの関連情報を付加。
- `logger.exception()` でスタックトレースを記録。
- 短く、必要な情報を含むメッセージ。

```python
# Good: Informative with context
logger.debug(f"Applying threshold {threshold} to image {filepath}")
logger.warning(f"Metadata missing 'exposure_time' in {filepath}, using default")
logger.error(f"Failed to save to {output_path}: {e}")
```

**避けること**：

- APIキーや認証情報などの機密情報の記録。
- ログの代わりに `print()` を使う。
- 高頻度ループでの過剰なログ（性能に影響）。
- ログを残すためだけに例外を送る。

```python
# Bad: Using print
print(f"Loading {filepath}")  # Use logger.info() instead

# Bad: Logging in tight loops
for pixel in image.flatten():  # Millions of iterations!
    logger.debug(f"Processing pixel: {pixel}")

# Bad: Raising just to log
try:
    process()
except Exception as e:
    logger.error(str(e))
    raise  # If you're re-raising anyway, use logger.exception()

# Good: Use logger.exception() for full traceback
try:
    process()
except Exception as e:
    logger.exception(f"Processing failed")  # Includes traceback
    raise
```

<a id="internationalization-i18n-guidance"></a>

### 国際化（i18n）の指針

- **UI/UXだけを翻訳**：CLIの質問、進捗集計、エラー見出しなど、利用者向けの表示。
- **それ以外は英語**：ログ、デバッグ出力、開発者向け診断は、問題調査を一貫させるため英語を維持。

UI/UXには `meteor_core/locales/<locale>/messages.yaml` の共通カタログを `meteor_core.i18n.get_message` で使ってください。中核の保守担当者と調整せずに、プラグイン独自の翻訳ファイルを追加することは避けてください。

Slack出力ハンドラーに対応する項目の例：

```yaml
ui:
  notification:
    meteor_detected: "🌠 *Meteor Detected!*\n• File: `{filename}`\n• Score: {score}\n• Aspect Ratio: {aspect_ratio}"
    detection_complete: "✅ *Detection Complete*\n• Processed: {processed} images\n• Detected: {detected} candidates\n• Time: {minutes} minutes\n• Rate: {rate} images/sec"
```

`meteor_core/locales/ja/messages.yaml` など、ほかのロケールにも翻訳を追加してください。

<a id="using-diagnostic-reports"></a>

#### 診断レポートの利用

Issueとして報告される可能性のあるエラーには `format_for_issue()` を使います。

```python
from meteor_core.exceptions import MeteorLoadError

try:
    image = load_image(filepath)
except MeteorLoadError as e:
    # Get diagnostic report for GitHub issue
    diagnostic_report = e.format_for_issue()
    logger.error(f"Load failed. Diagnostic info:\n{diagnostic_report}")
    raise
```

<a id="65-performance-considerations"></a>

### 6.5 性能上の注意点

**入力ローダー**：

- 単一チャンネル配列を返し、通常はuint16、正規化する場合はfloat32を使用。
- 不要なコピーを避ける（`image.astype()` はコピーを生成）。
- 大きな画像ではメモリマップを検討。

**検出器**：

- PythonのループよりNumPyのベクトル化を使用。
- 可能なら配列を事前確保。
- OpenCVの最適化された関数を検討。

**出力ハンドラー**：

- フックが長時間停止しないようタイムアウトを設定。
- 必要なら通知をまとめてバッチ送信。
- ネットワーク処理には非同期I/Oも検討（高度な実装）。

<a id="66-thread-safety"></a>

### 6.6 スレッド安全性

パイプラインの並列処理は別プロセスで行い、ワーカー間でプラグインのインスタンス状態を共有しません。プラグイン自身がスレッドを作成する場合は、以下のように共有状態を保護してください。

```python
class MyHandler(DataclassOutputHandler[MyConfig]):
    def __init__(self, config):
        super().__init__(config)
        self._lock = threading.Lock()
        self._count = 0

    def on_candidate_detected(self, filename, saved, score, aspect_ratio):
        with self._lock:
            self._count += 1
```

<a id="67-type-safety-with-ty"></a>

### 6.7 tyによる型安全性

v1.6.8から、AstralのRust製型検査ツール[ty](https://docs.astral.sh/ty/)を使用します。プラグインでも利用し、型エラーを早期に発見することを推奨します。

<a id="running-ty-on-your-plugin"></a>

#### プラグインにtyを実行する

```bash
# Install ty (if not already installed)
uv add ty --dev

# Run type checker on your plugin
uv run ty check your_plugin/

# Or via pre-commit
uv run pre-commit run ty-check --all-files
```

<a id="type-hints-best-practices"></a>

#### 型ヒントの推奨事項

**推奨**：

- 公開メソッドに型ヒントを明示。

  ```python
  from meteor_core.schema import DetectionContext, DetectionResult


  def detect(self, context: DetectionContext) -> DetectionResult:
      # ty will validate your implementation
      ...
  ```

- `invalid-method-override` を避けるため、基底クラスのシグネチャに正確に一致させる。

  ```python
  # Good: Exact signature match
  def save_candidate(
      self,
      source_path: str,
      filename: str,
      debug_image: Optional[np.ndarray] = None,
      roi_polygon: Optional[List[List[int]]] = None,
  ) -> OutputResult: ...
  ```

- 契約に対応する型を `meteor_core.schema` からインポート。

  ```python
  from meteor_core.schema import (
      InputContext,
      DetectionContext,
      DetectionResult,
      OutputResult,
      RuntimeParams,
  )
  ```

**避けること**：

- 公開メソッドの戻り値型を省略。
- オーバーライドで互換性のない引数型を使用。
- tyのエラーを無視する（実際の問題を示すことが多い）。

<a id="common-ty-errors-and-fixes"></a>

#### よくあるtyのエラーと修正

| エラー | 原因 | 修正 |
|--------|------|------|
| `invalid-return-type` | 戻り値が宣言と不一致 | return文と宣言した型を照合 |
| `invalid-method-override` | 基底クラスと非互換 | 引数型を正確に一致させる |
| `invalid-assignment` | 代入値が宣言と不一致 | 正しい型または型の絞り込み |
| `call-non-callable` | 呼び出し不能な値 | 呼び出し可能かを確認 |

<a id="example-type-safe-detector"></a>

#### 例：型安全な検出器

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from meteor_core.detectors import DataclassDetector
from meteor_core.schema import DetectionContext, DetectionResult
from meteor_core.utils import ensure_numpy


@dataclass
class MyDetectorConfig:
    threshold: float = 0.5


class MyDetector(DataclassDetector[MyDetectorConfig]):
    plugin_name = "my_detector"
    ConfigType = MyDetectorConfig

    def detect(self, context: DetectionContext) -> DetectionResult:
        # Type-safe image conversion
        current: np.ndarray = ensure_numpy(context.current_image)
        previous: np.ndarray = ensure_numpy(context.previous_image)

        # Type-safe parameter extraction
        global_params, detector_params = self.split_runtime_params(
            context.runtime_params
        )

        # ... detection logic ...

        return DetectionResult(
            is_candidate=True,
            score=85.0,
            lines=[(10, 20, 100, 120)],
            aspect_ratio=3.5,
            debug_image=None,
            extras={},
            metrics={"duration_ms": 15.2},
        )

    def compute_line_score(
        self,
        mask: np.ndarray,
        hough_params: Dict[str, int],
    ) -> Tuple[float, List[Tuple[int, int, int, int]]]:
        # Implementation with correct return type
        return 0.0, []
```

<a id="68-worker-limit-configuration"></a>

### 6.8 ワーカー上限の設定

`meteor_core` が公開する `MAX_NUM_WORKERS` は `max(1, multiprocessing.cpu_count())` です。既定ワーカー数は `max(1, CPU数 - 1)` です。

```python
from meteor_core import MAX_NUM_WORKERS

# Upper bound: max(1, multiprocessing.cpu_count())
print(f"Maximum workers allowed: {MAX_NUM_WORKERS}")
```

**PipelineConfigの検証**：

直接生成時に `num_workers` の範囲（1～`MAX_NUM_WORKERS`）と `batch_size >= 1` を検証し、不正な値には `ValueError` を送出します。

```python
from meteor_core import MAX_NUM_WORKERS
from meteor_core.schema import DetectionParams, PipelineConfig

try:
    config = PipelineConfig(
        target_folder="./raw",
        output_folder="./candidates",
        debug_folder="./debug",
        params=DetectionParams(),
        num_workers=MAX_NUM_WORKERS + 1,
    )
except ValueError as exc:
    print(exc)
```

設定ファイルの読み込みに使う `from_partial_dict()` は、既定設定を生成した後に属性を代入し、同じ範囲検証を再実行しません。ファイルにも上記範囲と `hook_error_mode: warn|raise` を指定してください。

**作者への留意点**：

- 無制限のワーカープロセスを前提にしない。
- 状態を持たない、またはスレッド安全な実装にする（[6.6 スレッド安全性](#66-thread-safety)参照）。
- 性能の推奨設定を記すときは `MAX_NUM_WORKERS` を考慮。

---

<a id="7-step-by-step-tutorial"></a>

## 7. 段階的なチュートリアル

<a id="71-step-by-step-creating-a-plugin"></a>

### 7.1 プラグインを作成する手順

<a id="step-1-choose-plugin-type-and-base-class"></a>

#### 手順1：種別と基底クラスを選ぶ

```python
# For input loaders with dataclass config
from meteor_core.inputs import DataclassInputLoader, BaseMetadataExtractor

# For detectors with dataclass config
from meteor_core.detectors import DataclassDetector

# For output handlers with dataclass config
from meteor_core.outputs import DataclassOutputHandler
```

<a id="step-2-define-configuration-optional"></a>

#### 手順2：設定を定義する（任意）

```python
from dataclasses import dataclass

import numpy as np

from meteor_core.exceptions import MeteorLoadError
from meteor_core.inputs import DataclassInputLoader
from meteor_core.schema import InputContext


@dataclass
class MyPluginConfig:
    normalize: bool = False
```

<a id="step-3-implement-the-plugin-class"></a>

#### 手順3：クラスを実装する

手順2の設定クラスと以下のコードを `my_plugin.py` に保存します。このローダーは明示した `.npy` ファイルを読み込みます。

```python
class MyPlugin(DataclassInputLoader[MyPluginConfig]):
    plugin_name = "my_plugin"
    name = "NumPy Frame Loader"
    version = "1.0.0"
    ConfigType = MyPluginConfig

    def load(self, filepath: str) -> InputContext:
        """Load one 2D uint16 NumPy frame with optional normalization."""
        try:
            image = np.load(filepath, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise MeteorLoadError(
                "Could not read NumPy frame",
                filepath=filepath,
                original_error=exc,
            ) from exc

        if image.ndim != 2 or image.dtype != np.uint16:
            raise MeteorLoadError("Expected a 2D uint16 frame", filepath=filepath)

        if self.config.normalize:
            image = image.astype(np.float32) / 65535

        return InputContext(image, filepath, loader_info=self.get_info())
```

<a id="step-4-register-the-plugin"></a>

#### 手順4：登録する

**方法A：実行時登録**

```python
from meteor_core.inputs import LoaderRegistry
from my_plugin import MyPlugin

LoaderRegistry.register(MyPlugin)
```

**方法B：パッケージのエントリーポイント**

```toml
# pyproject.toml
[project.entry-points."detect_meteors.input"]
my_plugin = "my_plugin:MyPlugin"
```

**方法C：利用者のプラグインディレクトリ**

```bash
# Save as ~/.detect_meteors/input_plugins/my_plugin.py
```

<a id="72-testing-your-plugin"></a>

### 7.2 プラグインのテスト

<a id="unit-test-example"></a>

#### ユニットテストの例

`my_plugin.py` の隣に `test_my_plugin.py` として保存し、`uv run python -m unittest test_my_plugin` を実行します。

```python
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from my_plugin import MyPlugin, MyPluginConfig


class TestMyPlugin(unittest.TestCase):
    def test_load_returns_uint16_or_normalized_float32(self):
        pixels = np.array([[0, 65535]], dtype=np.uint16)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "frame.npy"
            np.save(path, pixels)

            original = MyPlugin(MyPluginConfig()).load(str(path))
            normalized = MyPlugin(MyPluginConfig(normalize=True)).load(str(path))

        np.testing.assert_array_equal(original.image_data, pixels)
        self.assertEqual(original.image_data.dtype, np.uint16)
        np.testing.assert_array_equal(normalized.image_data, [[0.0, 1.0]])
        self.assertEqual(normalized.image_data.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
```

<a id="integration-test"></a>

#### レジストリでの動作確認

```python
from meteor_core.inputs import LoaderRegistry
from my_plugin import MyPlugin


LoaderRegistry.register(MyPlugin)
assert "my_plugin" in LoaderRegistry.list_available()

loader = LoaderRegistry.create("my_plugin", {"normalize": True})
assert loader.config.normalize
```

<a id="73-debugging-tips"></a>

### 7.3 デバッグのヒント

**詳しいログを有効にする**：

```python
import logging

logging.basicConfig(level=logging.DEBUG)
```

**プラグイン情報を確認する**：

```python
from meteor_core.inputs import LoaderRegistry

LoaderRegistry.discover()
for name in LoaderRegistry.list_available():
    instance = LoaderRegistry.create(name)
    print(instance.get_info())
```

**設定変換を確認する**：

```python
# Test that dict config works
handler = OutputHandlerRegistry.create("my_handler", {"option": "value"})

# Test that ConfigType instance works
config = MyConfig(option="value")
handler = OutputHandlerRegistry.create("my_handler", config)

# Test default config
handler = OutputHandlerRegistry.create("my_handler")  # Uses ConfigType()
```

---

<a id="see-also"></a>

## 関連資料

**ドキュメント**：

- [INSTALL_DEV.md](INSTALL_DEV_ja.md) — 開発環境の構築
- [CHANGELOG.md](CHANGELOG_ja.md) — リリース履歴
- [README.md](../README_ja.md) — 利用者向けドキュメント

**参考用の組み込み実装**：

- [`meteor_core/inputs/raw.py`](../meteor_core/inputs/raw.py) — RawImageLoader
- [`meteor_core/detectors/hough_default.py`](../meteor_core/detectors/hough_default.py) — HoughDetector
- [`meteor_core/outputs/file_handler.py`](../meteor_core/outputs/file_handler.py) — FileOutputHandler
