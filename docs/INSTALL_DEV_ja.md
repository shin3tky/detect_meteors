# 開発者向けインストールガイド

[English](INSTALL_DEV.md)

Detect Meteors CLIの開発者・貢献者向けに、環境構築の手順を説明します。

このガイドのコマンドは `docs/` 内ではなく、リポジトリのルート、または展開したソース配布版のルートで実行してください。

基本的なインストールは[INSTALL_ja.md](INSTALL_ja.md)、プラグイン開発は[PLUGIN_AUTHOR_GUIDE_ja.md](PLUGIN_AUTHOR_GUIDE_ja.md)を参照してください。

## 前提条件

Gitとuvを用意し、リポジトリまたは自分のフォークをクローンして、開発対象のブランチに切り替えます。OSごとの準備とmacOSでの配置先は [INSTALL_ja.md](INSTALL_ja.md) を参照してください。

```bash
git clone https://github.com/shin3tky/detect_meteors.git
cd detect_meteors
```

バージョンと依存関係はチェックアウトしたソースに従います。1.6.12の開発ブランチでは、現在のCLIとパッケージのバージョン表示は `1.6.11` です。

## 開発環境の構築

### 手順1：開発用の依存ライブラリをインストールする

チェックアウトのルートで、ロックファイルに従って実行時・開発用の依存ライブラリをインストールします。`.python-version` はPython 3.12を指定しています。CIではPython 3.12と3.13でテストを実行します。

```bash
uv python install 3.12
uv sync --locked --all-extras
uv run detect-meteors --version
uv run detect-meteors --help
```

`.venv` にプロジェクトを編集可能モードでインストールし、Ruff、ty、pre-commit、coverageも導入します。Python 3.13で確認するときは `uv python install 3.13` を実行し、`uv sync` と `uv run` に `--python 3.13` を付けます。

### 開発ツールのバージョン（v1.6.11）

| ライブラリ | 固定バージョン |
| --- | --- |
| `ruff` | `0.17.4` |
| `ty` | `0.0.99` |
| `pre-commit` | `4.7.0` |
| `coverage` | `7.17.2` |

`pyproject.toml` の `dev` と `uv.lock` に従います。実行時依存は
[インストールガイド](INSTALL_ja.md#依存ライブラリのバージョンv1611)を参照。

ブランチを切り替えたときやロックファイルを更新したときは、チェックアウトのルートで `uv sync --locked --all-extras` を再実行します。

### 手順2：pre-commitフックを設定する

macOSまたはLinuxのGitチェックアウトでpre-commitフックを導入します。フックは `.venv/bin` のRuffとtyを実行します。Windowsネイティブ環境や展開したソース配布版では、手順3のコマンドで直接検査します。

```bash
# Install the git hooks
uv run pre-commit install

# Verify installation
uv run pre-commit --version
```

### 手順3：環境構築の結果を確認する

ソースルートで静的解析、整形チェック、型検査、テストを実行します。

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run python run_tests.py
```

POSIX形式の仮想環境を使うGitチェックアウトでは、`uv run pre-commit run --all-files` でCIのLintジョブと同じフックを実行できます。

### IDEのPythonインタープリター

VS CodeやPyCharmではリポジトリのルートを開き、macOS/Linuxでは `.venv/bin/python`、Windowsでは `.venv\Scripts\python.exe` を既存のインタープリターとして選びます。チェックアウト内の環境を選ぶと、インポート、docstringの参照、テスト実行に同期済みの環境を使えます。uvが使用するパスは `uv run python -c "import sys; print(sys.executable)"` で確認できます。

## pre-commitフック

### 動作の仕組み

設定後は、コミット前にRuffの静的解析・整形とtyの型検査を実行します。

1. 変更を加え、`git commit` を実行します。
2. ステージ済みPythonファイルに対してRuffの自動修正と整形を実行し、続いて設定対象のソース全体にtyを実行します。
3. ファイルが修正された場合やエラーが見つかった場合、コミットを中止します。
4. 変更をステージし直し、再度コミットします。

### 作業の例

```bash
# Make your changes
vim detect_meteors_cli.py

# Stage and commit
git add detect_meteors_cli.py
git commit -m "Add new feature"

# If Ruff makes changes or finds errors, you'll see:
# - Files were modified by this hook
# - or linting errors
# - Commit aborted

# Stage the changes and commit again
git add detect_meteors_cli.py
git commit -m "Add new feature"
```

### 手動での整形と静的解析

```bash
# Run lint, formatting, and type checking via pre-commit
uv run pre-commit run --all-files

# Or run Ruff directly
uv run ruff check .              # Lint
uv run ruff check --fix .        # Lint with auto-fix
uv run ruff format .             # Format
uv run ruff format --check .     # Check format without changing
```

### 型検査

型検査には[ty](https://docs.astral.sh/ty/)を使用します。

```bash
# Run type checker
uv run ty check

# Check specific file or directory
uv run ty check meteor_core/
```

型検査のルールと対象パスは `pyproject.toml` の `[tool.ty.*]` にあります。`ty-check` フックは、設定された対象パス全体を検査します。

## テストの実行

### テストランナーを使う

```bash
uv run python run_tests.py
```

### unittestを直接使う

```bash
# Run all tests
uv run python -m unittest discover -s tests -p "test_*.py" -v

# Run specific test file
uv run python -m unittest tests.test_calculations_v1x -v

# Run specific test class
uv run python -m unittest tests.test_calculations_v1x.TestCalculateNPFRule -v
```

### テストカバレッジ

```bash
# Run tests with coverage measurement
uv run coverage run -m unittest discover tests

# View coverage report
uv run coverage report

# Generate HTML report (outputs to htmlcov/)
uv run coverage html
```

### テストファイルの一覧

#### 基本計算と例外

| ファイル | テスト数 | 内容 |
|----------|----------|------|
| `test_calculations_v1x.py` | 54 | NPF Rule、画素ピッチ、星・流星の光跡推定 |
| `test_exceptions_v1x.py` | 70 | 例外の階層と診断情報 |
| `test_schema_context_results_v1x.py` | 4 | スキーマのコンテキスト・結果ヘルパー |

#### センサー、光学、画像補正

| ファイル | テスト数 | 内容 |
|----------|----------|------|
| `test_fisheye_v1x.py` | 27 | 魚眼レンズ補正 |
| `test_sensor_npf_integration_v1x.py` | 16 | センサーとNPFの連携 |
| `test_sensor_presets_v1x.py` | 38 | センサータイプのプリセット |
| `test_sensor_validation_v1x.py` | 27 | センサー設定の上書き検証 |

#### 入出力と画像I/O

| ファイル | テスト数 | 内容 |
|----------|----------|------|
| `test_image_io_helpers_v1x.py` | 4 | RAWビニングとEXIFヘルパー |
| `test_inputs_base_v1x.py` | 12 | 入力ローダーの基底クラス |
| `test_inputs_logging_v1x.py` | 8 | ログ設定 |
| `test_loader_registry_v1x.py` | 30 | 入力ローダーのレジストリ |
| `test_output_handler_registry_v1x.py` | 53 | 出力ハンドラーのレジストリ |
| `test_outputs_base_v1x.py` | 7 | 出力ハンドラーの基底クラス |
| `test_raw_loader_v1x.py` | 23 | RAW画像ローダー |

#### パイプラインの実行と統合

| ファイル | テスト数 | 内容 |
|----------|----------|------|
| `test_infrastructure_v1x.py` | 27 | ROI、進捗、ファイル収集、パラメータ推定 |
| `test_integration_v1x.py` | 44 | 流星検出の一連の処理 |
| `test_memory_batch_size_v1x.py` | 6 | メモリに基づくバッチサイズ調整 |
| `test_pipeline_execution_v1x.py` | 7 | パイプラインの実行と結果 |
| `test_pipeline_helpers_v1x.py` | 12 | パイプラインのヘルパー |
| `test_pipeline_hooks_v1x.py` | 12 | フックの順序・状態・実行場所・中断と再開 |
| `test_pipeline_hook_errors_v1x.py` | 11 | フックのエラー方針、ワーカー・保存失敗 |
| `test_aircraft_trail_hook_v1x.py` | 3 | 飛行機の追跡メタデータと進捗への保存 |

#### プラグイン、レジストリ、データ契約

| ファイル | テスト数 | 内容 |
|----------|----------|------|
| `test_detector_plugin_v1x.py` | 26 | 検出器のプラグイン構成 |
| `test_detector_registry_v1x.py` | 30 | 検出器レジストリ |
| `test_detectors_base_v1x.py` | 12 | 検出器の基底クラス |
| `test_discovery_parity.py` | 2 | プラグイン検出処理の一貫性 |
| `test_plugin_contract_helpers_v1x.py` | 5 | プラグイン契約のヘルパー |
| `test_plugin_contract_validation_v1x.py` | 3 | プラグイン契約の検証 |
| `test_plugin_registry_base.py` | 4 | レジストリの基底クラス |
| `test_registry_default_contracts_v1x.py` | 2 | レジストリの既定契約 |

#### CLI、設定、多言語対応、ユーティリティ

| ファイル | テスト数 | 内容 |
|----------|----------|------|
| `test_cli_options_v1x.py` | 4 | CLIオプションの解析 |
| `test_config_io_v1x.py` | 1 | 設定ファイルの読み込み |
| `test_i18n.py` | 5 | 翻訳メッセージの検索と複数形 |
| `test_roi_selector_helpers_v1x.py` | 2 | ROI選択のヘルパー |
| `test_roi_selector_ui_v1x.py` | 2 | ROI選択UIの処理 |
| `test_utils_display_width_v1x.py` | 3 | Unicodeの表示幅 |
| `test_utils_roi_hash_v1x.py` | 5 | ROIハッシュのヘルパー |

#### 取りこぼし分析・調整・保存

| ファイル | テスト数 | 内容 |
| --- | --- | --- |
| `test_hough_analysis_v1x.py` | 7 | 計測・判定・描画の共通基盤、正規化入力 |
| `test_analysis_reproduction_v1x.py` | 14 | 読み取り専用のペア再現と出所 |
| `test_analysis_cli_v1x.py` | 27 | 分析・調整CLIとROIの変更 |
| `test_tuning_v1x.py` | 16 | 上限付きの調整案と比較 |
| `test_analysis_reports_v1x.py` | 20 | 証拠保存・識別情報・設定保存と再実行 |

**現在のチェックアウトでは685テスト（期待失敗なし）。**

## コードスタイル

<a id="building-a-release"></a>

### リリース版のビルド

リリースするバージョンを `pyproject.toml` と `meteor_core/schema.py` に設定し、`uv.lock` を再生成します。setuptoolsのバックエンドがCLIモジュールをパッケージ化し、`detect-meteors` コマンドをインストールします。`MANIFEST.in` はソース配布版にドキュメント、設定例、テストを含めます。

```bash
uv lock
uv sync --locked --all-extras
uv run python run_tests.py
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv build
```

`dist/` にリリースバージョンを含むファイル名で生成されます。wheelの確認には別ディレクトリに仮想環境を作成し、そこで `uv pip install /path/to/the-built.whl` を実行します。その環境を有効化し、同じディレクトリから `detect-meteors --version` と `detect-meteors --help` を実行します。

ソースアーカイブに設定例、ドキュメント、テスト、飛行機サンプルのREADMEとチェックサムが含まれることを確認します。RAW画像と生成済みの候補は配布物の対象外です。手元の飛行機サンプルを使う検証手順は [飛行機ガイド](aircraft_light_trails_hook_design.md) を参照してください。

### 基準

- 静的解析と整形には **Ruff** を使用（1行88文字）。
- Markdown内のコード例はRuffの整形対象から除外し、行末の `#` コメントの桁位置を手動で揃えて視認性を保ちます。
- **Python 3.12または3.13** が必要（`>=3.12,<3.14`）。
- **Google形式のdocstring**。
- 全体で **型ヒント** を使用。

### 例

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class NPFMetrics:
    """NPF Rule calculation results."""

    pixel_pitch_um: Optional[float] = None
    npf_recommended_sec: Optional[float] = None
    compliance_level: str = "UNKNOWN"


def estimate_star_trail_length(
    focal_length_mm: float,
    exposure_time_sec: float,
    image_width_px: int,
) -> float:
    """Estimate star trail length in pixels during exposure.

    Args:
        focal_length_mm: Focal length in 35mm equivalent (mm)
        exposure_time_sec: Exposure time in seconds
        image_width_px: Image width in pixels

    Returns:
        Star trail length in pixels
    """
    ...
```

## ローカライズとログ

- 既定のロケールは英語（`en`）です。ロケールコードを正規化し（例：`en_US` → `en-us`）、地域を含まない言語コード、英語の順にフォールバックします。翻訳がない場合はメッセージキー自体を表示し、未翻訳箇所を見つけやすくします。
- `get_message` や `log_warning` でメッセージを整形してからロガーへ渡すため、ログレコードには `%` のプレースホルダーではなく、完成した文字列が入ります。

## 国際化（i18n）

- 利用者向け文字列のキーは `ui.*` 配下に置きます（例：`ui.error.header`、`ui.run.summary`）。技術的なログは `log.*` 配下に置き、ロケール設定にかかわらず英語を維持します。
- 翻訳は `meteor_core/locales/<locale>/messages.yaml` のJSON互換YAMLにあります。言語間でプレースホルダー名を揃え、ドット区切りのキーに対応する入れ子構造を使ってください。
- `{path}` のようなICU形式のプレースホルダーや、`{count, plural, =0 {Complete! No candidates extracted} one {Complete! # candidate extracted} other {Complete! # candidates extracted}}` のような複数形テンプレートを使います。`#` は数値に置き換えます。
- `meteor_core.i18n.get_message(key, locale=..., params={...})` でメッセージを取得します（`meteor_core.get_message` としても公開）。ロケールやキーがない場合は英語へフォールバックし、不明なプレースホルダーはデバッグしやすいように残します。
- 新しい文字列を追加するときは、`locales/en/messages.yaml` を更新し、他のロケールにも同じキーを追加してください（仮の英語でも可）。引数や複数形を使うメッセージにはユニットテストも追加します。

## プロジェクト構成

```
detect_meteors/
├── _detect_meteors_cli            # Generated completion output (for install)
├── detect_meteors_cli.py          # CLI interface
├── detect_meteors_cli_completion.bash  # Bash completion script
├── meteor_core/                   # Core logic modules
│   ├── __init__.py                # Package entry point
│   ├── i18n.py                    # Locale resolution and message formatting
│   ├── schema.py                  # Type definitions, constants
│   ├── config_io.py               # YAML/JSON pipeline configuration
│   ├── exceptions.py              # Custom exception hierarchy
│   ├── pipeline.py                # Pipeline orchestration
│   ├── analysis.py                # Reproduce a Hough decision for an explicit pair
│   ├── analysis_cli.py            # Analysis input selection and CLI reporting
│   ├── analysis_reports.py        # Analysis bundles and replay configuration
│   ├── tuning.py                   # Bounded parameter search and comparisons
│   ├── image_io.py                # Image IO, EXIF utilities
│   ├── roi_selector.py            # ROI selection
│   ├── utils.py                   # Utility functions
│   ├── messages.py                # User-facing message helpers
│   ├── plugin_registry_base.py    # Base class for plugin registries
│   ├── plugin_registry.py         # Unified plugin registry
│   ├── plugin_contract.py         # Plugin contract definitions
│   ├── inputs/                    # Input loader plugins
│   ├── detectors/                 # Detection algorithm plugins
│   ├── outputs/                   # Output handler plugins
│   ├── hooks/                     # Pipeline hooks, including aircraft trail analysis
│   ├── locales/                   # Translations
│   └── templates/                 # Report templates and assets
├── candidates/                    # Default output folder for detections
├── config_examples/               # Sample configuration files
├── docs/                          # Guides, changelogs, release notes, and translations
│   ├── INSTALL_DEV.md             # Developer setup guide
│   └── PLUGIN_AUTHOR_GUIDE.md     # Plugin development guide
├── debug_masks/                   # Debug output masks
├── rawfiles/                      # Sample/raw image inputs
├── tests/                         # Test suite
├── pyproject.toml                 # Project configuration
├── run_tests.py                   # Test runner helper
├── README.md                      # English overview
└── README_ja.md                   # Japanese overview
```

## プラグイン開発

入力ローダー、検出器、出力ハンドラー、パイプラインフックの作成方法は[PLUGIN_AUTHOR_GUIDE_ja.md](PLUGIN_AUTHOR_GUIDE_ja.md)を参照してください。

> ⚠️ **補足**：プラグイン構成は実験的であり、v2.0より前に変更される可能性があります。

## 貢献の手順

1. リポジトリを **Fork** します。
2. Forkしたリポジトリをローカルに **Clone** します。
3. このガイドに従って **開発環境を構築** します。
4. 作業用ブランチを **作成** します：`git checkout -b feature/your-feature`
5. **変更を加え**、テストを追加します。
6. **テストを実行** します：`uv run python run_tests.py`
7. **カバレッジを測定** します：`uv run coverage run -m unittest discover tests && uv run coverage report`
8. **コミット** します（インストール済みのフックがRuffとtyを実行）。
9. Forkしたリポジトリへ **Push** します。
10. **Pull Requestを作成** します。

## 問題への対処

### Ruffのバージョンを更新する

Ruffのバージョンは `pyproject.toml` の開発依存にある `ruff==X.Y.Z` で固定しています。`.pre-commit-config.yaml` のローカルフックは `.venv/bin/ruff` と `.venv/bin/ty` を実行します。依存バージョンを変更したら、`uv lock` でロックファイルを再生成し、`uv sync --locked --all-extras` でフックが使うツールを更新します。

### pre-commitの問題

```bash
# Reinstall hooks
uv run pre-commit uninstall
uv run pre-commit install

# Refresh the locally pinned tools
uv sync --locked --all-extras
```

### テストの問題

```bash
# Ensure dependencies are installed
uv sync --locked --all-extras

# Reinstall if needed
uv sync --locked --all-extras --reinstall
```

### インポートエラー

```bash
# Test imports
uv run python -c "from meteor_core import BaseInputLoader, BaseOutputHandler, BaseDetector"
```

## ライセンス

このプロジェクトは **Apache License 2.0** で公開しています。詳細は[LICENSE](../LICENSE)を参照してください。

再配布時には[NOTICE](../NOTICE)を含めてください。

## 参考資料

- [PLUGIN_AUTHOR_GUIDE_ja.md](PLUGIN_AUTHOR_GUIDE_ja.md) — プラグイン開発
- [README_ja.md](../README_ja.md) — 利用者向けドキュメント
- [CHANGELOG_ja.md](CHANGELOG_ja.md) — リリース履歴
- [Ruffドキュメント](https://docs.astral.sh/ruff/)
- [uvドキュメント](https://docs.astral.sh/uv/)
- [pre-commitドキュメント](https://pre-commit.com/)
