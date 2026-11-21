# インストールガイド

[English](INSTALL.md)

📺 Detect Meteors 2.1.1 は2026年11月21日（世界テレビ・デー）に正式公開しました。
このガイドでは、macOSとWindowsでのインストール手順を説明します。
[GitHub Release 2.1.1](https://github.com/shin3tky/detect_meteors/releases/tag/2.1.1) · [リリースノート](RELEASE_NOTES_2.1_ja.md#version-211)

> [!IMPORTANT]
> **macOSでは、iCloud Driveの同期対象外にチェックアウトしてください。**
> iCloud Driveで「書類」と「デスクトップ」を同期している場合、
> `~/Documents` または `~/Desktop` の配下には `detect_meteors` をクローンしないでください。
> ソースアーカイブの展開先も同様です。
> `~/Development/shin3tky/detect_meteors` など、iCloud Driveの同期対象外の場所を使ってください。
> 同期対象内では `.venv` 内のファイルに非表示属性が再付与され、
> 導入に成功していても `ModuleNotFoundError` で起動できなくなる場合があります。

<a id="python-requirements"></a>

## Python の動作要件

uv と Python 3.12、3.13、3.14 のいずれかを使います。公開したwheelとソース配布物は3つの版で
検証済みです。Gitのチェックアウトでは既定で Python 3.14 を使います。
依存パッケージの取得にはパッケージインデックスへの接続が必要です。

## 2.1.1 のソースからインストールする

下のOS別の手順でGitとuvを用意した後、リリースタグを取得し、ロックファイルに従って
インストールします。

```bash
git clone --branch 2.1.1 --single-branch https://github.com/shin3tky/detect_meteors.git
cd detect_meteors
uv sync --locked
uv run detect-meteors --version
uv run detect-meteors --help
```

ソース配布物からもインストールできます。ダウンロード後にチェックサムを検証してから展開します。
`pyproject.toml` があるディレクトリでターミナルを開き、上記の `uv sync --locked` 以降の
コマンドを実行します。
[ソース配布物をダウンロード](https://github.com/shin3tky/detect_meteors/releases/download/2.1.1/detect_meteors-2.1.1.tar.gz) · [SHA256SUMSで検証](#verify-release-downloads)

`uv sync --locked` はプロジェクトと `detect-meteors` コマンドを `.venv` にインストールします。
版番号の表示が `2.1.1` であることを確認してください。手元のRAW画像の全画面を検出対象にする場合は
`/path/to/rawfiles` を画像のあるディレクトリに置き換えます。

```bash
uv run detect-meteors --target /path/to/rawfiles --no-roi
```

Gitのチェックアウトとソース配布物の両方に、ドキュメント、設定例、テストが含まれます。

### リリースの wheel からインストールする

wheelをダウンロードしてチェックサムを検証します。プロジェクトのチェックアウト外に保存し、
そのディレクトリでターミナルを開いて実行します。
[wheelをダウンロード](https://github.com/shin3tky/detect_meteors/releases/download/2.1.1/detect_meteors-2.1.1-py3-none-any.whl) · [SHA256SUMSで検証](#verify-release-downloads)

```bash
uv venv --python 3.14
uv pip install ./detect_meteors-2.1.1-py3-none-any.whl
uv run detect-meteors --version
uv run detect-meteors --help
```

版番号の表示が `2.1.1` であることを確認してください。ここで作成した `.venv` がある
ディレクトリから `uv run detect-meteors` を実行できます。下の設定例を試す場合は、
Gitのチェックアウトかソース配布物を使います。

### サンプル RAW 画像

RAW 画像、目視ラベル、チェックサムは [detect_meteors_assets の rawfiles](https://github.com/shin3tky/detect_meteors_assets/tree/main/rawfiles) で公開しています。
Git LFSを導入し、`detect_meteors` のソースディレクトリでシェルに合うコマンドを実行すると、
航空機が写った12枚のサンプルを取得できます。

macOSまたはLinuxでは次のコマンドを実行します。

```bash
git lfs install --skip-repo
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/shin3tky/detect_meteors_assets.git ../detect_meteors_assets
git -C ../detect_meteors_assets lfs pull --include='rawfiles/2024GEMINI_AIRCRAFT/**'
mkdir -p rawfiles
cp -R ../detect_meteors_assets/rawfiles/2024GEMINI_AIRCRAFT rawfiles/
```

Windows PowerShellでは次のコマンドを実行します。

```powershell
git lfs install --skip-repo
$env:GIT_LFS_SKIP_SMUDGE = '1'
git clone https://github.com/shin3tky/detect_meteors_assets.git ../detect_meteors_assets
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
git -C ../detect_meteors_assets lfs pull --include='rawfiles/2024GEMINI_AIRCRAFT/**'
New-Item -ItemType Directory -Force -Path rawfiles | Out-Null
Copy-Item -Recurse ../detect_meteors_assets/rawfiles/2024GEMINI_AIRCRAFT -Destination rawfiles
```

隣に `detect_meteors_assets` をクローン済みの場合は `git lfs pull` から実行してください。
`2021GEMINI` または `2024GEMINI` を使う場合は、取得とコピーのコマンド内の系列名を置き換えてください。

### 検出設定の切り替え

チェックアウトまたは展開したソースの `rawfiles/2024GEMINI_AIRCRAFT/` に航空機サンプルを配置し、
ソースのルートで次のコマンドを実行します。YAML内の相対パスはコマンドを実行した
ディレクトリが基準です。

```bash
uv run detect-meteors --config config_examples/hough_legacy_sample.yaml --debug-image
```

試したい処理に応じて `--config` に渡すファイルを替えます。

| 処理 | `config_examples/` 内の設定ファイル |
| --- | --- |
| 従来の絶対差分ハフ（既定） | [hough_legacy_sample.yaml](../config_examples/hough_legacy_sample.yaml) |
| ROI内の恒星に合わせて直前画像を回転・平行移動し、絶対差分ハフで検出 | [hough_registered_sample.yaml](../config_examples/hough_registered_sample.yaml) |
| 位置合わせなしで、現在画像が明るくなった正の差分だけをハフ検出に使用 | [hough_positive_sample.yaml](../config_examples/hough_positive_sample.yaml) |
| 位置合わせなしの正の差分ハフで棄却したペアをFRTで再検査（実験用） | [hough_frt_retry_sample.yaml](../config_examples/hough_frt_retry_sample.yaml) |

従来の検出に戻すには `hough_legacy_sample.yaml` を指定します。4つの設定例は入力、ROI、ハフ閾値が
共通で、候補・デバッグ画像・進捗ファイルは設定ごとに別の場所に保存されます。手元のRAW画像を
使う場合は、設定例をコピーして `target_folder` と `roi_polygon` を撮影系列に合わせて変更します。
[設定の切り替えと移行の詳細](RELEASE_NOTES_2.1_ja.md#設定の切り替えと移行)

恒星位置合わせを追加すると処理時間が増え、FRT再検査では処理時間とワーカーあたりのメモリ使用量が
増えます。2025LEOの比較ではFRT再検査によって偽陽性も増えました。
[位置合わせの仕組みと使い方](star_registration_workflow_ja.md) · [FRTの検証結果と処理負荷](frt_retry_workflow_ja.md)

<a id="verify-release-downloads"></a>

### ダウンロードした配布物を検証する

wheel、ソース配布物 `detect_meteors-2.1.1.tar.gz`、`SHA256SUMS` をファイル名を変えずに
同じフォルダに保存します。チェックサムの対象はリリースに添付したwheelとソース配布物です。
[GitHub Release 2.1.1](https://github.com/shin3tky/detect_meteors/releases/tag/2.1.1) · [SHA256SUMSをダウンロード](https://github.com/shin3tky/detect_meteors/releases/download/2.1.1/SHA256SUMS)

macOS では、保存先のフォルダでターミナルを開き、次を実行します。

```sh
shasum -a 256 -c SHA256SUMS
```

Linuxでは `sha256sum -c SHA256SUMS` を実行します。両方のファイルに `OK` と表示されれば検証完了です。

Windows では、保存先のフォルダで PowerShell を開き、次を実行します。

```powershell
Get-Content -LiteralPath .\SHA256SUMS
Get-FileHash -Algorithm SHA256 -LiteralPath .\detect_meteors-2.1.1-py3-none-any.whl, .\detect_meteors-2.1.1.tar.gz
```

`Hash` 列の値を、`SHA256SUMS` に記載されている同じファイル名のチェックサムと照合します。
英字の大文字と小文字は区別しません。
一致しない場合は、該当するファイルをダウンロードし直し、インストール前にもう一度検証してください。

<a id="dependency-versions-v1613"></a>
<a id="dependency-versions-v211"></a>

## 依存ライブラリのバージョン（2.1.1）

実行時の依存ライブラリは `pyproject.toml` で固定し、`uv.lock` に記録しています。

| ライブラリ | 固定バージョン |
| --- | --- |
| `numpy` | `2.5.5` |
| `opencv-python` | `5.0.0.93` |
| `rawpy` | `0.28.0` |
| `psutil` | `7.2.2` |
| `pillow` | `13.0.0` |
| `pydantic` | `2.14.4` |
| `pyyaml` | `6.0.4` |

2.1.1ではPydanticとPyYAMLを更新しました。
[変更履歴](CHANGELOG_ja.md#v211) · [開発ツールのバージョン](INSTALL_DEV_ja.md#development-tool-versions)

## 2.1.1 への更新

2.1.1のソースに切り替えた後、プロジェクトルートで用途に合うコマンドを実行します。

```bash
uv sync --locked
# 開発用。pre-commitとその依存関係も含める
uv sync --locked --all-extras
```

wheelで導入している場合は、2.1.1のwheelをダウンロードして検証します。使用中の `.venv` がある
ディレクトリで、ダウンロードしたwheelのパスを指定して次を実行します。
[wheelをダウンロード](https://github.com/shin3tky/detect_meteors/releases/download/2.1.1/detect_meteors-2.1.1-py3-none-any.whl) · [SHA256SUMSで検証](#verify-release-downloads)

```bash
uv pip install --upgrade ./detect_meteors-2.1.1-py3-none-any.whl
uv run detect-meteors --version
```

wheelでは直接の実行時依存を固定し、間接依存は各パッケージの制約に従って解決します。
設定ファイルの互換性と設定変更時の進捗ファイルの扱いは、リリースノートを参照してください。
[2.1.1の移行手順](RELEASE_NOTES_2.1_ja.md#設定の切り替えと移行)

<a id="upgrading-from-1612"></a>

1.6.12から1.6.13への更新手順は、当時のガイドを参照してください。
[1.6.13のインストールガイド](https://github.com/shin3tky/detect_meteors/blob/1.6.13/docs/INSTALL_ja.md#upgrading-from-1612)

## macOSへのインストール

### 手順1：Homebrewをインストールする

Homebrewがない場合は、https://brew.sh の手順に従ってインストールしてください。

ターミナルを開き、HomebrewのWebサイトに掲載されているインストールコマンドを実行します。

### 手順2：Gitをインストールする

ターミナルでHomebrewを使い、Gitをインストールします。

```bash
brew install git
```

### 手順3：リポジトリをクローンする

```bash
# Clone the repository
git clone --branch 2.1.1 --single-branch https://github.com/shin3tky/detect_meteors.git
cd detect_meteors
```

### 手順4：uvをインストールする

高速なPythonパッケージ・プロジェクト管理ツールであるuvをインストールします。

```bash
brew install uv
```

### 手順5：システム依存ライブラリをインストールする

Pythonパッケージのソースビルドなどで必要な場合に、システムライブラリをインストールします。
対応するwheelを使う通常の構成では、この手順を省略できます。

```bash
brew install opencv libraw
```

### 手順6：Python環境を構築する

uvはPythonのバージョンと仮想環境を自動管理します。

```bash
# 最新のPython 3.14.xをインストール
uv python install 3.14

# Create virtual environment and install dependencies
uv sync --locked
```

この操作で、次の処理を行います。

- 最新のPython 3.14.xパッチリリースをダウンロードしてインストール。
- `.venv` に仮想環境を作成。
- `uv.lock` に従って実行時の依存パッケージをインストール。

### 手順7：動作を確認する

次のコマンドでインストール結果を確認します。

```bash
uv run detect-meteors --help
```

利用可能なすべてのオプションを含むヘルプが表示されます。

**補足**：プロジェクトルートで `uv run detect-meteors` を実行すると、uvが `.venv` の実行環境を選択します。

### 任意：シェル補完（Bash/Zsh）

Gitで取得したソースとリリースページのソースアーカイブにはBash用とZsh標準の補完を収録しています。
wheelだけを使う場合はソースアーカイブから補完ファイルを取り出してください。
補完はインストール後の `detect-meteors` に登録します。`uv run` の後ろでの補完は、シェル側のuv補完との連携が別途必要です。

**Bash (`~/.bashrc`)**

```bash
source /path/to/detect_meteors/detect_meteors_cli_completion.bash
```

**Zsh (`~/.zshrc`)**

まず、補完ファイルを配置します。

```zsh
mkdir -p ~/.zsh/completions
cp /path/to/detect_meteors/_detect_meteors_cli ~/.zsh/completions/
```

`~/.zshrc` では既存の `compinit` 実行より前に `fpath` を設定します。

```zsh
fpath=(~/.zsh/completions $fpath)
autoload -Uz compinit
compinit
```

**uv経由の実行で補完を使う**

上記の補完を設定した後、Bashでは `~/.bashrc`、Zshでは `~/.zshrc` に次のシェル関数を追加します。
両方のシェルで同じ定義を使えます。

```sh
detect-meteors() {
    uv run detect-meteors "$@"
}
```

新しいシェルを起動し、プロジェクトルートで `detect-meteors` を実行します。

```sh
detect-meteors --sensor-type MFT
```

例えば `detect-meteors --sen` まで入力してTabキーを押すと、`--sensor-type` や
`--sensor-width` を補完できます。分析・調整・保存オプションも補完できます。
関数はすべての引数を `uv run detect-meteors` に渡すため、uvによるプロジェクト環境の選択と同期も維持されます。
補完を使うときは関数名から直接入力してください。`uv run detect-meteors` と入力する場合は、
引き続きuv補完との別途連携が必要です。

---

## Windowsへのインストール

### 手順1：Gitをインストールする

PowerShellまたはコマンドプロンプトを開き、WingetでGitをインストールします。

```powershell
winget install --id Git.Git -e --source winget
```

インストール後はターミナルを再起動するか、新しく開いて、PATHにGitが反映されるようにしてください。

### 手順2：uvをインストールする

Wingetでuvをインストールします。

```powershell
winget install --id astral-sh.uv -e --source winget
```

インストール後はターミナルを再起動するか、新しく開いて、PATHにuvが反映されるようにしてください。

### 手順3：リポジトリをクローンする

```powershell
# Clone the repository
git clone --branch 2.1.1 --single-branch https://github.com/shin3tky/detect_meteors.git
cd detect_meteors
```

### 手順4：システム依存ライブラリを確認する

**補足**：Windowsでは、次の手順でuvがOpenCVをインストールします。LibRawに必要なバイナリは `rawpy` に含まれるため、別途インストールする必要はありません。

### 手順5：Python環境を構築する

uvはPythonのバージョンと仮想環境を自動管理します。

```powershell
# 最新のPython 3.14.xをインストール
uv python install 3.14

# Create virtual environment and install dependencies
uv sync --locked
```

この操作で、次の処理を行います。

- 最新のPython 3.14.xパッチリリースをダウンロードしてインストール。
- `.venv` に仮想環境を作成。
- `uv.lock` に従って実行時の依存パッケージをインストール。

### 手順6：動作を確認する

次のコマンドでインストール結果を確認します。

```powershell
uv run detect-meteors --help
```

利用可能なすべてのオプションを含むヘルプが表示されます。

**補足**：プロジェクトルートで `uv run detect-meteors` を実行すると、uvが `.venv` の実行環境を選択します。

---

## よくある問題

### macOS

- **Homebrewが見つからない**：Homebrewがインストール済みで、`/usr/local/bin`（Intel）または `/opt/homebrew/bin`（Apple Silicon）がPATHに含まれることを確認してください。
- **uvが見つからない**：インストール後にターミナルを再起動するか、`brew link uv` を実行してください。

### Windows

- **Wingetを利用できない**：Windows 10（バージョン1809以降）またはWindows 11を使用していることを確認し、必要ならWindowsを更新してください。
- **GitやuvがPATHにない**：インストール後にターミナルを再起動するか、インストール先のディレクトリを手動でPATHへ追加してください。
- **PowerShellの実行ポリシー**：補足のwheel導入で手動の仮想環境有効化が必要な場合は、`Set-ExecutionPolicy RemoteSigned -Scope Process` で現在のセッションにだけ適用できます。

## 次の手順

インストール後は[README](../README_ja.md)の使い方と実行例を参照してください。
