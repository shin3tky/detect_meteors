# インストールガイド

[English](INSTALL.md)

Detect Meteors CLIをmacOSとWindowsにインストールする手順を説明します。

> [!IMPORTANT]
> **macOSでは、iCloud Driveの同期対象外にチェックアウトしてください。**
> iCloud Driveで「書類」と「デスクトップ」を同期している場合、
> `~/Documents` または `~/Desktop` の配下には `detect_meteors` をクローンしないでください。
> ソースアーカイブの展開先も同様です。
> `~/Development/shin3tky/detect_meteors` など、iCloud Driveの同期対象外の場所を使ってください。
> 同期対象内では `.venv` 内のファイルに非表示属性が再付与され、
> 導入に成功していても `ModuleNotFoundError` で起動できなくなる場合があります。

## v1.6.12 のソースからインストールする

以下のOS別の手順でGitとuvを用意した後、リリース済みのソースを取得し、
ロックファイルに従って依存パッケージをインストールします。
Python 3.12または3.13が必要です。依存パッケージの取得にはパッケージインデックスへの接続が必要です。

```bash
git clone --branch 1.6.12 --single-branch https://github.com/shin3tky/detect_meteors.git
cd detect_meteors
uv python install 3.12
uv sync --locked
uv run detect-meteors --version
uv run detect-meteors --help
```

Git を使わずに取得する場合は、[v1.6.12 のソース配布物](https://github.com/shin3tky/detect_meteors/releases/download/1.6.12/detect_meteors-1.6.12.tar.gz)か、
[リリースページ](https://github.com/shin3tky/detect_meteors/releases/tag/1.6.12)にある GitHub の **Source code** アーカイブをダウンロードしてください。
ソース配布物を選んだ場合は、展開する前に [SHA256SUMS で検証](#verify-release-downloads)してください。
ダウンロードしたファイルを展開し、`pyproject.toml` があるディレクトリでターミナルを開いて、
上記の `uv python install 3.12` 以降のコマンドを実行します。

`uv sync --locked` はプロジェクトと `detect-meteors` コマンドを `.venv` にインストールします。
取得したソースのディレクトリから、次のように実行できます。

```bash
uv run detect-meteors --target /path/to/rawfiles --hooks aircraft_trail --no-roi
```

以降のアプリ起動例も `uv run detect-meteors` に統一しています。コマンド名はハイフン付きです。
ソースにはドキュメント、設定例、テストが含まれます。`rawfiles/2024GEMINI_AIRCRAFT` のサンプル RAW 画像は、Git でクローンしたソースか GitHub の **Source code** アーカイブで取得できます。

### リリースの wheel からインストールする

[v1.6.12 の wheel](https://github.com/shin3tky/detect_meteors/releases/download/1.6.12/detect_meteors-1.6.12-py3-none-any.whl)をダウンロードし、保存先でターミナルを開いてください。
インストールする前に [SHA256SUMS で検証](#verify-release-downloads)してください。

```bash
uv venv --python 3.12
uv pip install ./detect_meteors-1.6.12-py3-none-any.whl
```

wheelを別の場所に保存した場合は、実際のファイルパスに置き換えてください。
macOSとLinuxでは `source .venv/bin/activate`、Windows PowerShellでは
`.venv\Scripts\Activate.ps1` で仮想環境を有効にし、次のコマンドを実行します。

```bash
uv run detect-meteors --version
uv run detect-meteors --help
```

wheelにはコマンドを含みますが、ソース一式、RAWサンプル、生成済み候補は含みません。
ドキュメント、設定例、テスト、サンプル画像を使う場合は、上記のソース取得手順に従ってください。

<a id="verify-release-downloads"></a>

### ダウンロードした配布物を検証する

[GitHub Release 1.6.12](https://github.com/shin3tky/detect_meteors/releases/tag/1.6.12) から、
wheel、ソース配布物 `detect_meteors-1.6.12.tar.gz`、
[`SHA256SUMS`](https://github.com/shin3tky/detect_meteors/releases/download/1.6.12/SHA256SUMS) を
ファイル名を変えずに同じフォルダに保存します。`SHA256SUMS` に記載されているのはこの2つの配布物のチェックサムです。
GitHub が自動生成する **Source code** アーカイブは、この一覧の検証対象に含まれません。

macOS では、保存先のフォルダでターミナルを開き、次を実行します。

```sh
shasum -a 256 -c SHA256SUMS
```

両方のファイルに `OK` と表示されれば、検証は完了です。

Windows では、保存先のフォルダで PowerShell を開き、次を実行します。

```powershell
Get-Content -LiteralPath .\SHA256SUMS
Get-FileHash -Algorithm SHA256 -LiteralPath .\detect_meteors-1.6.12-py3-none-any.whl, .\detect_meteors-1.6.12.tar.gz
```

`Hash` 列の値を、`SHA256SUMS` に記載されている同じファイル名のチェックサムと照合します。
英字の大文字と小文字は区別しません。
一致しない場合は、該当するファイルをダウンロードし直し、インストール前にもう一度検証してください。

<a id="dependency-versions-v1612"></a>

## 依存ライブラリのバージョン（v1.6.12）

Python **3.12／3.13** に対応します。以下は `pyproject.toml` と `uv.lock` の固定値です。

| ライブラリ | 固定バージョン |
| --- | --- |
| `numpy` | `2.5.4` |
| `opencv-python` | `5.0.0.93` |
| `rawpy` | `0.28.0` |
| `psutil` | `7.2.2` |
| `pillow` | `13.0.0` |
| `pydantic` | `2.14.3` |
| `pyyaml` | `6.0.3` |

実行時の依存ライブラリは v1.6.11 と同じです。開発用の `ty` は `0.0.100` に更新しました。[変更履歴](CHANGELOG_ja.md#v1612)と[開発ツールのバージョン](INSTALL_DEV_ja.md#development-tool-versions)を参照してください。

<a id="upgrading-from-1611"></a>

## 1.6.11 からの更新

1.6.12 のソースへ切り替えた後、プロジェクトルートで実行します。

```bash
uv sync --locked
# Development environment, including pre-commit and its dependencies:
uv sync --locked --all-extras
```

用途に合わせてどちらかを選びます。`uv.lock` に記録した直接・間接依存を反映します。
wheel で導入している場合は、リリースページから v1.6.12 の wheel をダウンロードし、
[SHA256SUMS で検証](#verify-release-downloads)してください。
その後、使用中の仮想環境を有効にして、wheel の保存先で次を実行します。
別の場所に保存した場合は実際のパスを指定してください。

```bash
uv pip install --upgrade ./detect_meteors-1.6.12-py3-none-any.whl
```

wheelでは直接の実行時依存を固定し、間接依存は各パッケージの制約に従って解決します。
設定の互換性や検出器エラーからの再開については、[v1.6.12 のリリースノート](RELEASE_NOTES_1.6_ja.md#version-1612)を参照してください。

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
git clone --branch 1.6.12 --single-branch https://github.com/shin3tky/detect_meteors.git
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
# Install the latest Python 3.12.x
uv python install 3.12

# Create virtual environment and install dependencies
uv sync --locked
```

この操作で、次の処理を行います。

- 最新のPython 3.12.xパッチリリースをダウンロードしてインストール。
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
git clone --branch 1.6.12 --single-branch https://github.com/shin3tky/detect_meteors.git
cd detect_meteors
```

### 手順4：システム依存ライブラリを確認する

**補足**：Windowsでは、次の手順でuvがOpenCVをインストールします。LibRawに必要なバイナリは `rawpy` に含まれるため、別途インストールする必要はありません。

### 手順5：Python環境を構築する

uvはPythonのバージョンと仮想環境を自動管理します。

```powershell
# Install the latest Python 3.12.x
uv python install 3.12

# Create virtual environment and install dependencies
uv sync --locked
```

この操作で、次の処理を行います。

- 最新のPython 3.12.xパッチリリースをダウンロードしてインストール。
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
