# インストールガイド

[English](INSTALL.md)

Detect Meteors CLIをmacOSとWindowsにインストールする手順を説明します。

## v1.6.11の配布ファイルからインストールする

リリースのwheelファイルを持っている場合は、Python 3.12または3.13の仮想環境へインストールしてください。依存ライブラリのインストールにはパッケージインデックスへの接続が必要です。

```bash
uv venv --python 3.12
uv pip install dist/detect_meteors-1.6.11-py3-none-any.whl
```

ダウンロードしたwheelが `dist/` にない場合は、実際の保存先を指定してください。macOSとLinuxでは `source .venv/bin/activate`、Windows PowerShellでは `.venv\Scripts\Activate.ps1` で仮想環境を有効にし、次のコマンドを実行します。

```bash
detect-meteors --version
detect-meteors --help
detect-meteors --target /path/to/rawfiles --hooks aircraft_trail --no-roi
```

wheelには `detect-meteors` コマンドを含めています。従来のソースからの実行（`uv run python detect_meteors_cli.py ...`）も利用できます。
ソース配布版にはドキュメント、設定例、テストが含まれます。`detect_meteors-1.6.11.tar.gz` を展開してそのディレクトリへ移動し、以下の `uv sync` の手順に従ってください。
これらの配布ファイルにRAW画像や生成済み候補は含めません。飛行機サンプルを再現するには、チェックサムとともに `rawfiles/2024GEMINI_AIRCRAFT` を管理しているリポジトリをクローンするか、そのディレクトリを展開済みソースへコピーしてください。

## 依存ライブラリのバージョン（v1.6.11）

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

開発用ツールは[開発環境ガイド](INSTALL_DEV_ja.md#開発ツールのバージョンv1611)を参照。

<a id="upgrading-from-1610"></a>

## 1.6.10からの更新

1.6.11のソースへ切り替えた後、プロジェクトルートで実行します。

```bash
uv sync --locked
# Development environment, including pre-commit and its dependencies:
uv sync --locked --all-extras
```

用途に合わせてどちらかを選びます。`uv.lock` に記録した直接・間接依存を反映します。
wheelで導入している場合は、使用中の仮想環境へ新しい配布ファイルをインストールしてください。

```bash
uv pip install --upgrade /path/to/detect_meteors-1.6.11-py3-none-any.whl
```

wheelでは直接の実行時依存を固定し、間接依存は各パッケージの制約に従って解決します。
[バージョン差分](CHANGELOG_ja.md#dependency-updates-1610-to-1611)と
[セキュリティ修正の根拠](CHANGELOG_ja.md#security-updates-1611)も参照してください。

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
git clone https://github.com/shin3tky/detect_meteors.git
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
uv sync
```

この操作で、次の処理を行います。

- 最新のPython 3.12.xパッチリリースをダウンロードしてインストール。
- `.venv` に仮想環境を作成。
- `pyproject.toml` の依存ライブラリをすべてインストール。

### 手順7：動作を確認する

次のコマンドでインストール結果を確認します。

```bash
uv run python detect_meteors_cli.py --help
```

利用可能なすべてのオプションを含むヘルプが表示されます。

**補足**：仮想環境内のコマンドは `uv run` で実行できます。手動で仮想環境を有効にして実行することもできます。

```bash
source .venv/bin/activate
python detect_meteors_cli.py --help
```

### 任意：シェル補完（Bash/Zsh）

リポジトリとソース配布版にはBash用とZsh標準の補完を収録しています。
wheelだけを使う場合はソース配布版から補完ファイルを取り出してください。
インストール後の `detect-meteors` とソースの `detect_meteors_cli.py` の両方に対応します。
`python`／`python3` 全体の補完は変更しません。

**Bash (`~/.bashrc`)**

```bash
source /path/to/detect_meteors/detect_meteors_cli_completion.bash

# Optional source-command alias
alias detect_meteors='uv run python /path/to/detect_meteors/detect_meteors_cli.py'
complete -F _detect_meteors_cli_completion detect_meteors
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

# Optional source-command alias
alias detect_meteors='uv run python /path/to/detect_meteors/detect_meteors_cli.py'
compdef _detect_meteors_cli detect_meteors
```

設定後に新しいシェルを起動します。分析・調整・保存オプションも補完できます。

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
git clone https://github.com/shin3tky/detect_meteors.git
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
uv sync
```

この操作で、次の処理を行います。

- 最新のPython 3.12.xパッチリリースをダウンロードしてインストール。
- `.venv` に仮想環境を作成。
- `pyproject.toml` の依存ライブラリをすべてインストール。

### 手順6：動作を確認する

次のコマンドでインストール結果を確認します。

```powershell
uv run python detect_meteors_cli.py --help
```

利用可能なすべてのオプションを含むヘルプが表示されます。

**補足**：仮想環境内のコマンドは `uv run` で実行できます。手動で仮想環境を有効にして実行することもできます。

```powershell
# Allow script execution for this terminal session only
Set-ExecutionPolicy RemoteSigned -Scope Process

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

python detect_meteors_cli.py --help
```

---

## よくある問題

### macOS

- **Homebrewが見つからない**：Homebrewがインストール済みで、`/usr/local/bin`（Intel）または `/opt/homebrew/bin`（Apple Silicon）がPATHに含まれることを確認してください。
- **uvが見つからない**：インストール後にターミナルを再起動するか、`brew link uv` を実行してください。

### Windows

- **Wingetを利用できない**：Windows 10（バージョン1809以降）またはWindows 11を使用していることを確認し、必要ならWindowsを更新してください。
- **GitやuvがPATHにない**：インストール後にターミナルを再起動するか、インストール先のディレクトリを手動でPATHへ追加してください。
- **PowerShellの実行ポリシー**：手順6の `Set-ExecutionPolicy RemoteSigned -Scope Process` を使ってください。現在のターミナルセッションにのみ適用され、影響を限定できます。

## 次の手順

インストール後は[README](../README_ja.md)の使い方と実行例を参照してください。
