# Installation Guide

[日本語版](INSTALL_ja.md)

This guide provides detailed installation instructions for Detect Meteors CLI on macOS and Windows.

> [!IMPORTANT]
> **macOS: choose a checkout location outside iCloud Drive.**
> If iCloud Drive syncs your Documents and Desktop folders, do not clone or
> extract `detect_meteors` anywhere under `~/Documents` or `~/Desktop`.
> Use a location outside iCloud Drive, such as
> `~/Development/shin3tky/detect_meteors`.
> In synced locations, hidden attributes can be reapplied to files in `.venv`,
> causing startup to fail with `ModuleNotFoundError` even after a successful installation.

## Install v1.6.12 from Source

Install Git and uv using the platform-specific steps below, then obtain the
released source and install its locked dependencies. Python 3.12 or 3.13 is
required; dependency installation needs access to the package index.

```bash
git clone --branch 1.6.12 --single-branch https://github.com/shin3tky/detect_meteors.git
cd detect_meteors
uv python install 3.12
uv sync --locked
uv run detect-meteors --version
uv run detect-meteors --help
```

Alternatively, download the [v1.6.12 source distribution](https://github.com/shin3tky/detect_meteors/releases/download/1.6.12/detect_meteors-1.6.12.tar.gz)
or one of GitHub's **Source code** archives from the
[release page](https://github.com/shin3tky/detect_meteors/releases/tag/1.6.12).
If you downloaded the source distribution, [verify it with SHA256SUMS](#verify-release-downloads) before extracting it.
Extract it, open a terminal in the extracted directory containing `pyproject.toml`,
and run the commands above starting with `uv python install 3.12`.

`uv sync --locked` installs the project and the `detect-meteors` command into
`.venv`. From the source directory, run detection with:

```bash
uv run detect-meteors --target /path/to/rawfiles --hooks aircraft_trail --no-roi
```

All application examples below use `uv run detect-meteors` (with a hyphen)
from the source directory. The source includes documentation, configuration
examples, and tests. The RAW images for the `rawfiles/2024GEMINI_AIRCRAFT`
sample are available in the Git checkout or GitHub's **Source code** archives.

### Install the Release Wheel

Download the [v1.6.12 wheel](https://github.com/shin3tky/detect_meteors/releases/download/1.6.12/detect_meteors-1.6.12-py3-none-any.whl)
and open a terminal in the directory where you saved it.
[Verify the wheel with SHA256SUMS](#verify-release-downloads) before installing it.

```bash
uv venv --python 3.12
uv pip install ./detect_meteors-1.6.12-py3-none-any.whl
```

If you saved the wheel elsewhere, replace the filename with its actual path.
On macOS or Linux, activate with `source .venv/bin/activate`; on Windows
PowerShell, use `.venv\Scripts\Activate.ps1`. Then run:

```bash
uv run detect-meteors --version
uv run detect-meteors --help
```

The wheel includes the command but not the source tree, RAW sample, or generated
candidates. Use the source instructions above for documentation, configuration
examples, tests, and sample images.

<a id="verify-release-downloads"></a>

### Verify the Release Downloads

From [GitHub Release 1.6.12](https://github.com/shin3tky/detect_meteors/releases/tag/1.6.12),
save the wheel, the `detect_meteors-1.6.12.tar.gz` source distribution, and
[`SHA256SUMS`](https://github.com/shin3tky/detect_meteors/releases/download/1.6.12/SHA256SUMS)
in the same directory. Keep the original filenames. The checksums cover these two
distribution files. GitHub's automatically generated **Source code** archives
are separate downloads.

On macOS, open a terminal in that directory and run:

```sh
shasum -a 256 -c SHA256SUMS
```

Both files should report `OK`.

On Windows, open PowerShell in the same directory and run:

```powershell
Get-Content -LiteralPath .\SHA256SUMS
Get-FileHash -Algorithm SHA256 -LiteralPath .\detect_meteors-1.6.12-py3-none-any.whl, .\detect_meteors-1.6.12.tar.gz
```

Compare each value in the `Hash` column with the checksum for the same filename
in `SHA256SUMS`. Uppercase and lowercase letters are equivalent.
If a checksum differs, download the affected file again and verify it before installing.

## Dependency Versions (v1.6.12)

Python **3.12 and 3.13** are supported. The following runtime package versions
are pinned in `pyproject.toml` and recorded in `uv.lock`.

| Package | Pinned version |
| --- | --- |
| `numpy` | `2.5.4` |
| `opencv-python` | `5.0.0.93` |
| `rawpy` | `0.28.0` |
| `psutil` | `7.2.2` |
| `pillow` | `13.0.0` |
| `pydantic` | `2.14.3` |
| `pyyaml` | `6.0.3` |

The runtime pins are the same as in v1.6.11. The development dependency `ty`
was updated to `0.0.100`. See the [changelog](CHANGELOG.md#v1612) and
[development tool versions](INSTALL_DEV.md#development-tool-versions).

<a id="upgrading-from-1611"></a>

## Upgrading from 1.6.11

After switching to the 1.6.12 source, run from the project root:

```bash
uv sync --locked
# Development environment, including pre-commit and its dependencies:
uv sync --locked --all-extras
```

Choose the command for your environment. It installs direct and transitive
versions recorded in `uv.lock`. For a wheel installation, download the v1.6.12
wheel from the release page and [verify it with SHA256SUMS](#verify-release-downloads).
Then activate your existing virtual environment and run the following from the
directory where you saved the wheel. Use its actual path if it is saved elsewhere:

```bash
uv pip install --upgrade ./detect_meteors-1.6.12-py3-none-any.whl
```

The wheel pins direct runtime dependencies; transitive packages resolve according
to their package constraints. See the [v1.6.12 release notes](RELEASE_NOTES_1.6.md#version-1612)
for configuration compatibility and recovery after detector errors.

## macOS Installation

### Step 1: Install Homebrew

If you don't have Homebrew installed, follow the instructions at https://brew.sh to install it.

Open Terminal and run the installation command provided on the Homebrew website.

### Step 2: Install Git

Open Terminal and install Git using Homebrew:

```bash
brew install git
```

### Step 3: Clone the Repository

```bash
# Clone the repository
git clone --branch 1.6.12 --single-branch https://github.com/shin3tky/detect_meteors.git
cd detect_meteors
```

### Step 4: Install uv

Install uv, a fast Python package and project manager:

```bash
brew install uv
```

### Step 5: Install System Dependencies

Install system libraries when needed for source builds. This step can be
skipped when compatible Python wheels are available:

```bash
brew install opencv libraw
```

### Step 6: Set Up Python Environment

uv manages Python versions and virtual environments automatically:

```bash
# Install the latest Python 3.12.x
uv python install 3.12

# Create virtual environment and install dependencies
uv sync --locked
```

This will:
- Download and install the latest Python 3.12.x patch release
- Create a virtual environment in `.venv`
- Install runtime dependencies according to `uv.lock`

### Step 7: Verification

Verify the installation:

```bash
uv run detect-meteors --help
```

You should see the help message with all available options.

**Note**: Run `uv run detect-meteors` from the project root; uv selects the
project’s `.venv` environment.

### Optional: Shell Completion (Bash/Zsh)

The Git checkout and release-page source archives include Bash and native Zsh
completion. For wheel-only installations, obtain the completion files from a
source archive. The scripts register completion for the installed
`detect-meteors` command. Completing its arguments after `uv run` additionally
requires integration with your shell’s uv completion.

**Bash (`~/.bashrc`)**

```bash
source /path/to/detect_meteors/detect_meteors_cli_completion.bash
```

**Zsh (`~/.zshrc`)**

First place the completion file in a directory for your Zsh functions:

```zsh
mkdir -p ~/.zsh/completions
cp /path/to/detect_meteors/_detect_meteors_cli ~/.zsh/completions/
```

In `~/.zshrc`, set `fpath` before any existing `compinit` invocation:

```zsh
fpath=(~/.zsh/completions $fpath)
autoload -Uz compinit
compinit
```

**Use completion while running through uv**

After configuring completion above, add this shell function to `~/.bashrc`
(Bash) or `~/.zshrc` (Zsh). The same definition works in both shells:

```sh
detect-meteors() {
    uv run detect-meteors "$@"
}
```

Start a new shell, then run `detect-meteors` from the project root:

```sh
detect-meteors --sensor-type MFT
```

For example, type `detect-meteors --sen` and press Tab to complete
`--sensor-type` or `--sensor-width`. Analysis, tuning, and report/session options
are also included. The function forwards all arguments to `uv run detect-meteors`,
so uv still selects and synchronizes the project environment. Use the function
name directly to get completion; typing `uv run detect-meteors` still requires
separate integration with uv's completion.

---

## Windows Installation

### Step 1: Install Git

Open PowerShell or Command Prompt and install Git using Winget:

```powershell
winget install --id Git.Git -e --source winget
```

After installation, restart your terminal or open a new one to ensure Git is in your PATH.

### Step 2: Install uv

Install uv using Winget:

```powershell
winget install --id astral-sh.uv -e --source winget
```

After installation, restart your terminal or open a new one to ensure uv is in your PATH.

### Step 3: Clone the Repository

```powershell
# Clone the repository
git clone --branch 1.6.12 --single-branch https://github.com/shin3tky/detect_meteors.git
cd detect_meteors
```

### Step 4: Install System Dependencies

**Note**: On Windows, OpenCV will be installed via uv in the next steps. For LibRaw support, `rawpy` includes the necessary binaries, so no separate installation is required.

### Step 5: Set Up Python Environment

uv manages Python versions and virtual environments automatically:

```powershell
# Install the latest Python 3.12.x
uv python install 3.12

# Create virtual environment and install dependencies
uv sync --locked
```

This will:
- Download and install the latest Python 3.12.x patch release
- Create a virtual environment in `.venv`
- Install runtime dependencies according to `uv.lock`

### Step 6: Verification

Verify the installation:

```powershell
uv run detect-meteors --help
```

You should see the help message with all available options.

**Note**: Run `uv run detect-meteors` from the project root; uv selects the
project’s `.venv` environment.

---

## Common Issues

### macOS

- **Homebrew not found**: Ensure Homebrew is installed and `/usr/local/bin` (Intel) or `/opt/homebrew/bin` (Apple Silicon) is in your PATH.
- **uv not found**: After installation, restart your terminal or run `brew link uv`.

### Windows

- **Winget not available**: Ensure you're running Windows 10 (version 1809 or later) or Windows 11. Update Windows if necessary.
- **Git or uv not in PATH**: After installation, restart your terminal or add the installation directories to your PATH manually.
- **PowerShell execution policy**: If the optional wheel setup requires manual environment activation, `Set-ExecutionPolicy RemoteSigned -Scope Process` applies to the current terminal session only.

## Next Steps

After installation, proceed to the [README](../README.md) for usage instructions and examples.
