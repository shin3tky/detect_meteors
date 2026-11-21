# Installation Guide

[日本語版](INSTALL_ja.md)

📺 Detect Meteors 2.1.1 was released on November 21, 2026 (World Television Day).
This guide covers installation on macOS and Windows.
[GitHub Release 2.1.1](https://github.com/shin3tky/detect_meteors/releases/tag/2.1.1) · [Release notes](RELEASE_NOTES_2.1.md#version-211)

> [!IMPORTANT]
> **macOS: choose a checkout location outside iCloud Drive.**
> If iCloud Drive syncs your Documents and Desktop folders, do not clone or
> extract `detect_meteors` anywhere under `~/Documents` or `~/Desktop`.
> Use a location outside iCloud Drive, such as
> `~/Development/shin3tky/detect_meteors`.
> In synced locations, hidden attributes can be reapplied to files in `.venv`,
> causing startup to fail with `ModuleNotFoundError` even after a successful installation.

## Python Requirements

Use uv with Python 3.12, 3.13, or 3.14. The release wheel and source
distribution were verified on all three versions. The Git checkout defaults
to Python 3.14. Installing dependencies requires access to the package index.

## Install 2.1.1 from Source

Set up Git and uv using the platform-specific steps below, then check out
the release tag and install from the lockfile.

```bash
git clone --branch 2.1.1 --single-branch https://github.com/shin3tky/detect_meteors.git
cd detect_meteors
uv sync --locked
uv run detect-meteors --version
uv run detect-meteors --help
```

You can also download the source distribution. Verify it before extracting
it, open a terminal in the extracted directory containing `pyproject.toml`,
and run the commands above starting with `uv sync --locked`.
[Download the source distribution](https://github.com/shin3tky/detect_meteors/releases/download/2.1.1/detect_meteors-2.1.1.tar.gz) · [Verify with SHA256SUMS](#verify-release-downloads)

`uv sync --locked` installs the project and the `detect-meteors` command into
`.venv`. The version command should report `2.1.1`. To run detection on the
full frame of your own RAW images, replace `/path/to/rawfiles` with their
directory.

```bash
uv run detect-meteors --target /path/to/rawfiles --no-roi
```

The Git checkout and source distribution include the documentation,
configuration examples, and tests.

### Install the Release Wheel

Download and verify the wheel, then open a terminal in the directory where
you saved it, outside a project checkout.
[Download the wheel](https://github.com/shin3tky/detect_meteors/releases/download/2.1.1/detect_meteors-2.1.1-py3-none-any.whl) · [Verify with SHA256SUMS](#verify-release-downloads)

```bash
uv venv --python 3.14
uv pip install ./detect_meteors-2.1.1-py3-none-any.whl
uv run detect-meteors --version
uv run detect-meteors --help
```

The version command should report `2.1.1`. Run `uv run detect-meteors`
from the directory containing this `.venv`. To try the configuration examples
below, use the Git checkout or source distribution.

### Sample RAW Images

The [detect_meteors_assets repository](https://github.com/shin3tky/detect_meteors_assets/tree/main/rawfiles)
contains the RAW sequences, labels, and checksums. Install Git LFS to download
the images. Run the commands for your shell from the `detect_meteors` source
directory to download the 12-frame aircraft sample.

Run this block on macOS or Linux.

```bash
git lfs install --skip-repo
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/shin3tky/detect_meteors_assets.git ../detect_meteors_assets
git -C ../detect_meteors_assets lfs pull --include='rawfiles/2024GEMINI_AIRCRAFT/**'
mkdir -p rawfiles
cp -R ../detect_meteors_assets/rawfiles/2024GEMINI_AIRCRAFT rawfiles/
```

Run this block in Windows PowerShell.

```powershell
git lfs install --skip-repo
$env:GIT_LFS_SKIP_SMUDGE = '1'
git clone https://github.com/shin3tky/detect_meteors_assets.git ../detect_meteors_assets
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
git -C ../detect_meteors_assets lfs pull --include='rawfiles/2024GEMINI_AIRCRAFT/**'
New-Item -ItemType Directory -Force -Path rawfiles | Out-Null
Copy-Item -Recurse ../detect_meteors_assets/rawfiles/2024GEMINI_AIRCRAFT -Destination rawfiles
```

If you already cloned `detect_meteors_assets` alongside the source, start with
`git lfs pull` in that checkout. To use `2021GEMINI` or `2024GEMINI`, substitute
the sequence name in the pull and copy commands.

### Switch Detection Configurations

Place the aircraft sample in `rawfiles/2024GEMINI_AIRCRAFT/` in the checkout or
extracted source distribution. Run the following command from that source
directory. Relative paths in the YAML files are resolved from the working
directory.

```bash
uv run detect-meteors --config config_examples/hough_legacy_sample.yaml --debug-image
```

Change the file passed to `--config` to select a workflow.

| Workflow | File in `config_examples/` |
| --- | --- |
| Conventional absolute-difference Hough (default) | [hough_legacy_sample.yaml](../config_examples/hough_legacy_sample.yaml) |
| Align the previous frame to stars in the ROI, then run absolute-difference Hough | [hough_registered_sample.yaml](../config_examples/hough_registered_sample.yaml) |
| Run Hough on positive differences without alignment | [hough_positive_sample.yaml](../config_examples/hough_positive_sample.yaml) |
| Run experimental FRT retry on pairs rejected by unaligned positive-difference Hough | [hough_frt_retry_sample.yaml](../config_examples/hough_frt_retry_sample.yaml) |

Select `hough_legacy_sample.yaml` again to return to conventional detection.
The examples share the input, ROI, and Hough thresholds. Each writes
candidates, debug images, and progress to its own paths. To use your own RAW
sequence, copy an example and update `target_folder` and `roi_polygon`.
[Configuration and migration details](RELEASE_NOTES_2.1.md#switching-configurations-and-upgrading)

Stellar registration adds preprocessing time. FRT retry adds processing time
and memory use per worker. The 2025LEO comparison produced more false positives
with FRT retry.
[Registration workflow (Japanese)](star_registration_workflow_ja.md) · [FRT results and cost (Japanese)](frt_retry_workflow_ja.md)

<a id="verify-release-downloads"></a>

### Verify the Release Downloads

Save the wheel, `detect_meteors-2.1.1.tar.gz`, and `SHA256SUMS` in the
same directory with their original filenames. The listed checksums are for
the wheel and source distribution attached to the release.
[GitHub Release 2.1.1](https://github.com/shin3tky/detect_meteors/releases/tag/2.1.1) · [Download SHA256SUMS](https://github.com/shin3tky/detect_meteors/releases/download/2.1.1/SHA256SUMS)

On macOS, open a terminal in that directory and run:

```sh
shasum -a 256 -c SHA256SUMS
```

On Linux, use `sha256sum -c SHA256SUMS`. Both files should report `OK`.

On Windows, open PowerShell in the same directory and run:

```powershell
Get-Content -LiteralPath .\SHA256SUMS
Get-FileHash -Algorithm SHA256 -LiteralPath .\detect_meteors-2.1.1-py3-none-any.whl, .\detect_meteors-2.1.1.tar.gz
```

Compare each value in the `Hash` column with the checksum for the same filename
in `SHA256SUMS`. Uppercase and lowercase letters are equivalent.
If a checksum differs, download the affected file again and verify it before installing.

<a id="dependency-versions-v1613"></a>
<a id="dependency-versions-v211"></a>

## Dependency Versions (2.1.1)

The following runtime package versions are pinned in `pyproject.toml` and
recorded in `uv.lock`.

| Package | Pinned version |
| --- | --- |
| `numpy` | `2.5.5` |
| `opencv-python` | `5.0.0.93` |
| `rawpy` | `0.28.0` |
| `psutil` | `7.2.2` |
| `pillow` | `13.0.0` |
| `pydantic` | `2.14.4` |
| `pyyaml` | `6.0.4` |

Pydantic and PyYAML were updated in 2.1.1.
[Changelog](CHANGELOG.md#v211) · [Development tool versions](INSTALL_DEV.md#development-tool-versions)

## Upgrading to 2.1.1

After switching to the 2.1.1 source, run the command for your environment
from the project root.

```bash
uv sync --locked
# Development environment, including pre-commit and its dependencies:
uv sync --locked --all-extras
```

For a wheel installation, download and verify the 2.1.1 wheel. From the
directory containing your wheel installation's `.venv`, run the following
with the path to the downloaded wheel.
[Download the wheel](https://github.com/shin3tky/detect_meteors/releases/download/2.1.1/detect_meteors-2.1.1-py3-none-any.whl) · [Verify with SHA256SUMS](#verify-release-downloads)

```bash
uv pip install --upgrade ./detect_meteors-2.1.1-py3-none-any.whl
uv run detect-meteors --version
```

The wheel pins direct runtime dependencies. Transitive packages resolve
according to their package constraints. See the release notes for configuration
compatibility and how changes affect progress files.
[2.1.1 migration notes](RELEASE_NOTES_2.1.md#switching-configurations-and-upgrading)

<a id="upgrading-from-1612"></a>

For the earlier upgrade from 1.6.12 to 1.6.13, see the
[1.6.13 installation guide](https://github.com/shin3tky/detect_meteors/blob/1.6.13/docs/INSTALL.md#upgrading-from-1612).

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
git clone --branch 2.1.1 --single-branch https://github.com/shin3tky/detect_meteors.git
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
# Install the latest Python 3.14.x
uv python install 3.14

# Create virtual environment and install dependencies
uv sync --locked
```

This will:
- Download and install the latest Python 3.14.x patch release
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
git clone --branch 2.1.1 --single-branch https://github.com/shin3tky/detect_meteors.git
cd detect_meteors
```

### Step 4: Install System Dependencies

**Note**: On Windows, OpenCV will be installed via uv in the next steps. For LibRaw support, `rawpy` includes the necessary binaries, so no separate installation is required.

### Step 5: Set Up Python Environment

uv manages Python versions and virtual environments automatically:

```powershell
# Install the latest Python 3.14.x
uv python install 3.14

# Create virtual environment and install dependencies
uv sync --locked
```

This will:
- Download and install the latest Python 3.14.x patch release
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
