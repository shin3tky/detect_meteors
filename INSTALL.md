# Installation Guide

This guide provides detailed installation instructions for Detect Meteors CLI on macOS and Windows.

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
git clone https://github.com/shin3tky/detect_meteors.git
cd detect_meteors
```

### Step 4: Install uv

Install uv, a fast Python package and project manager:

```bash
brew install uv
```

### Step 5: Install System Dependencies

Install OpenCV and LibRaw:

```bash
brew install opencv libraw
```

### Step 6: Set Up Python Environment

uv manages Python versions and virtual environments automatically:

```bash
# Install the latest Python 3.12.x
uv python install 3.12

# Create virtual environment and install dependencies
uv sync
```

This will:
- Download and install the latest Python 3.12.x patch release
- Create a virtual environment in `.venv`
- Install all dependencies from `pyproject.toml`

### Step 7: Verification

Verify the installation:

```bash
uv run python detect_meteors_cli.py --help
```

You should see the help message with all available options.

**Note**: Use `uv run` to execute commands within the virtual environment, or activate it manually:
```bash
source .venv/bin/activate
python detect_meteors_cli.py --help
```

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
git clone https://github.com/shin3tky/detect_meteors.git
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
uv sync
```

This will:
- Download and install the latest Python 3.12.x patch release
- Create a virtual environment in `.venv`
- Install all dependencies from `pyproject.toml`

### Step 6: Verification

Verify the installation:

```powershell
uv run python detect_meteors_cli.py --help
```

You should see the help message with all available options.

**Note**: Use `uv run` to execute commands within the virtual environment, or activate it manually:
```powershell
# Allow script execution for this terminal session only
Set-ExecutionPolicy RemoteSigned -Scope Process

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

python detect_meteors_cli.py --help
```

---

## Common Issues

### macOS

- **Homebrew not found**: Ensure Homebrew is installed and `/usr/local/bin` (Intel) or `/opt/homebrew/bin` (Apple Silicon) is in your PATH.
- **uv not found**: After installation, restart your terminal or run `brew link uv`.

### Windows

- **Winget not available**: Ensure you're running Windows 10 (version 1809 or later) or Windows 11. Update Windows if necessary.
- **Git or uv not in PATH**: After installation, restart your terminal or add the installation directories to your PATH manually.
- **PowerShell execution policy**: Use `Set-ExecutionPolicy RemoteSigned -Scope Process` as mentioned in Step 6. This only affects the current terminal session and is the safest approach.

## Next Steps

After installation, proceed to the [README](README.md) for usage instructions and examples.
