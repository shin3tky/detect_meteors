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

### Step 4: Install Python

Install Python 3.12 using Homebrew:

```bash
brew install python@3.12
```

### Step 5: Install System Dependencies

Install OpenCV and LibRaw:

```bash
brew install opencv libraw
```

### Step 6: Create Virtual Environment

Create and activate a Python virtual environment:

```bash
# Create virtual environment
$(brew --prefix python@3.12)/bin/python3.12 -m venv venv

# Activate virtual environment
source ./venv/bin/activate
```

**Note**: This command works on both Intel Macs (`/usr/local`) and Apple Silicon Macs (`/opt/homebrew`). On subsequent uses, you only need to activate the environment:
```bash
source ./venv/bin/activate
```

### Step 7: Install Python Dependencies

Install required Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

This will install:
- `numpy` - Numerical computing
- `matplotlib` - Plotting and visualization
- `opencv-python` - Image processing
- `rawpy` - RAW image reading
- `psutil` - System utilities
- `pillow` - Image handling and EXIF extraction
- `pydantic` - Configuration and data validation

### Step 8: Verification

Verify the installation:

```bash
python detect_meteors_cli.py --help
```

You should see the help message with all available options.

---

## Windows Installation

### Step 1: Install Git

Open PowerShell or Command Prompt and install Git using Winget:

```powershell
winget install --id Git.Git -e --source winget
```

After installation, restart your terminal or open a new one to ensure Git is in your PATH.

### Step 2: Install Python

Install Python 3.12 using Winget:

```powershell
winget install --id Python.Python.3.12 -e --source winget
```

After installation, restart your terminal or open a new one to ensure Python is in your PATH.

### Step 3: Clone the Repository

```powershell
# Clone the repository
git clone https://github.com/shin3tky/detect_meteors.git
cd detect_meteors
```

### Step 4: Install System Dependencies

Install OpenCV and LibRaw:

**Note**: On Windows, OpenCV will be installed via pip in the next steps. For LibRaw support, `rawpy` includes the necessary binaries, so no separate installation is required.

### Step 5: Create Virtual Environment

Create and activate a Python virtual environment:

```powershell
# Create virtual environment
python -m venv venv

# Allow script execution for this terminal session only
Set-ExecutionPolicy RemoteSigned -Scope Process

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

**Note**: The `Set-ExecutionPolicy` command with `-Scope Process` only affects the current terminal session and doesn't change your system-wide policy. This is the safest approach.

On subsequent uses, you need to run both commands:
```powershell
Set-ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\Activate.ps1
```

### Step 6: Install Python Dependencies

Install required Python packages from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

This will install:
- `numpy` - Numerical computing
- `matplotlib` - Plotting and visualization
- `opencv-python` - Image processing
- `rawpy` - RAW image reading
- `psutil` - System utilities
- `pillow` - Image handling and EXIF extraction

### Step 7: Verification

Verify the installation:

```powershell
python detect_meteors_cli.py --help
```

You should see the help message with all available options.

---

## Common Issues

### macOS

- **Homebrew not found**: Ensure Homebrew is installed and `/usr/local/bin` (Intel) or `/opt/homebrew/bin` (Apple Silicon) is in your PATH.
- **Python version issues**: If `python3.12` is not found, check the Homebrew installation path with `brew --prefix python@3.12`.

### Windows

- **Winget not available**: Ensure you're running Windows 10 (version 1809 or later) or Windows 11. Update Windows if necessary.
- **Git or Python not in PATH**: After installation, restart your terminal or add the installation directories to your PATH manually.
- **PowerShell execution policy**: Use `Set-ExecutionPolicy RemoteSigned -Scope Process` as mentioned in Step 5. This only affects the current terminal session and is the safest approach.

## Next Steps

After installation, proceed to the [README](README.md) for usage instructions and examples.
