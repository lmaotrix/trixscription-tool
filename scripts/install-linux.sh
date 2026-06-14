#!/usr/bin/env bash
# Install system dependencies and Python environment for Trixscription on Linux.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=== Trixscription Linux installer ==="

install_system_deps() {
    if command -v pacman >/dev/null 2>&1; then
        echo "Detected Arch-based system (pacman)."
        sudo pacman -S --needed ffmpeg python python-pip
        if pacman -Ss cudnn >/dev/null 2>&1; then
            read -r -p "Install cuDNN for NVIDIA GPU acceleration? [y/N] " reply
            if [[ "${reply,,}" == "y" ]]; then
                sudo pacman -S --needed cudnn
            fi
        fi
    elif command -v apt-get >/dev/null 2>&1; then
        echo "Detected Debian/Ubuntu system (apt)."
        sudo apt-get update
        sudo apt-get install -y ffmpeg python3 python3-pip python3-venv
    elif command -v dnf >/dev/null 2>&1; then
        echo "Detected Fedora/RHEL system (dnf)."
        sudo dnf install -y ffmpeg python3 python3-pip
    else
        echo "Could not detect a supported package manager."
        echo "Install FFmpeg and Python 3.11+ manually, then re-run this script."
        exit 1
    fi
}

find_python() {
    if command -v python3 >/dev/null 2>&1; then
        echo python3
    elif command -v python >/dev/null 2>&1; then
        echo python
    else
        echo ""
    fi
}

install_system_deps

PYTHON="$(find_python)"
if [[ -z "$PYTHON" ]]; then
    echo "Python 3 not found. Install Python 3.11+ and re-run."
    exit 1
fi

PY_VERSION="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "Using Python $PY_VERSION"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "FFmpeg not found in PATH after system install. Add it to PATH and re-run."
    exit 1
fi

echo "Creating virtual environment..."
"$PYTHON" -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate

pip install --upgrade pip

echo "Installing PyTorch with CUDA 12.8 support..."
if pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128; then
    echo "PyTorch (CUDA) installed."
else
    echo "CUDA PyTorch install failed; installing CPU-only PyTorch."
    pip install torch torchaudio
fi

pip install -r requirements.txt

echo
echo "=== Verification ==="
ffmpeg -version | head -n 1
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "from faster_whisper import WhisperModel; print('faster-whisper OK')"

echo
echo "Installation complete."
echo "Activate the environment with: source venv/bin/activate"
echo "Run transcription with: python transcribe.py --help"
