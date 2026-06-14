# Trixscription Tool: Local AI Audio Transcription CLI

# Overview

Trixscription is a command-line interface (CLI) tool for local audio transcription using AI, powered by OpenAI's Whisper model via the `faster-whisper` library. It runs entirely on your machine, leveraging your GPU for fast inference, without relying on cloud services like ChatGPT or Claude. This tool is designed for users who need accurate, multilingual speech-to-text capabilities for audio files, directories of files, or live microphone input.

Key advantages:

- **Privacy-Focused**: All processing is local; no data is sent to external servers.
- **Efficient**: Optimized for NVIDIA GPUs (e.g., RTX 3050 Laptop GPU) with CUDA acceleration.
- **Flexible**: Supports various audio formats (WAV, MP3, M4A) via FFmpeg, and multiple Whisper model sizes for trade-offs between speed and accuracy.
- **Batch Processing**: Transcribe entire directories or stream live audio.
- **Cross-Platform**: Works on Linux and Windows 11.

This project is ideal for developers, researchers, journalists, or anyone handling audio content who wants an offline solution.

## Features

- **Single File Transcription**: Transcribe individual audio files and save results as TXT with language detection and confidence scores.
- **Directory Transcription**: Batch process all supported audio files in a folder, saving individual TXT files for each.
- **Live Microphone Transcription**: Record from your microphone for a specified duration and transcribe in real-time, saving the result to a timestamped TXT file.
- **Model Selection**: Choose from Whisper models (`tiny`, `base`, `small`, `medium`) for different accuracy/speed levels.
- **GPU Acceleration**: Uses CUDA and cuDNN for fast inference on NVIDIA GPUs; falls back to CPU if GPU is unavailable.
- **Voice Activity Detection (VAD)**: Automatically skips silent parts for efficient processing.
- **Error Handling**: Robust handling for file not found, invalid formats, and model loading issues.
- **Customizable Output**: Saves transcriptions to a specified directory (default: `outputs`).
- **Multilingual Support**: Whisper handles over 100 languages with auto-detection.

## Hardware Requirements and Optimization

- **Recommended**:

  - GPU: NVIDIA with CUDA support (e.g., RTX 3050 Laptop GPU with 4GB VRAM).
  - CPU: Intel Core i7 12th Gen or equivalent.
  - RAM: 16GB+.
  - Storage: Enough space for model downloads (~145MB for "base" model).

- **Optimization Notes**:

  - The tool uses `float16` precision for GPU inference to reduce VRAM usage and speed up processing.
  - For limited VRAM (e.g., 4GB), use smaller models like "base" or "small". For larger models, switch to `int8` quantization in the code:

    ```python
    model = WhisperModel(args.model, device="cuda", compute_type="int8")
    ```
  - Expected performance: ~5-10x real-time transcription on RTX 3050 for the "base" model (e.g., 1 minute of audio in 6-12 seconds).

## Installation

### Quick install (recommended)

Use the platform install script from the project root. Each script installs FFmpeg (where possible), creates a Python virtual environment, and installs dependencies.

**Linux (Arch, Debian/Ubuntu, Fedora):**

```bash
chmod +x scripts/install-linux.sh
./scripts/install-linux.sh
source venv/bin/activate
```

**Windows 11 (PowerShell):**

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\install-windows.ps1
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks the script, run the `Set-ExecutionPolicy` line above first, or use:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install-windows.ps1
```

### Manual installation

#### System dependencies

| Dependency | Linux | Windows 11 |
|------------|-------|------------|
| **FFmpeg** | `sudo pacman -S ffmpeg` (Arch) or `sudo apt install ffmpeg` (Debian/Ubuntu) | `winget install Gyan.FFmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html) |
| **NVIDIA drivers** | `yay -S nvidia nvidia-utils` (Arch) | Install from [NVIDIA](https://www.nvidia.com/Download/index.aspx) or Windows Update |
| **cuDNN** | `sudo pacman -S cudnn` (Arch) | Bundled with PyTorch CUDA wheels; no separate install usually needed |
| **Python 3.11+** | `pacman -S python` or `apt install python3 python3-venv` | `winget install Python.Python.3.12` |

Verify FFmpeg and GPU:

```bash
# Linux / Windows (in Git Bash or PowerShell)
ffmpeg -version
nvidia-smi
```

#### Python dependencies

1. **Set up virtual environment**:

   **Linux:**

   ```bash
   cd ~/projects/trixscription-tool
   python -m venv venv
   source venv/bin/activate
   ```

   **Windows 11:**

   ```powershell
   cd C:\Users\you\projects\trixscription-tool
   py -3 -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Install packages**:

   ```bash
   pip install -r requirements.txt
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```

   For CPU-only systems (no NVIDIA GPU):

   ```bash
   pip install torch torchaudio
   ```

   Verify CUDA:

   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

   Should print `True` when an NVIDIA GPU and drivers are configured correctly.

## Usage

Run the tool with `python transcribe.py`. Use `--help` for options:

```
CLI Audio Transcription Tool using Whisper

options:
  --file PATH            Path to a single audio file (WAV, MP3, etc.)
  --dir PATH             Directory containing audio files to transcribe
  --output-dir PATH      Output directory for transcriptions (default: outputs)
  --live SECONDS         Transcribe live audio for specified seconds
  --timestamps           Include timestamps in transcription output
  --model {tiny,base,small,medium}
                         Whisper model size (default: small)
```

### Examples

1. **Transcribe a Single File**:

   ```bash
   python transcribe.py --file sample_audio.wav --model small --output-dir output
   ```

   - Saves to `output/sample_audio.wav.txt` with language and transcription.

2. **Transcribe a Directory**:

   ```bash
   python transcribe.py --dir audio_folder --model base
   ```

   - Processes all WAV/MP3/M4A files in `audio_folder`, saving TXT files in `outputs/`.

3. **Live Microphone Transcription**:

   ```bash
   python transcribe.py --live 10 --model base
   ```

   - Records for 10 seconds, transcribes, and saves to a timestamped TXT file (e.g., `live_transcription_20250927_120000.txt`).

4. **Help Menu**:

   ```bash
   python transcribe.py --help
   ```

## Troubleshooting

### Linux

- **cuDNN Errors**: Ensure cuDNN is installed for your CUDA version (e.g., 12.8). Run `sudo ldconfig` and reboot.
- **CUDA Not Available**: Check NVIDIA drivers with `nvidia-smi`. Reinstall `torch` with the correct index.
- **Live Mode Fails**: Check microphone settings with `pavucontrol` (install via `yay -S pavucontrol`).
- **Python Version Issues**: If using Python 3.13, switch to 3.11 for better compatibility: `yay -S python311`.

### Windows 11

- **CUDA Not Available**: Run `nvidia-smi` in PowerShell. Reinstall PyTorch with the CUDA index URL, or use CPU-only `pip install torch torchaudio`.
- **FFmpeg Not Found**: Ensure FFmpeg is in PATH. After `winget install Gyan.FFmpeg`, restart the terminal or add the install directory to PATH.
- **Script Execution Blocked**: Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before the install script.
- **Live Mode Fails**: Open **Settings → System → Sound** and confirm the default input device is set. Close apps that may lock the microphone.
- **Virtual Environment Activation**: Use `.\venv\Scripts\Activate.ps1` in PowerShell, or `venv\Scripts\activate.bat` in Command Prompt.

### All platforms

- **VRAM Out of Memory**: Use smaller models or `compute_type="int8"` in the code.
- **Audio Format Issues**: Ensure FFmpeg is in your PATH. Test with WAV files first.
- **Model Download Fails**: Ensure write permissions in the Whisper cache directory (`~/.cache/whisper` on Linux, `%USERPROFILE%\.cache\whisper` on Windows). Models download on first run.

If problems persist, check the GitHub issues. The app automatically falls back to CPU when CUDA is unavailable (slower but works without a GPU).

## Contributing

Contributions are welcome! Fork the repository, make changes, and submit a pull request. Please include:

- Bug fixes with tests.
- New features with documentation.
- Improvements to GPU optimization or model support.

## License

MIT License

Copyright (c) 2025 [lmaotrix]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
