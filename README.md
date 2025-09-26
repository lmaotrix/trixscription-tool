# Trixscription Tool: Local AI Audio Transcription CLI

# Overview

Trixscription is a command-line interface (CLI) tool for local audio transcription using AI, powered by OpenAI's Whisper model via the `faster-whisper` library. It runs entirely on your machine, leveraging your GPU for fast inference, without relying on cloud services like ChatGPT or Claude. This tool is designed for users who need accurate, multilingual speech-to-text capabilities for audio files, directories of files, or live microphone input.

Key advantages:

- **Privacy-Focused**: All processing is local; no data is sent to external servers.
- **Efficient**: Optimized for NVIDIA GPUs (e.g., RTX 3050 Laptop GPU) with CUDA acceleration.
- **Flexible**: Supports various audio formats (WAV, MP3, M4A) via FFmpeg, and multiple Whisper model sizes for trade-offs between speed and accuracy.
- **Batch Processing**: Transcribe entire directories or stream live audio.

This project is ideal for developers, researchers, journalists, or anyone handling audio content who wants an offline solution.

## Features

- **Single File Transcription**: Transcribe individual audio files and save results as TXT with language detection and confidence scores.
- **Directory Transcription**: Batch process all supported audio files in a folder, saving individual TXT files for each.
- **Live Microphone Transcription**: Record from your microphone for a specified duration and transcribe in real-time, saving the result to a timestamped TXT file.
- **Model Selection**: Choose from Whisper models (`tiny`, `base`, `small`, `medium`) for different accuracy/speed levels.
- **GPU Acceleration**: Uses CUDA and cuDNN for fast inference on NVIDIA GPUs; falls back to CPU if GPU is unavailable.
- **Voice Activity Detection (VAD)**: Automatically skips silent parts for efficient processing.
- **Error Handling**: Robust handling for file not found, invalid formats, and model loading issues.
- **Customizable Output**: Saves transcriptions to a specified directory (default: `transcriptions`).
- **Multilingual Support**: Whisper handles over 100 languages with auto-detection.

## Hardware Requirements and Optimization

- **Recommended**:

  - GPU: NVIDIA with CUDA support (e.g., RTX 3050 Laptop GPU with 4GB VRAM).
  - CPU: Intel Core i7 12th Gen or equivalent.
  - RAM: 16GB+.
  - Storage: Enough space for model downloads (\~145MB for "base" model).

- **Optimization Notes**:

  - The tool uses `float16` precision for GPU inference to reduce VRAM usage and speed up processing.
  - For limited VRAM (e.g., 4GB), use smaller models like "base" or "small". For larger models, switch to `int8` quantization in the code:

    ```python
    model = WhisperModel(args.model, device="cuda", compute_type="int8")
    ```
  - Expected performance: \~5-10x real-time transcription on RTX 3050 for the "base" model (e.g., 1 minute of audio in 6-12 seconds).

## Installation

### System Dependencies

1. **FFmpeg**: For audio format handling.

   ```bash
   yay -S ffmpeg
   ```

2. **cuDNN**: For GPU acceleration (NVIDIA only).

   ```bash
   sudo pacman -S cudnn
   ```

   Verify:

   ```bash
   ls /usr/lib | grep cudnn
   ```

3. **NVIDIA Drivers** (if using GPU):

   ```bash
   yay -S nvidia nvidia-utils
   ```

   Verify:

   ```bash
   nvidia-smi
   ```

### Python Dependencies

1. **Set Up Virtual Environment**:

   ```bash
   cd ~/projects/trixscription-tool
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install Packages**:

   ```bash
   pip install -r requirements.txt
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```

   - `requirements.txt` contents:

     ```
     faster-whisper>=1.0.3
     torch>=2.5.1
     torchaudio>=2.5.1
     sounddevice>=0.5.2
     numpy>=2.0.0
     ```

   Verify CUDA:

   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

   Should print `True`.

## Usage

Run the tool with `python transcribe.py`. Use `--help` for options:

```
CLI Audio Transcription Tool using Whisper

options:
  --file PATH            Path to a single audio file (WAV, MP3, etc.)
  --dir PATH             Directory containing audio files to transcribe
  --output-dir PATH      Output directory for transcriptions (default: transcriptions)
  --live SECONDS         Transcribe live audio for specified seconds
  --model {tiny,base,small,medium}
                         Whisper model size (default: base)
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

   - Processes all WAV/MP3/M4A files in `audio_folder`, saving TXT files in `transcriptions/`.

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

- **cuDNN Errors**: Ensure cuDNN is installed for your CUDA version (e.g., 12.8). Run `sudo ldconfig` and reboot.
- **CUDA Not Available**: Check NVIDIA drivers with `nvidia-smi`. Reinstall `torch` with the correct index.
- **VRAM Out of Memory**: Use smaller models or `compute_type="int8"` in the code.
- **Audio Format Issues**: Ensure FFmpeg is in your PATH (`which ffmpeg`). Test with WAV files first.
- **Live Mode Fails**: Check microphone settings with `pavucontrol` (install via `yay -S pavucontrol`).
- **Model Download Fails**: Ensure write permissions in `~/.cache/whisper`. Models download on first run.
- **Python Version Issues**: If using Python 3.13, switch to 3.11 for better compatibility: `yay -S python311`.

If problems persist, check the GitHub issues or run with CPU fallback: change `device="cuda"` to `device="cpu"` in the code (slower).

## Contributing

Contributions are welcome! Fork the repository, make changes, and submit a pull request. Please include:

- Bug fixes with tests.
- New features with documentation.
- Improvements to GPU optimization or model support.

## License

MIT License

Copyright (c) 2025 \[Your Name or Username\]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
