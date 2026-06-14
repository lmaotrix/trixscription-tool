#!/usr/bin/env python
import argparse
import os
import sys
import time
from datetime import datetime

# Avoid noisy Windows symlink warning from huggingface_hub on first model download.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}


def format_duration(seconds):
    if seconds is None:
        return "unknown length"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def get_audio_duration(file_path):
    try:
        import av

        with av.open(file_path) as container:
            stream = container.streams.audio[0]
            if stream.duration:
                return float(stream.duration * stream.time_base)
            if container.duration:
                return container.duration / 1_000_000.0
    except Exception:
        pass
    return None


def describe_input(file_path):
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    duration = get_audio_duration(file_path)
    return size_mb, duration


def load_whisper_model(model_name):
    """Load WhisperModel with CUDA when available, otherwise CPU."""
    print(f"Loading Whisper '{model_name}' model...", flush=True)
    print("First run downloads model files from Hugging Face; this can take a few minutes.", flush=True)

    started = time.time()
    try:
        import torch

        if torch.cuda.is_available():
            print("Using GPU (CUDA) for transcription.", flush=True)
            model = WhisperModel(model_name, device="cuda", compute_type="float16")
            print(f"Model ready in {time.time() - started:.1f}s.", flush=True)
            return model, "cuda"
    except ImportError:
        pass

    print("CUDA unavailable; using CPU (slower). Install PyTorch with CUDA for GPU acceleration.", flush=True)
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    print(f"Model ready in {time.time() - started:.1f}s.", flush=True)
    return model, "cpu"


def transcribe_segments(model, file_path, beam_size, quiet=False):
    """Run transcription and stream segment progress."""
    size_mb, duration = describe_input(file_path)
    duration_text = format_duration(duration)
    print(f"Transcribing {file_path}", flush=True)
    print(f"  File size: {size_mb:.1f} MB | Duration: {duration_text}", flush=True)
    if duration and duration > 600:
        print("  Long audio on CPU can take a while. Consider --model tiny for faster drafts.", flush=True)

    started = time.time()
    segments, info = model.transcribe(file_path, beam_size=beam_size, vad_filter=True)
    print(
        f"  Detected language: {info.language} (probability {info.language_probability:.2f})",
        flush=True,
    )
    print("  Processing segments...", flush=True)

    collected = []
    last_report = started
    for index, segment in enumerate(segments, start=1):
        collected.append(segment)
        now = time.time()
        if not quiet and (index == 1 or index % 10 == 0 or now - last_report >= 15):
            elapsed = now - started
            print(
                f"  ... segment {index} | audio ~{segment.end:.0f}s | elapsed {elapsed:.0f}s",
                flush=True,
            )
            last_report = now

    print(f"  Finished {len(collected)} segments in {time.time() - started:.1f}s.", flush=True)
    return collected, info


def transcribe_file(model, file_path, output_dir, beam_size, quiet=False):
    """Transcribe a single audio file and save the result."""
    try:
        segments, info = transcribe_segments(model, file_path, beam_size, quiet=quiet)
        transcription = " ".join(segment.text for segment in segments)
        output_file = os.path.join(output_dir, f"{os.path.basename(file_path)}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Language: {info.language} (Probability: {info.language_probability:.2f})\n")
            f.write(f"transcription: {transcription}\n")
        print(f"Saved transcription -> {output_file}", flush=True)
        return transcription
    except Exception as e:
        print(f"Error transcribing {file_path}: {e}", flush=True)
        return None


def transcribe_with_timestamps(model, file_path, output_dir, beam_size, quiet=False):
    """Transcribe a single audio file with timestamps and save the result."""
    try:
        segments, info = transcribe_segments(model, file_path, beam_size, quiet=quiet)
        timestamped_transcription = "\n".join(
            f"[{segment.start:.2f}-{segment.end:.2f}] {segment.text}" for segment in segments
        )
        output_file = os.path.join(output_dir, f"{os.path.basename(file_path)}_timestamped.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Language: {info.language} (Probability: {info.language_probability:.2f})\n")
            f.write(f"Timestamped Transcription:\n{timestamped_transcription}\n")
        print(f"Saved timestamped transcription -> {output_file}", flush=True)
        return timestamped_transcription
    except Exception as e:
        print(f"Error transcribing with timestamps {file_path}: {e}", flush=True)
        return None


def transcribe_directory(model, dir_path, output_dir, timestamps=False, beam_size=5, quiet=False):
    """Transcribe all supported audio files in a directory."""
    files = sorted(
        os.path.join(dir_path, name)
        for name in os.listdir(dir_path)
        if os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        print(f"No supported audio files found in {dir_path}", flush=True)
        return

    for index, file_path in enumerate(files, start=1):
        print(f"\n[{index}/{len(files)}] {file_path}", flush=True)
        if timestamps:
            transcribe_with_timestamps(model, file_path, output_dir, beam_size, quiet=quiet)
        else:
            transcribe_file(model, file_path, output_dir, beam_size, quiet=quiet)


def transcribe_live(model, duration_seconds, output_dir, beam_size):
    """Record from the default microphone and transcribe."""
    sample_rate = 16000
    print(f"Recording for {duration_seconds} seconds...", flush=True)
    recording = sd.rec(
        int(duration_seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("Recording complete. Transcribing...", flush=True)

    temp_wav = os.path.join(output_dir, "_live_recording.wav")
    try:
        import scipy.io.wavfile as wavfile

        wavfile.write(temp_wav, sample_rate, (recording * 32767).astype(np.int16))
    except ImportError:
        import wave

        with wave.open(temp_wav, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes((recording * 32767).astype(np.int16).tobytes())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"live_transcription_{timestamp}.txt")
    try:
        segments, info = transcribe_segments(model, temp_wav, beam_size)
        transcription = " ".join(segment.text for segment in segments)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Language: {info.language} (Probability: {info.language_probability:.2f})\n")
            f.write(f"transcription: {transcription}\n")
        print(f"Live transcription saved to {output_file}", flush=True)
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)


def validate_file_arg(file_path):
    if not os.path.isfile(file_path):
        print(f"Error: file not found: {file_path}", flush=True)
        sys.exit(1)
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        print(f"Warning: {ext} is not a commonly tested format. FFmpeg must be installed.", flush=True)


def validate_dir_arg(dir_path):
    if not os.path.isdir(dir_path):
        print(f"Error: directory not found: {dir_path}", flush=True)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="CLI Audio Transcription Tool using Whisper")
    parser.add_argument("--file", help="Path to a single audio file (WAV, MP3, etc.)")
    parser.add_argument("--dir", help="Directory containing audio files to transcribe")
    parser.add_argument("--output-dir", default="outputs", help="Output directory for transcriptions")
    parser.add_argument("--timestamps", action="store_true", help="Include timestamps in transcription output")
    parser.add_argument("--live", type=int, metavar="SECONDS", help="Transcribe live audio for specified seconds")
    parser.add_argument(
        "--model",
        default="small",
        choices=["tiny", "base", "small", "medium"],
        help="Whisper model size (default: small)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=None,
        help="Beam search size (default: 5 on GPU, 1 on CPU)",
    )
    parser.add_argument("--quiet", action="store_true", help="Hide segment progress output")
    args = parser.parse_args()

    if not any([args.file, args.dir, args.live]):
        parser.print_help()
        sys.exit(1)

    if args.file:
        validate_file_arg(args.file)
    if args.dir:
        validate_dir_arg(args.dir)

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        model, device = load_whisper_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}", flush=True)
        sys.exit(1)

    beam_size = args.beam_size if args.beam_size is not None else (5 if device == "cuda" else 1)
    if args.beam_size is None and device == "cpu":
        print("Using beam_size=1 on CPU for faster transcription.", flush=True)

    if args.live:
        transcribe_live(model, args.live, args.output_dir, beam_size)
    elif args.dir:
        transcribe_directory(
            model,
            args.dir,
            args.output_dir,
            timestamps=args.timestamps,
            beam_size=beam_size,
            quiet=args.quiet,
        )
    elif args.file:
        if args.timestamps:
            transcribe_with_timestamps(
                model,
                args.file,
                args.output_dir,
                beam_size,
                quiet=args.quiet,
            )
        else:
            transcribe_file(model, args.file, args.output_dir, beam_size, quiet=args.quiet)


if __name__ == "__main__":
    main()
