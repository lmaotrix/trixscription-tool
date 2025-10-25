#!/usr/bin/env python
import argparse
import os
from faster_whisper import WhisperModel
import sounddevice as sd 
import numpy as np 
from datetime import datetime

def transcribe_file(model, file_path, output_dir):
    """transcribe a single audio file and save the result"""
    try:
        segments, info = model.transcribe(file_path, beam_size = 5, vad_filter = True)
        transcription = " ".join([segment.text for segment in segments])
        output_file = os.path.join(output_dir, f"{os.path.basename(file_path)}.txt")
        with open(output_file, "w") as f:
            f.write(f"Language: {info.language} (Probability: {info.language_probability:.2f})\n")
            f.write(f"transcription: {transcription}\n")
        print(f"transcribed {file_path} -> {output_file}")
        return transcription
    except Exception as e:
        print(f"Error transcribing {file_path}: {e}")
        return None

def transcribe_with_timestamps(model, file_path, output_dir):
    """transcribe single audio file with timestamps and save result"""
    try:
        segments, info = model.transcribe(file_path, beam_size = 5, vad_filter = True)
        timestamped_transcription = "\n".join([f"[{segment.start:.2f}-{segment.end:.2f}] {segment.text}" for segment in segments])
        output_file = os.path.join(output_dir, f"{os.path.basename(file_path)}_timestamped.txt")
        with open(output_file, "w") as f:
            f.write(f"Language: {info.language} (Probability: {info.language_probability:.2f})\n")
            f.write(f"Timestamped Transcription:\n{timestamped_transcription}\n")
            print(f"Transcribed with timestamps {file_path} -> {output_file}")
            return timestamped_transcription
    except Exception as e:
        print(f"Error transcribing with timestamps {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="CLI Audio Transcription Tool using Whisper")
    parser.add_argument("--file", help="Path to a single audio file (WAV, MP3, etc.)")
    parser.add_argument("--output-dir", default="outputs", help="Output directory for transcriptions")
    parser.add_argument("--timestamps", action="store_true", help="Include timestamps in transcription output")
    parser.add_argument("--model", default="small", choices=["tiny", "base", "small", "medium"], help="Whisper model size (default: base)")
    args = parser.parse_args()

    try:
        model = WhisperModel(args.model, device="cuda", compute_type="float16")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    if args.file:
        if args.timestamps:
            transcribe_with_timestamps(model, args.file, args.output_dir)
        else:
            transcribe_file(model, args.file, args.output_dir)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

