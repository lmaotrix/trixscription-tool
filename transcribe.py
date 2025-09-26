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

if __name__ == "__main__":
    model = WhisperModel("base", device="cuda", compute_type="float16")
    transcribe_file(model, "harvard.wav", "outputs")
