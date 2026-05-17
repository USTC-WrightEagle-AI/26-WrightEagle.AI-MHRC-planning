#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to play back debug audio saved by tts_node.py
Usage: python3 play_debug_audio.py [path_to_pkl_file]
"""

import pickle
import sys
import os

import sounddevice as sd
import numpy as np


def play_debug_audio(pkl_path=None):
    """Load and play debug audio from pickle file."""
    
    if pkl_path is None:
        # Default path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pkl_path = os.path.join(script_dir, "tts_debug_audio.pkl")
    
    if not os.path.exists(pkl_path):
        print(f"Error: File not found: {pkl_path}")
        print("Please run the TTS node first to generate the debug audio file.")
        sys.exit(1)
    
    # Load audio data
    print(f"Loading audio from: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    audio = data['audio']
    sample_rate = data['sample_rate']
    device = data['device']
    device_name = data['device_name']
    
    print(f"Audio info:")
    print(f"  - Sample rate: {sample_rate} Hz")
    print(f"  - Device ID: {device}")
    print(f"  - Device name: {device_name}")
    print(f"  - Audio samples: {len(audio)}")
    print(f"  - Duration: {len(audio) / sample_rate:.2f} seconds")
    print(f"  - Audio dtype: {audio.dtype}")
    print(f"  - Audio shape: {audio.shape if hasattr(audio, 'shape') else 'scalar'}")
    
    # Check volume levels
    audio_np = np.array(audio)
    max_amplitude = np.max(np.abs(audio_np))
    rms_amplitude = np.sqrt(np.mean(audio_np**2))
    print(f"  - Max amplitude: {max_amplitude:.6f}")
    print(f"  - RMS amplitude: {rms_amplitude:.6f}")
    print(f"  - Min value: {np.min(audio_np):.6f}")
    print(f"  - Max value: {np.max(audio_np):.6f}")
    
    if max_amplitude < 0.01:
        print("  WARNING: Audio amplitude is very low! This might be why you hear no sound.")
    elif max_amplitude > 1.0:
        print("  WARNING: Audio amplitude is > 1.0, might be clipped.")
    print()
    
    # List available devices
    print("Available audio devices:")
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        marker = " >>>" if i == device else "    "
        print(f"{marker} [{i}] {d['name']}")
    print()
    
    # Play audio
    print(f"Playing audio on device [{device}] {device_name}...")
    try:
        sd.play(audio, sample_rate, device=device)
        sd.wait()
        print("Playback finished successfully!")
    except Exception as e:
        print(f"Error playing audio: {e}")
        print()
        print("Trying with default device...")
        try:
            sd.play(audio, sample_rate)
            sd.wait()
            print("Playback finished with default device!")
        except Exception as e2:
            print(f"Error playing with default device: {e2}")
    
    print()
    print("=== Troubleshooting ===")
    print("If you still hear no sound, try:")
    print("1. Check volume: pactl set-sink-volume 0 100%")
    print("2. Check mute: pactl set-sink-mute 0 0")
    print("3. Test with paplay: paplay /home/nvidia/huyanshen/26-WrightEagle.AI-MHRC-planning/src/asr_tts/output.wav")
    print("4. Check sink info: pactl list sinks")


if __name__ == "__main__":
    pkl_path = sys.argv[1] if len(sys.argv) > 1 else None
    play_debug_audio(pkl_path)
