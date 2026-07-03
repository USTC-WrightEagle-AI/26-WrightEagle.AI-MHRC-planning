#!/usr/bin/env python3
"""Standalone microphone record/playback tester.

This script does not depend on ROS. It records from a selected input device,
saves a 16-bit mono WAV file, plays it back, and prints basic audio statistics
that help diagnose clipping, low volume, or silence.
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "recordings" / "manual_tests"
)


def parse_device(value):
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def print_devices():
    print("=== sounddevice devices ===")
    print(sd.query_devices())
    print()
    print(f"default input/output device: {sd.default.device}")


def dbfs(value):
    if value <= 0:
        return float("-inf")
    return 20.0 * math.log10(value)


def analyze_audio(audio, samplerate):
    mono = audio.reshape(-1)
    abs_audio = np.abs(mono)
    peak = float(abs_audio.max()) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
    active_threshold = max(0.01, peak * 0.1)
    active_mask = abs_audio >= active_threshold if mono.size else np.array([], dtype=bool)
    active_ratio = float(np.mean(active_mask)) if mono.size else 0.0
    if np.any(active_mask):
        active_audio = mono[active_mask]
        active_rms = float(np.sqrt(np.mean(np.square(active_audio))))
    else:
        active_rms = 0.0
    clipped_ratio = float(np.mean(abs_audio >= 0.999)) if mono.size else 0.0
    near_silence_ratio = float(np.mean(abs_audio < 0.005)) if mono.size else 0.0
    exact_zero_ratio = float(np.mean(abs_audio == 0.0)) if mono.size else 0.0
    frame_size = max(1, int(0.02 * samplerate))
    frame_count = len(mono) // frame_size
    if frame_count:
        framed = mono[: frame_count * frame_size].reshape(frame_count, frame_size)
        frame_rms = np.sqrt(np.mean(np.square(framed), axis=1))
        silent_frames = frame_rms < 0.001
        silent_frame_ratio = float(np.mean(silent_frames))
        max_silent_run = 0
        current_run = 0
        for is_silent in silent_frames:
            if is_silent:
                current_run += 1
                max_silent_run = max(max_silent_run, current_run)
            else:
                current_run = 0
        max_silent_run_ms = max_silent_run * 20
    else:
        silent_frame_ratio = 0.0
        max_silent_run_ms = 0
    dc_offset = float(np.mean(mono)) if mono.size else 0.0
    return {
        "peak": peak,
        "peak_dbfs": dbfs(peak),
        "rms": rms,
        "rms_dbfs": dbfs(rms),
        "active_threshold": active_threshold,
        "active_ratio": active_ratio,
        "active_rms": active_rms,
        "active_rms_dbfs": dbfs(active_rms),
        "clipped_ratio": clipped_ratio,
        "near_silence_ratio": near_silence_ratio,
        "exact_zero_ratio": exact_zero_ratio,
        "silent_frame_ratio": silent_frame_ratio,
        "max_silent_run_ms": max_silent_run_ms,
        "dc_offset": dc_offset,
    }


def print_stats(stats):
    print("=== audio stats ===")
    print(f"peak: {stats['peak']:.4f} ({stats['peak_dbfs']:.1f} dBFS)")
    print(f"rms : {stats['rms']:.4f} ({stats['rms_dbfs']:.1f} dBFS)")
    print(
        f"active rms: {stats['active_rms']:.4f} "
        f"({stats['active_rms_dbfs']:.1f} dBFS), "
        f"active samples: {stats['active_ratio'] * 100:.1f}%"
    )
    print(f"clipped/near-clipped samples: {stats['clipped_ratio'] * 100:.3f}%")
    print(f"near-silence samples       : {stats['near_silence_ratio'] * 100:.1f}%")
    print(f"exact-zero samples        : {stats['exact_zero_ratio'] * 100:.1f}%")
    print(
        f"silent 20ms frames        : {stats['silent_frame_ratio'] * 100:.1f}% "
        f"(longest {stats['max_silent_run_ms']} ms)"
    )
    print(f"dc offset                  : {stats['dc_offset']:.5f}")

    print("=== quick diagnosis ===")
    if stats["clipped_ratio"] > 0.001 or stats["peak"] >= 0.999:
        print("WARNING: input is clipping/overloaded. Lower mic/PulseAudio gain and disable AGC.")
    elif stats["peak_dbfs"] > -3.0 or stats["active_rms_dbfs"] > -14.0:
        print("WARNING: input level is very hot. Lower mic gain if playback sounds harsh.")
    elif stats["peak_dbfs"] < -18.0 or stats["active_rms_dbfs"] < -35.0:
        print("WARNING: input is too quiet or mostly silence. Check selected mic and gain.")
    elif stats["active_ratio"] < 0.05:
        print("WARNING: very little speech detected in this recording. Speak during most of the test.")
    elif stats["exact_zero_ratio"] > 0.1 or stats["max_silent_run_ms"] > 500:
        print("WARNING: recording has hard digital silence. If you spoke continuously, the input source is gating/droping audio.")
    else:
        print("OK: level looks usable from simple statistics.")


def save_wav(path, audio, samplerate):
    path.parent.mkdir(parents=True, exist_ok=True)
    mono = audio.reshape(-1)
    pcm = np.clip(mono, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(pcm16.tobytes())


def read_wav(path):
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        samplerate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio.reshape(-1, 1), samplerate


def play_with_system_player(wav_path):
    if wav_path is None:
        return False
    for player in ("paplay", "aplay"):
        exe = shutil.which(player)
        if exe:
            print(f"Using system player: {exe}")
            return subprocess.run([exe, str(wav_path)], check=False).returncode == 0
    return False


def play_audio(audio, samplerate, output_device, wav_path=None, playback_backend="paplay"):
    print("Playing audio...")
    if playback_backend in ("paplay", "auto") and output_device is None:
        if play_with_system_player(wav_path):
            return
        if playback_backend == "paplay":
            print("System player failed; falling back to sounddevice.")

    try:
        sd.play(audio.reshape(-1), samplerate=samplerate, device=output_device)
        sd.wait()
        return
    except Exception as exc:
        print(f"sounddevice playback failed: {exc}")

    if wav_path is None:
        raise RuntimeError("No wav file available for fallback playback")

    if play_with_system_player(wav_path):
        return
    raise RuntimeError("No playback method worked. Install/use paplay, aplay, or sounddevice output.")


def record_audio(duration, samplerate, channels, input_device):
    print(f"Recording {duration:.1f}s at {samplerate} Hz, channels={channels}, input={input_device or 'default'}")
    print("Start speaking after the countdown.")
    for number in (3, 2, 1):
        print(number)
        time.sleep(1.0)
    print("Recording...")
    frames = int(duration * samplerate)
    audio = sd.rec(
        frames,
        samplerate=samplerate,
        channels=channels,
        dtype="float32",
        device=input_device,
    )
    sd.wait()
    print("Recording finished.")
    if channels > 1:
        audio = audio.mean(axis=1, keepdims=True)
    return audio


def make_default_output_path(output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"mic_test_{timestamp}.wav"


def main():
    parser = argparse.ArgumentParser(description="Record, save, analyze, and play back microphone audio.")
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices and exit.")
    parser.add_argument("--duration", type=float, default=5.0, help="Recording duration in seconds.")
    parser.add_argument("--rate", type=int, default=16000, help="Sample rate. Use 16000 to match ASR.")
    parser.add_argument("--channels", type=int, default=1, help="Input channel count before mono conversion.")
    parser.add_argument("--input-device", type=parse_device, default=None, help="Input device index or name.")
    parser.add_argument("--output-device", type=parse_device, default=None, help="Output device index or name.")
    parser.add_argument("--output", type=Path, default=None, help="Output WAV path.")
    parser.add_argument("--no-play", action="store_true", help="Record and save only; do not play back.")
    parser.add_argument("--play-only", type=Path, default=None, help="Play and analyze an existing WAV file.")
    parser.add_argument(
        "--playback-backend",
        choices=("paplay", "sounddevice", "auto"),
        default="paplay",
        help="Playback backend. paplay avoids PortAudio playback underruns on this robot.",
    )
    args = parser.parse_args()

    if args.list_devices:
        print_devices()
        return 0

    try:
        print_devices()
        print()

        if args.play_only:
            wav_path = args.play_only.expanduser().resolve()
            audio, samplerate = read_wav(wav_path)
            print(f"Loaded: {wav_path}")
        else:
            wav_path = args.output
            if wav_path is None:
                wav_path = make_default_output_path(DEFAULT_OUTPUT_DIR)
            wav_path = wav_path.expanduser().resolve()
            audio = record_audio(args.duration, args.rate, args.channels, args.input_device)
            samplerate = args.rate
            save_wav(wav_path, audio, samplerate)
            print(f"Saved: {wav_path}")
            print(f"Size : {os.path.getsize(wav_path)} bytes")

        stats = analyze_audio(audio, samplerate)
        print_stats(stats)

        if not args.no_play:
            play_audio(audio, samplerate, args.output_device, wav_path, args.playback_backend)
            print("Playback finished.")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
