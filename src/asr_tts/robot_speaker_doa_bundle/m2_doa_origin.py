import math
import os

import numpy as np


SPEED_OF_SOUND = 343.0


def load_interleaved_pcm(path, channels=8, dtype=np.int16):
    data = np.fromfile(path, dtype=dtype)
    usable = (len(data) // channels) * channels
    if usable == 0:
        raise ValueError("pcm file is empty or shorter than one frame")
    if usable != len(data):
        data = data[:usable]
    return data.reshape(-1, channels)


def circle_geometry(num_mics=6, radius_m=0.035, angle_offset_deg=0.0):
    pts = []
    for i in range(num_mics):
        theta = math.radians(angle_offset_deg + i * 360.0 / num_mics)
        pts.append([radius_m * math.cos(theta), radius_m * math.sin(theta)])
    return np.asarray(pts, dtype=np.float64)


def line_geometry(num_mics=6, spacing_m=0.028):
    center = (num_mics - 1) / 2.0
    pts = []
    for i in range(num_mics):
        pts.append([(i - center) * spacing_m, 0.0])
    return np.asarray(pts, dtype=np.float64)


def parse_channel_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def build_geometry(mode, num_mics, radius_m, spacing_m, angle_offset_deg):
    if mode == "circle6":
        return circle_geometry(num_mics=num_mics, radius_m=radius_m, angle_offset_deg=angle_offset_deg)
    if mode == "line6":
        return line_geometry(num_mics=num_mics, spacing_m=spacing_m)
    raise ValueError(f"unsupported geometry mode: {mode}")


def select_loudest_segment(samples, sample_rate, seconds=0.5):
    seg_len = max(1, int(round(seconds * sample_rate)))
    if len(samples) <= seg_len:
        return samples
    mono = np.mean(np.abs(samples.astype(np.float64)), axis=1)
    energy = np.convolve(mono, np.ones(seg_len, dtype=np.float64), mode="valid")
    start = int(np.argmax(energy))
    return samples[start : start + seg_len]


def gcc_phat(sig, refsig, fs, interp=16, max_tau=None):
    n = sig.shape[0] + refsig.shape[0]
    nfft = 1
    while nfft < n:
        nfft <<= 1

    sig_f = np.fft.rfft(sig, n=nfft)
    ref_f = np.fft.rfft(refsig, n=nfft)
    cross = sig_f * np.conj(ref_f)
    denom = np.abs(cross)
    denom[denom < 1e-12] = 1e-12
    cross /= denom

    cc = np.fft.irfft(cross, n=nfft * interp)
    max_shift = int(interp * nfft / 2)
    if max_tau is not None:
        max_shift = min(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    shifts = np.arange(-max_shift, max_shift + 1, dtype=np.float64) / float(interp * fs)
    return shifts, cc


def sample_correlation_at_delay(delays, corr, target_delay):
    if target_delay <= delays[0]:
        return float(corr[0])
    if target_delay >= delays[-1]:
        return float(corr[-1])
    return float(np.interp(target_delay, delays, corr))


def estimate_doa(samples, sample_rate, geometry_xy, angle_grid_deg, interp=16):
    num_mics = geometry_xy.shape[0]
    if samples.shape[1] != num_mics:
        raise ValueError(f"samples channels ({samples.shape[1]}) != geometry microphones ({num_mics})")

    signals = samples.astype(np.float64)
    signals -= np.mean(signals, axis=0, keepdims=True)
    pair_indices = [(i, j) for i in range(num_mics) for j in range(i + 1, num_mics)]

    pair_corr = {}
    for i, j in pair_indices:
        baseline = geometry_xy[i] - geometry_xy[j]
        max_tau = np.linalg.norm(baseline) / SPEED_OF_SOUND
        delays, corr = gcc_phat(signals[:, i], signals[:, j], sample_rate, interp=interp, max_tau=max_tau)
        pair_corr[(i, j)] = (delays, corr)

    scores = []
    for deg in angle_grid_deg:
        theta = math.radians(deg)
        unit = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float64)
        score = 0.0
        for i, j in pair_indices:
            baseline = geometry_xy[i] - geometry_xy[j]
            tau = float(np.dot(baseline, unit) / SPEED_OF_SOUND)
            delays, corr = pair_corr[(i, j)]
            score += sample_correlation_at_delay(delays, corr, tau)
        scores.append(score)

    scores = np.asarray(scores, dtype=np.float64)
    best_idx = int(np.argmax(scores))
    return {"angle_deg": float(angle_grid_deg[best_idx]), "score": float(scores[best_idx]), "scores": scores}


def estimate_doa_from_pcm(
    pcm_path,
    sample_rate=16000,
    total_channels=8,
    active_channels="0,1,2,3,4,5",
    geometry="circle6",
    num_mics=6,
    radius_m=0.035,
    spacing_m=0.028,
    angle_offset_deg=0.0,
    segment_seconds=0.5,
    angle_step_deg=2.0,
):
    active = parse_channel_list(active_channels)
    if len(active) != int(num_mics):
        raise ValueError("active_channels count must match num_mics")

    raw = load_interleaved_pcm(pcm_path, channels=int(total_channels))
    picked = raw[:, active]
    segment = select_loudest_segment(picked, int(sample_rate), seconds=float(segment_seconds))
    geometry_xy = build_geometry(
        mode=geometry,
        num_mics=int(num_mics),
        radius_m=float(radius_m),
        spacing_m=float(spacing_m),
        angle_offset_deg=float(angle_offset_deg),
    )
    angle_grid = np.arange(0.0, 360.0, float(angle_step_deg), dtype=np.float64)
    result = estimate_doa(segment, int(sample_rate), geometry_xy, angle_grid)
    top_idx = np.argsort(result["scores"])[-5:][::-1]
    top_candidates = [{"angle_deg": float(angle_grid[i]), "score": float(result["scores"][i])} for i in top_idx]
    return {
        "pcm_path": os.path.abspath(pcm_path),
        "estimated_angle_deg": result["angle_deg"],
        "score": result["score"],
        "top_candidates": top_candidates,
    }
