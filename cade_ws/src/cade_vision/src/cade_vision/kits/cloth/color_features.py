"""Shared color feature extraction for training and runtime inference."""

import cv2
import numpy as np


FEATURE_VERSION = "lab_hsv_v2"
LAB_HIST_BINS = 16
AB_JOINT_BINS = 8
H_HIST_BINS = 18
FEATURE_STAT_NAMES = ("mean", "std", "median", "p05", "p25", "p75", "p95")
MAX_FEATURE_PIXELS = 20000


def feature_names():
    names = []
    for channel in ("L", "a", "b"):
        for bin_index in range(LAB_HIST_BINS):
            names.append(f"{channel}_hist_{bin_index:02d}")

    for a_bin in range(AB_JOINT_BINS):
        for b_bin in range(AB_JOINT_BINS):
            names.append(f"ab_hist_{a_bin:02d}_{b_bin:02d}")

    for channel in ("L", "a", "b"):
        for stat in FEATURE_STAT_NAMES:
            names.append(f"{channel}_{stat}")

    for channel in ("S", "V"):
        for stat in FEATURE_STAT_NAMES:
            names.append(f"{channel}_{stat}")

    for bin_index in range(H_HIST_BINS):
        names.append(f"H_hist_sat_{bin_index:02d}")

    for stat in FEATURE_STAT_NAMES:
        names.append(f"lab_chroma_{stat}")

    names.extend(
        [
            "ratio_lab_L_le_60",
            "ratio_lab_L_le_90",
            "ratio_lab_L_ge_200",
            "ratio_hsv_S_le_40",
            "ratio_hsv_S_ge_80",
            "ratio_dark_low_sat",
            "ratio_dark_high_sat",
            "ratio_lab_chroma_le_10",
            "ratio_lab_chroma_ge_30",
        ]
    )
    return names


FEATURE_NAMES = tuple(feature_names())
FEATURE_DIM = len(FEATURE_NAMES)


def _sample_pixels(bgr_pixels):
    if len(bgr_pixels) <= MAX_FEATURE_PIXELS:
        return bgr_pixels
    indices = np.linspace(
        0,
        len(bgr_pixels) - 1,
        MAX_FEATURE_PIXELS,
        dtype=np.int32,
    )
    return bgr_pixels[indices]


def _normalized_hist(values, bins, value_range):
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    hist = hist.astype(np.float32)
    hist /= max(1.0, float(hist.sum()))
    return hist


def _stats(values):
    values = values.astype(np.float32)
    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.median(values)),
        float(np.percentile(values, 5)),
        float(np.percentile(values, 25)),
        float(np.percentile(values, 75)),
        float(np.percentile(values, 95)),
    ]


def extract_color_features(bgr_values, return_debug=False):
    bgr_pixels = np.asarray(bgr_values, dtype=np.uint8).reshape(-1, 3)
    if len(bgr_pixels) == 0:
        raise ValueError("empty pixel set")

    total_pixels = int(len(bgr_pixels))
    sampled_pixels = _sample_pixels(bgr_pixels)
    lab = cv2.cvtColor(sampled_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    hsv = cv2.cvtColor(sampled_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)

    features = []
    for channel_index in range(3):
        features.extend(
            _normalized_hist(lab[:, channel_index], LAB_HIST_BINS, (0, 256)).tolist()
        )

    ab_hist, _, _ = np.histogram2d(
        lab[:, 1],
        lab[:, 2],
        bins=(AB_JOINT_BINS, AB_JOINT_BINS),
        range=((0, 256), (0, 256)),
    )
    ab_hist = ab_hist.astype(np.float32)
    ab_hist /= max(1.0, float(ab_hist.sum()))
    features.extend(ab_hist.reshape(-1).tolist())

    for channel_index in range(3):
        features.extend(_stats(lab[:, channel_index]))

    for channel_index in (1, 2):
        features.extend(_stats(hsv[:, channel_index]))

    saturated = hsv[:, 1] >= 30
    if np.any(saturated):
        h_hist = _normalized_hist(hsv[saturated, 0], H_HIST_BINS, (0, 180))
    else:
        h_hist = np.zeros(H_HIST_BINS, dtype=np.float32)
    features.extend(h_hist.tolist())

    lab_float = lab.astype(np.float32)
    chroma = np.hypot(lab_float[:, 1] - 128.0, lab_float[:, 2] - 128.0)
    features.extend(_stats(chroma))

    l_values = lab[:, 0]
    s_values = hsv[:, 1]
    features.extend(
        [
            float(np.mean(l_values <= 60)),
            float(np.mean(l_values <= 90)),
            float(np.mean(l_values >= 200)),
            float(np.mean(s_values <= 40)),
            float(np.mean(s_values >= 80)),
            float(np.mean((l_values <= 90) & (s_values <= 50))),
            float(np.mean((l_values <= 90) & (s_values > 50))),
            float(np.mean(chroma <= 10.0)),
            float(np.mean(chroma >= 30.0)),
        ]
    )

    features = np.asarray(features, dtype=np.float32)
    if features.shape[0] != FEATURE_DIM:
        raise ValueError(f"unexpected feature length: {features.shape[0]}")
    if not np.isfinite(features).all():
        raise ValueError("feature contains NaN/Inf")

    if not return_debug:
        return features
    return features, {
        "feature_version": FEATURE_VERSION,
        "feature_dim": FEATURE_DIM,
        "total_pixels": total_pixels,
        "sampled_pixels": int(len(sampled_pixels)),
        "low_light_ratio": float(features[FEATURE_NAMES.index("ratio_lab_L_le_90")]),
        "low_saturation_ratio": float(features[FEATURE_NAMES.index("ratio_hsv_S_le_40")]),
        "dark_low_saturation_ratio": float(features[FEATURE_NAMES.index("ratio_dark_low_sat")]),
        "high_chroma_ratio": float(features[FEATURE_NAMES.index("ratio_lab_chroma_ge_30")]),
    }
