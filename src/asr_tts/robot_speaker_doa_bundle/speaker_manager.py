import os
import pickle
import uuid
import wave

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

try:
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:
    from speechbrain.pretrained import EncoderClassifier


class SpeakerManager:
    def __init__(self, db_path, threshold=0.3, model_dir=None):
        self.db_path = db_path
        self.threshold = float(threshold)
        self.model_dir = model_dir
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=model_dir,
        )
        self.db = self._load_db()

    def _load_db(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_db(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump(self.db, f)

    def _get_embedding(self, wav_path):
        waveform, _ = self._load_mono_16k(wav_path)
        waveform = waveform.unsqueeze(0).to(torch.float32)
        with torch.no_grad():
            emb = self.classifier.encode_batch(waveform)
        emb = emb.squeeze(0).cpu().numpy()
        emb = emb / np.linalg.norm(emb)
        return emb

    def _resample_waveform(self, waveform, src_sr, dst_sr):
        if src_sr == dst_sr:
            return waveform

        if waveform.numel() == 0:
            return waveform

        duration = waveform.shape[0] / float(src_sr)
        target_len = max(1, int(round(duration * float(dst_sr))))
        src_positions = np.linspace(0.0, waveform.shape[0] - 1, num=waveform.shape[0], dtype=np.float32)
        dst_positions = np.linspace(0.0, waveform.shape[0] - 1, num=target_len, dtype=np.float32)
        resampled = np.interp(dst_positions, src_positions, waveform.cpu().numpy()).astype(np.float32)
        return torch.from_numpy(resampled)

    def _load_mono_16k(self, wav_path):
        with wave.open(wav_path, "rb") as wf:
            channels = wf.getnchannels()
            sr = wf.getframerate()
            sample_width = wf.getsampwidth()
            num_frames = wf.getnframes()
            raw = wf.readframes(num_frames)

        if sample_width == 1:
            audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sample_width == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"unsupported wav sample width: {sample_width}")

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)

        waveform = torch.from_numpy(audio.astype(np.float32, copy=False))
        target_sr = 16000
        if sr != target_sr:
            waveform = self._resample_waveform(waveform, sr, target_sr)
            sr = target_sr
        return waveform.to(torch.float32), sr

    def detect_speech(self, wav_path, min_speech_seconds=0.8):
        waveform, sr = self._load_mono_16k(wav_path)
        x = waveform.cpu().numpy().astype(np.float32)
        if x.size == 0:
            return {
                "speech_like": False,
                "reason": "empty_audio",
                "rms": 0.0,
                "active_ratio": 0.0,
                "speech_seconds": 0.0,
                "band_ratio": 0.0,
            }

        frame_len = max(1, int(0.03 * sr))
        hop = max(1, int(0.01 * sr))
        if x.size < frame_len:
            pad = np.zeros(frame_len - x.size, dtype=np.float32)
            x = np.concatenate([x, pad])

        frames = []
        for start in range(0, x.size - frame_len + 1, hop):
            frames.append(x[start : start + frame_len])
        frames = np.stack(frames, axis=0)

        frame_rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
        noise_floor = float(np.percentile(frame_rms, 20))
        active_threshold = max(noise_floor * 2.5, 0.003)
        active_mask = frame_rms > active_threshold
        active_ratio = float(np.mean(active_mask))
        speech_seconds = float(np.sum(active_mask) * hop / sr)
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))

        spec = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(x.size, d=1.0 / sr)
        total_energy = float(np.sum(spec) + 1e-12)
        band_mask = (freqs >= 300.0) & (freqs <= 3400.0)
        band_ratio = float(np.sum(spec[band_mask]) / total_energy)

        speech_like = (
            rms >= 0.003
            and active_ratio >= 0.12
            and speech_seconds >= float(min_speech_seconds)
            and band_ratio >= 0.45
        )

        reason = None
        if not speech_like:
            if speech_seconds < float(min_speech_seconds):
                reason = "too_little_active_speech"
            elif band_ratio < 0.45:
                reason = "speech_band_ratio_too_low"
            elif rms < 0.003:
                reason = "audio_too_quiet"
            else:
                reason = "not_speech_like"

        return {
            "speech_like": bool(speech_like),
            "reason": reason,
            "rms": rms,
            "active_ratio": active_ratio,
            "speech_seconds": speech_seconds,
            "band_ratio": band_ratio,
            "active_threshold": active_threshold,
        }

    def _compute_centroids(self):
        centroids = {}
        for spk, embs in self.db.items():
            arr = np.vstack(embs)
            c = arr.mean(axis=0)
            c = c / np.linalg.norm(c)
            centroids[spk] = c
        return centroids

    def _new_speaker_id(self):
        return "spk_" + uuid.uuid4().hex[:8]

    def enroll(self, wav_path):
        emb = self._get_embedding(wav_path)
        if not self.db:
            new_id = self._new_speaker_id()
            self.db[new_id] = [emb]
            self._save_db()
            return {"action": "new", "speaker_id": new_id, "score": None}

        centroids = self._compute_centroids()
        ids = list(centroids.keys())
        mats = np.vstack([centroids[i] for i in ids])
        sims = cosine_similarity(mats, emb.reshape(1, -1)).reshape(-1)
        best_idx = np.argmax(sims)
        best_score = float(sims[best_idx])
        best_id = ids[best_idx]

        if best_score >= self.threshold:
            self.db[best_id].append(emb)
            self._save_db()
            return {"action": "assign", "speaker_id": best_id, "score": best_score}

        new_id = self._new_speaker_id()
        self.db[new_id] = [emb]
        self._save_db()
        return {"action": "new", "speaker_id": new_id, "score": best_score}

    def classify(self, wav_path):
        if not self.db:
            return {"error": "empty_db"}
        emb = self._get_embedding(wav_path)
        centroids = self._compute_centroids()
        ids = list(centroids.keys())
        mats = np.vstack([centroids[i] for i in ids])
        sims = cosine_similarity(mats, emb.reshape(1, -1)).reshape(-1)
        best_idx = np.argmax(sims)
        best_score = float(sims[best_idx])
        best_id = ids[best_idx]
        if best_score >= self.threshold:
            return {"speaker_id": best_id, "score": best_score}
        return {"speaker_id": None, "score": best_score}
