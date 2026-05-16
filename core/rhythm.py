import math
import os
import time
from collections import deque

import numpy as np
import torch

from core.rhythm_mamba import RhythmMamba


class RhythmRuntime:
    def __init__(
        self,
        model_path="weights/rhythm_mamba.pth",
        mock_data_path="core/data/rhythm_mock.npz",
        num_classes=12,
        eps=1e-7,
    ):
        self.num_classes = num_classes
        self.eps = eps
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.seq_len = 96
        self.history = deque(maxlen=self.seq_len)
        self.recent_surprises = deque(maxlen=256)

        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=self.device)
            config = ckpt.get("config", {})
            self.seq_len = config.get("seq_len", 96)
            self.history = deque(maxlen=self.seq_len)
            self.model = RhythmMamba(
                input_dim=config.get("input_dim", 14),
                hidden_dim=config.get("hidden_dim", 64),
                num_classes=config.get("num_classes", 12),
            ).to(self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()

        self._seed_history(mock_data_path)

    def _seed_history(self, mock_data_path):
        if os.path.exists(mock_data_path):
            data = np.load(mock_data_path, allow_pickle=True)
            features = np.concatenate([data["probs"], data["time_features"]], axis=1).astype(np.float32)
            for row in features[-self.seq_len:]:
                self.history.append(row)
        while len(self.history) < self.seq_len:
            self.history.append(np.r_[np.ones(self.num_classes) / self.num_classes, [0.0, 1.0]].astype(np.float32))

    def _time_features(self, timestamp=None):
        now = timestamp or time.time()
        local = time.localtime(now)
        minutes = local.tm_hour * 60 + local.tm_min
        angle = 2.0 * math.pi * minutes / (24 * 60)
        return np.array([math.sin(angle), math.cos(angle)], dtype=np.float32), local.tm_hour

    def _normalize(self, values):
        arr = np.asarray(values, dtype=np.float32)
        arr = np.clip(arr, self.eps, 1.0)
        return arr / arr.sum()

    def _fallback_prior(self):
        hist = np.stack(list(self.history), axis=0)[:, :self.num_classes]
        weights = np.linspace(0.25, 1.0, len(hist), dtype=np.float32)
        prior = (hist * weights[:, None]).sum(axis=0) / weights.sum()
        return self._normalize(prior)

    def observe(self, probs, timestamp=None):
        probs = self._normalize(probs)
        time_feats, hour = self._time_features(timestamp)
        window = np.stack(list(self.history), axis=0).astype(np.float32)

        if self.model is not None:
            with torch.no_grad():
                tensor = torch.from_numpy(window).unsqueeze(0).to(self.device)
                prior = self.model.predict_prior(tensor).cpu().numpy()[0]
            prior = self._normalize(prior)
        else:
            prior = self._fallback_prior()

        kl_raw = float(np.sum(probs * np.log(probs / np.clip(prior, self.eps, 1.0))))
        surprise = float(np.log1p(max(0.0, kl_raw)))
        self.recent_surprises.append(surprise)
        baseline = float(np.mean(self.recent_surprises))
        std = float(np.std(self.recent_surprises)) if len(self.recent_surprises) > 1 else 0.0
        anomaly = max(0.0, (surprise - baseline) / (std + self.eps)) if std > 0 else 0.0

        self.history.append(np.concatenate([probs, time_feats]).astype(np.float32))
        return {
            "hour": hour,
            "top_idx": int(np.argmax(probs)),
            "activity_score": float(1.0 - probs[:6].sum()),
            "surprise": surprise,
            "kl_raw": kl_raw,
            "anomaly": float(min(anomaly, 10.0)),
            "prior": prior.astype(np.float32),
            "model": "RhythmMamba" if self.model is not None else "OnlinePrior",
        }
