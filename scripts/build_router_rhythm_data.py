import argparse
import os
import sys

import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.rhythm_mamba import RhythmMamba


def load_rhythm_model(path, device):
    ckpt = torch.load(path, map_location=device)
    config = ckpt.get("config", {})
    model = RhythmMamba(
        input_dim=config.get("input_dim", 14),
        hidden_dim=config.get("hidden_dim", 64),
        num_classes=config.get("num_classes", 12),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config.get("seq_len", 96)


def kl_div(p, q, eps=1e-7):
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    return np.sum(p * np.log(p / q), axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-x", default="core/data/data/X_router_enhanced.npy")
    parser.add_argument("--router-y", default="core/data/data/y_router_enhanced.npy")
    parser.add_argument("--rhythm-data", default="core/data/rhythm_mock.npz")
    parser.add_argument("--rhythm-model", default="weights/rhythm_mamba.pth")
    parser.add_argument("--output-x", default="core/data/data/X_router_rhythm.npy")
    parser.add_argument("--output-y", default="core/data/data/y_router_rhythm.npy")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = np.load(args.router_x).astype(np.float32)
    y = np.load(args.router_y).astype(np.float32)
    rhythm = np.load(args.rhythm_data, allow_pickle=True)
    features = np.concatenate([rhythm["probs"], rhythm["time_features"]], axis=1).astype(np.float32)
    anomaly = rhythm["anomaly"].astype(np.int64)

    model, seq_len = load_rhythm_model(args.rhythm_model, device)
    normal_candidates = np.where((anomaly == 0) & (np.arange(len(anomaly)) >= seq_len))[0]
    anomaly_candidates = np.where((anomaly == 1) & (np.arange(len(anomaly)) >= seq_len))[0]
    if len(anomaly_candidates) == 0:
        raise RuntimeError("rhythm mock data contains no anomaly candidates")

    surprise = np.zeros(len(X), dtype=np.float32)
    observed_labels = np.zeros(len(X), dtype=np.int64)

    with torch.no_grad():
        for idx, target in enumerate(y):
            if target > 0.5:
                source_idx = int(rng.choice(anomaly_candidates))
            else:
                if rng.random() < 0.06:
                    source_idx = int(rng.choice(anomaly_candidates))
                else:
                    source_idx = int(rng.choice(normal_candidates))

            window = torch.from_numpy(features[source_idx - seq_len:source_idx]).unsqueeze(0).to(device)
            prior = model.predict_prior(window).cpu().numpy()[0]
            observed = rhythm["probs"][source_idx].astype(np.float32)
            observed_labels[idx] = int(observed.argmax())
            surprise[idx] = float(kl_div(observed[None, :], prior[None, :])[0])

    surprise = np.log1p(surprise).astype(np.float32)
    X_aug = np.concatenate([X, surprise[:, None]], axis=1).astype(np.float32)
    os.makedirs(os.path.dirname(args.output_x), exist_ok=True)
    np.save(args.output_x, X_aug)
    np.save(args.output_y, y)
    print(f"saved {args.output_x} {X_aug.shape}")
    print(f"saved {args.output_y} {y.shape}")
    print(f"surprise min={surprise.min():.4f} mean={surprise.mean():.4f} max={surprise.max():.4f}")
    print(f"positive mean surprise={surprise[y > 0.5].mean():.4f}")
    print(f"negative mean surprise={surprise[y <= 0.5].mean():.4f}")


if __name__ == "__main__":
    main()
