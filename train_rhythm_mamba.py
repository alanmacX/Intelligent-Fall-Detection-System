import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from core.rhythm_mamba import RhythmMamba


def get_wandb(project, name, config):
    try:
        import wandb
        os.environ.setdefault("WANDB_MODE", "offline")
        return wandb.init(project=project, name=name, config=config)
    except Exception as exc:
        print(f"wandb disabled: {exc}")
        return None


def build_windows(data_path, seq_len):
    data = np.load(data_path, allow_pickle=True)
    features = np.concatenate([data["probs"], data["time_features"]], axis=1).astype(np.float32)
    targets = data["probs"].astype(np.float32)
    anomaly = data["anomaly"].astype(np.int64)
    xs, ys, flags = [], [], []
    for idx in range(seq_len, len(features)):
        xs.append(features[idx - seq_len:idx])
        ys.append(targets[idx])
        flags.append(anomaly[idx])
    return np.stack(xs), np.stack(ys), np.array(flags, dtype=np.int64)


def top1_acc(logits, targets):
    pred = logits.argmax(dim=1)
    gt = targets.argmax(dim=1)
    return (pred == gt).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="core/data/rhythm_mock.npz")
    parser.add_argument("--output", default="weights/rhythm_mamba.pth")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, y, flags = build_windows(args.data, args.seq_len)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_x = torch.from_numpy(X_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)

    model = RhythmMamba(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    run = get_wandb("fall-rhythm", "rhythm-mamba-demo", vars(args))
    best_val = float("inf")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = F.kl_div(F.log_softmax(logits, dim=1), by, reduction="batchmean")
            loss.backward()
            optimizer.step()
            total += loss.item()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = F.kl_div(F.log_softmax(val_logits, dim=1), val_y, reduction="batchmean").item()
            acc = top1_acc(val_logits, val_y)

        metrics = {
            "train/loss": total / max(len(train_loader), 1),
            "val/loss": val_loss,
            "val/top1_acc": acc,
            "epoch": epoch,
        }
        if run:
            run.log(metrics)
        print(f"epoch {epoch:02d} train_loss={metrics['train/loss']:.4f} val_loss={val_loss:.4f} acc={acc:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "input_dim": 14,
                        "hidden_dim": args.hidden_dim,
                        "num_classes": 12,
                        "seq_len": args.seq_len,
                    },
                },
                args.output,
            )
            print(f"saved {args.output}")

    if run:
        run.finish()


if __name__ == "__main__":
    main()
