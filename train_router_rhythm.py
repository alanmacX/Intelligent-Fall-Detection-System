import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


class RhythmRouter(nn.Module):
    def __init__(self, input_dim=515, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def get_wandb(project, name, config):
    try:
        import wandb
        os.environ.setdefault("WANDB_MODE", "offline")
        return wandb.init(project=project, name=name, config=config)
    except Exception as exc:
        print(f"wandb disabled: {exc}")
        return None


def auc_score(y_true, y_score):
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score)
    pos = y_score[y_true]
    neg = y_score[~y_true]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    pos_ranks = ranks[:len(pos)]
    return float((pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def metrics(pred, target):
    pred_np = pred.detach().cpu().numpy().reshape(-1)
    target_np = target.detach().cpu().numpy().reshape(-1)
    hard = (pred_np >= 0.5).astype(np.float32)
    tp = int(((hard == 1) & (target_np == 1)).sum())
    tn = int(((hard == 0) & (target_np == 0)).sum())
    fp = int(((hard == 1) & (target_np == 0)).sum())
    fn = int(((hard == 0) & (target_np == 1)).sum())
    recall = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {
        "val/recall": recall,
        "val/precision": precision,
        "val/f1": f1,
        "val/auc": auc_score(target_np, pred_np),
        "val/tp": tp,
        "val/tn": tn,
        "val/fp": fp,
        "val/fn": fn,
        "val/positive_rate": float(hard.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", default="core/data/data/X_router_rhythm.npy")
    parser.add_argument("--y", default="core/data/data/y_router_rhythm.npy")
    parser.add_argument("--output", default="weights/router_rhythm_best.pth")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = np.load(args.x).astype(np.float32)
    y = np.load(args.y).astype(np.float32)
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y).unsqueeze(1)

    targets = y_tensor.view(-1).long()
    class_count = torch.bincount(targets, minlength=2)
    sample_weight = (1.0 / class_count.float().clamp_min(1))[targets]
    sampler = WeightedRandomSampler(sample_weight, len(sample_weight), replacement=True)
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=args.batch_size, sampler=sampler)

    model = RhythmRouter(input_dim=X.shape[1], hidden_dim=args.hidden_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    run = get_wandb("fall-router", "router-rhythm-515", {**vars(args), "input_dim": X.shape[1], "positive": int(y.sum()), "negative": int((1 - y).sum())})

    best_recall = -1.0
    best_f1 = -1.0
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    full_x = X_tensor.to(device)
    full_y = y_tensor.to(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            total += loss.item()

        model.eval()
        with torch.no_grad():
            full_pred = model(full_x)
            val_loss = criterion(full_pred, full_y).item()
        m = metrics(full_pred, full_y)
        log = {"epoch": epoch, "train/loss": total / max(len(loader), 1), "val/loss": val_loss, **m}
        if run:
            run.log(log)
        print(
            f"epoch {epoch:02d} loss={log['train/loss']:.4f} "
            f"recall={m['val/recall']:.3f} precision={m['val/precision']:.3f} f1={m['val/f1']:.3f} auc={m['val/auc']:.3f}"
        )
        if m["val/recall"] >= best_recall and m["val/f1"] >= best_f1:
            best_recall = m["val/recall"]
            best_f1 = m["val/f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {"input_dim": X.shape[1], "hidden_dim": args.hidden_dim},
                    "metrics": m,
                },
                args.output,
            )
            print(f"saved {args.output}")

    if run:
        run.finish()


if __name__ == "__main__":
    main()
