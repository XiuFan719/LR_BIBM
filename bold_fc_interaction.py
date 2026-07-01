"""BOLD-vs-FC representation/complexity control.

Tests whether high-rank raw BOLD needs a temporal model while low-rank FC is
already linearly decodable. Uses the cached flattened BOLD_TS80 representation
(built by bold_prep.py), reshapes it back to N x ROI x T, and trains a small
temporal CNN under the 3 seeds x 3 folds protocol.
Outputs -> out/3x3_1002/bold_temporal_*.csv, bold_fc_interaction_summary.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from fc_lib import OUT, inner_val_split, make_folds


OUT_NEW = Path(OUT) / "3x3_1002"
T0 = 80
SEEDS = [1, 2, 3]
N_SPLITS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NICE = {"hcp_sex": "HCP-sex", "abide": "ABIDE", "adhd": "ADHD-200"}
ROI_DIMS = {"hcp_sex": 246, "abide": 116, "adhd": 116}

EPOCH_PATH = OUT_NEW / "bold_temporal_epoch_metrics.csv"
SUMMARY_PATH = OUT_NEW / "bold_temporal_run_summary.csv"
INTERACTION_PATH = OUT_NEW / "bold_fc_interaction_summary.csv"
COMPLETED_PATH = OUT_NEW / "bold_temporal_completed_runs.csv"


def auc_safe(y, score):
    try:
        return float(roc_auc_score(y, score))
    except ValueError:
        return float("nan")


def acc_score(y, prob):
    return float(((prob >= 0.5).astype(np.float32) == y.astype(np.float32)).mean())


class TemporalBlock(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        pad = 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=5, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class TemporalCNN(nn.Module):
    def __init__(self, n_roi, channels=64, dropout=0.25):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_roi, channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            TemporalBlock(channels, dilation=1, dropout=dropout),
            TemporalBlock(channels, dilation=2, dropout=dropout),
            TemporalBlock(channels, dilation=4, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels, 1),
        )

    def forward(self, x):
        z = self.blocks(self.stem(x))
        avg = z.mean(dim=-1)
        mx = z.amax(dim=-1)
        return self.head(torch.cat([avg, mx], dim=1)).squeeze(-1)


def load_bold_timeseries(dataset):
    path = OUT_NEW / f"{dataset}_bold_ts{T0}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run bold_prep.py first")
    d = np.load(path, allow_pickle=True)
    Xflat = d["X"].astype(np.float32)
    y = d["y"].astype(np.float32)
    n_roi = ROI_DIMS[dataset]
    if Xflat.shape[1] != T0 * n_roi:
        raise RuntimeError(f"{dataset}: expected D={T0 * n_roi}, got {Xflat.shape[1]}")
    X = Xflat.reshape(len(Xflat), T0, n_roi).transpose(0, 2, 1).astype(np.float32)
    return X, y


def standardize_channelwise(Xtr, Xva, Xte):
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd


def eval_split(model, lossf, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss = lossf(logits, y).item()
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        yn = y.detach().cpu().numpy()
    return loss, auc_safe(yn, prob), acc_score(yn, prob)


def train_one(X, y, tr_idx, va_idx, te_idx, seed, channels, epochs, lr, weight_decay, dropout):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    Xtr, Xva, Xte = standardize_channelwise(X[tr_idx], X[va_idx], X[te_idx])
    Xtr_t = torch.tensor(Xtr, device=DEVICE)
    Xva_t = torch.tensor(Xva, device=DEVICE)
    Xte_t = torch.tensor(Xte, device=DEVICE)
    ytr_t = torch.tensor(y[tr_idx], device=DEVICE)
    yva_t = torch.tensor(y[va_idx], device=DEVICE)
    yte_t = torch.tensor(y[te_idx], device=DEVICE)

    model = TemporalCNN(X.shape[1], channels=channels, dropout=dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.BCEWithLogitsLoss()
    rows = []
    best = {"val_auroc": -np.inf, "test_auroc_at_best_val": np.nan, "epoch": -1}

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = lossf(model(Xtr_t), ytr_t)
        loss.backward()
        opt.step()

        tr_loss, tr_auc, tr_acc = eval_split(model, lossf, Xtr_t, ytr_t)
        va_loss, va_auc, va_acc = eval_split(model, lossf, Xva_t, yva_t)
        te_loss, te_auc, te_acc = eval_split(model, lossf, Xte_t, yte_t)
        rows.append({
            "epoch": ep + 1,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "test_loss": te_loss,
            "train_auroc": tr_auc,
            "val_auroc": va_auc,
            "test_auroc": te_auc,
            "train_acc": tr_acc,
            "val_acc": va_acc,
            "test_acc": te_acc,
        })
        if np.isfinite(va_auc) and va_auc > best["val_auroc"]:
            best = {"val_auroc": va_auc, "test_auroc_at_best_val": te_auc, "epoch": ep + 1}

    n_params = sum(p.numel() for p in model.parameters())
    return rows, best, n_params


def append_df(path, df):
    df.to_csv(path, mode="a", header=not Path(path).exists(), index=False)


def load_completed():
    if not COMPLETED_PATH.exists():
        return set()
    return set(pd.read_csv(COMPLETED_PATH)["run_id"].astype(str))


def summarize_temporal():
    if not SUMMARY_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(SUMMARY_PATH)
    rows = []
    for (dataset, model), g in df.groupby(["dataset", "model"]):
        seed_means = g.groupby("seed")["test_auroc_at_best_val"].mean()
        rows.append({
            "dataset": dataset,
            "representation": f"BOLD_TS{T0}_timeseries",
            "model": model,
            "test_AUROC": round(float(seed_means.mean()), 4),
            "std": round(float(seed_means.std(ddof=0)), 4),
            "selection": "best validation AUROC epoch",
            "n_runs": int(len(g)),
        })
    return pd.DataFrame(rows)


def write_interaction_summary():
    rows = []
    ridge_path = OUT_NEW / "bold_vs_fc_ridge.csv"
    if ridge_path.exists():
        ridge = pd.read_csv(ridge_path)
        for r in ridge.itertuples():
            rows.append({
                "dataset": r.dataset,
                "representation": r.representation,
                "model": r.model,
                "test_AUROC": r.test_AUROC,
                "std": r.std,
                "selection": "3x3 seed mean",
                "n_runs": 9,
            })

    arch_path = OUT_NEW / "exp6_architecture_with_bnt.csv"
    if arch_path.exists():
        arch = pd.read_csv(arch_path)
        keep = arch[arch["model"].isin(["MLP(1024)", "BrainNetworkTransformer"])]
        for r in keep.itertuples():
            rows.append({
                "dataset": r.dataset,
                "representation": "FC_FisherZ_uppertri",
                "model": r.model,
                "test_AUROC": r.test_AUROC,
                "std": r.test_std,
                "selection": "architecture table",
                "n_runs": 3,
            })

    temporal = summarize_temporal()
    if not temporal.empty:
        rows.extend(temporal.to_dict("records"))
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["dataset", "representation", "model"])
        out.to_csv(INTERACTION_PATH, index=False)
    return out


def run(args):
    OUT_NEW.mkdir(parents=True, exist_ok=True)
    completed = load_completed()
    all_summaries = []
    for dataset in args.datasets:
        X, y = load_bold_timeseries(dataset)
        folds_by_seed = [(sd, make_folds(y, mode="subject", n_splits=N_SPLITS, seed=sd)) for sd in SEEDS]
        print(f"[bold-temporal] {dataset}: X={X.shape}, device={DEVICE}, epochs={args.epochs}", flush=True)
        for sd, folds in folds_by_seed:
            for fi, (tr0, te) in enumerate(folds):
                tr, va = inner_val_split(tr0, y, seed=sd * 100 + fi, frac=0.2)
                run_id = f"bold_temporal__{dataset}__TemporalCNN_c{args.channels}__seed{sd}__fold{fi}"
                if run_id in completed:
                    print(f"[skip] {run_id}", flush=True)
                    continue
                rows, best, n_params = train_one(
                    X, y, tr, va, te,
                    seed=sd * 1000 + fi,
                    channels=args.channels,
                    epochs=args.epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    dropout=args.dropout,
                )
                epoch_df = pd.DataFrame(rows)
                epoch_df["dataset"] = dataset
                epoch_df["representation"] = f"BOLD_TS{T0}_timeseries"
                epoch_df["model"] = f"TemporalCNN_c{args.channels}"
                epoch_df["seed"] = sd
                epoch_df["fold"] = fi
                epoch_df["run_id"] = run_id
                epoch_df["n_train"] = len(tr)
                epoch_df["n_val"] = len(va)
                epoch_df["n_test"] = len(te)
                epoch_df["n_params"] = n_params
                append_df(EPOCH_PATH, epoch_df)

                last = epoch_df.iloc[-1]
                summary = {
                    "dataset": dataset,
                    "representation": f"BOLD_TS{T0}_timeseries",
                    "model": f"TemporalCNN_c{args.channels}",
                    "seed": sd,
                    "fold": fi,
                    "run_id": run_id,
                    "n_train": len(tr),
                    "n_val": len(va),
                    "n_test": len(te),
                    "n_params": n_params,
                    "best_val_epoch": best["epoch"],
                    "best_val_auroc": best["val_auroc"],
                    "test_auroc_at_best_val": best["test_auroc_at_best_val"],
                    "test_auroc_final": last["test_auroc"],
                    "test_auroc_max": float(epoch_df["test_auroc"].max()),
                    "train_loss_final": last["train_loss"],
                    "val_loss_final": last["val_loss"],
                    "test_loss_final": last["test_loss"],
                    "train_auroc_final": last["train_auroc"],
                    "val_auroc_final": last["val_auroc"],
                    "train_acc_final": last["train_acc"],
                    "val_acc_final": last["val_acc"],
                    "test_acc_final": last["test_acc"],
                }
                append_df(SUMMARY_PATH, pd.DataFrame([summary]))
                append_df(COMPLETED_PATH, pd.DataFrame([{"run_id": run_id}]))
                completed.add(run_id)
                all_summaries.append(summary)
                print(
                    f"[done] {run_id}: best-val ep={best['epoch']} "
                    f"val={best['val_auroc']:.3f} test={best['test_auroc_at_best_val']:.3f} "
                    f"final={last['test_auroc']:.3f}",
                    flush=True,
                )

    inter = write_interaction_summary()
    if not inter.empty:
        print(inter.to_string(index=False), flush=True)
    return pd.DataFrame(all_summaries)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["hcp_sex"], choices=["hcp_sex", "abide", "adhd"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.25)
    return ap.parse_args()


if __name__ == "__main__":
    torch.set_num_threads(8)
    run(parse_args())
