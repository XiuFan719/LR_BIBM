"""Top-k PC decoding curve.

For each dataset, PCA is fit on the TRAIN fold only, the top-k principal
components are kept, a logistic decoder is trained, and test AUROC is read as a
function of k over {5,10,20,50,100, ER, 2*ER, ...} plus the full-FC ceiling.
A mid-band control decodes the PCs ranked [ER : 2*ER] to show the tail past the
effective rank carries little class information.
Outputs -> out/3x3_1002/exp_topk_pc_decoding.{csv,pdf} + _summary.csv.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from fc_lib import load_dataset, make_folds, effective_rank, OUT

OUT_NEW = Path(OUT) / "3x3_1002"
OUT_NEW.mkdir(parents=True, exist_ok=True)

SEEDS = [1, 2, 3]
N_SPLITS = 3
DATASETS = ["hcp_sex", "abide", "adhd"]
NICE = {"hcp_sex": "HCP-sex", "abide": "ABIDE", "adhd": "ADHD-200"}
BASE_KS = [5, 10, 20, 50, 100]
C_LOGREG = 1.0


def decode(Xtr, ytr, Xte, yte, k=None, drop_first=0):
    """Logistic AUROC on top-k PCs (or full FC if k is None). drop_first>0 decodes
    the band [drop_first : drop_first+k]. PCA + scaler fit on train only."""
    sc0 = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc0.transform(Xtr), sc0.transform(Xte)
    if k is None:
        Ztr, Zte = Xtr_s, Xte_s
    else:
        n_comp = drop_first + k
        n_comp = min(n_comp, Xtr_s.shape[0] - 1, Xtr_s.shape[1])
        pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=0).fit(Xtr_s)
        Ztr = pca.transform(Xtr_s)[:, drop_first:]
        Zte = pca.transform(Xte_s)[:, drop_first:]
    sc = StandardScaler().fit(Ztr)
    clf = LogisticRegression(max_iter=1000, C=C_LOGREG).fit(sc.transform(Ztr), ytr)
    if len(set(yte.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(yte, clf.predict_proba(sc.transform(Zte))[:, 1]))


def run():
    rows = []
    er_by_ds = {}
    for name in DATASETS:
        d = load_dataset(name)
        X, y = d["X"], d["y"]
        er = float(effective_rank(X)[0])
        er_by_ds[name] = er
        kER = int(round(er))
        # cap the k grid at the smallest train fold's rank so every point is a true top-k decode
        folds_all = [tr for sd in SEEDS for tr, _ in make_folds(y, mode="subject", n_splits=N_SPLITS, seed=sd)]
        max_k = min(min(len(tr) for tr in folds_all) - 1, X.shape[1])
        mult = {int(round(m * er)): f"{m}ER" for m in (1, 2, 3, 4, 6, 8)}
        cand = sorted(set([k for k in BASE_KS if k <= max_k] + [k for k in mult if k <= max_k]))
        labels = []
        for k in cand:
            tag = mult.get(k, "")
            labels.append((k, f"k={k}" + (f" ({tag})" if tag else ""), 0))
        labels.append((None, "full FC", 0))
        labels.append((kER, f"band[{kER}:{2 * kER}]", kER))   # mid-band [ER:2ER] control

        for k, label, drop in labels:
            aucs = []
            for sd in SEEDS:
                for tr, te in make_folds(y, mode="subject", n_splits=N_SPLITS, seed=sd):
                    aucs.append(decode(X[tr], y[tr], X[te], y[te], k=k, drop_first=drop))
            aucs = np.array(aucs, dtype=float)
            rows.append({
                "dataset": name,
                "label": label,
                "k_value": ("full" if k is None else (k if drop == 0 else f"{drop}-{drop + k}")),
                "is_band": drop > 0,
                "ER": round(er, 1),
                "auroc_mean": round(float(np.nanmean(aucs)), 4),
                "auroc_std": round(float(np.nanstd(aucs)), 4),
                "n_runs": int((~np.isnan(aucs)).sum()),
            })
            print(f"[{name:8s}] {label:18s} -> AUROC {rows[-1]['auroc_mean']:.4f} "
                  f"+/- {rows[-1]['auroc_std']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_NEW / "exp_topk_pc_decoding.csv", index=False)

    # saturation summary: smallest top-k reaching 95% / 99% of full-FC AUROC
    summ = []
    for name in DATASETS:
        sub = df[(df.dataset == name) & (~df.is_band)]
        full = float(sub[sub.k_value == "full"]["auroc_mean"].iloc[0])
        curve = sub[sub.k_value != "full"].copy()
        curve["k_int"] = curve["k_value"].astype(int)
        curve = curve.sort_values("k_int")
        band = df[(df.dataset == name) & (df.is_band)]["auroc_mean"].iloc[0]

        def smallest_k(frac):
            hit = curve[curve["auroc_mean"] >= frac * full]
            return int(hit["k_int"].iloc[0]) if len(hit) else None

        er = er_by_ds[name]
        peak_row = curve.loc[curve["auroc_mean"].idxmax()]
        peak_k = int(peak_row["k_int"])
        peak_auroc = float(peak_row["auroc_mean"])
        max_k_meas = int(curve["k_int"].max())
        summ.append({
            "dataset": name,
            "ER": round(er, 1),
            "max_k_measured": max_k_meas,
            "full_FC_AUROC": round(full, 4),
            "AUROC_at_k=ER": round(float(curve.iloc[(curve["k_int"] - round(er)).abs().argmin()]["auroc_mean"]), 4),
            "peak_k": peak_k,
            "peak_auroc": round(peak_auroc, 4),
            "peak_over_ER": round(peak_k / er, 2),
            "interior_peak_gt_full": bool(peak_auroc > full and peak_k < max_k_meas),
            "k95": smallest_k(0.95),
            "k99": smallest_k(0.99),
            "k95_le_ER": (smallest_k(0.95) is not None and smallest_k(0.95) <= er),
            "midband[ER:2ER]_AUROC": round(float(band), 4),
        })
    sdf = pd.DataFrame(summ)
    sdf.to_csv(OUT_NEW / "exp_topk_pc_decoding_summary.csv", index=False)
    print("\n=== saturation summary ===")
    print(sdf.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, name in zip(axes, DATASETS):
        sub = df[(df.dataset == name) & (~df.is_band) & (df.k_value != "full")].copy()
        sub["k_int"] = sub["k_value"].astype(int)
        sub = sub.sort_values("k_int")
        full = float(df[(df.dataset == name) & (df.k_value == "full")]["auroc_mean"].iloc[0])
        ax.errorbar(sub["k_int"], sub["auroc_mean"], yerr=sub["auroc_std"],
                    fmt="o-", capsize=2.5, c="tab:blue", label="top-k PC decode")
        ax.axhline(full, ls="--", c="0.4", lw=1.1, label="full FC")
        ax.axhline(0.5, ls=":", c="tab:red", lw=0.9, label="chance")
        ax.axvline(er_by_ds[name], ls="--", c="tab:green", lw=1.2, label=f"ER={er_by_ds[name]:.0f}")
        ax.set_xscale("log")
        ax.set_xlabel("# top PCs (k, log)")
        if name == DATASETS[0]:
            ax.set_ylabel("test AUROC")
        ax.set_title(NICE[name], fontsize=10)
        ax.grid(alpha=0.3)
        if name == DATASETS[0]:
            ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_NEW / "exp_topk_pc_decoding.pdf", bbox_inches="tight")
    plt.close()
    print(f"\nsaved -> {OUT_NEW}/exp_topk_pc_decoding.{{csv,pdf}} + _summary.csv")


if __name__ == "__main__":
    run()
