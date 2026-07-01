"""Within-dataset pipeline: effective rank (Table I), control/leakage checks
(Table III), and aggregation of the architecture comparison (Table II).

All decoding uses one L2-logistic decoder (StandardScaler + LogisticRegression,
C=1.0, max_iter=1000), identical to the full-FC point in topk_decoding.py and the
L2-logistic row in architecture.py, so the reference AUROC is consistent across
Table II, Table III, and the top-k figure. CV: 3 seeds x 3-fold subject-stratified;
the HCP family control uses GroupKFold; the ABIDE control uses leave-one-site-out.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from fc_lib import OUT, effective_rank, load_dataset, make_folds

OUT_NEW = Path(OUT) / "3x3_1002"
OUT_NEW.mkdir(parents=True, exist_ok=True)

SEEDS = [1, 2, 3]
N_SPLITS = 3
DATASETS = ["hcp_sex", "abide", "adhd"]
NICE = {"hcp_sex": "HCP-sex", "abide": "ABIDE", "adhd": "ADHD-200"}


def logit_auroc(Xtr, ytr, Xte, yte):
    """The canonical L2-logistic decoder: StandardScaler + logistic regression."""
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(Xtr), ytr)
    if len(set(yte.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1]))


def cv_subject(X, y):
    """Mean test AUROC over 3 seeds x 3 subject-stratified folds."""
    a = []
    for sd in SEEDS:
        for tr, te in make_folds(y, mode="subject", n_splits=N_SPLITS, seed=sd):
            a.append(logit_auroc(X[tr], y[tr], X[te], y[te]))
    return float(np.nanmean(a))


def run_effective_rank():
    rows, spectra = [], {}
    for name in DATASETS:
        d = load_dataset(name)
        X = d["X"]
        sh, pr, lam = effective_rank(X)
        rows.append({"Dataset": name, "N": X.shape[0], "D": X.shape[1],
                     "Shannon_ER": round(sh, 2), "Participation_ratio": round(pr, 2),
                     "ER_over_D": round(sh / X.shape[1], 5), "ER_over_N": round(sh / X.shape[0], 4)})
        spectra[name] = np.cumsum(lam / lam.sum())
    pd.DataFrame(rows).to_csv(OUT_NEW / "exp3_effective_rank_table.csv", index=False)

    plt.figure(figsize=(5, 3.8))
    for name in DATASETS:
        c = spectra[name]
        plt.plot(np.arange(1, min(len(c), 300) + 1), c[:300], label=NICE[name])
    plt.axhline(0.9, ls="--", c="gray", lw=0.8)
    plt.xlabel("principal component index")
    plt.ylabel("cumulative variance")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_NEW / "fig1_erank_spectrum.pdf")
    plt.close()
    print("[done] effective rank", flush=True)


def run_controls():
    rows = []
    for name in DATASETS:
        d = load_dataset(name)
        X, y = d["X"], d["y"]
        real = cv_subject(X, y)
        ysh = y.copy()
        np.random.default_rng(0).shuffle(ysh)
        shuf = cv_subject(X, ysh)
        Xr = np.random.default_rng(1).standard_normal(X.shape).astype(np.float32)
        rand = cv_subject(Xr, y)
        rows.append({"exp": "4a/4b", "dataset": name, "real_AUROC": round(real, 3),
                     "shuffle_AUROC": round(shuf, 3), "randfeat_AUROC": round(rand, 3)})
        print(f"[control] {name}: real={real:.3f} shuffle={shuf:.3f} randfeat={rand:.3f}", flush=True)

    real_by_ds = {r["dataset"]: r["real_AUROC"] for r in rows}

    # HCP family control: GroupKFold over families (deterministic; no seed needed)
    d = load_dataset("hcp_sex")
    X, y, g = d["X"], d["y"], d["groups"]
    fam = [logit_auroc(X[tr], y[tr], X[te], y[te])
           for tr, te in make_folds(y, groups=g, mode="group", n_splits=N_SPLITS)]
    rows.append({"exp": "4c", "dataset": "hcp_sex", "subject_CV": real_by_ds["hcp_sex"],
                 "family_CV": round(float(np.nanmean(fam)), 3), "n_families": len(set(g))})
    print(f"[control] hcp_sex family_CV={np.nanmean(fam):.3f}", flush=True)

    # ABIDE leave-one-site-out
    d = load_dataset("abide")
    X, y, g = d["X"], d["y"], d["groups"]
    loso = [logit_auroc(X[tr], y[tr], X[te], y[te]) for tr, te in make_folds(y, groups=g, mode="loso")]
    rows.append({"exp": "4d", "dataset": "abide", "random3x3": real_by_ds["abide"],
                 "LOSO": round(float(np.nanmean(loso)), 3), "LOSO_std": round(float(np.nanstd(loso)), 3)})
    print(f"[control] abide LOSO={np.nanmean(loso):.3f}+/-{np.nanstd(loso):.3f}", flush=True)

    pd.DataFrame(rows).to_csv(OUT_NEW / "exp4_controls.csv", index=False)
    print("[done] controls", flush=True)


def write_architecture_tables():
    """Aggregate the architecture comparison (run architecture.py and bnt.py first)."""
    arch_src = Path(OUT) / "exp6_architecture_comparison.csv"
    bnt_src = Path(OUT) / "exp6c_bnt.csv"
    if not (arch_src.exists() and bnt_src.exists()):
        print("[arch] architecture CSVs not found; run architecture.py and bnt.py first", flush=True)
        return
    full = pd.concat([pd.read_csv(arch_src), pd.read_csv(bnt_src)], ignore_index=True)
    full.to_csv(OUT_NEW / "exp6_architecture_with_bnt.csv", index=False)
    print("[done] architecture aggregation", flush=True)


def main():
    run_effective_rank()
    run_controls()
    write_architecture_tables()
    print(f"ALL DONE -> {OUT_NEW}", flush=True)


if __name__ == "__main__":
    main()
