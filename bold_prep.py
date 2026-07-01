"""Build the raw-BOLD representation and the BOLD-vs-FC linear controls.

Caches a flattened first-80-frame BOLD feature per dataset, then reports
effective rank of FC vs BOLD and 3x3 ridge AUROC on each, plus ABIDE per-site
effective rank.
Outputs -> out/3x3_1002/{dataset}_bold_ts80.npz, bold_vs_fc_er.csv,
bold_vs_fc_ridge.csv, abide_per_site_er.csv.
"""
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import config
from fc_lib import OUT, _raw_abide, _read_1d, effective_rank, load_dataset, make_folds


OUT_NEW = Path(OUT) / "3x3_1002"
OUT_NEW.mkdir(parents=True, exist_ok=True)

SEEDS = [1, 2, 3]
N_SPLITS = 3
T0 = 80
DATASETS = ["hcp_sex", "abide", "adhd"]


def flatten_bold(sig, t0=T0):
    """First t0 frames of ROI signals, per-ROI standardized, flattened."""
    z = sig[:t0].astype(np.float32)
    z = (z - z.mean(0, keepdims=True)) / (z.std(0, keepdims=True) + 1e-6)
    return z.reshape(-1).astype(np.float32)


def raw_hcp_bold(t0=T0):
    info = pd.read_csv(config.HCP_INFO_CSV)
    gmap, fmap = {}, {}
    for _, r in info.iterrows():
        if pd.isna(r["subject"]):
            continue
        sid = str(int(r["subject"]))
        g = str(r.get("gender")).strip().lower()
        if g.startswith("m"):
            gmap[sid] = 1.0
        elif g.startswith("f"):
            gmap[sid] = 0.0
        else:
            continue
        fmap[sid] = int(r["Family_ID"]) if pd.notna(r.get("Family_ID")) else -1

    X, y, ids, groups = [], [], [], []
    for p in sorted(glob.glob(os.path.join(config.HCP_ROOT, "**", "*.mat"), recursive=True)):
        sid = os.path.basename(p).split("_")[0]
        if sid not in gmap:
            continue
        sig = loadmat(p, mat_dtype=True)["ROISignals"]
        if sig.shape[0] >= t0 and sig.shape[1] == 246:
            X.append(flatten_bold(sig, t0))
            y.append(gmap[sid])
            ids.append(sid)
            groups.append(fmap.get(sid, -1))
    return {"X": np.stack(X).astype(np.float32), "y": np.asarray(y, dtype=np.float32),
            "ids": np.asarray(ids), "groups": np.asarray(groups)}


def raw_abide_bold(t0=T0):
    rgx = re.compile(r"^(.*)_(\d{6,7})_rois_aal\.1D$")
    X, y, ids, groups = [], [], [], []
    for cls, lab in (("Healthy", 0.0), ("ASD", 1.0)):
        for p in sorted(glob.glob(os.path.join(config.ABIDE_ROOT, cls, "*_rois_aal.1D"))):
            m = rgx.match(os.path.basename(p))
            if not m:
                continue
            sig = _read_1d(p, "abide")
            if sig.shape[0] >= t0 and sig.shape[1] == 116:
                X.append(flatten_bold(sig, t0))
                y.append(lab)
                ids.append(m.group(2))
                groups.append(m.group(1))
    return {"X": np.stack(X).astype(np.float32), "y": np.asarray(y, dtype=np.float32),
            "ids": np.asarray(ids), "groups_raw": np.asarray(groups)}


def raw_adhd_bold(t0=T0):
    pheno = pd.read_csv(config.ADHD_PHENO, sep="\t")
    site = {int(r["ScanDir ID"]): int(r["Site"]) for _, r in pheno.iterrows() if pd.notna(r["ScanDir ID"])}
    seen = {}
    for cls, lab in (("0", 0.0), ("1", 1.0)):
        for p in sorted(glob.glob(os.path.join(config.ADHD_ROOT, cls, "*session_1_rest_1*aal_TCs.1D"))):
            m = re.search(r"da(\d+)_", os.path.basename(p))
            if m:
                seen.setdefault(m.group(1), (lab, p))
    X, y, ids, groups = [], [], [], []
    for sid, (lab, p) in seen.items():
        if int(sid) not in site:
            continue
        sig = _read_1d(p, "adhd")
        if sig.shape[0] >= t0 and sig.shape[1] == 116:
            X.append(flatten_bold(sig, t0))
            y.append(lab)
            ids.append(sid)
            groups.append(site[int(sid)])
    return {"X": np.stack(X).astype(np.float32), "y": np.asarray(y, dtype=np.float32),
            "ids": np.asarray(ids), "groups_raw": np.asarray(groups)}


def load_bold_dataset(name):
    cache_path = OUT_NEW / f"{name}_bold_ts{T0}.npz"
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        return {k: d[k] for k in d.files}
    d = {"hcp_sex": raw_hcp_bold, "abide": raw_abide_bold, "adhd": raw_adhd_bold}[name](T0)
    np.savez(cache_path, **d)
    return d


def rank_row(dataset, representation, X):
    er, pr, _ = effective_rank(X)
    return {"dataset": dataset, "representation": representation, "N": X.shape[0], "D": X.shape[1],
            "Shannon_ER": round(er, 4), "Participation_ratio": round(pr, 4),
            "ER_over_D": round(er / X.shape[1], 6), "ER_over_N": round(er / X.shape[0], 6)}


def auc_safe(y, score):
    try:
        return float(roc_auc_score(y, score))
    except ValueError:
        return float("nan")


def ridge_3x3(dataset, representation, X, y):
    rows = []
    for sd in SEEDS:
        for fi, (tr, te) in enumerate(make_folds(y, mode="subject", n_splits=N_SPLITS, seed=sd)):
            sc = StandardScaler().fit(X[tr])
            clf = RidgeClassifier(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
            score = clf.decision_function(sc.transform(X[te]))
            rows.append({"dataset": dataset, "representation": representation, "model": "RidgeClassifier",
                         "seed": sd, "fold": fi, "n_train": len(tr), "n_test": len(te),
                         "test_AUROC": auc_safe(y[te], score)})
    return rows


def write_bold_vs_fc():
    er_rows, fold_rows = [], []
    for name in DATASETS:
        fc = load_dataset(name)
        bold = load_bold_dataset(name)
        print(f"[bold-vs-fc] {name}: FC {fc['X'].shape}, BOLD {bold['X'].shape}", flush=True)
        er_rows.append(rank_row(name, "FC_FisherZ_uppertri", fc["X"]))
        er_rows.append(rank_row(name, f"BOLD_TS{T0}_flatten", bold["X"]))
        fold_rows.extend(ridge_3x3(name, "FC_FisherZ_uppertri", fc["X"], fc["y"]))
        fold_rows.extend(ridge_3x3(name, f"BOLD_TS{T0}_flatten", bold["X"], bold["y"]))

    pd.DataFrame(er_rows).to_csv(OUT_NEW / "bold_vs_fc_er.csv", index=False)
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(OUT_NEW / "bold_vs_fc_ridge_folds.csv", index=False)
    summary = []
    for (ds, rep), g in folds.groupby(["dataset", "representation"]):
        seed_means = g.groupby("seed")["test_AUROC"].mean()
        summary.append({"dataset": ds, "representation": rep, "model": "RidgeClassifier",
                        "test_AUROC": round(float(seed_means.mean()), 4),
                        "std": round(float(seed_means.std(ddof=0)), 4),
                        "fold_AUROC_mean": round(float(g["test_AUROC"].mean()), 4),
                        "fold_AUROC_std": round(float(g["test_AUROC"].std(ddof=0)), 4)})
    pd.DataFrame(summary).sort_values(["dataset", "representation"]).to_csv(
        OUT_NEW / "bold_vs_fc_ridge.csv", index=False)
    print("[done] bold-vs-fc rank and ridge controls", flush=True)


def write_abide_per_site_er():
    d = load_dataset("abide")
    raw_rows = _raw_abide()
    site_labels = sorted(set(str(r[3]) for r in raw_rows))
    site_name = {i: s for i, s in enumerate(site_labels)}
    rows = []
    for g in sorted(set(d["groups"])):
        mask = d["groups"] == g
        y = d["y"][mask]
        er, pr, _ = effective_rank(d["X"][mask])
        rows.append({"site_id": int(g), "site_label": site_name.get(int(g), str(g)),
                     "N": int(mask.sum()), "n_class0": int((y == 0).sum()), "n_class1": int((y == 1).sum()),
                     "D": int(d["X"].shape[1]), "Shannon_ER": round(er, 4),
                     "Participation_ratio": round(pr, 4), "ER_over_D": round(er / d["X"].shape[1], 6),
                     "ER_over_N": round(er / mask.sum(), 6),
                     "included_in_16site_loso": bool(mask.sum() >= 20 and len(set(y.tolist())) == 2)})
    pd.DataFrame(rows).to_csv(OUT_NEW / "abide_per_site_er.csv", index=False)
    print("[done] ABIDE per-site ER", flush=True)


def main():
    write_bold_vs_fc()
    write_abide_per_site_er()
    print(f"ALL DONE -> {OUT_NEW}", flush=True)


if __name__ == "__main__":
    main()
