"""Shared infrastructure: FC feature construction, effective rank, MLP, CV.

Raw ROI time series -> Pearson correlation -> Fisher-z -> upper-triangle vector,
cached per dataset under ./cache. Datasets: hcp_sex, abide, adhd.
"""
import os
import re
import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold

import config

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(HERE, "out")
os.makedirs(CACHE, exist_ok=True)
os.makedirs(OUT, exist_ok=True)


# ---------- feature extraction ----------
def fc_features(sig, tri):
    """corrcoef -> Fisher-z -> upper-tri vector. sig: (T, V)."""
    c = np.corrcoef(sig.T)
    c = np.nan_to_num(c, nan=0.0)
    c = np.clip(c, -0.9999, 0.9999)
    z = np.arctanh(c)
    return z[tri].astype(np.float32)


# ---------- raw loaders ----------
def _read_mat(p):
    return loadmat(p, mat_dtype=True)["ROISignals"].astype(np.float64)


def _read_1d(p, kind):
    df = pd.read_csv(p, sep=r"\s+", header=0)
    if kind == "adhd":
        return df.filter(like="Mean_").to_numpy(dtype=np.float64)
    return df.to_numpy(dtype=np.float64)            # abide: all columns are ROIs


def _raw_hcp():
    """HCP sex classification. Labels and family ids come from the phenotype csv."""
    df = pd.read_csv(config.HCP_INFO_CSV)
    gmap, fmap = {}, {}
    for _, r in df.iterrows():
        if pd.isna(r["subject"]):
            continue
        sid = str(int(r["subject"]))
        g = str(r.get("gender")).strip().lower()
        gmap[sid] = 1 if g.startswith("m") else (0 if g.startswith("f") else None)
        fmap[sid] = int(r["Family_ID"]) if pd.notna(r.get("Family_ID")) else -1
    rows = []
    for p in sorted(glob.glob(os.path.join(config.HCP_ROOT, "**", "*.mat"), recursive=True)):
        sid = os.path.basename(p).split("_")[0]
        if gmap.get(sid) is None:
            continue
        rows.append((sid, _read_mat(p), gmap[sid], fmap.get(sid, -1)))
    return rows


def _raw_abide():
    rgx = re.compile(r"^(.*)_(\d{6,7})_rois_aal\.1D$")
    rows = []
    for cls, lab in (("Healthy", 0), ("ASD", 1)):
        for p in sorted(glob.glob(os.path.join(config.ABIDE_ROOT, cls, "*_rois_aal.1D"))):
            m = rgx.match(os.path.basename(p))
            if not m:
                continue
            sig = _read_1d(p, "abide")
            if sig.shape[1] == 116 and sig.shape[0] >= 80:
                rows.append((m.group(2), sig, lab, m.group(1)))   # group = site string
    return rows


def _raw_adhd():
    pheno = pd.read_csv(config.ADHD_PHENO, sep="\t")
    site = {int(r["ScanDir ID"]): int(r["Site"]) for _, r in pheno.iterrows() if pd.notna(r["ScanDir ID"])}
    seen = {}
    for cls, lab in (("0", 0), ("1", 1)):
        for p in sorted(glob.glob(os.path.join(config.ADHD_ROOT, cls, "*session_1_rest_1*aal_TCs.1D"))):
            m = re.search(r"da(\d+)_", os.path.basename(p))
            if m:
                seen.setdefault(m.group(1), (lab, p))
    rows = []
    for sid, (lab, p) in seen.items():
        if int(sid) not in site:
            continue
        sig = _read_1d(p, "adhd")
        if sig.shape[1] == 116 and sig.shape[0] >= 80:
            rows.append((sid, sig, lab, site[int(sid)]))
    return rows


_SPEC = {
    "hcp_sex": (_raw_hcp,   "bin"),
    "abide":   (_raw_abide, "bin"),
    "adhd":    (_raw_adhd,  "bin"),
}


def load_dataset(name, cache=True):
    """Return dict: X (N,D) float32, y (N,), groups (N,) int, task, V, ids."""
    cf = os.path.join(CACHE, f"{name}.npz")
    if cache and os.path.exists(cf):
        d = np.load(cf, allow_pickle=True)
        return dict(X=d["X"], y=d["y"], groups=d["groups"], task=str(d["task"]),
                    V=int(d["V"]), ids=d["ids"])
    raw_fn, task = _SPEC[name]
    rows = raw_fn()
    V = rows[0][1].shape[1]
    tri = np.triu_indices(V, k=1)
    X = np.stack([fc_features(sig, tri) for _, sig, _, _ in rows])
    y = np.array([r[2] for r in rows], dtype=np.float32)
    ids = np.array([r[0] for r in rows])
    graw = [r[3] for r in rows]
    uniq = {g: i for i, g in enumerate(sorted(set(map(str, graw))))}
    groups = np.array([uniq[str(g)] for g in graw], dtype=np.int64)
    # 1 row == 1 subject (ids globally unique) so subject-level CV cannot leak
    assert len(set(ids.tolist())) == len(ids), f"{name}: non-unique subject ids"
    d = dict(X=X.astype(np.float32), y=y, groups=groups, task=task, V=V, ids=ids)
    if cache:
        np.savez(cf, **d)
    return d


# ---------- effective rank ----------
def effective_rank(X):
    """Shannon effective rank and participation ratio on centered X (N, D)."""
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2
    lam = lam[lam > 0]
    p = lam / lam.sum()
    shannon = float(np.exp(-(p * np.log(p)).sum()))
    pr = float((lam.sum() ** 2) / (lam ** 2).sum())
    return shannon, pr, lam


# ---------- model ----------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, task="bin", dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _metric(task, y, pred):
    if task == "bin":
        try:
            return roc_auc_score(y, pred)
        except ValueError:
            return float("nan")
    return r2_score(y, pred)


def train_with_curves(Xtr, ytr, Xval, yval, Xte, yte, task="bin", hidden=256,
                      epochs=500, lr=1e-3, weight_decay=0.0, dropout=0.0, seed=0,
                      record=True, interp_thresh=1e-4):
    """Full-batch Adam MLP. Returns per-epoch train/val/test loss+metric, the
    interpolation epoch, and final metrics."""
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6

    def norm(A):
        return (A - mu) / sd

    Xtr_t = torch.tensor(norm(Xtr), device=DEV)
    Xval_t = torch.tensor(norm(Xval), device=DEV)
    Xte_t = torch.tensor(norm(Xte), device=DEV)
    if task == "reg":
        ym, ys = ytr.mean(), ytr.std() + 1e-6
    else:
        ym, ys = 0.0, 1.0
    ytr_t = torch.tensor((ytr - ym) / ys, device=DEV, dtype=torch.float32)
    m = MLP(Xtr.shape[1], hidden, task, dropout).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.BCEWithLogitsLoss() if task == "bin" else nn.MSELoss()
    hist = {k: [] for k in ("tr_loss", "va_loss", "te_loss", "tr_m", "va_m", "te_m")}
    interp = None

    def evalset(Xt, y_raw):
        m.eval()
        with torch.no_grad():
            o = m(Xt)
            if task == "bin":
                loss = nn.functional.binary_cross_entropy_with_logits(
                    o, torch.tensor(y_raw, device=DEV, dtype=torch.float32)).item()
                pred = torch.sigmoid(o).cpu().numpy()
            else:
                pred_raw = o.cpu().numpy() * ys + ym
                loss = float(np.mean((pred_raw - y_raw) ** 2))
                pred = pred_raw
        return loss, _metric(task, y_raw, pred)

    for ep in range(epochs):
        m.train()
        opt.zero_grad()
        out = m(Xtr_t)
        loss = lossf(out, ytr_t)
        loss.backward()
        opt.step()
        if record or ep == epochs - 1:
            trl, trm = evalset(Xtr_t, ytr)
            val, vam = evalset(Xval_t, yval)
            tel, tem = evalset(Xte_t, yte)
            for k, v in zip(hist, (trl, val, tel, trm, vam, tem)):
                hist[k].append(v)
            if interp is None and trl < interp_thresh:
                interp = ep
    final = dict(tr_loss=hist["tr_loss"][-1], te_m=hist["te_m"][-1],
                 va_m=hist["va_m"][-1], tr_m=hist["tr_m"][-1],
                 gap=hist["tr_m"][-1] - hist["te_m"][-1], interp=interp,
                 n_params=sum(p.numel() for p in m.parameters()))
    return dict(hist=hist, **final)


# ---------- CV splitters ----------
def make_folds(y, groups=None, mode="subject", n_splits=5, seed=1):
    """Yield (train_idx, test_idx). mode: subject (stratified) | group | loso."""
    idx = np.arange(len(y))
    if mode == "subject":
        yb = (y > np.median(y)).astype(int) if len(set(y.tolist())) > 10 else y.astype(int)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(skf.split(idx, yb))
    if mode == "group":
        return list(GroupKFold(n_splits=n_splits).split(idx, y, groups))
    if mode == "loso":
        # Keep sites with >= 20 subjects, matching the cross-site MIN_N=20 so LOSO
        # and the subspace/Mantel analyses run on the same 16-site ABIDE subset.
        out = []
        for g in sorted(set(groups)):
            te = groups == g
            if te.sum() >= 20 and (len(set(y[te])) > 1):
                out.append((idx[~te], idx[te]))
        return out
    raise ValueError(mode)


def inner_val_split(train_idx, y, seed=1, frac=0.2):
    """Carve a validation set out of train_idx."""
    rng = np.random.default_rng(seed)
    ti = train_idx.copy()
    rng.shuffle(ti)
    n_val = max(2, int(frac * len(ti)))
    return ti[n_val:], ti[:n_val]
