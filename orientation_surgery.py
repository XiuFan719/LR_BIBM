"""Orientation surgery: a within-site causal test of whether subspace ORIENTATION
drives cross-site transfer failure.

Within one ABIDE site, subjects are split into source / target halves. The target
half is perturbed by an orthogonal rotation that moves only orientation -- the
covariance spectrum, mean, and sample identity are preserved (verified). Three
arms:
  Arm A   : rotate the unsupervised leading-k subspace (the cross-site overlap
            diagnostic), swept over angle theta.
  active  : rotate the discriminative direction within the leading subspace.
  placebo : rotate an equal-variance label-orthogonal direction by the same angle
            (the discriminative direction is untouched); active and placebo carry
            matched per-sample displacement, reported empirically.
A logistic decoder trained on the source half is transferred to the rotated target;
transfer AUROC is read as a function of theta / overlap.
Output -> out/3x3_1002/orientation_surgery_points.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from fc_lib import OUT, load_dataset

OUT_NEW = Path(OUT) / "3x3_1002"
OUT_NEW.mkdir(parents=True, exist_ok=True)

K = 10                      # rotated subspace dimension (matches the k=10 overlap diagnostic)
THETAS = [0, 15, 30, 45, 60, 75, 90]
N_SPLITS = 10
MIN_N = 42                  # need each half >= 2K subjects to define the rotation


def lead(Xc, k):
    _, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    lam = (sv ** 2) / (len(Xc) - 1)
    return Vt[:k].T, Vt[k:2 * k].T, lam[:k]            # leading (D,k), next (D,k), eigvals(k)


def overlap(U1, U2):
    s = np.linalg.svd(U1.T @ U2, compute_uv=False)
    return float((np.clip(s, 0, 1) ** 2).mean())


def disc_dir(Z, y):
    e = Z[y == 1].mean(0) - Z[y == 0].mean(0)
    return e / np.linalg.norm(e)


def eq_var_nulls(e_d, lam):
    """Two unit directions (PC coords) orthogonal to e_d and to each other, each
    with data-variance == var(e_d), used to build the matched placebo plane."""
    Kk = len(lam)
    P = np.eye(Kk) - np.outer(e_d, e_d)
    w, Q = np.linalg.eigh(P @ np.diag(lam) @ P)
    keep = w > 1e-9 * w.max()
    w, Q = w[keep], Q[:, keep]
    o = np.argsort(w)[::-1]
    w, Q = w[o], Q[:, o]
    vd = float(e_d @ (lam * e_d))

    def combo(i, j):
        hi, lo = w[i], w[j]
        t = np.clip((vd - lo) / (hi - lo + 1e-12), 0, 1)
        v = np.sqrt(t) * Q[:, i] + np.sqrt(1 - t) * Q[:, j]
        return v / np.linalg.norm(v)

    return combo(0, len(w) - 1), combo(1, len(w) - 2)


def rot_subspace(Xc, M, G):
    delta = (Xc @ M) @ (G - np.eye(M.shape[1]))
    return Xc + delta @ M.T, float(np.sqrt((delta ** 2).sum(1)).mean())


def givens(theta, k):
    G = np.eye(2 * k)
    c, s = np.cos(theta), np.sin(theta)
    for i in range(k):
        G[i, i] = c; G[k + i, k + i] = c; G[i, k + i] = -s; G[k + i, i] = s
    return G


def rot_plane(Xc, p, q, theta):
    """Rotate axis p toward q by theta in the orthonormal (p,q) plane."""
    a, b = Xc @ p, Xc @ q
    c, s = np.cos(theta), np.sin(theta)
    na, nb = a * c - b * s, a * s + b * c
    Xr = Xc + np.outer(na - a, p) + np.outer(nb - b, q)
    return Xr, float(np.sqrt((na - a) ** 2 + (nb - b) ** 2).mean())


def main():
    d = load_dataset("abide")
    X, y, site = d["X"].astype(np.float64), d["y"].astype(np.float64), d["groups"]
    cnt = pd.Series(site).value_counts()
    sites = sorted(s for s in set(site.tolist()) if cnt[s] >= MIN_N)
    print(f"sites (n>={MIN_N}): {sites}")
    rng = np.random.default_rng(0)
    rows, san = [], []
    for s in sites:
        idx = np.where(site == s)[0]
        ys = y[idx]
        for rep in range(N_SPLITS):
            tr, te = [], []
            for c in (0, 1):
                ci = rng.permutation(idx[ys == c]); h = len(ci) // 2
                tr += list(ci[:h]); te += list(ci[h:])
            tr, te = np.array(tr), np.array(te)
            if len(te) < 2 * K or len(tr) < K or min(np.bincount(y[te].astype(int))) < 2:
                continue
            Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
            mu = Xte.mean(0); Xc = Xte - mu
            Ul, Un, lam = lead(Xc, K); Usrc = lead(Xtr - Xtr.mean(0), K)[0]
            Z = Xc @ Ul; e_d = disc_dir(Z, yte); e_a, e_b = eq_var_nulls(e_d, lam)
            p_d, p_a, p_b = Ul @ e_d, Ul @ e_a, Ul @ e_b
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(Xtr), ytr)
            M = np.hstack([Ul, Un])
            for deg in THETAS:
                th = np.deg2rad(deg)
                XA, dA = rot_subspace(Xc, M, givens(th, K)); XA = XA + mu
                XAct, dAct = rot_plane(Xc, p_d, p_a, th); XAct = XAct + mu
                XPla, dPla = rot_plane(Xc, p_a, p_b, th); XPla = XPla + mu

                def tr_auc(Xr):
                    return roc_auc_score(yte, clf.predict_proba(sc.transform(Xr))[:, 1])

                rows.append({"site": int(s), "rep": rep, "theta": deg,
                             "armA": tr_auc(XA), "active": tr_auc(XAct), "placebo": tr_auc(XPla),
                             "ovA": overlap(Usrc, lead(XA - XA.mean(0), K)[0]),
                             "disp_armA": dA, "disp_active": dAct, "disp_placebo": dPla})
                if deg == 90:
                    def top_eig(A):                # top-5 eigvals via small n x n gram
                        G = (A @ A.T) / (A.shape[0] - 1); return np.sort(np.linalg.eigvalsh(G)[-5:])
                    e0 = top_eig(Xc)
                    for nm, Xr in [("A", XA), ("active", XAct), ("placebo", XPla)]:
                        e1 = top_eig(Xr - Xr.mean(0))
                        san.append({"arm": nm, "d_eig": np.abs(e0 - e1).max(), "d_mean": np.abs(mu - Xr.mean(0)).max()})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_NEW / "orientation_surgery_points.csv", index=False)

    print("\n=== invariant sanity (orthogonal rotation; per arm) ===")
    print(pd.DataFrame(san).groupby("arm").agg(max_eig_delta=("d_eig", "max"), max_mean_delta=("d_mean", "max")).to_string())

    dd = df.groupby("theta").agg(d_armA=("disp_armA", "mean"), d_active=("disp_active", "mean"),
                                 d_placebo=("disp_placebo", "mean"), t_armA=("armA", "mean"),
                                 t_active=("active", "mean"), t_placebo=("placebo", "mean")).reset_index()
    print("\n=== mean ||dx|| and transfer per theta (3 arms) ===")
    print(dd.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    base = dd.t_armA.iloc[0]
    common = min(dd.d_armA.max(), dd.d_active.max(), dd.d_placebo.max())

    def drop_at(dc, tc):
        return base - float(np.interp(common, dd[dc], dd[tc]))
    print(f"matched-displacement readout at ||dx||={common:.2f}: "
          f"armA={drop_at('d_armA','t_armA'):+.3f} active={drop_at('d_active','t_active'):+.3f} placebo={drop_at('d_placebo','t_placebo'):+.3f}")

    # Pooled dose-response over all point-level rows. INTERNAL DIAGNOSTIC ONLY:
    # the n = sites x splits x angles points are pseudo-replicated (splits within a
    # site are not independent), so this Spearman p is NOT the statistic reported
    # in the paper. The paper cites the per-site paired test below.
    n = len(df)
    print(f"\n=== pooled dose-response [internal diagnostic, not cited] "
          f"(n={n} = {df.site.nunique()} sites x {df.rep.nunique()} splits x {df.theta.nunique()} angles) ===")
    for col in ["armA", "active", "placebo"]:
        rho, p = stats.spearmanr(df["theta"], df[col])
        drop = df[df.theta == 0][col].mean() - df[df.theta == 90][col].mean()
        print(f"  {col:8s}: drop={drop:+.3f}  Spearman rho={rho:+.3f}  p={p:.1e}")

    # Per-site paired test -- the statistic reported in the paper. One value per
    # site (no pseudo-replication): the theta 0->90 drop of the discriminative arm
    # vs the displacement-matched label-orthogonal control, compared across sites.
    per_site = []
    for s, sub in df.groupby("site"):
        b0, b9 = sub[sub.theta == 0], sub[sub.theta == 90]
        per_site.append({"site": int(s),
                         "act_drop": b0.active.mean() - b9.active.mean(),
                         "pla_drop": b0.placebo.mean() - b9.placebo.mean()})
    ps = pd.DataFrame(per_site)
    ps.to_csv(OUT_NEW / "orientation_surgery_per_site.csv", index=False)
    tt = stats.ttest_rel(ps.act_drop, ps.pla_drop)
    try:
        wil_p = stats.wilcoxon(ps.act_drop, ps.pla_drop).pvalue
    except ValueError:                          # all paired differences zero
        wil_p = float("nan")
    print(f"\n=== per-site paired test [reported] "
          f"(discriminative vs label-orthogonal drop, theta 0->90; n={len(ps)} sites) ===")
    print(f"  mean drop: active={ps.act_drop.mean():+.3f}  placebo={ps.pla_drop.mean():+.3f}")
    print(f"  active-drop>0 in {(ps.act_drop > 0).sum()}/{len(ps)} sites")
    print(f"  paired-t p={tt.pvalue:.4f}   Wilcoxon p={wil_p:.4f}")


if __name__ == "__main__":
    main()
