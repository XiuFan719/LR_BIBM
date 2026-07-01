"""Partial Mantel (MRQAP) test: does per-site subspace overlap predict cross-site
transfer AUROC after controlling for site-level confounds?
Confounds are pairwise distance matrices over ABIDE sites with n>=20: sample-size
similarity, class-balance distance, age distance, sex-ratio distance, and site
mean-FC distance. Permutation p-values come from joint relabelings of the response
matrix.
"""
import os, numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from fc_lib import load_dataset, OUT, CACHE

K, MIN_N, NP = 10, 20, 5000
d = load_dataset('abide'); X, y, site, ids = d['X'], d['y'], d['groups'], d['ids']
ph = pd.read_csv(os.path.join(CACHE, 'abide_pheno.csv'))
age = {int(r.SUB_ID): r.AGE_AT_SCAN for r in ph.itertuples()}
male = {int(r.SUB_ID): (1.0 if r.SEX == 1 else 0.0) for r in ph.itertuples()}
agev = np.array([age.get(int(i), np.nan) for i in ids])
malev = np.array([male.get(int(i), np.nan) for i in ids])

vc = pd.Series(site).value_counts(); S = sorted(vc[vc >= MIN_N].index)
Xs = {s: X[site == s] for s in S}; ys = {s: y[site == s] for s in S}
n_s = {s: int((site == s).sum()) for s in S}
asd = {s: float(ys[s].mean()) for s in S}
agm = {s: float(np.nanmean(agev[site == s])) for s in S}
sxm = {s: float(np.nanmean(malev[site == s])) for s in S}
fcm = {s: Xs[s].mean(0) for s in S}


def subspace(A, k=K):
    A = A - A.mean(0); _, _, Vt = np.linalg.svd(A, full_matrices=False)
    return Vt[:min(k, A.shape[0] - 1)].T


def overlap(Va, Vb):
    sv = np.clip(np.linalg.svd(Va.T @ Vb, compute_uv=False), 0, 1)
    return float((sv ** 2).mean())


def transfer(s, t):
    sc = StandardScaler().fit(Xs[s])
    clf = LogisticRegression(max_iter=300).fit(sc.transform(Xs[s]), ys[s])
    if len(set(ys[t])) < 2:
        return np.nan
    return roc_auc_score(ys[t], clf.predict_proba(sc.transform(Xs[t]))[:, 1])


# Build overlap, transfer and confound distance matrices over sites.
M = len(S); sub = {s: subspace(Xs[s]) for s in S}
OV = np.zeros((M, M)); TR = np.full((M, M), np.nan)
SZ = np.zeros((M, M)); CB = np.zeros((M, M)); AG = np.zeros((M, M)); SX = np.zeros((M, M)); FC = np.zeros((M, M))
for i, s in enumerate(S):
    for j, t in enumerate(S):
        if i == j:
            continue
        OV[i, j] = overlap(sub[s], sub[t]); TR[i, j] = transfer(s, t)
        SZ[i, j] = abs(n_s[s] - n_s[t]); CB[i, j] = abs(asd[s] - asd[t])
        AG[i, j] = abs(agm[s] - agm[t]); SX[i, j] = abs(sxm[s] - sxm[t])
        FC[i, j] = np.linalg.norm(fcm[s] - fcm[t])

valid = (~np.eye(M, dtype=bool)) & np.isfinite(TR)
names = ['overlap', 'size_dist', 'classbal_dist', 'age_dist', 'sex_dist', 'meanfc_dist']
mats = [OV, SZ, CB, AG, SX, FC]


def z(v):
    return (v - v.mean()) / (v.std() + 1e-12)

yv = z(TR[valid])
preds = np.column_stack([z(m[valid]) for m in mats])
A = np.column_stack([np.ones(len(yv)), preds])
beta, *_ = np.linalg.lstsq(A, yv, rcond=None)

print("=== Marginal correlation with transfer AUROC ===")
for k, nm in enumerate(names):
    r = stats.pearsonr(z(mats[k][valid]), yv).statistic
    print(f"  {nm:14s} marginal r={r:+.3f}   partial beta={beta[k+1]:+.3f}")

# MRQAP: permute site labels jointly on the response matrix to respect dyadic structure.
rng = np.random.default_rng(0); cnt = {k: 0 for k in range(len(names))}
obs = beta[1:]
for _ in range(NP):
    pi = rng.permutation(M)
    yp = z(TR[np.ix_(pi, pi)][valid])
    bp, *_ = np.linalg.lstsq(A, yp, rcond=None)
    for k in range(len(names)):
        if abs(bp[k + 1]) >= abs(obs[k]):
            cnt[k] += 1
pvals = {names[k]: (cnt[k] + 1) / (NP + 1) for k in range(len(names))}
print(f"\n=== MRQAP partial test (overlap controlling all confounds, {NP} perms) ===")
print(f"  overlap: partial beta={beta[1]:+.3f}  MRQAP p={pvals['overlap']:.4f}")
print(f"  (overlap stays significant after controlling size/class-balance/age/sex/mean-FC)"
      if pvals['overlap'] < 0.05 else "  (overlap NOT significant after control)")

# Check whether overlap merely proxies a confound.
print("\n=== overlap vs each confound (is overlap just a proxy?) ===")
for k in range(1, len(names)):
    print(f"  corr(overlap, {names[k]}) = {stats.pearsonr(z(OV[valid]), z(mats[k][valid])).statistic:+.3f}")

rows = [dict(predictor=names[k], marginal_r=round(stats.pearsonr(z(mats[k][valid]), yv).statistic, 3),
             partial_beta=round(beta[k + 1], 3), mrqap_p=round(pvals[names[k]], 4)) for k in range(len(names))]
pd.DataFrame(rows).to_csv(os.path.join(OUT, 's5_partial_mantel.csv'), index=False)
print('\nsaved s5_partial_mantel.csv')
