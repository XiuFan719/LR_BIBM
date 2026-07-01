"""ABIDE cross-site alignment analysis.

Computes per-site principal-subspace overlap and pairwise cross-site transfer
AUROC, tests whether overlap predicts transfer via Mantel and partial-Mantel
(MRQAP) controlling for site-level confounds, and writes the result matrices,
tables and figures.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from fc_lib import CACHE, OUT, load_dataset


OUT_NEW = Path(OUT) / "3x3_1002"
OUT_NEW.mkdir(parents=True, exist_ok=True)

K_MAIN = 10
KS = [5, 10, 20]
MIN_N = 20
NP = 5000


def z(v):
    v = np.asarray(v, dtype=float)
    return (v - v.mean()) / (v.std() + 1e-12)


def auc_safe(y, score):
    try:
        return float(roc_auc_score(y, score))
    except ValueError:
        return float("nan")


def site_subspace(X, k):
    A = X - X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    return Vt[:min(k, A.shape[0] - 1)].T


def overlap(Va, Vb):
    sv = np.clip(np.linalg.svd(Va.T @ Vb, compute_uv=False), 0, 1)
    return float((sv ** 2).mean())


def transfer_auroc(Xs, ys, s, t):
    sc = StandardScaler().fit(Xs[s])
    clf = LogisticRegression(max_iter=500, C=1.0).fit(sc.transform(Xs[s]), ys[s])
    if len(set(ys[t].tolist())) < 2:
        return float("nan")
    score = clf.predict_proba(sc.transform(Xs[t]))[:, 1]
    return auc_safe(ys[t], score)


def matrix_csv(path, mat, sites, labels):
    cols = [f"{int(s)}:{labels[int(s)]}" for s in sites]
    df = pd.DataFrame(mat, index=cols, columns=cols)
    df.index.name = "source_or_row_site"
    df.to_csv(path)


def mantel_p(pred, response, valid, n_perm=NP, seed=0):
    # Permute site labels jointly on rows/cols to respect the dyadic structure.
    obs = stats.pearsonr(pred[valid], response[valid]).statistic
    rng = np.random.default_rng(seed)
    m = pred.shape[0]
    cnt = 0
    for _ in range(n_perm):
        pi = rng.permutation(m)
        rp = stats.pearsonr(pred[np.ix_(pi, pi)][valid], response[valid]).statistic
        if abs(rp) >= abs(obs):
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


def main():
    d = load_dataset("abide")
    X, y, site, ids = d["X"], d["y"], d["groups"], d["ids"]

    vc = pd.Series(site).value_counts()
    sites = sorted(vc[vc >= MIN_N].index.tolist())
    labels = {int(s): str(int(s)) for s in sites}
    per_er_path = OUT_NEW / "abide_per_site_er.csv"
    if per_er_path.exists():
        per_er = pd.read_csv(per_er_path)
        labels.update({int(r.site_id): str(r.site_label) for r in per_er.itertuples()})
    else:
        per_er = pd.DataFrame()

    Xs = {s: X[site == s] for s in sites}
    ys = {s: y[site == s] for s in sites}
    M = len(sites)
    print(f"[cross-site] ABIDE sites with n>={MIN_N}: {M}", flush=True)

    # Transfer is k-independent.
    TR = np.full((M, M), np.nan)
    for i, s in enumerate(sites):
        for j, t in enumerate(sites):
            if i != j:
                TR[i, j] = transfer_auroc(Xs, ys, s, t)
    matrix_csv(OUT_NEW / "s1_pairwise_transfer_matrix.csv", TR, sites, labels)

    # Overlap matrices across sensitivity ks.
    overlap_mats = {}
    alignment_rows = []
    valid = (~np.eye(M, dtype=bool)) & np.isfinite(TR)
    for k in KS:
        sub = {s: site_subspace(Xs[s], k) for s in sites}
        OV = np.eye(M)
        for i, s in enumerate(sites):
            for j, t in enumerate(sites):
                if i != j:
                    OV[i, j] = overlap(sub[s], sub[t])
        overlap_mats[k] = OV
        matrix_csv(OUT_NEW / f"s1_pairwise_overlap_matrix_k{k}.csv", OV, sites, labels)

        pear = stats.pearsonr(OV[valid], TR[valid])
        spear = stats.spearmanr(OV[valid], TR[valid])
        row = {
            "k": k,
            "n_sites": M,
            "n_ordered_pairs": int(valid.sum()),
            "mean_overlap": round(float(OV[valid].mean()), 4),
            "mean_transfer_AUROC": round(float(np.nanmean(TR[valid])), 4),
            "pearson_r": round(float(pear.statistic), 4),
            "pearson_p_naive": round(float(pear.pvalue), 6),
            "spearman_rho": round(float(spear.correlation), 4),
            "spearman_p_naive": round(float(spear.pvalue), 6),
        }
        if k == K_MAIN:
            r_obs, p = mantel_p(OV, TR, valid)
            row["mantel_r"] = round(float(r_obs), 4)
            row["mantel_p"] = round(float(p), 4)
        alignment_rows.append(row)

    pd.DataFrame(alignment_rows).to_csv(OUT_NEW / "s1_cross_site_alignment.csv", index=False)

    # Pairwise long table with effective-rank covariates.
    er_map = {}
    ern_map = {}
    if not per_er.empty:
        er_map = {int(r.site_id): float(r.Shannon_ER) for r in per_er.itertuples()}
        ern_map = {int(r.site_id): float(r.ER_over_N) for r in per_er.itertuples()}
    pair_rows = []
    for i, s in enumerate(sites):
        for j, t in enumerate(sites):
            if i == j:
                continue
            row = {
                "source_site": int(s),
                "source_label": labels[int(s)],
                "target_site": int(t),
                "target_label": labels[int(t)],
                "transfer_AUROC": TR[i, j],
            }
            for k in KS:
                row[f"overlap_k{k}"] = overlap_mats[k][i, j]
            if er_map:
                row["source_ER"] = er_map[int(s)]
                row["target_ER"] = er_map[int(t)]
                row["mean_ER"] = 0.5 * (er_map[int(s)] + er_map[int(t)])
                row["absdiff_ER"] = abs(er_map[int(s)] - er_map[int(t)])
                row["source_ER_over_N"] = ern_map[int(s)]
                row["target_ER_over_N"] = ern_map[int(t)]
                row["mean_ER_over_N"] = 0.5 * (ern_map[int(s)] + ern_map[int(t)])
                row["absdiff_ER_over_N"] = abs(ern_map[int(s)] - ern_map[int(t)])
            pair_rows.append(row)
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(OUT_NEW / "s1_pairwise_transfer_long.csv", index=False)

    # MRQAP confound distance matrices.
    ph = pd.read_csv(Path(CACHE) / "abide_pheno.csv")
    age = {int(r.SUB_ID): r.AGE_AT_SCAN for r in ph.itertuples()}
    male = {int(r.SUB_ID): (1.0 if r.SEX == 1 else 0.0) for r in ph.itertuples()}
    agev = np.array([age.get(int(i), np.nan) for i in ids])
    malev = np.array([male.get(int(i), np.nan) for i in ids])
    n_s = {s: int((site == s).sum()) for s in sites}
    asd = {s: float(ys[s].mean()) for s in sites}
    agm = {s: float(np.nanmean(agev[site == s])) for s in sites}
    sxm = {s: float(np.nanmean(malev[site == s])) for s in sites}
    fcm = {s: Xs[s].mean(0) for s in sites}

    SZ = np.zeros((M, M))
    CB = np.zeros((M, M))
    AG = np.zeros((M, M))
    SX = np.zeros((M, M))
    FC = np.zeros((M, M))
    ERmean = np.zeros((M, M))
    ERdiff = np.zeros((M, M))
    for i, s in enumerate(sites):
        for j, t in enumerate(sites):
            if i == j:
                continue
            SZ[i, j] = abs(n_s[s] - n_s[t])
            CB[i, j] = abs(asd[s] - asd[t])
            AG[i, j] = abs(agm[s] - agm[t])
            SX[i, j] = abs(sxm[s] - sxm[t])
            FC[i, j] = np.linalg.norm(fcm[s] - fcm[t])
            if er_map:
                ERmean[i, j] = 0.5 * (er_map[int(s)] + er_map[int(t)])
                ERdiff[i, j] = abs(er_map[int(s)] - er_map[int(t)])

    OV = overlap_mats[K_MAIN]

    def fit_mrqap(model_name, names, mats):
        yv = z(TR[valid])
        preds = np.column_stack([z(m[valid]) for m in mats])
        A = np.column_stack([np.ones(len(yv)), preds])
        beta, *_ = np.linalg.lstsq(A, yv, rcond=None)

        # Permute site labels jointly to build the dyadic null.
        rng = np.random.default_rng(0)
        cnt = np.zeros(len(names), dtype=int)
        for _ in range(NP):
            pi = rng.permutation(M)
            yp = z(TR[np.ix_(pi, pi)][valid])
            bp, *_ = np.linalg.lstsq(A, yp, rcond=None)
            cnt += (np.abs(bp[1:]) >= np.abs(beta[1:]))
        pvals = (cnt + 1) / (NP + 1)

        out = []
        for idx, name in enumerate(names):
            out.append({
                "model": model_name,
                "predictor": name,
                "marginal_r": round(float(stats.pearsonr(z(mats[idx][valid]), yv).statistic), 4),
                "partial_beta": round(float(beta[idx + 1]), 4),
                "mrqap_p": round(float(pvals[idx]), 4),
            })
        return out

    core_names = ["overlap", "size_dist", "classbal_dist", "age_dist", "sex_dist", "meanfc_dist"]
    core_mats = [OV, SZ, CB, AG, SX, FC]
    rows = fit_mrqap("core_confounds", core_names, core_mats)
    names = list(core_names)
    mats = list(core_mats)
    if er_map:
        er_names = core_names + ["mean_ER", "absdiff_ER"]
        er_mats = core_mats + [ERmean, ERdiff]
        rows.extend(fit_mrqap("with_er_sensitivity", er_names, er_mats))
        names = er_names
        mats = er_mats
    pd.DataFrame(rows).to_csv(OUT_NEW / "s5_partial_mantel.csv", index=False)

    design_rows = []
    for i, s in enumerate(sites):
        for j, t in enumerate(sites):
            if i == j or not np.isfinite(TR[i, j]):
                continue
            rec = {
                "source_site": int(s),
                "target_site": int(t),
                "transfer_AUROC": TR[i, j],
            }
            for name, mat in zip(names, mats):
                rec[name] = mat[i, j]
            design_rows.append(rec)
    pd.DataFrame(design_rows).to_csv(OUT_NEW / "s5_mrqap_design_long.csv", index=False)

    # Effective-rank vs transfer diagnostic.
    er_rows = []
    er_predictors = []
    if er_map:
        er_predictors = [
            ("source_ER", np.array([[er_map[int(s)] if i != j else np.nan for j, t in enumerate(sites)] for i, s in enumerate(sites)])),
            ("target_ER", np.array([[er_map[int(t)] if i != j else np.nan for j, t in enumerate(sites)] for i, s in enumerate(sites)])),
            ("mean_ER", ERmean),
            ("absdiff_ER", ERdiff),
            ("mean_ER_over_N", np.array([[0.5 * (ern_map[int(s)] + ern_map[int(t)]) if i != j else np.nan for j, t in enumerate(sites)] for i, s in enumerate(sites)])),
            ("absdiff_ER_over_N", np.array([[abs(ern_map[int(s)] - ern_map[int(t)]) if i != j else np.nan for j, t in enumerate(sites)] for i, s in enumerate(sites)])),
        ]
        for name, mat in er_predictors:
            r, p = mantel_p(mat, TR, valid)
            er_rows.append({
                "predictor": name,
                "pearson_r": round(float(stats.pearsonr(mat[valid], TR[valid]).statistic), 4),
                "qap_r": round(float(r), 4),
                "qap_p": round(float(p), 4),
            })
        pd.DataFrame(er_rows).to_csv(OUT_NEW / "s1_er_vs_transfer.csv", index=False)

    # Figures.
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    im = axes[0].imshow(OV, vmin=0, vmax=float(np.nanpercentile(OV[valid], 98)), cmap="viridis")
    axes[0].set_title("Subspace overlap (k=10)")
    axes[0].set_xticks(range(M))
    axes[0].set_yticks(range(M))
    axes[0].set_xticklabels([str(int(s)) for s in sites], rotation=90, fontsize=6)
    axes[0].set_yticklabels([str(int(s)) for s in sites], fontsize=6)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    x = OV[valid]
    yy = TR[valid]
    axes[1].scatter(x, yy, s=16, alpha=0.55)
    b, a = np.polyfit(x, yy, 1)
    xx = np.linspace(x.min(), x.max(), 100)
    axes[1].plot(xx, a + b * xx, color="black", lw=1)
    axes[1].set_xlabel("Subspace overlap")
    axes[1].set_ylabel("Transfer AUROC")
    main_row = [r for r in alignment_rows if r["k"] == K_MAIN][0]
    axes[1].set_title(f"Mantel r={main_row['mantel_r']:.3f}, p={main_row['mantel_p']:.3f}")

    # Residualize overlap and transfer on confounds before plotting.
    cov_mats = [SZ, CB, AG, SX, FC]
    C = np.column_stack([np.ones(valid.sum())] + [z(m[valid]) for m in cov_mats])
    rx = z(OV[valid]) - C @ np.linalg.lstsq(C, z(OV[valid]), rcond=None)[0]
    ry = z(TR[valid]) - C @ np.linalg.lstsq(C, z(TR[valid]), rcond=None)[0]
    axes[2].scatter(rx, ry, s=16, alpha=0.55)
    b, a = np.polyfit(rx, ry, 1)
    xx = np.linspace(rx.min(), rx.max(), 100)
    axes[2].plot(xx, a + b * xx, color="black", lw=1)
    axes[2].set_xlabel("Residualized overlap")
    axes[2].set_ylabel("Residualized transfer")
    overlap_row = pd.DataFrame(rows).query("predictor == 'overlap'").iloc[0]
    axes[2].set_title(f"MRQAP beta={overlap_row.partial_beta:.3f}, p={overlap_row.mrqap_p:.3f}")
    plt.tight_layout()
    plt.savefig(OUT_NEW / "fig4_subspace.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    if er_map:
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
        mean_er = pair_df["mean_ER"].to_numpy()
        diff_er = pair_df["absdiff_ER"].to_numpy()
        tr = pair_df["transfer_AUROC"].to_numpy()
        for ax, vals, title, xlabel in [
            (axes[0], mean_er, "Mean site ER vs transfer", "Mean source/target ER"),
            (axes[1], diff_er, "ER difference vs transfer", "|source ER - target ER|"),
        ]:
            ax.scatter(vals, tr, s=16, alpha=0.55)
            b, a = np.polyfit(vals, tr, 1)
            xx = np.linspace(vals.min(), vals.max(), 100)
            ax.plot(xx, a + b * xx, color="black", lw=1)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Transfer AUROC")
            ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(OUT_NEW / "fig4_er_vs_transfer.pdf", bbox_inches="tight", dpi=300)
        plt.close()

    print("[done] cross-site alignment rerun", flush=True)
    print(pd.DataFrame(alignment_rows).to_string(index=False), flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    if er_map:
        print(pd.DataFrame(er_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
