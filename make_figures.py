"""Regenerate the three data figures from the analysis CSVs under out/3x3_1002.

  fig_bold_fc_interaction.pdf  (paper: interaction.pdf) <- run_main.py + bold_prep.py + bold_fc_interaction.py
  fig_topk_decoding.pdf        (paper: pca.pdf)         <- topk_decoding.py
  fig_cross_site_subspace.pdf  (paper: fig4_subspace.pdf) <- cross_site_alignment.py + partial_mantel.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from fc_lib import OUT as _OUT

OUT = Path(_OUT) / "3x3_1002"
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

COL_FC = "#9EC3D8"
COL_BOLD = "#46688A"
COL_TEMP = "#009E73"
COL_GRAY = "#6B7280"
COL_FULL = "#111827"


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=7, length=3, width=0.7)
    ax.grid(axis="y", color="#E5E7EB", lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax, label):
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left")


def make_rank_interaction():
    er = pd.read_csv(OUT / "bold_vs_fc_er.csv")
    inter = pd.read_csv(OUT / "bold_fc_interaction_summary.csv")
    tests_path = OUT / "bold_fc_interaction_tests.csv"
    tests = pd.read_csv(tests_path) if tests_path.exists() else None

    nice = {"hcp_sex": "HCP-sex", "abide": "ABIDE", "adhd": "ADHD-200"}
    order = ["hcp_sex", "abide", "adhd"]
    labels = [nice[d] for d in order]

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55),
                             gridspec_kw={"wspace": 0.36, "width_ratios": [0.9, 1.55]})

    ax = axes[0]
    x = np.arange(len(order))
    width = 0.34
    fc_vals, bold_vals = [], []
    for d in order:
        sub = er[er["dataset"] == d]
        fc_vals.append(float(sub[sub["representation"].str.startswith("FC")]["ER_over_N"].iloc[0]) * 100)
        bold_vals.append(float(sub[sub["representation"].str.startswith("BOLD")]["ER_over_N"].iloc[0]) * 100)
    ax.bar(x - width / 2, fc_vals, width, color=COL_FC, edgecolor="white", linewidth=0.6, label="FC", zorder=3)
    ax.bar(x + width / 2, bold_vals, width, color=COL_BOLD, edgecolor="white", linewidth=0.6, label="BOLD", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Effective rank / N (%)", fontsize=8)
    ax.set_ylim(0, 92)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2)
    style_axes(ax)
    panel_label(ax, "a")

    ax = axes[1]

    def series(rep, model):
        vals, errs = [], []
        for d in order:
            row = inter[(inter["dataset"] == d) & (inter["representation"] == rep) & (inter["model"] == model)]
            if row.empty:
                vals.append(np.nan)
                errs.append(0)
            else:
                vals.append(float(row["test_AUROC"].iloc[0]))
                errs.append(float(row["std"].iloc[0]))
        return np.array(vals), np.array(errs)

    fc, fc_err = series("FC_FisherZ_uppertri", "RidgeClassifier")
    bold_ridge, bold_ridge_err = series("BOLD_TS80_flatten", "RidgeClassifier")
    bold_temp, bold_temp_err = series("BOLD_TS80_timeseries", "TemporalCNN_c64")
    off_fc, off_br, off_bt = -0.22, 0.00, 0.22

    ax.errorbar(x + off_fc, fc, yerr=fc_err, color=COL_FC, marker="o", mfc=COL_FC, mec="#2F465E",
                mew=0.6, lw=1.5, ls="-", capsize=2, zorder=4)
    ax.errorbar(x + off_br, bold_ridge, yerr=bold_ridge_err, color=COL_BOLD, marker="o", mfc="white",
                mec=COL_BOLD, mew=0.9, lw=1.3, ls="--", capsize=2, zorder=4)
    ax.errorbar(x + off_bt, bold_temp, yerr=bold_temp_err, color=COL_BOLD, marker="s", mfc=COL_BOLD,
                mec=COL_BOLD, mew=0.6, lw=1.6, ls="-", capsize=2, zorder=4)
    ax.axhline(0.5, color="#9CA3AF", lw=0.8, ls="--", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Test AUROC", fontsize=8)
    ax.set_ylim(0.45, 0.96)
    ax.set_xlim(-0.55, len(order) - 0.45)

    def sig_text(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

    if tests is not None:
        for d in order:
            r1 = tests[(tests["dataset"] == d) & (tests["comparison"] == "TemporalCNN_vs_BOLD_Ridge")]
            if r1.empty:
                continue
            r1 = r1.iloc[0]
            i = order.index(d)
            x1, x2 = x[i] + off_br, x[i] + off_bt
            y = max(bold_ridge[i] + bold_ridge_err[i], bold_temp[i] + bold_temp_err[i]) + 0.025
            h = 0.010
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=COL_BOLD, lw=0.75, clip_on=False)
            ax.text((x1 + x2) / 2, y + h + 0.006, sig_text(float(r1["p"])), ha="center", va="bottom",
                    fontsize=7, color=COL_BOLD)

    handles = [
        Line2D([0], [0], color=COL_FC, marker="o", mfc=COL_FC, mec="#2F465E", lw=1.5, label="FC, ridge"),
        Line2D([0], [0], color=COL_BOLD, marker="o", mfc="white", mec=COL_BOLD, lw=1.3, ls="--", label="BOLD, ridge"),
        Line2D([0], [0], color=COL_BOLD, marker="s", mfc=COL_BOLD, mec=COL_BOLD, lw=1.6, label="BOLD, temporal CNN"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper left",
              bbox_to_anchor=(0.0, 1.18), ncol=3, columnspacing=1.1, handlelength=2.0)
    style_axes(ax)
    panel_label(ax, "b")

    fig.savefig(FIG / "fig_bold_fc_interaction.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig_bold_fc_interaction.pdf")


def _read_matrix(path):
    df = pd.read_csv(path, index_col=0)
    labels = [s.split(":", 1)[0] for s in df.index.astype(str)]
    return df.to_numpy(dtype=float), labels


def make_cross_site():
    ov, labels = _read_matrix(OUT / "s1_pairwise_overlap_matrix_k10.csv")
    tr, _ = _read_matrix(OUT / "s1_pairwise_transfer_matrix.csv")
    valid = (~np.eye(len(labels), dtype=bool)) & np.isfinite(tr)
    design = pd.read_csv(OUT / "s5_mrqap_design_long.csv")

    y = design["transfer_AUROC"].to_numpy(dtype=float)
    x = design["overlap"].to_numpy(dtype=float)
    covars = ["size_dist", "classbal_dist", "age_dist", "sex_dist", "meanfc_dist"]
    C = design[covars].to_numpy(dtype=float)
    C = (C - C.mean(0)) / (C.std(0) + 1e-12)
    A = np.column_stack([np.ones(len(C)), C])
    xr = (x - x.mean()) / (x.std() + 1e-12) - A @ np.linalg.lstsq(A, (x - x.mean()) / (x.std() + 1e-12), rcond=None)[0]
    yr = (y - y.mean()) / (y.std() + 1e-12) - A @ np.linalg.lstsq(A, (y - y.mean()) / (y.std() + 1e-12), rcond=None)[0]

    align = pd.read_csv(OUT / "s1_cross_site_alignment.csv")
    main = align[align["k"] == 10].iloc[0]
    mrqap = pd.read_csv(OUT / "s5_partial_mantel.csv")
    core = mrqap[(mrqap["model"] == "core_confounds") & (mrqap["predictor"] == "overlap")].iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.25), gridspec_kw={"wspace": 0.38})

    ax = axes[0]
    im = ax.imshow(ov, cmap="viridis", vmin=0.08, vmax=0.32)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_title("Subspace overlap", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=6)
    panel_label(ax, "a")

    ax = axes[1]
    xx, yy = ov[valid], tr[valid]
    ax.scatter(xx, yy, s=13, color=COL_FC, alpha=0.6, edgecolor="none")
    b, a = np.polyfit(xx, yy, 1)
    grid = np.linspace(xx.min(), xx.max(), 100)
    ax.plot(grid, a + b * grid, color="black", lw=1)
    ax.set_xlabel("Principal-angle overlap", fontsize=8)
    ax.set_ylabel("Transfer AUROC", fontsize=8)
    ax.set_title(f"Mantel r={main.mantel_r:.3f}, p={main.mantel_p:.3f}", fontsize=8)
    style_axes(ax)
    panel_label(ax, "b")

    ax = axes[2]
    ax.scatter(xr, yr, s=13, color=COL_TEMP, alpha=0.65, edgecolor="none")
    b, a = np.polyfit(xr, yr, 1)
    grid = np.linspace(xr.min(), xr.max(), 100)
    ax.plot(grid, a + b * grid, color="black", lw=1)
    ax.set_xlabel("Residualized overlap", fontsize=8)
    ax.set_ylabel("Residualized transfer", fontsize=8)
    ax.set_title(f"MRQAP beta={core.partial_beta:.3f}, p={core.mrqap_p:.3f}", fontsize=8)
    style_axes(ax)
    panel_label(ax, "c")

    fig.savefig(FIG / "fig_cross_site_subspace.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig_cross_site_subspace.pdf")


def make_topk_decoding():
    df = pd.read_csv(OUT / "exp_topk_pc_decoding.csv")
    nice = {"hcp_sex": "HCP", "abide": "ABIDE", "adhd": "ADHD-200"}
    order = ["hcp_sex", "abide", "adhd"]
    colors = {"hcp_sex": COL_FC, "abide": COL_TEMP, "adhd": COL_BOLD}

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw={"wspace": 0.5})
    for col, (ax, ds) in enumerate(zip(axes, order)):
        sub = df[df["dataset"] == ds]
        er = float(sub["ER"].iloc[0])
        k2 = round(2 * er)
        c = colors[ds]

        topk = sub[(~sub["is_band"]) & (sub["k_value"] != "full")].copy()
        topk["k_int"] = topk["k_value"].astype(int)
        topk = topk[topk["k_int"] <= k2].sort_values("k_int")
        kx = topk["k_int"].to_numpy(float)
        ky = topk["auroc_mean"].to_numpy(float)
        ks = topk["auroc_std"].to_numpy(float)

        full = sub[sub["k_value"] == "full"].iloc[0]
        fa = float(full["auroc_mean"])

        yall = np.r_[ky, fa, 0.5]
        lo = max(0.44, float(yall.min()) - 0.03)
        hi = float(yall.max()) + 0.075
        ax.set_ylim(lo, hi)

        ax.set_xscale("log")
        x_hi = k2 * 1.7
        ax.set_xlim(3.6, x_hi)

        ax.plot(kx, ky, marker="o", ms=3.5, lw=1.5, color=c, zorder=4)
        ax.fill_between(kx, ky - ks, ky + ks, color=c, alpha=0.15, lw=0, zorder=2)
        ax.axhline(fa, color=COL_FULL, lw=1.1, ls="--", zorder=3)            # full-FC reference
        ax.axvline(er, color="#111827", lw=0.7, ls=":", zorder=1)           # ER
        ax.axvline(k2, color="#111827", lw=0.7, ls=":", zorder=1)           # 2ER
        ax.axhline(0.5, color="#9CA3AF", lw=0.8, ls="--", zorder=1)         # chance

        style_axes(ax)
        ax.tick_params(axis="both", labelsize=9, length=3, width=0.7)
        drop = {round(er), k2}
        ticks = [int(t) for t in kx]
        ticklabels = ["" if t in drop else str(t) for t in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels, fontsize=7, rotation=25, ha="right")
        from matplotlib.ticker import NullLocator
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="x", which="minor", length=0)
        ax.set_title(nice[ds], fontsize=10)
        ax.set_xlabel("Number of PCs ($k$)", fontsize=9.5)
        if col == 0:
            ax.set_ylabel("Test AUROC", fontsize=9.5)
        panel_label(ax, "ABC"[col])

    proxies = [
        Line2D([0], [0], color=COL_GRAY, marker="o", ms=3.5, lw=1.5, label="Top-$k$ PC decoding"),
        Line2D([0], [0], color=COL_FULL, lw=1.1, ls="--", label="Full FC"),
    ]
    fig.legend(proxies, [p.get_label() for p in proxies], loc="upper center", ncol=2,
               fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 1.10), columnspacing=1.5, handlelength=1.8)
    fig.savefig(FIG / "fig_topk_decoding.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig_topk_decoding.pdf")


def main():
    make_rank_interaction()
    make_topk_decoding()
    make_cross_site()
    print(f"[done] figures -> {FIG}")


if __name__ == "__main__":
    main()
