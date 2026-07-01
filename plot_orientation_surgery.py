"""Figures for the orientation-surgery analysis (paper fig5 / fig6).

Reads out/3x3_1002/orientation_surgery_points.csv (written by orientation_surgery.py)
and regenerates:

  fig5_orientation_doseresponse.pdf   two panels:
     (A) transfer AUROC vs principal-angle overlap when the leading k=10 subspace
         is rotated (arm A); an orange band marks the real cross-site overlap (k=10).
     (B) transfer AUROC vs rotation angle for the discriminative vs the
         displacement-matched label-orthogonal control.
  fig6_orientation_sitelaw.pdf        per-site "site law": orientation effect
     (-Spearman rho of transfer vs theta) against within-site transfer at theta=0,
     for the discriminative arm vs the label-orthogonal control.

CSV columns: site, rep, theta, armA, active, placebo, ovA,
             disp_armA, disp_active, disp_placebo.

NOTE (reconstruction): the paper figures were produced by a script that was not
kept; this regenerates them from the CSV. If the styling drifts from the archived
PDFs, adjust the constants below (REAL_OVERLAP_MEAN/STD, colours, band widths).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats

from fc_lib import OUT as _OUT

OUT = Path(_OUT) / "3x3_1002"
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

CSV = OUT / "orientation_surgery_points.csv"

# Colours: discriminative / arm A = slate blue; label-orthogonal control = teal.
COL_DISC = "#46688A"
COL_NULL = "#009E73"
COL_BAND = "#B7C4D2"
COL_ORANGE = "#E69F00"

# Mean +/- spread of the real cross-site principal-angle overlap at k=10
# (Fig. 4 reports mean overlap ~0.20). Used only to draw the reference band in 5A.
REAL_OVERLAP_MEAN = 0.20
REAL_OVERLAP_STD = 0.015


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8, length=3, width=0.7)
    ax.grid(color="#E5E7EB", lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax, label):
    """Bold panel letter in the top-left corner, no parentheses (matches make_figures)."""
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left")


def make_doseresponse(df):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), gridspec_kw={"wspace": 0.28})

    # ---- Panel A: transfer vs overlap under leading-subspace rotation (arm A) ----
    g = df.groupby("theta")
    ov_x = g["ovA"].mean().to_numpy()
    a_y = g["armA"].mean().to_numpy()
    a_s = np.nan_to_num(g["armA"].std(ddof=0).to_numpy())

    ax = axes[0]
    order = np.argsort(-ov_x)                       # high overlap (theta=0) first
    ov_x, a_y, a_s = ov_x[order], a_y[order], a_s[order]
    ax.fill_between(ov_x, a_y - a_s, a_y + a_s, color=COL_BAND, alpha=0.55, lw=0, zorder=2)
    ax.plot(ov_x, a_y, marker="o", ms=5, lw=1.8, color=COL_DISC, zorder=4)
    ax.axvspan(REAL_OVERLAP_MEAN - REAL_OVERLAP_STD, REAL_OVERLAP_MEAN + REAL_OVERLAP_STD,
               color=COL_ORANGE, alpha=0.30, lw=0, zorder=1)
    ax.text(REAL_OVERLAP_MEAN, 0.55, "real cross-site\noverlap (k=10)", color="#9A6A00",
            fontsize=7.5, ha="center", va="center")
    ax.axhline(0.5, color="#9CA3AF", lw=0.9, ls="--", zorder=1)
    ax.invert_xaxis()                               # overlap decreases as theta grows
    ax.set_xlabel("principal-angle overlap (k=10)", fontsize=9)
    ax.set_ylabel("transfer AUROC", fontsize=9)
    ax.set_title("on-thesis: subspace orientation (k=10)", fontsize=9.5)
    handles = [
        Line2D([0], [0], color=COL_DISC, marker="o", lw=1.8, label="rotate leading subspace"),
        Line2D([0], [0], color="#9CA3AF", lw=0.9, ls="--", label="chance"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower left")
    style_axes(ax)
    panel_label(ax, "A")

    # ---- Panel B: transfer vs rotation angle, discriminative vs control ----
    g = df.groupby("theta")
    th = g["active"].mean().index.to_numpy(dtype=float)
    d_y = g["active"].mean().to_numpy()
    d_s = np.nan_to_num(g["active"].std(ddof=0).to_numpy())
    n_y = g["placebo"].mean().to_numpy()
    n_s = np.nan_to_num(g["placebo"].std(ddof=0).to_numpy())

    ax = axes[1]
    for y, s, col in [(d_y, d_s, COL_DISC), (n_y, n_s, COL_NULL)]:
        ax.fill_between(th, y - s, y + s, color=col, alpha=0.16, lw=0, zorder=2)
    ax.plot(th, d_y, marker="o", ms=5, lw=1.8, color=COL_DISC, zorder=4)
    ax.plot(th, n_y, marker="o", ms=5, lw=1.8, color=COL_NULL, zorder=4)
    ax.axhline(0.5, color="#9CA3AF", lw=0.9, ls="--", zorder=1)
    ax.set_xlabel(r"rotation angle $\theta$ (deg)", fontsize=9)
    ax.set_ylabel("transfer AUROC", fontsize=9)
    ax.set_title("variance/displacement control", fontsize=9.5)
    handles = [
        Line2D([0], [0], color=COL_DISC, marker="o", lw=1.8, label="rotate discriminative direction"),
        Line2D([0], [0], color=COL_NULL, marker="o", lw=1.8, label="rotate label-orthogonal direction"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower left")
    style_axes(ax)
    panel_label(ax, "B")

    fig.savefig(FIG / "fig5_orientation_doseresponse.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig5_orientation_doseresponse.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("[fig] fig5_orientation_doseresponse")


def _site_law(df, arm):
    """One point per site (no pseudo-replication over splits): x = within-site
    transfer at theta=0 (mean over splits); y = -Spearman(theta, arm) over all of
    that site's split x angle points."""
    xs, ys = [], []
    for _, sub in df.groupby("site"):
        base = sub.loc[sub["theta"] == 0, arm]
        if base.empty or sub["theta"].nunique() < 3 or sub[arm].nunique() < 2:
            continue
        rho, _ = stats.spearmanr(sub["theta"], sub[arm])
        if not np.isfinite(rho) or not np.isfinite(base.mean()):
            continue
        xs.append(float(base.mean()))
        ys.append(-float(rho))
    return np.array(xs), np.array(ys)


def make_sitelaw(df):
    fig, ax = plt.subplots(figsize=(6.0, 5.0))

    xd, yd = _site_law(df, "active")            # discriminative arm
    xn, yn = _site_law(df, "placebo")           # label-orthogonal control

    ax.axhline(0.0, color="#9CA3AF", lw=0.8, ls=":", zorder=1)
    ax.scatter(xd, yd, s=55, color=COL_DISC, alpha=0.85, edgecolor="none", zorder=4)
    ax.scatter(xn, yn, s=55, marker="s", color=COL_NULL, alpha=0.85, edgecolor="none", zorder=4)

    def fit_line(x, y, col, ls):
        if len(x) < 2:
            return
        b, a = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 100)
        ax.plot(grid, a + b * grid, color=col, lw=2.0, ls=ls, zorder=3)

    fit_line(xd, yd, COL_DISC, "-")
    fit_line(xn, yn, COL_NULL, "--")

    if len(xd) >= 3:
        pr = stats.pearsonr(xd, yd)
        r_txt = f"r={pr.statistic:.2f} (n={len(xd)})"
        p_val = pr.pvalue
    else:
        r_txt, p_val = "r=n/a", float("nan")
    ax.text(0.97, 0.06, r_txt, transform=ax.transAxes, color=COL_DISC,
            fontsize=10, ha="right", va="bottom")

    ax.set_xlabel(r"within-site transfer ($\theta=0$)", fontsize=10)
    ax.set_ylabel(r"orientation effect ($-\rho$)", fontsize=10)
    handles = [
        Line2D([0], [0], color=COL_DISC, marker="o", lw=2.0, label="rotate discriminative direction"),
        Line2D([0], [0], color=COL_NULL, marker="s", lw=2.0, ls="--",
               label="rotate label-orthogonal direction (matched displacement)"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper left")
    style_axes(ax)

    fig.savefig(FIG / "fig6_orientation_sitelaw.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig6_orientation_sitelaw.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[fig] fig6_orientation_sitelaw  (discriminative {r_txt}, p={p_val:.4f})")


def main():
    if not CSV.exists():
        raise SystemExit(f"missing {CSV}; run orientation_surgery.py first")
    df = pd.read_csv(CSV)
    make_doseresponse(df)
    make_sitelaw(df)
    print(f"[done] orientation figures -> {FIG}")


if __name__ == "__main__":
    main()
