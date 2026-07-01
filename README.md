# Effective Rank and Subspace Alignment in Functional Connectivity Classification

Code for the effective-rank and cross-site subspace-alignment diagnostics. The
analysis covers three public resting-state fMRI datasets (HCP, ABIDE, ADHD-200).
No data is included.

## Setup

```bash
pip install -r requirements.txt
export DATA_ROOT=/path/to/datasets      # see config.py for the expected layout
```

Datasets are obtained from their official sources (HCP S1200; ABIDE; ADHD-200).
`config.py` defines where each dataset lives; edit it or set `DATA_ROOT`.
Per-subject features are cached under `./cache` on first load; all intermediate
tables/figures are written under `./out` and `./figures`.

## Pipeline

| Step | Command | Produces (paper artifact) |
|------|---------|---------------------------|
| Effective rank, reference fit, controls | `python run_main.py` | `exp3_effective_rank_table.csv` (Table I), `exp4_controls.csv` (Table III) |
| Architecture comparison | `python architecture.py` then `python bnt.py` | `exp6_architecture_comparison.csv`, `exp6c_bnt.csv` (Table II) |
| Top-k PC decoding | `python topk_decoding.py` | `exp_topk_pc_decoding.csv` (Fig. top-k) |
| Raw-BOLD prep + BOLD-vs-FC ridge | `python bold_prep.py` | `bold_vs_fc_er.csv`, `bold_vs_fc_ridge.csv` |
| BOLD temporal CNN + interaction | `python bold_fc_interaction.py --datasets hcp_sex abide adhd` | `bold_fc_interaction_summary.csv` (Fig. BOLD-FC) |
| Cross-site subspace + Mantel | `python cross_site_alignment.py` | `s1_*`, `s5_mrqap_design_long.csv` (Fig. cross-site) |
| Partial Mantel / MRQAP | `python partial_mantel.py` | `s5_partial_mantel.csv` |
| Per-site effective rank (sample-size matched) | `python per_site_er_subsampled.py` | `abide_per_site_er_subsampled.csv` |
| Orientation surgery (causal cross-site test) | `python orientation_surgery.py` | `orientation_surgery_points.csv`, `orientation_surgery_per_site.csv` (per-site paired test) |
| Figures | `python make_figures.py` | `figures/fig_topk_decoding.pdf` (= paper `pca.pdf`), `figures/fig_bold_fc_interaction.pdf` (= `interaction.pdf`), `figures/fig_cross_site_subspace.pdf` (= `fig4_subspace.pdf`) |
| Orientation figures | `python plot_orientation_surgery.py` | `figures/fig5_orientation_doseresponse.pdf`, `figures/fig6_orientation_sitelaw.pdf` |

`run_main.py` aggregates the architecture table only after `architecture.py` and
`bnt.py` have been run. `make_figures.py` reads the CSVs produced above; the
significance markers on the BOLD-FC panel are drawn only if
`out/3x3_1002/bold_fc_interaction_tests.csv` is present.

## Shared modules

- `config.py` — dataset paths.
- `fc_lib.py` — FC feature construction, Shannon effective rank, the MLP, and the
  cross-validation splitters (subject-stratified, family-group, leave-one-site-out).

## Notes for anonymous submission

This folder contains no author, institution, or local-path information. If you
publish it for double-blind review, initialise a fresh repository (do not import
existing git history, which carries commit-author metadata) and do not commit
`data/`, `cache/`, or `out/` (see `.gitignore`). A front-end proxy such as
`anonymous.4open.science` hides the hosting account but not repository contents,
so the cleaning above is still required.
