"""Per-site effective rank after matching sample size.

Whole-site effective rank grows with the number of subjects, so it is not
comparable across sites. This script subsamples every ABIDE site (n>=20) down to
a common n, recomputes the Shannon effective rank, repeats over random draws, and
reports the per-site mean and the across-site spread -- isolating site identity
from sample-size. Output -> out/3x3_1002/abide_per_site_er_subsampled.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from fc_lib import OUT, effective_rank, load_dataset

OUT_NEW = Path(OUT) / "3x3_1002"
OUT_NEW.mkdir(parents=True, exist_ok=True)

MIN_N = 20      # sites kept for the cross-site analysis
SUB_N = 20      # common subsample size
N_DRAWS = 30


def main():
    d = load_dataset("abide")
    X, g = d["X"].astype(np.float64), d["groups"]
    counts = pd.Series(g).value_counts()
    sites = sorted(s for s in set(g.tolist()) if counts[s] >= MIN_N)
    rng = np.random.default_rng(0)

    rows = []
    for s in sites:
        idx = np.where(g == s)[0]
        ers = [effective_rank(X[rng.choice(idx, SUB_N, replace=False)])[0] for _ in range(N_DRAWS)]
        rows.append({"site_id": int(s), "N_full": int(len(idx)),
                     "ER_subsampled_mean": round(float(np.mean(ers)), 2),
                     "ER_subsampled_sd": round(float(np.std(ers)), 2)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_NEW / "abide_per_site_er_subsampled.csv", index=False)

    m = df["ER_subsampled_mean"].to_numpy()
    print(df.to_string(index=False))
    print(f"\nacross {len(sites)} sites at n={SUB_N}: "
          f"mean={m.mean():.2f}  min={m.min():.2f}  max={m.max():.2f}  "
          f"spread={m.max()-m.min():.2f}  std={m.std():.2f}", flush=True)


if __name__ == "__main__":
    main()
