"""BrainNetworkTransformer (BNT, Kan et al. NeurIPS 2022) baseline comparison.
Uses the same data, splits, metric, and training budget as the other graph/connectome baselines
(Adam lr=1e-3, weight decay 1e-4, mini-batch 64, 100 epochs, 3 seeds x 3-fold subject CV): per-node
connection-profile tokens feed a 2-layer Transformer encoder, an Orthonormal Clustering Readout
(OCRead), and an MLP head. Also logs ABIDE convergence and compares the ABIDE result to the value
reported in the original BNT paper. -> exp6c_bnt.csv
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '8'
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from fc_lib import load_dataset, make_folds, inner_val_split, OUT, DEV
from architecture import fc_matrices   # identical FC->(V,V) matrix builder used by the other baselines

DATASETS = ['hcp_sex', 'abide', 'adhd']
SEEDS = [1, 2, 3]
EPOCHS = 100              # unified with the other graph/connectome baselines
ORTHO_LAMBDA = 1e-2
# original BNT paper (Kan et al. 2022) ABIDE result, for a sanity comparison (different atlas/preproc)
BNT_PAPER_ABIDE_AUROC = 0.766


class BrainNetworkTransformer(nn.Module):
    def __init__(self, V, d=64, heads=4, layers=2, K=8, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(V, d)
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=dropout,
                                         batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(enc, layers)
        self.Wc = nn.Linear(d, K, bias=False)        # OCRead cluster projection
        nn.init.orthogonal_(self.Wc.weight)
        self.head = nn.Sequential(nn.Linear(K * d, 64), nn.LeakyReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, M):                            # M: (B,V,V), node i feature = FC row i
        Z = self.enc(self.proj(M))                   # (B,V,d)
        A = torch.softmax(self.Wc(Z), dim=1)         # (B,V,K) soft node->cluster assignment
        G = A.transpose(1, 2) @ Z                    # (B,K,d) cluster embeddings
        return self.head(G.flatten(1)).squeeze(-1)

    def ortho_loss(self):                            # encourage orthonormal cluster directions
        W = self.Wc.weight                           # (K,d)
        I = torch.eye(W.shape[0], device=W.device)
        return ((W @ W.t() - I) ** 2).mean()


def train_bnt(Mtr, ytr, Mte, yte, seed, V, epochs=EPOCHS, bs=64, log_every=0):
    torch.manual_seed(seed); m = BrainNetworkTransformer(V).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4); lossf = nn.BCEWithLogitsLoss()
    Mtr_t = torch.tensor(Mtr, device=DEV); ytr_t = torch.tensor(ytr, device=DEV); n = len(Mtr)

    def pred(Mx):
        out = []
        with torch.no_grad():
            for i in range(0, len(Mx), 128):
                out.append(torch.sigmoid(m(torch.tensor(Mx[i:i + 128], device=DEV))).cpu().numpy())
        return np.concatenate(out)

    log = []
    for ep in range(1, epochs + 1):
        m.train(); perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(m(Mtr_t[idx]), ytr_t[idx]) + ORTHO_LAMBDA * m.ortho_loss()
            loss.backward(); opt.step()
        if log_every and (ep == 1 or ep % log_every == 0 or ep == epochs):
            m.eval(); log.append((ep, roc_auc_score(yte, pred(Mte))))
    m.eval()
    return roc_auc_score(yte, pred(Mte)), roc_auc_score(ytr, pred(Mtr)), log


def main():
    rows = []
    for name in DATASETS:
        d = load_dataset(name); X, y, V = d['X'], d['y'], d['V']; Mfc = fc_matrices(d)
        per_seed_te, per_seed_tr = [], []
        for sd in SEEDS:
            fte, ftr = [], []
            for tr_all, te in make_folds(y, mode='subject', n_splits=3, seed=sd):
                tr, _va = inner_val_split(tr_all, y, seed=sd)
                a, b, _ = train_bnt(Mfc[tr], y[tr], Mfc[te], y[te], sd, V)
                fte.append(a); ftr.append(b)
            per_seed_te.append(np.mean(fte)); per_seed_tr.append(np.mean(ftr))
        te = np.array(per_seed_te); tr = np.array(per_seed_tr)
        rows.append(dict(dataset=name, model='BrainNetworkTransformer',
                         test_AUROC=round(te.mean(), 3), test_std=round(te.std(), 3),
                         train_AUROC=round(tr.mean(), 3), gen_gap=round(tr.mean() - te.mean(), 3), params='~0.2M'))
        print(f"{name:8s} BNT test={te.mean():.3f}±{te.std():.3f} gap={tr.mean()-te.mean():+.3f}")

    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'exp6c_bnt.csv'), index=False)

    # convergence check on ABIDE (seed=1, 3-fold avg, 300 epochs)
    d = load_dataset('abide'); y, V = d['y'], d['V']; Mfc = fc_matrices(d)
    conv = []
    for tr_all, te in make_folds(y, mode='subject', n_splits=3, seed=1):
        tr, _va = inner_val_split(tr_all, y, seed=1)
        _, _, lg = train_bnt(Mfc[tr], y[tr], Mfc[te], y[te], 1, V, epochs=300, log_every=10)
        conv.append(dict(lg))
    eps = sorted(conv[0]); te100 = np.mean([c[100] for c in conv]);
    tail = max(abs(np.mean([c[e] for c in conv]) - te100) for e in eps if e >= 100)
    pd.DataFrame({'epoch': eps,
                  'abide_BNT_test_AUROC': [round(float(np.mean([c[e] for c in conv])), 4) for e in eps]}
                 ).to_csv(os.path.join(OUT, 'exp6c_bnt_abide_convergence.csv'), index=False)
    print(f"\nABIDE BNT convergence (seed1, 3-fold avg): test@100={te100:.3f}  "
          f"max|test-test@100| over 100-300ep={tail:.3f}")

    abide_ours = [r['test_AUROC'] for r in rows if r['dataset'] == 'abide'][0]
    print(f"\n=== ABIDE comparison vs original BNT paper ===")
    print(f"  ours (AAL116, our preprocessing): {abide_ours:.3f}")
    print(f"  Kan et al. 2022 (reported):       {BNT_PAPER_ABIDE_AUROC:.3f}")
    print(f"  difference:                       {abide_ours - BNT_PAPER_ABIDE_AUROC:+.3f}")
    print('saved exp6c_bnt.csv')


if __name__ == '__main__':
    main()
