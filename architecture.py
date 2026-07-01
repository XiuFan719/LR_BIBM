"""Architecture and baseline comparison on functional connectivity features.
Compares traditional baselines (PCA+LogReg, ridge-logistic, linear/RBF SVM), MLPs, and
hand-written graph networks (GCN, GAT, BrainNetCNN) under 3 seeds x 3-fold subject CV,
reporting test AUROC mean +/- std and the train-test gap. -> exp6_architecture_comparison.csv"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '8'                      # limit BLAS threads to avoid oversubscription
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from fc_lib import load_dataset, train_with_curves, make_folds, inner_val_split, OUT, DEV

DATASETS = ['hcp_sex', 'abide', 'adhd']
SEEDS = [1, 2, 3]
EPOCHS = 500              # MLP epochs
GRAPH_EPOCHS = 100        # graph nets are heavy; 100 epochs suffice for the comparison


def fc_matrices(d):
    V = d['V']; tri = np.triu_indices(V, k=1); X = d['X']
    M = np.zeros((len(X), V, V), dtype=np.float32)
    M[:, tri[0], tri[1]] = X; M[:, tri[1], tri[0]] = X
    return M


def norm_adj(M):
    A = M.abs() + torch.eye(M.shape[1], device=M.device)
    deg = A.sum(-1).clamp(min=1e-6).pow(-0.5)
    return deg.unsqueeze(-1) * A * deg.unsqueeze(-2)


class GCN(nn.Module):
    def __init__(self, V, h=64):
        super().__init__(); self.w1 = nn.Linear(V, h); self.w2 = nn.Linear(h, h); self.out = nn.Linear(h, 1)
    def forward(self, M):
        Ah = norm_adj(M); H = torch.relu(Ah @ self.w1(M)); H = torch.relu(Ah @ self.w2(H))
        return self.out(H.mean(1)).squeeze(-1)


class E2E(nn.Module):
    """Edge-to-Edge cross-shaped filter (Kawahara et al. 2017)."""
    def __init__(self, cin, cout, V):
        super().__init__(); self.V = V
        self.row = nn.Conv2d(cin, cout, (1, V)); self.col = nn.Conv2d(cin, cout, (V, 1))
    def forward(self, x):                                   # (B,cin,V,V)
        r = self.row(x); c = self.col(x)
        return r.expand(-1, -1, -1, self.V) + c.expand(-1, -1, self.V, -1)


class BrainNetCNN(nn.Module):
    def __init__(self, V, c1=16, c2=32, cg=64):
        super().__init__()
        self.e2e = E2E(1, c1, V); self.e2n = nn.Conv2d(c1, c2, (1, V))
        self.n2g = nn.Conv2d(c2, cg, (V, 1)); self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(cg, 32); self.fc2 = nn.Linear(32, 1)
    def forward(self, M):                                   # M: (B,V,V)
        x = torch.relu(self.e2e(M.unsqueeze(1)))
        x = torch.relu(self.e2n(x)); x = torch.relu(self.n2g(x)).flatten(1)
        return self.fc2(torch.relu(self.fc1(self.drop(x)))).squeeze(-1)


class GAT(nn.Module):
    def __init__(self, V, h=64):
        super().__init__(); self.w = nn.Linear(V, h); self.a1 = nn.Linear(h, 1); self.a2 = nn.Linear(h, 1)
        self.w2 = nn.Linear(h, h); self.out = nn.Linear(h, 1)
    def forward(self, M):
        mask = M.abs() + torch.eye(M.shape[1], device=M.device); H = self.w(M)
        e = torch.nn.functional.leaky_relu(self.a1(H) + self.a2(H).transpose(1, 2)).masked_fill(mask < 1e-6, -1e9)
        H = torch.relu(torch.softmax(e, -1) @ H); H = torch.relu(self.w2(H))
        return self.out(H.mean(1)).squeeze(-1)


def train_graph(model, Mtr, ytr, Mte, yte, seed, bs=64):
    # mini-batched: full-batch on V=246 blows up memory, so batch train and eval
    torch.manual_seed(seed); m = model.to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4); lossf = nn.BCEWithLogitsLoss()
    Mtr_t = torch.tensor(Mtr, device=DEV); ytr_t = torch.tensor(ytr, device=DEV)
    n = len(Mtr)
    for _ in range(GRAPH_EPOCHS):
        m.train(); perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); lossf(m(Mtr_t[idx]), ytr_t[idx]).backward(); opt.step()
    m.eval()

    def pred(Mx):
        out = []
        with torch.no_grad():
            for i in range(0, len(Mx), 128):
                out.append(torch.sigmoid(m(torch.tensor(Mx[i:i + 128], device=DEV))).cpu().numpy())
        return np.concatenate(out)
    return roc_auc_score(yte, pred(Mte)), roc_auc_score(ytr, pred(Mtr))


def sk_eval(pipe, Xtr, ytr, Xte, yte):
    pipe.fit(Xtr, ytr)
    sc = pipe.predict_proba(Xte)[:, 1] if hasattr(pipe[-1], 'predict_proba') else pipe.decision_function(Xte)
    sctr = pipe.predict_proba(Xtr)[:, 1] if hasattr(pipe[-1], 'predict_proba') else pipe.decision_function(Xtr)
    return roc_auc_score(yte, sc), roc_auc_score(ytr, sctr)


def sk(model):
    return Pipeline([('sc', StandardScaler()), ('m', model)])

SK_MODELS = {
    # canonical L2-logistic: identical decoder to topk_decoding.py (full FC) and
    # run_main.py controls, so the reference AUROC is consistent across Table II,
    # Table III, and the top-k figure.
    'L2-logistic': lambda: sk(LogisticRegression(max_iter=1000, C=1.0)),
    'PCA50+LogReg': lambda: Pipeline([('sc', StandardScaler()), ('pca', PCA(50, random_state=0)),
                                      ('m', LogisticRegression(max_iter=1000))]),
    'Linear-SVM': lambda: sk(LinearSVC(C=1.0, max_iter=2000)),
    # RBF SVM on PCA-50 (an RBF kernel on 30k raw dims is slow and poorly conditioned)
    'RBF-SVM': lambda: Pipeline([('sc', StandardScaler()), ('pca', PCA(50, random_state=0)),
                                 ('m', SVC(kernel='rbf', C=1.0))]),
}
PARAM = {'L2-logistic': 'D', 'PCA50+LogReg': '51', 'Linear-SVM': 'D',
         'RBF-SVM': 'kernel', 'MLP(64)': '~2M', 'MLP(1024)': '~31M', 'GCN': '~20K',
         'GAT-style': '~20K', 'BrainNetCNN': '~0.6M'}

def main():
    rows = []
    for name in DATASETS:
        d = load_dataset(name); X, y, V = d['X'], d['y'], d['V']; Mfc = fc_matrices(d)
        order = list(SK_MODELS) + ['MLP(64)', 'MLP(1024)', 'GCN', 'GAT-style', 'BrainNetCNN']
        per_seed = {m: {'te': [], 'tr': []} for m in order}
        for sd in SEEDS:
            folds = make_folds(y, mode='subject', n_splits=3, seed=sd)
            fte = {m: [] for m in order}; ftr = {m: [] for m in order}
            for tr_all, te in folds:
                tr, va = inner_val_split(tr_all, y, seed=sd)
                for mn, mk in SK_MODELS.items():
                    a, b = sk_eval(mk(), X[tr], y[tr], X[te], y[te]); fte[mn].append(a); ftr[mn].append(b)
                for tag, h in [('MLP(64)', 64), ('MLP(1024)', 1024)]:
                    r = train_with_curves(X[tr], y[tr], X[va], y[va], X[te], y[te], task='bin', hidden=h, epochs=EPOCHS, seed=sd, record=False)
                    fte[tag].append(r['te_m']); ftr[tag].append(r['tr_m'])
                for tag, Mod in [('GCN', GCN), ('GAT-style', GAT), ('BrainNetCNN', BrainNetCNN)]:
                    a, b = train_graph(Mod(V), Mfc[tr], y[tr], Mfc[te], y[te], sd); fte[tag].append(a); ftr[tag].append(b)
            for m in order:
                per_seed[m]['te'].append(np.mean(fte[m])); per_seed[m]['tr'].append(np.mean(ftr[m]))
        for m in order:
            te = np.array(per_seed[m]['te']); tr = np.array(per_seed[m]['tr'])
            rows.append(dict(dataset=name, model=m, test_AUROC=round(te.mean(), 3), test_std=round(te.std(), 3),
                             train_AUROC=round(tr.mean(), 3), gen_gap=round(tr.mean() - te.mean(), 3), params=PARAM[m]))
            print(f"{name:8s} {m:15s} test={te.mean():.3f}±{te.std():.3f} gap={tr.mean()-te.mean():+.3f}")

    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'exp6_architecture_comparison.csv'), index=False)
    print('saved exp6_architecture_comparison.csv')


if __name__ == '__main__':
    main()
