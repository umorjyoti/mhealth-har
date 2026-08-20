"""Three checks on whether the 0.978 out-of-fold score is real.

1. Nested selection - feature ranking recomputed inside each fold, so no fold ever
   sees its held-out subject during selection.
2. Label permutation - if the pipeline leaks, a model trained on shuffled labels still
   scores above chance. Two flavours: shuffle every window's label, and permute each
   subject's twelve activity labels as whole blocks (which keeps the within-block
   structure intact and is the harder test).
3. Group audit - assert no subject appears in both sides of any fold.
"""
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.data import cached
from src.features import extract
from src.models import classical

K = 40


def run(F, y, g, nested=True, cols_fixed=None):
    oof = np.zeros_like(y)
    for tr, te in LeaveOneGroupOut().split(F, y, g):
        assert not set(g[tr]) & set(g[te]), "subject on both sides of the fold"
        if nested:                                   # rank using training subjects only
            sc0 = StandardScaler().fit(F[tr])
            rf = classical("RandomForest").fit(sc0.transform(F[tr]), y[tr])
            cols = np.argsort(rf.feature_importances_)[::-1][:K]
        else:
            cols = cols_fixed
        sc = StandardScaler().fit(F[tr][:, cols])
        clf = classical("RandomForest").fit(sc.transform(F[tr][:, cols]), y[tr])
        oof[te] = clf.predict(sc.transform(F[te][:, cols]))
    p, r, f, _ = precision_recall_fscore_support(y, oof, average="macro", zero_division=0)
    return accuracy_score(y, oof), f, oof


def main():
    X, y, g = cached("results/windows.npz")
    F, names, _ = extract(X)
    rng = np.random.default_rng(0)

    rf = classical("RandomForest").fit(StandardScaler().fit_transform(F), y)
    cols_all = np.argsort(rf.feature_importances_)[::-1][:K]

    a1, f1, _ = run(F, y, g, nested=False, cols_fixed=cols_all)
    print(f"1. selection on all subjects (as shipped) : acc {a1:.4f}  macro F1 {f1:.4f}")

    a2, f2, _ = run(F, y, g, nested=True)
    print(f"2. selection inside each fold (nested)    : acc {a2:.4f}  macro F1 {f2:.4f}")

    y_shuf = rng.permutation(y)
    a3, f3, _ = run(F, y_shuf, g, nested=False, cols_fixed=cols_all)
    print(f"3. labels shuffled per window             : acc {a3:.4f}  (chance = {1/12:.4f})")

    # block permutation: each subject's twelve activity labels are relabelled as units
    y_blk = y.copy()
    for s in np.unique(g):
        m = g == s
        labs = np.unique(y[m])
        mapping = dict(zip(labs, rng.permutation(labs)))
        y_blk[m] = [mapping[v] for v in y[m]]
    a4, f4, _ = run(F, y_blk, g, nested=False, cols_fixed=cols_all)
    print(f"4. activity blocks relabelled per subject : acc {a4:.4f}  (chance = {1/12:.4f})")

    # how many independent units actually back each class
    print(f"\nwindows per class ~{np.bincount(y)[1:].mean():.0f}, "
          f"but they come from only {len(np.unique(g))} subjects "
          f"({np.bincount(y)[1:].mean()/len(np.unique(g)):.0f} windows per subject-block)")


if __name__ == "__main__":
    main()
