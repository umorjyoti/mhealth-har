"""Leave-One-Subject-Out ablation — the discriminating protocol.

The 60/20/20 and 2-held-out-subject splits both saturate at ~1.00 macro-F1 on this
dataset, so they cannot rank feature families. LOSO (10 folds, one subject held out
each time) has enough spread to answer "which features are better".
"""
import numpy as np, pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.data import cached
from src.features import extract
from src.models import classical

MODELS = ["SVM", "RandomForest", "MLP"]


def loso(F, y, g, cols, model):
    out = []
    for tr, te in LeaveOneGroupOut().split(F, y, g):
        sc = StandardScaler().fit(F[tr][:, cols])
        clf = classical(model).fit(sc.transform(F[tr][:, cols]), y[tr])
        pred = clf.predict(sc.transform(F[te][:, cols]))
        p, r, f, _ = precision_recall_fscore_support(y[te], pred, average="macro", zero_division=0)
        out.append([accuracy_score(y[te], pred), p, r, f])
    a = np.array(out)
    return dict(zip(["accuracy", "precision", "recall", "f1"], a.mean(0))) | \
           {"f1_std": a[:, 3].std(), "worst_subject_f1": a[:, 3].min()}


def main():
    X, y, g = cached("results/windows.npz")
    F, names, groups = extract(X)
    names = np.array(names)

    # importance ranking computed once, on all subjects, only to order features for the
    # top-k sweep (each LOSO fold still refits scaler + model on its own training subjects)
    rf = classical("RandomForest").fit(StandardScaler().fit_transform(F), y)
    order = np.argsort(rf.feature_importances_)[::-1]
    stat_rank = np.array([i for i in order if i in set(groups["stat"])])
    phys_rank = np.array([i for i in order if i in set(groups["physics"])])

    sets = {"stat-only (135)": groups["stat"],
            "physics-only (62)": groups["physics"],
            "stat+physics (197)": np.arange(F.shape[1])}
    for k in [10, 20, 40]:
        sets[f"top{k} mixed"] = order[:k]
        sets[f"top{k} stat"] = stat_rank[:k]
        sets[f"top{k} physics"] = phys_rank[:k]

    rows = []
    for sname, cols in sets.items():
        for m in MODELS:
            rows.append({"features": sname, "n_feat": len(cols), "model": m,
                         **loso(F, y, g, np.asarray(cols), m)})
            print(f"{sname:20s} {m:13s} f1={rows[-1]['f1']:.3f}")
    df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    df.to_csv("results/loso_ablation.csv", index=False)
    print("\n=== LOSO ablation (mean over 10 subjects) ===")
    print(df.round(3).to_string(index=False))
    print("\nby feature family (mean f1 over models):")
    print(df.groupby("features").f1.mean().sort_values(ascending=False).round(3).to_string())




def finalize():
    """Refit the best LOSO configuration on all 10 subjects and save it."""
    import pickle
    X, y, g = cached("results/windows.npz")
    F, names, groups = extract(X)
    names = np.array(names)
    best = pd.read_csv("results/loso_ablation.csv").sort_values("f1", ascending=False).iloc[0]

    rf = classical("RandomForest").fit(StandardScaler().fit_transform(F), y)
    order = np.argsort(rf.feature_importances_)[::-1]
    fam = {"stat": groups["stat"], "physics": groups["physics"]}
    kind = best.features.split()[-1]
    rank = order if kind == "mixed" else np.array([i for i in order if i in set(fam[kind])])
    cols = rank[:int(best.n_feat)]

    sc = StandardScaler().fit(F[:, cols])
    clf = classical(best.model).fit(sc.transform(F[:, cols]), y)
    with open("results/final_model.pkl", "wb") as fh:
        pickle.dump({"model": clf, "scaler": sc, "cols": cols,
                     "feature_names": list(names[cols]),
                     "config": best.to_dict()}, fh)
    print(f"final: {best.model} on {best.features} -> LOSO f1 {best.f1:.3f}")
    print("features:", list(names[cols]))


if __name__ == "__main__":
    import sys
    (finalize if "--finalize" in sys.argv else main)()
