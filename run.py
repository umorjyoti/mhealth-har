"""Full MHEALTH HAR pipeline: windows -> features -> analysis -> classification -> report.

Run: .venv/bin/python run.py
"""
import json, pickle, time
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)

from src.data import cached, ACTIVITIES, CH_NAMES
from src.features import extract
from src.models import classical, train_cnn, SEED

MODELS = ["SVM", "RandomForest", "MLP"]
OUT = "results"


# ---------------------------------------------------------------- step 6: splits
def make_splits(y, g):
    """Both evaluation protocols, as (train, val, test) index triples."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=SEED)
    tr, rest = next(sss.split(np.zeros(len(y)), y))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    va, te = (rest[i] for i in next(sss2.split(np.zeros(len(rest)), y[rest])))

    sub = (np.flatnonzero(np.isin(g, [1, 2, 3, 4, 5, 6])),
           np.flatnonzero(np.isin(g, [7, 8])),
           np.flatnonzero(np.isin(g, [9, 10])))
    return {"random": (tr, va, te), "subject-wise": sub}


def score(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",
                                                 zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred),
            "precision": p, "recall": r, "f1": f}


# ------------------------------------------------- step 9: redundancy / importance
def prune_redundant(Ftr, names, thr=0.95):
    """Drop the later of any feature pair with |Pearson r| > thr."""
    C = np.corrcoef(Ftr, rowvar=False)
    C = np.nan_to_num(np.abs(C))
    keep, dropped = [], []
    for j in range(C.shape[0]):
        if any(C[j, k] > thr for k in keep):
            dropped.append(names[j])
        else:
            keep.append(j)
    return np.array(keep), dropped


def rf_importance(Ftr, ytr):
    rf = classical("RandomForest").fit(Ftr, ytr)
    return rf.feature_importances_


# ------------------------------------------------------------------ main pipeline
def main():
    t0 = time.time()
    X, y, g = cached(f"{OUT}/windows.npz")                       # steps 1-5
    t_feat = time.time()
    F, names, groups = extract(X)                                # step 8
    ms_per_window = (time.time() - t_feat) / len(X) * 1000
    names = np.array(names)
    print(f"windows {X.shape}  features {F.shape}  ({ms_per_window:.2f} ms/window)")

    y0 = y - 1                                                   # 0-indexed for torch
    n_cls = y0.max() + 1
    splits = make_splits(y, g)
    rows, analysis, cnn_store = [], {}, {}

    for split_name, (tr, va, te) in splits.items():
        # ---- step 7: normalisation, fit on training data only
        fs = StandardScaler().fit(F[tr])
        Fs = fs.transform(F)

        # ---- step 9: redundancy -> importance -> selection (train only)
        keep, dropped = prune_redundant(Fs[tr], names)
        imp = rf_importance(Fs[tr][:, keep], y[tr])
        order = keep[np.argsort(imp)[::-1]]

        best_k, best_val = None, -1
        for k in [10, 20, 40, 80, len(order)]:
            m = classical("RandomForest").fit(Fs[tr][:, order[:k]], y[tr])
            v = accuracy_score(y[va], m.predict(Fs[va][:, order[:k]]))
            if v > best_val:
                best_k, best_val = k, v
        analysis[split_name] = {
            "n_features": len(names), "n_after_prune": len(keep),
            "n_dropped_redundant": len(dropped), "best_k": int(best_k),
            "top20": [{"feature": names[i], "importance": float(im)}
                      for i, im in zip(order[:20], np.sort(imp)[::-1][:20])],
            "selected": [str(n) for n in names[order[:best_k]]],
        }

        featsets = {
            "stat-only": groups["stat"],
            "physics-only": groups["physics"],
            "stat+physics": np.arange(F.shape[1]),
            f"selected-top{best_k}": order[:best_k],
        }

        # ---- steps 10-11: classical models on each feature family
        for fs_name, cols in featsets.items():
            for mname in MODELS:
                t = time.time()
                clf = classical(mname).fit(Fs[tr][:, cols], y[tr])
                fit_s = time.time() - t
                t = time.time()
                pred = clf.predict(Fs[te][:, cols])
                inf_ms = (time.time() - t) / len(te) * 1000
                rows.append({"split": split_name, "features": fs_name, "model": mname,
                             "n_feat": len(cols), **score(y[te], pred),
                             "fit_s": fit_s, "infer_ms_per_window": inf_ms,
                             "size_kb": len(pickle.dumps(clf)) / 1024})

        # ---- 1D CNN baseline on raw windows (own scaler, train-only fit)
        sc = StandardScaler().fit(X[tr].reshape(-1, X.shape[2]))
        Xs = sc.transform(X.reshape(-1, X.shape[2])).reshape(X.shape).astype(np.float32)
        t = time.time()
        pred0, model = train_cnn(Xs[tr], y0[tr], Xs[va], y0[va], Xs[te], n_cls)
        fit_s = time.time() - t
        import torch
        t = time.time()
        with torch.no_grad():
            model(torch.tensor(Xs[te]).permute(0, 2, 1))
        inf_ms = (time.time() - t) / len(te) * 1000
        n_par = sum(p.numel() for p in model.parameters())
        rows.append({"split": split_name, "features": "raw signal", "model": "1D CNN",
                     "n_feat": X.shape[1] * X.shape[2], **score(y0[te], pred0),
                     "fit_s": fit_s, "infer_ms_per_window": inf_ms,
                     "size_kb": n_par * 4 / 1024})
        cnn_store[split_name] = (y0[te], pred0)

    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/ablation.csv", index=False)

    # ---- step 12: feature invariance (random vs subject-wise degradation)
    piv = res.pivot_table(index=["features", "model"], columns="split", values="f1")
    piv["drop"] = piv["random"] - piv["subject-wise"]
    piv.to_csv(f"{OUT}/invariance.csv")

    # per-feature subject sensitivity: between-subject spread inside a class,
    # relative to that feature's overall spread. Lower = more subject-invariant.
    fs_all = StandardScaler().fit_transform(F)
    sens = np.zeros(F.shape[1])
    for c in np.unique(y):
        m = y == c
        means = np.array([fs_all[m & (g == s)].mean(0) for s in np.unique(g[m])])
        sens += means.std(0) / (fs_all[m].std(0) + 1e-9)
    sens /= len(np.unique(y))
    inv = pd.DataFrame({"feature": names, "subject_sensitivity": sens,
                        "family": ["stat"] * len(groups["stat"]) + ["physics"] * len(groups["physics"])})
    inv.sort_values("subject_sensitivity").to_csv(f"{OUT}/feature_invariance.csv", index=False)

    # ---- step 14: final compact model = best subject-wise config, LOSO-verified
    sw = res[(res.split == "subject-wise") & (res.model != "1D CNN")]
    best = sw.sort_values("f1", ascending=False).iloc[0]
    cols = {**{"stat-only": groups["stat"], "physics-only": groups["physics"],
               "stat+physics": np.arange(F.shape[1])},
            f"selected-top{analysis['subject-wise']['best_k']}":
                np.array([list(names).index(n) for n in analysis["subject-wise"]["selected"]])}[best.features]

    loso, logo = [], LeaveOneGroupOut()
    for tr_i, te_i in logo.split(F, y, g):
        sc = StandardScaler().fit(F[tr_i])
        clf = classical(best.model).fit(sc.transform(F[tr_i])[:, cols], y[tr_i])
        loso.append(score(y[te_i], clf.predict(sc.transform(F[te_i])[:, cols])))
    loso = pd.DataFrame(loso, index=[f"S{s}" for s in np.unique(g)])
    loso.loc["mean"] = loso.mean()
    loso.to_csv(f"{OUT}/loso.csv")

    sc = StandardScaler().fit(F)
    final = classical(best.model).fit(sc.transform(F)[:, cols], y)
    with open(f"{OUT}/final_model.pkl", "wb") as fh:
        pickle.dump({"model": final, "scaler": sc, "cols": cols,
                     "feature_names": list(names[cols])}, fh)

    # confusion matrix of the final configuration on the held-out subjects
    tr, va, te = splits["subject-wise"]
    sc2 = StandardScaler().fit(F[tr])
    clf2 = classical(best.model).fit(sc2.transform(F[tr])[:, cols], y[tr])
    pred2 = clf2.predict(sc2.transform(F[te])[:, cols])
    cm = confusion_matrix(y[te], pred2, labels=list(ACTIVITIES))
    pd.DataFrame(cm, index=list(ACTIVITIES.values()),
                 columns=list(ACTIVITIES.values())).to_csv(f"{OUT}/confusion_subjectwise.csv")
    with open(f"{OUT}/report_subjectwise.txt", "w") as fh:
        fh.write(classification_report(y[te], pred2, labels=list(ACTIVITIES),
                                       target_names=list(ACTIVITIES.values()), zero_division=0))

    analysis["efficiency"] = {"feature_extraction_ms_per_window": ms_per_window,
                             "total_runtime_s": time.time() - t0}
    analysis["best_config"] = {"model": best.model, "features": best.features,
                               "n_feat": int(best.n_feat), "subjectwise_f1": float(best.f1),
                               "loso_mean": loso.loc["mean"].to_dict()}
    with open(f"{OUT}/analysis.json", "w") as fh:
        json.dump(analysis, fh, indent=2, default=float)

    pd.set_option("display.width", 200)
    print("\n=== ablation (macro metrics on test) ===")
    print(res.round(3).to_string(index=False))
    print("\n=== invariance: macro-F1 random vs subject-wise ===")
    print(piv.round(3).to_string())
    print("\n=== LOSO,", best.model, "/", best.features, "===")
    print(loso.round(3).to_string())


if __name__ == "__main__":
    main()
