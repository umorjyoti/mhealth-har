"""Train the final compact HAR model and save it with an honest generalisation report.

The best configuration from results/loso_ablation.csv is refit on all ten subjects.
Its quality is reported from leave-one-subject-out out-of-fold predictions — every
window is scored by a model that never saw that subject — which is the number that
survives on a new person, unlike the saturated random split.
"""
import pickle, time
import numpy as np, pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.data import cached, ACTIVITIES
from src.features import extract
from src.models import classical

MODEL_PATH = "results/final_model.pkl"


def select_columns(F, y, names, groups, spec, n_feat):
    """Reproduce the ranking used by the ablation: RF importance over all subjects,
    restricted to the requested family."""
    rf = classical("RandomForest").fit(StandardScaler().fit_transform(F), y)
    order = np.argsort(rf.feature_importances_)[::-1]
    kind = spec.split()[-1]
    if kind == "mixed":
        return order[:n_feat]
    if kind in ("stat", "physics"):
        fam = set(groups[kind])
        return np.array([i for i in order if i in fam])[:n_feat]
    return np.arange(F.shape[1])          # "stat+physics (197)" and friends


def main():
    X, y, g = cached("results/windows.npz")
    F, names, groups = extract(X)
    names = np.array(names)

    best = pd.read_csv("results/loso_ablation.csv").sort_values("f1", ascending=False).iloc[0]
    cols = select_columns(F, y, names, groups, best.features, int(best.n_feat))
    print(f"config: {best.model} on {best.features} ({len(cols)} features)")

    # out-of-fold predictions: one model per held-out subject
    oof = np.zeros_like(y)
    for tr, te in LeaveOneGroupOut().split(F, y, g):
        sc = StandardScaler().fit(F[tr][:, cols])
        clf = classical(best.model).fit(sc.transform(F[tr][:, cols]), y[tr])
        oof[te] = clf.predict(sc.transform(F[te][:, cols]))

    labels = list(ACTIVITIES)
    rep = classification_report(y, oof, labels=labels,
                                target_names=list(ACTIVITIES.values()), digits=3, zero_division=0)
    cm = pd.DataFrame(confusion_matrix(y, oof, labels=labels),
                      index=list(ACTIVITIES.values()), columns=list(ACTIVITIES.values()))
    open("results/final_report_loso.txt", "w").write(rep)
    cm.to_csv("results/final_confusion_loso.csv")
    print(f"\nout-of-fold accuracy {accuracy_score(y, oof):.4f}\n{rep}")

    # final fit on every subject
    t = time.time()
    sc = StandardScaler().fit(F[:, cols])
    clf = classical(best.model).fit(sc.transform(F[:, cols]), y)
    fit_s = time.time() - t
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump({"model": clf, "scaler": sc, "cols": cols,
                     "feature_names": [str(n) for n in names[cols]],
                     "classes": ACTIVITIES, "config": best.to_dict(),
                     "oof_accuracy": float(accuracy_score(y, oof))}, fh)
    print(f"saved {MODEL_PATH}  (fit {fit_s:.2f}s, "
          f"{len(pickle.dumps(clf)) / 1024:.0f} KB)")


def predict(X_windows, path=MODEL_PATH):
    """X_windows: (n, 250, 15) raw sensor windows -> activity labels 1..12."""
    with open(path, "rb") as fh:
        b = pickle.load(fh)
    F, _, _ = extract(np.asarray(X_windows, dtype=np.float32))
    return b["model"].predict(b["scaler"].transform(F[:, b["cols"]]))


def self_check():
    """Reload the saved model and confirm it still classifies known windows."""
    X, y, _ = cached("results/windows.npz")
    idx = np.random.default_rng(0).choice(len(X), 200, replace=False)
    pred = predict(X[idx])
    acc = accuracy_score(y[idx], pred)
    assert pred.shape == (200,), pred.shape
    assert set(pred) <= set(ACTIVITIES), set(pred) - set(ACTIVITIES)
    assert acc > 0.95, f"reloaded model only scores {acc:.3f} on training windows"
    print(f"self-check ok: reloaded model, 200 windows, accuracy {acc:.3f}")


if __name__ == "__main__":
    main()
    self_check()
