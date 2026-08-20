"""Load MHEALTH, clean, select sensors, window."""
import numpy as np, pandas as pd
from pathlib import Path

FS = 50
WIN = 5 * FS                      # 5 s, non-overlapping
RAW_DIR = Path(__file__).resolve().parents[1] / "MHEALTHDATASET"

# 0-indexed columns from README (ECG=3,4 and magnetometers=11..13,20..22 dropped)
CHANNELS = [0, 1, 2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 19]
CH_NAMES = [
    "chest_ax", "chest_ay", "chest_az",
    "ankle_ax", "ankle_ay", "ankle_az",
    "ankle_gx", "ankle_gy", "ankle_gz",
    "arm_ax", "arm_ay", "arm_az",
    "arm_gx", "arm_gy", "arm_gz",
]
# (name, slice, kind) — physics features need triads, not loose columns
TRIADS = [
    ("chest_acc", slice(0, 3), "acc"),
    ("ankle_acc", slice(3, 6), "acc"),
    ("ankle_gyro", slice(6, 9), "gyro"),
    ("arm_acc", slice(9, 12), "acc"),
    ("arm_gyro", slice(12, 15), "gyro"),
]
ACTIVITIES = {
    1: "standing", 2: "sitting", 3: "lying", 4: "walking", 5: "stairs",
    6: "waist_bend", 7: "arm_elevation", 8: "knee_bend", 9: "cycling",
    10: "jogging", 11: "running", 12: "jump",
}


def load_subject(sid):
    df = pd.read_csv(RAW_DIR / f"mHealth_subject{sid}.log", sep="\t", header=None)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()          # step 2: cleaning
    df = df[df[23] != 0]                                          # step 3: drop null class
    return df[CHANNELS].to_numpy(np.float32), df[23].to_numpy(np.int64)


def window(X, y, sid, overlap=0.0):
    """Non-overlapping 5 s windows, never crossing a label or recording break."""
    step = max(1, int(WIN * (1 - overlap)))
    bounds = np.flatnonzero(np.diff(y) != 0) + 1                  # label transitions
    Xs, ys, gs = [], [], []
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(y)]):
        for s in range(a, b - WIN + 1, step):
            Xs.append(X[s:s + WIN])
            ys.append(y[s])
            gs.append(sid)
    return Xs, ys, gs


def build(overlap=0.0, subjects=range(1, 11)):
    Xs, ys, gs = [], [], []
    for sid in subjects:
        X, y = load_subject(sid)
        a, b, c = window(X, y, sid, overlap)
        Xs += a; ys += b; gs += c
    return np.stack(Xs), np.array(ys), np.array(gs)


def cached(path, overlap=0.0):
    path = Path(path)
    if path.exists():
        d = np.load(path)
        return d["X"], d["y"], d["g"]
    X, y, g = build(overlap)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y, g=g)
    return X, y, g


if __name__ == "__main__":
    X, y, g = cached("results/windows.npz")
    print("windows", X.shape, "labels", np.bincount(y)[1:], "subjects", np.bincount(g)[1:])
