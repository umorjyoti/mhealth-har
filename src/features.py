"""Handcrafted features: 9 basic statistics per channel + physics/motion per sensor triad.

All features are computed on RAW (un-normalised) signals — gravity, tilt and jerk are
physical quantities and per-channel z-scoring would destroy them. Feature-level scaling
(fit on train only) happens later in run.py.
"""
import numpy as np
from scipy.signal import butter, filtfilt
from src.data import FS, CH_NAMES, TRIADS

STAT_FUNCS = [
    ("mean", lambda v: v.mean(1)),
    ("std", lambda v: v.std(1)),
    ("min", lambda v: v.min(1)),
    ("max", lambda v: v.max(1)),
    ("range", lambda v: v.max(1) - v.min(1)),
    ("median", lambda v: np.median(v, 1)),
    ("q25", lambda v: np.percentile(v, 25, axis=1)),
    ("q75", lambda v: np.percentile(v, 75, axis=1)),
    ("rms", lambda v: np.sqrt((v ** 2).mean(1))),
]

_B, _A = butter(3, 0.3 / (FS / 2), btype="low")   # 0.3 Hz split: gravity vs body motion


def _mag(v):                       # (n, t, 3) -> (n, t)
    return np.linalg.norm(v, axis=2)


def _energy(v):                    # mean squared magnitude per window
    return (v ** 2).mean(1)


def stat_features(X):
    """X: (n, t, c) -> (n, c*9), names."""
    cols, names = [], []
    for fname, f in STAT_FUNCS:
        cols.append(f(X))                                   # (n, c)
        names += [f"{ch}_{fname}" for ch in CH_NAMES]
    return np.concatenate(cols, 1), names


def _acc_physics(a, tag):
    """a: (n, t, 3) raw acceleration."""
    grav = filtfilt(_B, _A, a, axis=1)
    body = a - grav
    jerk = np.diff(body, axis=1) * FS
    gmag, bmag, jmag = _mag(grav), _mag(body), _mag(jerk)
    tilt = np.arctan2(grav[:, :, 2], np.hypot(grav[:, :, 0], grav[:, :, 1]))
    feats = [
        np.abs(a).sum(2).mean(1),                           # signal magnitude area
        gmag.mean(1), gmag.std(1),                          # gravity magnitude
        bmag.mean(1), bmag.std(1), _energy(bmag),           # body-acceleration magnitude
        jerk[:, :, 0].std(1), jerk[:, :, 1].std(1), jerk[:, :, 2].std(1),   # jerk per axis
        jmag.mean(1), jmag.std(1), _energy(jmag),           # jerk magnitude + energy
        tilt.mean(1), tilt.std(1),                          # tilt angle
    ]
    names = ["sma", "grav_mag_mean", "grav_mag_std", "body_mag_mean", "body_mag_std",
             "body_mag_energy", "jerk_x_std", "jerk_y_std", "jerk_z_std",
             "jerk_mag_mean", "jerk_mag_std", "jerk_energy", "tilt_mean", "tilt_std"]
    return np.stack(feats, 1), [f"{tag}_{n}" for n in names]


def _gyro_physics(w, tag):
    """w: (n, t, 3) raw angular velocity."""
    dw = np.diff(w, axis=1) * FS
    dmag, wmag = _mag(dw), _mag(w)                          # gyro change / angular-motion proxy
    feats = [
        np.abs(w).sum(2).mean(1),                           # signal magnitude area
        dw[:, :, 0].std(1), dw[:, :, 1].std(1), dw[:, :, 2].std(1),
        dmag.mean(1), dmag.std(1), _energy(dmag),
        wmag.mean(1), wmag.std(1), _energy(wmag),
    ]
    names = ["sma", "dgyro_x_std", "dgyro_y_std", "dgyro_z_std", "dgyro_mag_mean",
             "dgyro_mag_std", "dgyro_energy", "angmot_mean", "angmot_std", "angmot_energy"]
    return np.stack(feats, 1), [f"{tag}_{n}" for n in names]


def physics_features(X):
    cols, names = [], []
    for tag, sl, kind in TRIADS:
        f, n = (_acc_physics if kind == "acc" else _gyro_physics)(X[:, :, sl], tag)
        cols.append(f); names += n
    return np.concatenate(cols, 1), names


def extract(X):
    """-> features (n, d), names, and the index ranges of each family."""
    s, sn = stat_features(X)
    p, pn = physics_features(X)
    F = np.concatenate([s, p], 1).astype(np.float64)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    groups = {"stat": np.arange(s.shape[1]),
              "physics": np.arange(s.shape[1], F.shape[1])}
    return F, sn + pn, groups
