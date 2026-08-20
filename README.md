# Compact HAR framework on the MHEALTH dataset

Handcrafted statistical + physics/motion features for 12-activity human activity
recognition, with a 1D CNN on raw signals as a baseline.

## Running

### 1. Get the dataset

The MHEALTH data is not in this repository. Download it from the UCI Machine Learning
Repository and unpack it into the project root, so the log files sit at
`MHEALTHDATASET/mHealth_subject1.log` … `mHealth_subject10.log`:

```bash
curl -L -o mhealth.zip https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip
unzip -q mhealth.zip && rm mhealth.zip
```

If you use the data, cite Banos et al., *mHealthDroid: a novel framework for agile
development of mobile health applications*, IWAAL 2014.

### 2. Install

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 3. Run

```bash
.venv/bin/python run.py               # steps 1-14: windows, features, analysis, all models
.venv/bin/python loso_ablation.py     # leave-one-subject-out ablation (the deciding experiment)
.venv/bin/python train_final.py       # train + save the final model, with LOSO report
.venv/bin/python plots.py             # figures into figs/
```

Run them in that order the first time — `loso_ablation.py` reads the windows cached by
`run.py`, and `train_final.py` reads the ranking written by `loso_ablation.py`.

`run.py` takes about 10 minutes, almost all of it the two 1D CNN baselines on CPU; the
other three scripts take under two minutes each. Windows are cached to
`results/windows.npz`, so re-runs skip parsing the 1.2 M raw rows.

`results/windows.npz` and `results/final_model.pkl` are gitignored — both are rebuilt by
the commands above.

## Pipeline

| Step | Where | Notes |
| --- | --- | --- |
| Load, clean, drop null class | `src/data.py` | 10 subjects, 50 Hz, drop NaN/inf, drop label 0 |
| Sensor selection | `src/data.py` | 15 channels: chest acc, ankle acc+gyro, arm acc+gyro. ECG and magnetometers dropped (magnetometers are orientation- and environment-dependent) |
| Windowing | `src/data.py` | 250 samples (5 s), non-overlapping, never crossing a label or recording boundary → 1335 windows |
| Splits | `run.py` | random 60/20/20 stratified, subject-wise (1–6 / 7–8 / 9–10), and 10-fold LOSO |
| Normalisation | `run.py` | `StandardScaler` fit on training data only — separately for raw CNN input and for the feature matrix |
| Features | `src/features.py` | 135 statistical + 62 physics = 197 |
| Feature analysis | `run.py` | \|r\| > 0.95 redundancy pruning (72 features dropped), RF Gini importance, top-k selection |
| Classification | `src/models.py` | SVM (RBF), RandomForest, MLP, 1D CNN |
| Evaluation, invariance, efficiency | `run.py`, `loso_ablation.py` | see below |

Physics features are computed on the raw signals, never on z-scored ones: gravity,
tilt and jerk are physical quantities that per-channel standardisation would destroy.
Gravity is separated with a 0.3 Hz third-order Butterworth low-pass; body acceleration
is the residual.

## Results

Both the random split and the two-held-out-subject split saturate at ~1.00 macro-F1 for
every feature family, so neither can rank features. Leave-one-subject-out is the
protocol that discriminates; all numbers below are LOSO means over 10 subjects
(`results/loso_ablation.csv`).

| Feature set | Model | Accuracy | Precision | Recall | Macro F1 |
| --- | --- | --- | --- | --- | --- |
| top-40 mixed | RandomForest | 0.979 | 0.980 | 0.980 | **0.977** |
| top-20 statistical | RandomForest | 0.979 | 0.980 | 0.980 | 0.976 |
| stat + physics (197) | RandomForest | 0.978 | 0.979 | 0.979 | 0.976 |
| statistical only (135) | RandomForest | 0.966 | 0.970 | 0.965 | 0.961 |
| physics only (62) | RandomForest | 0.949 | 0.955 | 0.949 | 0.941 |
| top-10 physics | SVM | 0.782 | 0.736 | 0.792 | 0.742 |

Statistical features alone beat physics features alone (0.961 vs 0.941 with a random
forest), but the two are complementary — the combined and mixed top-k sets are best on
all four metrics. Random forests dominate SVM and MLP under LOSO at every feature count.

### Feature invariance

`results/feature_invariance.csv` scores each feature by how much its within-activity
mean moves across subjects, relative to its overall spread (lower = more invariant).
Physics features average 0.754, statistical features 0.876, and the six most
subject-invariant features in the whole set are all physics ones: gravity-magnitude
standard deviation and tilt-angle standard deviation at all three sensor sites.

The same effect shows up at the model level. The 1D CNN on raw signals scores 0.997
macro-F1 on the random split but 0.883 on held-out subjects — a 0.113 drop — while every
handcrafted feature set stays within 0.01 of its random-split score. Handcrafted
features generalise across people; the CNN partly memorises them, at this dataset size.

### Efficiency

Feature extraction costs 0.37 ms per 5 s window for all 197 features. The final model
(40 features, random forest) infers in ~0.24 ms per window against ~1.25 ms for the CNN.

### Final model

```bash
.venv/bin/python train_final.py
```

`train_final.py` refits the best LOSO configuration — a random forest on the 40
highest-importance mixed features — on all ten subjects, and reports its quality from
leave-one-subject-out **out-of-fold** predictions, where every window is scored by a
model that never saw that subject:

```
out-of-fold accuracy 0.978   macro precision 0.981   macro recall 0.980   macro F1 0.980
```

Per class (`results/final_report_loso.txt`, confusion in `results/final_confusion_loso.csv`):
lying, walking, waist bends, arm elevation, cycling and jump are perfect; the only real
confusions are standing vs sitting (sitting recall 0.900) and jogging vs running
(running recall 0.917) — activity pairs that genuinely share a sensor signature.

Loading and predicting:

```python
from train_final import predict
labels = predict(windows)      # windows: (n, 250, 15) raw -> labels 1..12
```

`train_final.py` ends with a `self_check()` that reloads the pickle and asserts the
restored scaler/model/column set still classifies known windows — a load-integrity
check, not a generalisation number.

`results/final_model.pkl` holds the best LOSO configuration — a random forest on the 40
highest-importance mixed features, refit on all ten subjects — together with its scaler
and feature names. The selected set is dominated by ankle sensor features, plus body
acceleration magnitude, jerk and gyroscope-change energy from the physics family.

## Caveats

- 1335 windows total is small. The 40-window count for "jump front & back" (label 12)
  makes its per-class metrics noisy.
- Per-subject LOSO scores range from 0.83 (subjects 6 and 8) to 1.00 (subjects 5, 7, 9, 10);
  the mean hides real between-subject variation.
- The random split leaks context between train and test — 5 s windows cut from one
  continuous 1-minute recording are near-duplicates. Treat its 1.00 scores as an upper
  bound, not a result.
