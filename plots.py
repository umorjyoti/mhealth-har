"""Figures from results/*.csv. Run after run.py."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd, numpy as np

res = pd.read_csv("results/ablation.csv")
inv = pd.read_csv("results/feature_invariance.csv")
cm = pd.read_csv("results/confusion_subjectwise.csv", index_col=0)

# 1. macro-F1 per feature set / model, both splits
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, sp in zip(axes, ["random", "subject-wise"]):
    d = res[res.split == sp].pivot_table(index="features", columns="model", values="f1")
    d.plot.bar(ax=ax, rot=20, width=0.8)
    ax.set_title(f"{sp} split"); ax.set_ylabel("macro F1"); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=.3)
fig.suptitle("Macro-F1 by feature family")
fig.tight_layout(); fig.savefig("figs/f1_by_featureset.png", dpi=130)

# 2. subject-sensitivity distribution, stat vs physics
fig, ax = plt.subplots(figsize=(6, 4))
for fam, sub in inv.groupby("family"):
    ax.hist(sub.subject_sensitivity, bins=30, alpha=.6, label=f"{fam} (n={len(sub)})")
ax.set_xlabel("subject sensitivity (lower = more invariant)"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("figs/feature_invariance.png", dpi=130)

# 3. confusion matrix, subject-wise test subjects
fig, ax = plt.subplots(figsize=(8, 7))
n = cm.values / np.maximum(cm.values.sum(1, keepdims=True), 1)
im = ax.imshow(n, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(cm)), cm.columns, rotation=90); ax.set_yticks(range(len(cm)), cm.index)
for i in range(len(cm)):
    for j in range(len(cm)):
        if n[i, j] > .01:
            ax.text(j, i, f"{n[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if n[i, j] > .5 else "black")
ax.set_title("Confusion, held-out subjects"); fig.colorbar(im)
fig.tight_layout(); fig.savefig("figs/confusion_subjectwise.png", dpi=130)

# 4. top-20 importance (subject-wise split)
import json
a = json.load(open("results/analysis.json"))["subject-wise"]["top20"]
fig, ax = plt.subplots(figsize=(7, 6))
ax.barh([x["feature"] for x in a][::-1], [x["importance"] for x in a][::-1])
ax.set_xlabel("RF importance"); ax.grid(axis="x", alpha=.3)
fig.tight_layout(); fig.savefig("figs/top20_importance.png", dpi=130)
print("figs written")

# 5. LOSO ablation — the discriminating protocol
la = pd.read_csv("results/loso_ablation.csv")
fig, ax = plt.subplots(figsize=(10, 5))
la.pivot_table(index="features", columns="model", values="f1").loc[
    la.groupby("features").f1.mean().sort_values(ascending=False).index
].plot.bar(ax=ax, rot=30, width=.8)
ax.set_ylabel("LOSO macro F1"); ax.set_ylim(.6, 1); ax.grid(axis="y", alpha=.3)
ax.set_title("Leave-one-subject-out: macro F1 by feature set")
fig.tight_layout(); fig.savefig("figs/loso_ablation.png", dpi=130)
print("loso fig written")
