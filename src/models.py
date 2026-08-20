"""Classifiers: SVM / RandomForest / MLP on features, 1D CNN on raw windows."""
import numpy as np, torch, torch.nn as nn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

SEED = 0


def classical(name):
    if name == "SVM":
        return SVC(C=10, gamma="scale", kernel="rbf", random_state=SEED)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1500,
                             early_stopping=True, random_state=SEED)
    raise ValueError(name)


class CNN1D(nn.Module):
    """Baseline: raw 15-channel window in, activity out. Deliberately small — only ~800
    training windows exist, a deeper net just memorises them."""

    def __init__(self, n_ch, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_ch, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_cls),
        )

    def forward(self, x):
        return self.net(x)


def train_cnn(Xtr, ytr, Xva, yva, Xte, n_cls, epochs=120, lr=1e-3, bs=64):
    """X: (n, t, c) already scaled. Returns test predictions from the best-val epoch."""
    torch.manual_seed(SEED)
    to = lambda X: torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
    Xtr_, Xva_, Xte_ = to(Xtr), to(Xva), to(Xte)
    ytr_, yva_ = torch.tensor(ytr), torch.tensor(yva)

    model = CNN1D(Xtr.shape[2], n_cls)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    best, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            lossf(model(Xtr_[idx]), ytr_[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xva_).argmax(1) == yva_).float().mean().item()
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(Xte_).argmax(1).numpy(), model
