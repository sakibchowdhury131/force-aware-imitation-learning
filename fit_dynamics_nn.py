#!/usr/bin/env python3
"""
Fit B of the mass/Coriolis-residual comparison: a small MLP predicting the
SAME target (gravity-free torque, minus calibrated gravity, minus nominal
rigid-body dynamics) from the SAME (q, qdot, qddot) as the regressor -- no
physical structure imposed, just supervised regression, so the two are
compared on equal footing (same train/held-out data, same target).

Input features: [sin(q), cos(q), qdot(rad/s), qddot(rad/s^2)] (24-dim) --
sin/cos handles the continuous joints' unbounded/cumulative degree values
(J1/J4/J6 can exceed +-360) without the model seeing an arbitrary
discontinuity or extrapolating past its training range. Inputs standardized
(zero mean, unit std) using TRAINING-set statistics only.

Small network (2 hidden layers, 64 units) + weight decay, given real
hardware data is expensive to collect and this dataset is a few thousand
samples, not the tens/hundreds of thousands typical NN training assumes.
Train/val split WITHIN the training episodes for early stopping; held-out
episodes are never seen until final evaluation (same discipline as the
regressor and as the gravity task-pose experiment).
"""
import numpy as np
import torch
import torch.nn as nn


def make_features(q_deg, qdot_deg, qddot_deg):
    q = np.deg2rad(q_deg)
    qdot = np.deg2rad(qdot_deg)
    qddot = np.deg2rad(qddot_deg)
    return np.concatenate([np.sin(q), np.cos(q), qdot, qddot], axis=-1)


class ResidualMLP(nn.Module):
    def __init__(self, in_dim=24, hidden=64, out_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(dataset, val_frac=0.15, epochs=300, lr=1e-3, weight_decay=1e-4,
              hidden=64, seed=0, patience=30):
    rng = np.random.default_rng(seed)
    X = make_features(dataset['q'], dataset['qdot'], dataset['qddot']).astype(np.float32)
    Y = dataset['target'].astype(np.float32)
    n = len(X)
    idx = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    x_mean, x_std = X[train_idx].mean(0), X[train_idx].std(0) + 1e-6
    Xn = (X - x_mean) / x_std

    Xt = torch.tensor(Xn[train_idx]); Yt = torch.tensor(Y[train_idx])
    Xv = torch.tensor(Xn[val_idx]);   Yv = torch.tensor(Y[val_idx])

    model = ResidualMLP(in_dim=X.shape[1], hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val, best_state, bad_epochs = float('inf'), None, 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xt), Yt)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), Yv).item()
        if val_loss < best_val - 1e-6:
            best_val, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs > patience:
                print(f'  early stop at epoch {epoch}, best val MSE={best_val:.4f}')
                break
    model.load_state_dict(best_state)
    print(f'Trained MLP: {n - n_val} train / {n_val} val samples, best val MSE={best_val:.4f}')
    return model, x_mean, x_std


def predict(model, x_mean, x_std, dataset):
    X = make_features(dataset['q'], dataset['qdot'], dataset['qddot']).astype(np.float32)
    Xn = (X - x_mean) / x_std
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(Xn)).numpy()


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from dynamics_dataset import build_dataset

    phi_gravity = np.load('data/gravity_phi_task_only.npy')
    train_eps = ['004', '009', '014', '019', '024', '029']
    print('Building training dataset...')
    ds = build_dataset(train_eps, 'pastaTransfer4', phi_gravity)
    print(f'{len(ds["target"])} training samples')
    model, x_mean, x_std = train_mlp(ds)
    torch.save({'state_dict': model.state_dict(), 'x_mean': x_mean, 'x_std': x_std},
              'data/dynamics_residual_nn.pt')
    print('Saved -> data/dynamics_residual_nn.pt')
