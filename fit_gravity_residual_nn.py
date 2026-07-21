#!/usr/bin/env python3
"""
G_res(q): a small neural network capturing whatever the LINEAR gravity
regressor (gravity_phi_task_only.npy) still misses -- trained EXCLUSIVELY on
static/near-zero-velocity data, never on motion data, so it cannot be
confused with (or contaminate) the separate mass/Coriolis residual models.

This exists because the mass/Coriolis residual TARGET is built by subtracting
the gravity model from every motion sample's torque reading -- if the gravity
model still has pose-dependent error, that error is present additively in
EVERY training sample for the dynamics models (both the regressor and the
constrained NN), acting as pose-correlated noise neither can explain (the
constrained NN structurally can't absorb it into a q̇/q̈-gated output, but it
still degrades what target they're being asked to fit). Fixing it here first
gives both dynamics models a cleaner signal.

TRAIN/HELD-OUT DISCIPLINE: uses the original 40 dedicated task-calibration
poses PLUS near-static stretches extracted ONLY from the 6 TRAIN dynamics
episodes (004/009/014/019/024/029) -- NOT the 3 held-out episodes
(039/044/049), which stay completely untouched by gravity fitting too, so
the eventual dynamics-model held-out evaluation remains clean on every axis.
"""
import os
import numpy as np
import torch
import torch.nn as nn

from calibrate_gravity_residual import load_session_records
from extend_gravity_from_excitation import extract_static_stretches
from contact_detector import gravity_regressor, torque_to_wrench

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DYNAMICS_EPISODES = ['001', '004', '005', '009', '010', '011', '012', '014', '015',
                           '017', '018', '019', '021', '022', '023', '024', '028', '029',
                           '030', '031', '032', '033', '035', '037', '038', '040', '042',
                           '043', '045', '047']
HELDOUT_DYNAMICS_EPISODES = ['039', '044', '049']


def make_features(q_deg):
    q = np.deg2rad(q_deg)
    return np.concatenate([np.sin(q), np.cos(q)], axis=-1)


class GravityResidualNet(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 6),
        )

    def forward(self, x):
        return self.net(x)


def build_records(phi):
    """Returns (train_records, heldout_records) as lists of (q, target) where
    target = gravity_free_torque - gravity_regressor(q)@phi, i.e. what's left
    of the ORIGINAL linear gravity model on purely static data."""
    orig_path = os.path.join(PIPELINE_DIR, 'data', 'gravity_calibration_records_task_only.npz')
    orig_records = load_session_records(orig_path)   # 40 dedicated poses

    train_static = []
    for ep in TRAIN_DYNAMICS_EPISODES:
        train_static += extract_static_stretches(ep)
    heldout_static = []
    for ep in HELDOUT_DYNAMICS_EPISODES:
        heldout_static += extract_static_stretches(ep)

    def to_targets(records):
        out = []
        for q, tau_gf in records:
            target = tau_gf - gravity_regressor(q) @ phi
            out.append((q, target))
        return out

    train_records = to_targets(orig_records) + to_targets(train_static)
    heldout_records = to_targets(heldout_static)
    print(f'G_res training data: {len(orig_records)} dedicated poses + {len(train_static)} '
         f'train-episode static stretches = {len(train_records)} total')
    print(f'G_res held-out data: {len(heldout_records)} static stretches from '
         f'{HELDOUT_DYNAMICS_EPISODES} (never used for gravity OR dynamics fitting)')
    return train_records, heldout_records


def train(train_records, val_frac=0.15, epochs=500, lr=1e-3, weight_decay=1e-3,
         hidden=32, seed=0, patience=50):
    rng = np.random.default_rng(seed)
    q = np.stack([r[0] for r in train_records])
    target = np.stack([r[1] for r in train_records]).astype(np.float32)
    X = make_features(q).astype(np.float32)

    n = len(X)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    x_mean, x_std = X[train_idx].mean(0), X[train_idx].std(0) + 1e-6
    Xn = (X - x_mean) / x_std
    Xt, Yt = torch.tensor(Xn[train_idx]), torch.tensor(target[train_idx])
    Xv, Yv = torch.tensor(Xn[val_idx]), torch.tensor(target[val_idx])

    model = GravityResidualNet(hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val, best_state, bad = float('inf'), None, 0
    for epoch in range(epochs):
        model.train(); opt.zero_grad()
        loss = loss_fn(model(Xt), Yt)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xv), Yv).item()
        if vl < best_val - 1e-7:
            best_val, best_state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad > patience:
                print(f'  early stop at epoch {epoch}, best val MSE={best_val:.5f}')
                break
    model.load_state_dict(best_state)
    print(f'Trained GravityResidualNet: {n-n_val} train / {n_val} val samples, best val MSE={best_val:.5f}')
    return model, x_mean, x_std


def predict(model, x_mean, x_std, q_deg):
    X = make_features(np.atleast_2d(q_deg)).astype(np.float32)
    Xn = (X - x_mean) / x_std
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(Xn)).numpy()
    return out[0] if np.ndim(q_deg) == 1 else out


def main():
    phi = np.load(os.path.join(PIPELINE_DIR, 'data', 'gravity_phi_task_only.npy'))
    train_records, heldout_records = build_records(phi)

    model, x_mean, x_std = train(train_records)

    def report(records, label):
        before, after = [], []
        for q, target in records:
            before.append(np.linalg.norm(torque_to_wrench(q, target, damping=0.05)[:3]))
            g_res = predict(model, x_mean, x_std, q)
            after.append(np.linalg.norm(torque_to_wrench(q, target - g_res, damping=0.05)[:3]))
        before, after = np.array(before), np.array(after)
        print(f'\n{label} (n={len(records)}):')
        print(f'  before (linear phi only): mean ||F|| = {before.mean():.3f} N')
        print(f'  after  (+ G_res NN):      mean ||F|| = {after.mean():.3f} N  '
             f'({(before.mean()-after.mean())/before.mean()*100:+.1f}%)')

    report(train_records, 'TRAIN (dedicated poses + train-episode static stretches)')
    report(heldout_records, 'HELD-OUT (static stretches from 039/044/049)')

    torch.save({'state_dict': model.state_dict(), 'x_mean': x_mean, 'x_std': x_std},
              os.path.join(PIPELINE_DIR, 'data', 'gravity_residual_nn.pt'))
    print(f'\nSaved -> data/gravity_residual_nn.pt')


if __name__ == '__main__':
    main()
