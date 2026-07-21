#!/usr/bin/env python3
"""
CORRECTED neural-network residual model for mass/Coriolis compensation.

The original fit_dynamics_nn.py (ResidualMLP) predicted torque directly from
(sin q, cos q, qdot, qddot) as plain input features -- nothing stopped it
from learning a pose-only function disguised as a "dynamics residual", which
a zero-velocity probe confirmed it was doing (its output at synthetic
qdot=qddot=0 was nearly as large as its entire real-world correction).

This module fixes that by construction, matching the exact structure of the
true equation:

    tau_residual = M_res(q) @ qddot  +  C_res(q, qdot) @ qdot

  - M_res(q): a small net taking ONLY q (as sin/cos) -> a symmetric 6x6
    matrix. Mass matrices are physically a function of configuration only,
    never velocity -- so this network never sees qdot/qddot at all.
  - C_res(q, qdot): a small net taking (q, qdot) -> a general 6x6 matrix.

Both contributions are then multiplied by qddot / qdot respectively (in
rad/s, rad/s^2 -- matching contact_detector's internal convention), so the
combined output is EXACTLY zero whenever qdot=qddot=0, for ANY network
weights -- the same hard physical constraint full_dynamics_regressor already
has by construction (verified in contact_detector.py's self-test), which the
original monolithic NN lacked.
"""
import numpy as np
import torch
import torch.nn as nn

N_JOINTS = 6
# 21 independent entries of a symmetric 6x6 matrix (row, col) with col>=row
_SYM_INDICES = [(r, c) for r in range(N_JOINTS) for c in range(r, N_JOINTS)]


def make_q_features(q_deg):
    q = np.deg2rad(q_deg)
    return np.concatenate([np.sin(q), np.cos(q)], axis=-1)   # 12-dim


def make_qqdot_features(q_deg, qdot_deg):
    q = np.deg2rad(q_deg)
    qdot = np.deg2rad(qdot_deg)
    return np.concatenate([np.sin(q), np.cos(q), qdot], axis=-1)   # 18-dim


class _SymmetricMatrixHead(nn.Module):
    """q (as sin/cos) -> symmetric 6x6 matrix, via 21 independent entries."""
    def __init__(self, in_dim=12, hidden=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, len(_SYM_INDICES)),
        )

    def forward(self, x):
        vals = self.net(x)                      # (B, 21)
        B = vals.shape[0]
        M = torch.zeros(B, N_JOINTS, N_JOINTS, dtype=vals.dtype, device=vals.device)
        for k, (r, c) in enumerate(_SYM_INDICES):
            M[:, r, c] = vals[:, k]
            M[:, c, r] = vals[:, k]
        return M


class _GeneralMatrixHead(nn.Module):
    """(q, qdot) -> general (not necessarily symmetric) 6x6 matrix."""
    def __init__(self, in_dim=18, hidden=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_JOINTS * N_JOINTS),
        )

    def forward(self, x):
        B = x.shape[0]
        return self.net(x).view(B, N_JOINTS, N_JOINTS)


class ConstrainedResidualNet(nn.Module):
    """tau_residual = M_res(q) @ qddot + C_res(q,qdot) @ qdot -- EXACTLY zero
    at qdot=qddot=0 by construction (the multiplication, not the network
    weights, enforces this)."""
    def __init__(self, hidden=48):
        super().__init__()
        self.m_head = _SymmetricMatrixHead(hidden=hidden)
        self.c_head = _GeneralMatrixHead(hidden=hidden)

    def forward(self, q_feat, qc_feat, qdot_rad, qddot_rad):
        M = self.m_head(q_feat)                          # (B,6,6)
        C = self.c_head(qc_feat)                          # (B,6,6)
        tau_m = torch.bmm(M, qddot_rad.unsqueeze(-1)).squeeze(-1)
        tau_c = torch.bmm(C, qdot_rad.unsqueeze(-1)).squeeze(-1)
        return tau_m + tau_c


def train_constrained(dataset, val_frac=0.15, epochs=400, lr=1e-3, weight_decay=1e-4,
                      hidden=48, seed=0, patience=40):
    rng = np.random.default_rng(seed)
    q_feat = make_q_features(dataset['q']).astype(np.float32)
    qc_feat = make_qqdot_features(dataset['q'], dataset['qdot']).astype(np.float32)
    qdot_rad = np.deg2rad(dataset['qdot']).astype(np.float32)
    qddot_rad = np.deg2rad(dataset['qddot']).astype(np.float32)
    Y = dataset['target'].astype(np.float32)

    n = len(Y)
    idx = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    # standardize the FEATURE inputs only (not qdot_rad/qddot_rad -- those are
    # the physical multiplicative gate, must stay in real rad/s, rad/s^2 units)
    qf_mean, qf_std = q_feat[train_idx].mean(0), q_feat[train_idx].std(0) + 1e-6
    qcf_mean, qcf_std = qc_feat[train_idx].mean(0), qc_feat[train_idx].std(0) + 1e-6
    q_feat_n = (q_feat - qf_mean) / qf_std
    qc_feat_n = (qc_feat - qcf_mean) / qcf_std

    def to_t(arr, idx_):
        return torch.tensor(arr[idx_])

    model = ConstrainedResidualNet(hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val, best_state, bad_epochs = float('inf'), None, 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(to_t(q_feat_n, train_idx), to_t(qc_feat_n, train_idx),
                    to_t(qdot_rad, train_idx), to_t(qddot_rad, train_idx))
        loss = loss_fn(pred, to_t(Y, train_idx))
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(to_t(q_feat_n, val_idx), to_t(qc_feat_n, val_idx),
                             to_t(qdot_rad, val_idx), to_t(qddot_rad, val_idx))
            val_loss = loss_fn(val_pred, to_t(Y, val_idx)).item()
        if val_loss < best_val - 1e-6:
            best_val, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs > patience:
                print(f'  early stop at epoch {epoch}, best val MSE={best_val:.4f}')
                break
    model.load_state_dict(best_state)
    print(f'Trained ConstrainedResidualNet: {n - n_val} train / {n_val} val samples, '
         f'best val MSE={best_val:.4f}')
    return model, dict(qf_mean=qf_mean, qf_std=qf_std, qcf_mean=qcf_mean, qcf_std=qcf_std)


def predict(model, norm, dataset):
    q_feat = (make_q_features(dataset['q']).astype(np.float32) - norm['qf_mean']) / norm['qf_std']
    qc_feat = (make_qqdot_features(dataset['q'], dataset['qdot']).astype(np.float32)
              - norm['qcf_mean']) / norm['qcf_std']
    qdot_rad = np.deg2rad(dataset['qdot']).astype(np.float32)
    qddot_rad = np.deg2rad(dataset['qddot']).astype(np.float32)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(q_feat), torch.tensor(qc_feat),
                    torch.tensor(qdot_rad), torch.tensor(qddot_rad))
    return pred.numpy()
