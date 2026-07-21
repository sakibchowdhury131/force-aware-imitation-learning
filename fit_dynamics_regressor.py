#!/usr/bin/env python3
"""
Fit A of the mass/Coriolis-residual comparison: ridge regression on the full
linear dynamics regressor (contact_detector.full_dynamics_regressor), same
method as calibrate_gravity_residual.py's gravity fit, just a bigger Y (72
columns: 60 inertial + 12 friction) and a motion-derived target instead of a
static-pose one.
"""
import numpy as np


def ridge_fit_generic(Y_stacked: np.ndarray, target_stacked: np.ndarray, ridge_lambda: float):
    """Y_stacked: (N,), target_stacked: (N,) after flattening the per-sample
    (6, n_params) / (6,) blocks. Returns (pi, cond_number, S, poorly_identified, Vt)."""
    n_params = Y_stacked.shape[1]
    U, S, Vt = np.linalg.svd(Y_stacked, full_matrices=False)
    cond_number = float(S[0] / S[-1]) if S[-1] > 1e-12 else float('inf')
    poorly_identified = S < (0.01 * S[0])

    Y_aug = np.vstack([Y_stacked, np.sqrt(ridge_lambda) * np.eye(n_params)])
    tau_aug = np.concatenate([target_stacked, np.zeros(n_params)])
    pi, *_ = np.linalg.lstsq(Y_aug, tau_aug, rcond=None)
    return pi, cond_number, S, poorly_identified, Vt


def flatten_blocks(Y_blocks: np.ndarray, target_blocks: np.ndarray):
    """(N,6,P),(N,6) -> (6N,P),(6N,)"""
    N, six, P = Y_blocks.shape
    return Y_blocks.reshape(N * six, P), target_blocks.reshape(N * six)


def fit(dataset, ridge_lambda=0.1):
    Y_stacked, target_stacked = flatten_blocks(dataset['Y'], dataset['target'])
    pi, cond_number, S, poorly_identified, Vt = ridge_fit_generic(Y_stacked, target_stacked, ridge_lambda)
    n_poor = int(poorly_identified.sum())
    print(f'Fitted pi ({len(pi)} params, ridge_lambda={ridge_lambda}): '
         f'{n_poor}/{len(pi)} poorly identified (<1% of largest singular value)')
    return pi


def predict(dataset, pi):
    """Returns per-sample predicted residual torque (N,6)."""
    return np.einsum('nij,j->ni', dataset['Y'], pi)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from dynamics_dataset import build_dataset

    phi_gravity = np.load('data/gravity_phi_task_only.npy')
    train_eps = ['004', '009', '014', '019', '024', '029']
    print('Building training dataset...')
    ds = build_dataset(train_eps, 'pastaTransfer4', phi_gravity)
    print(f'{len(ds["target"])} training samples')
    pi = fit(ds)
    np.save('data/dynamics_residual_pi.npy', pi)
    print('Saved -> data/dynamics_residual_pi.npy')
