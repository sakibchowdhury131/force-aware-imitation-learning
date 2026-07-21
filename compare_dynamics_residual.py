#!/usr/bin/env python3
"""
Head-to-head comparison: linear regressor vs neural network, both fitting the
SAME mass/Coriolis-residual target from the SAME (q,qdot,qddot) data, both
evaluated on the SAME held-out (never-trained-on) pastaTransfer4 episodes.

Builds the training dataset and held-out dataset ONCE (expensive: ~47ms/sample
through full_dynamics_regressor) and reuses them for both fits, rather than
each fit script rebuilding independently.
"""
import time
import numpy as np
import torch

from dynamics_dataset import build_dataset
from contact_detector import torque_to_wrench
import fit_dynamics_regressor as regfit
import fit_dynamics_nn as nnfit

TRAIN_EPISODES = ['004', '009', '014', '019', '024', '029']
HELDOUT_EPISODES = ['039', '044', '049']
TASK = 'pastaTransfer4'


def eval_set(ds, pi, model, x_mean, x_std, label):
    """Reports mean ||F|| for: no correction (baseline residual), regressor-
    corrected, NN-corrected -- on the given dataset."""
    n = len(ds['target'])
    Fn_base, Fn_reg, Fn_nn = np.zeros(n), np.zeros(n), np.zeros(n)

    pred_reg = regfit.predict(ds, pi)
    pred_nn = nnfit.predict(model, x_mean, x_std, ds)

    for i in range(n):
        q = ds['q'][i]
        Fn_base[i] = np.linalg.norm(torque_to_wrench(q, ds['target'][i], damping=0.05)[:3])
        Fn_reg[i]  = np.linalg.norm(torque_to_wrench(q, ds['target'][i] - pred_reg[i], damping=0.05)[:3])
        Fn_nn[i]   = np.linalg.norm(torque_to_wrench(q, ds['target'][i] - pred_nn[i], damping=0.05)[:3])

    print(f'\n{"="*70}\n{label}  (n={n})\n{"="*70}')
    print(f'{"":24} {"mean":>8} {"std":>8} {"min":>8} {"max":>8}')
    for name, arr in [('no correction (baseline)', Fn_base),
                      ('regressor-corrected', Fn_reg),
                      ('NN-corrected', Fn_nn)]:
        print(f'{name:24} {arr.mean():>8.3f} {arr.std():>8.3f} {arr.min():>8.3f} {arr.max():>8.3f}')
    red_reg = (Fn_base.mean() - Fn_reg.mean()) / Fn_base.mean() * 100
    red_nn  = (Fn_base.mean() - Fn_nn.mean()) / Fn_base.mean() * 100
    print(f'\nReduction vs baseline: regressor {red_reg:+.1f}%   NN {red_nn:+.1f}%')
    return dict(base=Fn_base, reg=Fn_reg, nn=Fn_nn, red_reg=red_reg, red_nn=red_nn)


def main():
    phi_gravity = np.load('data/gravity_phi_task_only.npy')

    print(f'Building TRAIN dataset from episodes {TRAIN_EPISODES} ...')
    t0 = time.time()
    ds_train = build_dataset(TRAIN_EPISODES, TASK, phi_gravity)
    print(f'  {len(ds_train["target"])} samples in {time.time()-t0:.0f}s')

    print(f'\nBuilding HELD-OUT dataset from episodes {HELDOUT_EPISODES} '
         f'(never used for fitting or model selection) ...')
    t0 = time.time()
    ds_heldout = build_dataset(HELDOUT_EPISODES, TASK, phi_gravity)
    print(f'  {len(ds_heldout["target"])} samples in {time.time()-t0:.0f}s')

    print('\n--- Fitting A: ridge regression on full_dynamics_regressor ---')
    pi = regfit.fit(ds_train, ridge_lambda=0.1)
    np.save('data/dynamics_residual_pi.npy', pi)

    print('\n--- Fitting B: MLP on (sin q, cos q, qdot, qddot) ---')
    model, x_mean, x_std = nnfit.train_mlp(ds_train)
    torch.save({'state_dict': model.state_dict(), 'x_mean': x_mean, 'x_std': x_std},
              'data/dynamics_residual_nn.pt')

    train_results = eval_set(ds_train, pi, model, x_mean, x_std, 'TRAINING SET (6 episodes, 004-029)')
    heldout_results = eval_set(ds_heldout, pi, model, x_mean, x_std,
                               'HELD-OUT SET (3 episodes, 039/044/049 -- never trained on)')

    print(f'\n{"="*70}\nSUMMARY\n{"="*70}')
    print(f'{"":30} {"train reduction":>16} {"held-out reduction":>20}')
    print(f'{"regressor":30} {train_results["red_reg"]:>15.1f}% {heldout_results["red_reg"]:>19.1f}%')
    print(f'{"neural network":30} {train_results["red_nn"]:>15.1f}% {heldout_results["red_nn"]:>19.1f}%')

    np.savez('data/dynamics_comparison_results.npz',
             train_base=train_results['base'], train_reg=train_results['reg'], train_nn=train_results['nn'],
             heldout_base=heldout_results['base'], heldout_reg=heldout_results['reg'], heldout_nn=heldout_results['nn'],
             train_episodes=TRAIN_EPISODES, heldout_episodes=HELDOUT_EPISODES)
    print('\nSaved -> data/dynamics_comparison_results.npz')


if __name__ == '__main__':
    main()
