#!/usr/bin/env python3
"""
CORRECTED head-to-head: linear regressor vs a properly-constrained neural
network for the mass/Coriolis residual -- both now share the exact same hard
physical constraint (output == 0 whenever qdot=qddot=0), so this comparison
isolates true velocity/acceleration-dependent modeling capability instead of
being confounded by pose-only gravity leakage (see compare_dynamics_residual.py
and the zero-velocity probe that caught the original NN cheating).
"""
import time
import numpy as np

from dynamics_dataset import build_dataset
from contact_detector import torque_to_wrench
import fit_dynamics_regressor as regfit
from fit_dynamics_nn_constrained import train_constrained, predict as predict_constrained

TRAIN_EPISODES = ['004', '009', '014', '019', '024', '029']
HELDOUT_EPISODES = ['039', '044', '049']
TASK = 'pastaTransfer4'


def eval_set(ds, pi, model, norm, label):
    n = len(ds['target'])
    pred_reg = regfit.predict(ds, pi)
    pred_nn = predict_constrained(model, norm, ds)

    Fn_base, Fn_reg, Fn_nn = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        q = ds['q'][i]
        Fn_base[i] = np.linalg.norm(torque_to_wrench(q, ds['target'][i], damping=0.05)[:3])
        Fn_reg[i]  = np.linalg.norm(torque_to_wrench(q, ds['target'][i] - pred_reg[i], damping=0.05)[:3])
        Fn_nn[i]   = np.linalg.norm(torque_to_wrench(q, ds['target'][i] - pred_nn[i], damping=0.05)[:3])

    print(f'\n{"="*70}\n{label}  (n={n})\n{"="*70}')
    print(f'{"":30} {"mean":>8} {"std":>8} {"min":>8} {"max":>8}')
    for name, arr in [('no correction (baseline)', Fn_base),
                      ('regressor (already q̇=q̈=0-safe)', Fn_reg),
                      ('constrained NN', Fn_nn)]:
        print(f'{name:30} {arr.mean():>8.3f} {arr.std():>8.3f} {arr.min():>8.3f} {arr.max():>8.3f}')
    red_reg = (Fn_base.mean() - Fn_reg.mean()) / Fn_base.mean() * 100
    red_nn  = (Fn_base.mean() - Fn_nn.mean()) / Fn_base.mean() * 100
    print(f'\nReduction vs baseline: regressor {red_reg:+.1f}%   constrained NN {red_nn:+.1f}%')
    return dict(base=Fn_base, reg=Fn_reg, nn=Fn_nn, red_reg=red_reg, red_nn=red_nn)


def zero_velocity_probe(model, norm, ds_heldout):
    """Sanity check: predicted residual at REAL held-out poses but with
    qdot=qddot=0 substituted -- should be exactly 0 (unlike the original
    unconstrained NN, which output ~1.48N here)."""
    ds_zero = dict(ds_heldout)
    ds_zero = dict(q=ds_heldout['q'], qdot=np.zeros_like(ds_heldout['qdot']),
                   qddot=np.zeros_like(ds_heldout['qddot']))
    pred = predict_constrained(model, norm, ds_zero)
    print(f'\nZero-velocity probe (constrained NN): max|pred|={np.abs(pred).max():.2e}, '
         f'mean|pred|={np.abs(pred).mean():.2e}  (expect exactly 0)')


def main():
    phi_gravity = np.load('data/gravity_phi_task_only.npy')

    print(f'Building TRAIN dataset from episodes {TRAIN_EPISODES} ...')
    t0 = time.time()
    ds_train = build_dataset(TRAIN_EPISODES, TASK, phi_gravity)
    print(f'  {len(ds_train["target"])} samples in {time.time()-t0:.0f}s')

    print(f'\nBuilding HELD-OUT dataset from episodes {HELDOUT_EPISODES} ...')
    t0 = time.time()
    ds_heldout = build_dataset(HELDOUT_EPISODES, TASK, phi_gravity)
    print(f'  {len(ds_heldout["target"])} samples in {time.time()-t0:.0f}s')

    print('\n--- Fitting A: ridge regression on full_dynamics_regressor (unchanged) ---')
    pi = regfit.fit(ds_train, ridge_lambda=0.1)

    print('\n--- Fitting B: CONSTRAINED neural network ---')
    model, norm = train_constrained(ds_train)

    zero_velocity_probe(model, norm, ds_heldout)

    train_results = eval_set(ds_train, pi, model, norm, 'TRAINING SET (6 episodes, 004-029)')
    heldout_results = eval_set(ds_heldout, pi, model, norm,
                               'HELD-OUT SET (3 episodes, 039/044/049 -- never trained on)')

    print(f'\n{"="*70}\nSUMMARY (corrected, constrained comparison)\n{"="*70}')
    print(f'{"":30} {"train reduction":>16} {"held-out reduction":>20}')
    print(f'{"regressor":30} {train_results["red_reg"]:>15.1f}% {heldout_results["red_reg"]:>19.1f}%')
    print(f'{"constrained NN":30} {train_results["red_nn"]:>15.1f}% {heldout_results["red_nn"]:>19.1f}%')

    np.savez('data/dynamics_comparison_v2_results.npz',
             train_base=train_results['base'], train_reg=train_results['reg'], train_nn=train_results['nn'],
             heldout_base=heldout_results['base'], heldout_reg=heldout_results['reg'], heldout_nn=heldout_results['nn'])
    print('\nSaved -> data/dynamics_comparison_v2_results.npz')

    import torch
    torch.save({'state_dict': model.state_dict(), 'norm': norm}, 'data/dynamics_residual_nn_constrained.pt')
    print('Saved -> data/dynamics_residual_nn_constrained.pt')


if __name__ == '__main__':
    main()
