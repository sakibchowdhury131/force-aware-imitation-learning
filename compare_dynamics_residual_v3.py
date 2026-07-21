#!/usr/bin/env python3
"""
FINAL corrected head-to-head: linear regressor vs constrained neural network
for the mass/Coriolis residual, with the gravity-residual NN (fit_gravity_
residual_nn.py, static-only data) subtracted from the target FIRST -- so
neither dynamics model is being asked to explain pose-only leftover error.

Three fixes stacked on top of the original (flawed) comparison:
  1. gravity_residual_nn.py closes most of the linear gravity model's
     remaining pose-dependent gap (68.2% train / 41.3% held-out reduction,
     fit on STATIC data only, never motion data).
  2. The dynamics target now subtracts that too, before either dynamics
     model ever sees it.
  3. The neural network is now ConstrainedResidualNet -- mathematically zero
     at qdot=qddot=0 by construction, same hard constraint the regressor
     already had, so this is a fair like-for-like comparison.
"""
import time
import numpy as np
import torch

from dynamics_dataset import build_dataset
from contact_detector import torque_to_wrench
import fit_dynamics_regressor as regfit
from fit_dynamics_nn_constrained import train_constrained, predict as predict_constrained
from fit_gravity_residual_nn import GravityResidualNet, predict as predict_gravity_residual

TRAIN_EPISODES = ['001', '004', '005', '009', '010', '011', '012', '014', '015',
                  '017', '018', '019', '021', '022', '023', '024', '028', '029',
                  '030', '031', '032', '033', '035', '037', '038', '040', '042',
                  '043', '045', '047']
HELDOUT_EPISODES = ['039', '044', '049']
TASK = 'pastaTransfer4'


def load_gravity_residual_fn():
    ckpt = torch.load('data/gravity_residual_nn.pt', weights_only=False)
    model = GravityResidualNet(); model.load_state_dict(ckpt['state_dict']); model.eval()
    x_mean, x_std = ckpt['x_mean'], ckpt['x_std']
    return lambda q: predict_gravity_residual(model, x_mean, x_std, q)


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
                      ('regressor', Fn_reg),
                      ('constrained NN', Fn_nn)]:
        print(f'{name:30} {arr.mean():>8.3f} {arr.std():>8.3f} {arr.min():>8.3f} {arr.max():>8.3f}')
    red_reg = (Fn_base.mean() - Fn_reg.mean()) / Fn_base.mean() * 100
    red_nn  = (Fn_base.mean() - Fn_nn.mean()) / Fn_base.mean() * 100
    print(f'\nReduction vs baseline: regressor {red_reg:+.1f}%   constrained NN {red_nn:+.1f}%')
    return dict(base=Fn_base, reg=Fn_reg, nn=Fn_nn, red_reg=red_reg, red_nn=red_nn)


def zero_velocity_probe(model, norm, ds_heldout):
    ds_zero = dict(q=ds_heldout['q'], qdot=np.zeros_like(ds_heldout['qdot']),
                   qddot=np.zeros_like(ds_heldout['qddot']))
    pred = predict_constrained(model, norm, ds_zero)
    print(f'\nZero-velocity probe (constrained NN): max|pred|={np.abs(pred).max():.2e}, '
         f'mean|pred|={np.abs(pred).mean():.2e}  (expect exactly 0)')


def main():
    phi_gravity = np.load('data/gravity_phi_task_only.npy')
    gravity_residual_fn = load_gravity_residual_fn()

    print(f'Building TRAIN dataset from episodes {TRAIN_EPISODES} (with gravity-residual-NN correction) ...')
    t0 = time.time()
    ds_train = build_dataset(TRAIN_EPISODES, TASK, phi_gravity, gravity_residual_fn=gravity_residual_fn)
    print(f'  {len(ds_train["target"])} samples in {time.time()-t0:.0f}s')

    print(f'\nBuilding HELD-OUT dataset from episodes {HELDOUT_EPISODES} ...')
    t0 = time.time()
    ds_heldout = build_dataset(HELDOUT_EPISODES, TASK, phi_gravity, gravity_residual_fn=gravity_residual_fn)
    print(f'  {len(ds_heldout["target"])} samples in {time.time()-t0:.0f}s')

    print('\n--- Fitting A: ridge regression on full_dynamics_regressor ---')
    pi = regfit.fit(ds_train, ridge_lambda=0.1)

    print('\n--- Fitting B: CONSTRAINED neural network ---')
    model, norm = train_constrained(ds_train)

    zero_velocity_probe(model, norm, ds_heldout)

    train_results = eval_set(ds_train, pi, model, norm, 'TRAINING SET (6 episodes, 004-029)')
    heldout_results = eval_set(ds_heldout, pi, model, norm,
                               'HELD-OUT SET (3 episodes, 039/044/049 -- never trained on)')

    print(f'\n{"="*70}\nFINAL SUMMARY (gravity-NN-corrected target, constrained NN)\n{"="*70}')
    print(f'{"":30} {"train reduction":>16} {"held-out reduction":>20}')
    print(f'{"regressor":30} {train_results["red_reg"]:>15.1f}% {heldout_results["red_reg"]:>19.1f}%')
    print(f'{"constrained NN":30} {train_results["red_nn"]:>15.1f}% {heldout_results["red_nn"]:>19.1f}%')

    np.savez('data/dynamics_comparison_v3_results.npz',
             train_base=train_results['base'], train_reg=train_results['reg'], train_nn=train_results['nn'],
             heldout_base=heldout_results['base'], heldout_reg=heldout_results['reg'], heldout_nn=heldout_results['nn'])
    print('\nSaved -> data/dynamics_comparison_v3_results.npz')

    torch.save({'state_dict': model.state_dict(), 'norm': norm}, 'data/dynamics_residual_nn_constrained_v3.pt')
    print('Saved -> data/dynamics_residual_nn_constrained_v3.pt')


if __name__ == '__main__':
    main()
