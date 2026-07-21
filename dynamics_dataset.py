#!/usr/bin/env python3
"""
Shared dataset builder for the mass/Coriolis RESIDUAL comparison (regressor
vs neural network). Both fitting methods predict the exact same target from
the exact same input, built here once:

    target(q,qdot,qddot) = gravity_free_torque
                            - gravity_regressor(q) @ phi_gravity   (calibrated gravity, linear part)
                            - gravity_residual_fn(q)               (fit_gravity_residual_nn's NN, optional)
                            - rnea_no_gravity(q, qdot, qddot)      (NOMINAL/uncalibrated M*qddot + C*qdot)

The gravity_residual_fn term matters: without it, every training sample's
target still carries whatever pose-dependent error the linear gravity model
misses, which pollutes the mass/Coriolis fit as pose-correlated "noise" even
though a properly qdot/qddot-gated model structurally cannot absorb it into
its own output (see fit_dynamics_nn_constrained.py) -- fit_gravity_residual_nn.py
should be run FIRST, on static-only data, and its correction passed in here.

i.e. whatever's left after removing the calibrated gravity model and the
generic-URDF rigid-body dynamics term -- exactly the "mass residual(q)*qddot +
coriolis residual(q,qdot)*qdot + friction" the user described. In free-space
motion (no contact), this residual should reduce cleanly to a small,
learnable function of (q,qdot,qddot); anything left over after a correction
maps to an EEF force via torque_to_wrench for the final before/after readout.

Reads replay_episode.py's dense logs (<episode_dir>/replay/torque_log_dense.npz):
q_deg, qdot_deg logged live, t used to finite-difference qddot per-episode
(VelocityDifferentiator, reset between episodes since they're independent
time series -- concatenating across episodes without resetting would create
a bogus large "qddot" at the seam).
"""
import os
import numpy as np

from contact_detector import (
    gravity_regressor, rnea_no_gravity, full_dynamics_regressor, VelocityDifferentiator,
)

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_episode_dense(episode_dir, task, qddot_smoothing=0.3):
    """Returns arrays (t, q_deg, qdot_deg, qddot_deg, gravity_free_torque) for
    one episode's dense replay log, with qddot finite-differenced fresh
    (differentiator reset at the start of this episode)."""
    path = os.path.join(PIPELINE_DIR, 'data', 'episodes', task, episode_dir, 'replay', 'torque_log_dense.npz')
    d = np.load(path)
    t, q_deg, qdot_deg, gf = d['t'], d['q_deg'], d['qdot_deg'], d['gravity_free_torque']
    diff = VelocityDifferentiator(smoothing=qddot_smoothing)
    qddot_deg = np.stack([diff.update(qdot_deg[i], t[i]) for i in range(len(t))])
    return t, q_deg, qdot_deg, qddot_deg, gf


def build_dataset(episodes, task, phi_gravity, qddot_smoothing=0.3, skip_first_n=5,
                  gravity_residual_fn=None):
    """Stacks (Y_full, target, q, qdot, qddot) across all given episodes.
    skip_first_n drops the first few samples of each episode (qddot estimate
    needs a couple of qdot samples to warm up -- VelocityDifferentiator
    returns exactly 0 for the very first call). gravity_residual_fn(q) ->
    (6,), if given, is subtracted too (see module docstring)."""
    Y_list, target_list, q_list, qdot_list, qddot_list, ep_id_list = [], [], [], [], [], []
    for ep in episodes:
        t, q_deg, qdot_deg, qddot_deg, gf = load_episode_dense(ep, task, qddot_smoothing)
        n = len(t)
        for i in range(skip_first_n, n):
            q, qdot, qddot = q_deg[i], qdot_deg[i], qddot_deg[i]
            tau_after_gravity = gf[i] - gravity_regressor(q) @ phi_gravity
            if gravity_residual_fn is not None:
                tau_after_gravity = tau_after_gravity - gravity_residual_fn(q)
            nominal_dyn = rnea_no_gravity(q, qdot, qddot)
            target = tau_after_gravity - nominal_dyn
            Y_list.append(full_dynamics_regressor(q, qdot, qddot))
            target_list.append(target)
            q_list.append(q); qdot_list.append(qdot); qddot_list.append(qddot)
            ep_id_list.append(ep)
    return dict(
        Y=np.stack(Y_list),                 # (N, 6, 72)
        target=np.stack(target_list),       # (N, 6)
        q=np.stack(q_list), qdot=np.stack(qdot_list), qddot=np.stack(qddot_list),
        episode=np.array(ep_id_list),
    )
