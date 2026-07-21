#!/usr/bin/env python3
"""
Dynamics-compensated force analysis for a replay_episode.py run.

replay_episode.py's SPARSE torque_log.npz only samples one settled reading
per waypoint (after the arm has ~stopped), so treating (raw - gravity_free)
as "external force" there mostly just misses M(q)*qddot + C(q,qdot)*qdot —
it's naturally quasi-static. This script instead uses torque_log_dense.npz
(continuous samples taken DURING each move, including convergence-timeout
steps where the arm was still moving when logged), estimates qddot by
finite-differencing the logged qdot (contact_detector.VelocityDifferentiator
-- one differentiation of SDK-provided velocity, not a noisier double
difference of position), computes M(q)*qddot + C(q,qdot)*qdot analytically
via contact_detector.rnea_no_gravity, and subtracts it from the gravity-free
torque before mapping to an EEF wrench.

CAVEAT: rnea_no_gravity uses the same GENERIC/nominal link mass+COM values as
the rest of contact_detector.py (_LINK_DYNAMICS) -- unlike the gravity term
(fit to this specific arm via calibrate_gravity_residual.py's phi), the
mass/Coriolis model here is NOT empirically calibrated. Treat this as the
standard analytical rigid-body correction, directionally right, not a
hardware-tuned one.

Usage:
    python analyze_replay_dynamics.py --episode_dir data/episodes/newExperiment/001
"""
import os, sys, argparse
import numpy as np

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)
from contact_detector import torque_to_wrench, rnea_no_gravity, VelocityDifferentiator, compute_jacobian


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--episode_dir', default=None, help='Reads <episode_dir>/replay/torque_log_dense.npz')
    p.add_argument('--log_path', default=None, help='Explicit path (overrides --episode_dir)')
    p.add_argument('--damping', type=float, default=0.05)
    p.add_argument('--qddot_smoothing', type=float, default=0.5,
                   help='EMA smoothing for the finite-differenced qddot (0=none, default 0.5)')
    p.add_argument('--plot_path', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.log_path:
        log_path = args.log_path
    elif args.episode_dir:
        log_path = os.path.join(args.episode_dir, 'replay', 'torque_log_dense.npz')
    else:
        raise SystemExit('Pass --episode_dir or --log_path')

    d = np.load(log_path)
    t, q_deg, qdot_deg, gf = d['t'], d['q_deg'], d['qdot_deg'], d['gravity_free_torque']
    n = len(t)
    print(f'Loaded {log_path}: {n} dense samples over {t[-1]-t[0]:.1f}s '
          f'({n/(t[-1]-t[0]):.0f} Hz average)')

    qdiff = VelocityDifferentiator(smoothing=args.qddot_smoothing)
    Fn_static = np.zeros(n)    # gravity-free only (what analyze_replay_forces.py would show)
    Fn_dyn    = np.zeros(n)    # gravity-free minus M*qddot + C*qdot
    qdot_norm = np.zeros(n)
    qddot_norm = np.zeros(n)
    dyn_torque_norm = np.zeros(n)

    for i in range(n):
        qddot_i = qdiff.update(qdot_deg[i], t[i])
        tau_dyn = rnea_no_gravity(q_deg[i], qdot_deg[i], qddot_i)
        tau_ext = gf[i] - tau_dyn

        F_static = torque_to_wrench(q_deg[i], gf[i], damping=args.damping)
        F_dyn    = torque_to_wrench(q_deg[i], tau_ext, damping=args.damping)
        Fn_static[i] = np.linalg.norm(F_static[:3])
        Fn_dyn[i]    = np.linalg.norm(F_dyn[:3])
        qdot_norm[i] = np.linalg.norm(qdot_deg[i])
        qddot_norm[i] = np.linalg.norm(qddot_i)
        dyn_torque_norm[i] = np.linalg.norm(tau_dyn)

    print(f'\n{"i":>5} {"t(s)":>7} {"|qdot|":>8} {"|qddot|":>9} {"|tau_dyn|":>10} '
          f'{"||F|| static":>13} {"||F|| dyn-comp":>15}')
    for i in range(0, n, max(1, n // 30)):
        print(f'{i:>5} {t[i]:>7.2f} {qdot_norm[i]:>8.2f} {qddot_norm[i]:>9.1f} '
              f'{dyn_torque_norm[i]:>10.3f} {Fn_static[i]:>13.3f} {Fn_dyn[i]:>15.3f}')

    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'{"":20} {"mean":>8} {"std":>8} {"min":>8} {"max":>8} {"95th%":>8}')
    for label, arr in [('||F|| static (gf only)', Fn_static),
                       ('||F|| dyn-compensated', Fn_dyn),
                       ('|qdot| (deg/s)', qdot_norm),
                       ('|tau_dyn| (N*m)', dyn_torque_norm)]:
        print(f'{label:20} {arr.mean():>8.3f} {arr.std():>8.3f} {arr.min():>8.3f} '
              f'{arr.max():>8.3f} {np.percentile(arr,95):>8.3f}')

    reduction = Fn_static.mean() - Fn_dyn.mean()
    print(f'\nMean ||F|| reduction from dynamics compensation: {reduction:+.3f} N '
          f'({100*reduction/Fn_static.mean():.1f}% of static mean)')
    worst_i = int(np.argmax(qdot_norm))
    print(f'Fastest sample: i={worst_i}, t={t[worst_i]:.2f}s, |qdot|={qdot_norm[worst_i]:.1f} deg/s, '
          f'static ||F||={Fn_static[worst_i]:.2f}N, dyn-comp ||F||={Fn_dyn[worst_i]:.2f}N')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        axes[0].plot(t, Fn_static, color='#f58231', label='static (gravity-free only)')
        axes[0].plot(t, Fn_dyn, color='#3cb44b', label='dynamics-compensated')
        axes[0].set_ylabel('||F|| (N)'); axes[0].legend()
        axes[0].set_title('EEF force: static vs dynamics-compensated')
        axes[1].plot(t, qdot_norm, color='#4363d8')
        axes[1].set_ylabel('|qdot| (deg/s)'); axes[1].set_title('Joint speed')
        axes[2].plot(t, dyn_torque_norm, color='#911eb4')
        axes[2].set_ylabel('|tau_dyn| (N*m)'); axes[2].set_xlabel('time (s)')
        axes[2].set_title('||M(q)*qddot + C(q,qdot)*qdot||')
        fig.tight_layout()
        plot_path = args.plot_path or os.path.join(os.path.dirname(log_path), 'replay_dynamics.png')
        fig.savefig(plot_path, dpi=110)
        print(f'\nPlot -> {plot_path}')
    except Exception as e:
        print(f'\n(plot skipped: {e!r})')


if __name__ == '__main__':
    main()
