#!/usr/bin/env python3
"""
Held-out evaluation of a fitted gravity-residual phi on REAL task-episode
poses that were NOT used to fit it. Pure evaluation -- never touches the
training dataset, never refits, never saves phi.

Since the workspace is cleared (per the user: no objects for the EEF to
contact during these episodes), the honest expectation is that ||F|| after
correction should approach the general sensor/model noise floor (~1-2N,
established in test_residual_repeatability.py / verify_gravity_params.py) --
there is no real external force to recover, so anything above that floor is
leftover model error, not "detected contact."

Usage:
    python evaluate_task_poses.py --task pastaTransfer4 --episodes 006 016 026 --phi data/gravity_phi_task_only.npy
    python evaluate_task_poses.py --task pastaTransfer4 --episodes 006 016 026 --phi data/gravity_phi_task_only.npy --execute
"""
import os, sys, argparse
import numpy as np

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)

from calibrate_firmware_gravity import get_q, capture_visit
from contact_detector import torque_to_wrench, recover_external_force, compute_jacobian
from collect_task_poses import pick_frames, order_for_travel, move_cartesian_to
from replay_episode import connect, load_tool_trajectory, load_T_tool_eef


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', required=True)
    p.add_argument('--episodes', nargs='+', required=True)
    p.add_argument('--frames_per_episode', type=int, default=3)
    p.add_argument('--phi', required=True, help='Fitted phi .npy to evaluate')
    p.add_argument('--mesh', default=os.path.join(PIPELINE_DIR, 'spoon.obj'))
    p.add_argument('--T_eef_spoon', default='data/T_eef_spoon.npy')
    p.add_argument('--robot_extrinsics', default='data/robot_extrinsics.npy')
    p.add_argument('--n_samples', type=int, default=200)
    p.add_argument('--hz', type=float, default=100.0)
    p.add_argument('--arm_trans_speed', type=float, default=0.03)
    p.add_argument('--damping', type=float, default=0.05)
    p.add_argument('--execute', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    phi = np.load(os.path.join(PIPELINE_DIR, args.phi))
    print(f'Loaded phi from {args.phi}: {np.round(phi, 4)}')

    T_tool_eef = load_T_tool_eef(args.mesh, args.T_eef_spoon)

    all_targets, all_labels = [], []
    for ep in args.episodes:
        ep_dir = os.path.join(PIPELINE_DIR, 'data', 'episodes', args.task, ep)
        traj = load_tool_trajectory(ep_dir, os.path.join(PIPELINE_DIR, args.robot_extrinsics))
        picked = pick_frames(traj, args.frames_per_episode)
        for fid, T_base_tool in picked:
            all_targets.append(T_base_tool @ T_tool_eef)
            all_labels.append(f'{args.task}/{ep}#f{fid}')

    ordered_targets, order = order_for_travel(all_targets)
    ordered_labels = [all_labels[i] for i in order]
    print(f'\n{len(ordered_targets)} HELD-OUT poses (never used to fit this phi):')
    for lbl in ordered_labels:
        print(f'  {lbl}')

    if not args.execute:
        print('\n*** DRY RUN — no hardware connection, no motion. Add --execute to run for real. ***')
        return

    api = connect()
    try:
        records = []
        for i, (lbl, T_target) in enumerate(zip(ordered_labels, ordered_targets)):
            print(f'\n[{i+1}/{len(ordered_targets)}] {lbl}', end='  ')
            dist, converged = move_cartesian_to(api, T_target, arm_trans_speed=args.arm_trans_speed)
            if not converged:
                print(f'(dist={dist*100:.1f}cm) WARNING: convergence timed out — sampling anyway', end='  ')
            tau_gf = capture_visit(api, n_samples=args.n_samples, hz=args.hz)
            q_actual = get_q(api)
            cond = np.linalg.cond(compute_jacobian(q_actual))
            F_before = torque_to_wrench(q_actual, tau_gf, damping=args.damping)
            _tau_ext, F_after = recover_external_force(q_actual, tau_gf, phi, damping=args.damping)
            Fn_before = float(np.linalg.norm(F_before[:3]))
            Fn_after = float(np.linalg.norm(F_after[:3]))
            print(f'cond(J)={cond:.1f}  ||F|| before={Fn_before:.3f}N  after={Fn_after:.3f}N')
            records.append(dict(label=lbl, Fn_before=Fn_before, Fn_after=Fn_after, cond=cond))

        print('\n' + '=' * 78)
        print('HELD-OUT TASK-POSE EVALUATION (no fitting, no accumulation)')
        print('=' * 78)
        print(f'{"episode/frame":30s} {"before (N)":>11} {"after (N)":>10} {"change":>9} {"cond(J)":>9}')
        for r in records:
            pct = (r['Fn_before'] - r['Fn_after']) / r['Fn_before'] * 100.0 if r['Fn_before'] > 1e-9 else 0.0
            print(f'{r["label"]:30s} {r["Fn_before"]:>11.3f} {r["Fn_after"]:>10.3f} {pct:>8.1f}% {r["cond"]:>9.1f}')

        mb = float(np.mean([r['Fn_before'] for r in records]))
        ma = float(np.mean([r['Fn_after'] for r in records]))
        print(f'\nMean ||F|| before: {mb:.3f} N')
        print(f'Mean ||F|| after : {ma:.3f} N  ({(mb-ma)/mb*100:.1f}% average reduction)')
        print('\n(Workspace was cleared of contact objects for these episodes -- any remaining '
             '||F|| after correction reflects leftover gravity/dynamics model error, not real '
             'external force. Compare against the ~1-2N noise floor established earlier this '
             'session, e.g. test_residual_repeatability.py / verify_gravity_params.py.)')

    finally:
        api.CloseAPI()
        print('\nAPI closed.')


if __name__ == '__main__':
    main()
