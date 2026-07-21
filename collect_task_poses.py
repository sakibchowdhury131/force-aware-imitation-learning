#!/usr/bin/env python3
"""
Add REAL task-episode poses (e.g. pastaTransfer4) to the gravity-residual
training set, so calibrate_gravity_residual.py's phi fits well specifically
in the workspace region the deployed policy actually visits -- not just the
generic ANCHOR_POSES_DEG region calibrate_gravity_residual.py generates
poses around.

METHOD
  1. Load the tracked tool trajectory for each requested episode (same
     loader as replay_episode.py: augmented/tool_poses_base.npz + T_tool_eef
     derived from --mesh/--T_eef_spoon), and pick --frames_per_episode
     evenly-spaced frames from each as target EEF (Cartesian) poses.
  2. Order all picked poses for smooth travel (greedy nearest-neighbor on
     EEF position, same idea as calibrate_gravity_residual.order_poses_for_travel
     but in Cartesian space instead of joint space).
  3. Move the arm to each target via CARTESIAN position control (single
     commanded move + convergence wait -- same "safe to do in one shot"
     reasoning replay_episode.py uses for its pre-position step, since
     --arm_trans_speed already caps physical speed), then capture
     GetAngularForceGravityFree (200 samples) and read back the resulting
     JOINT angles.
  4. Appends these (q, tau_gf) records into the SAME accumulating dataset
     calibrate_gravity_residual.py uses (data/gravity_calibration_records.npz
     by default) and re-fits phi with the identical ridge_fit/report
     functions, so task poses and the generic anchor poses are fit together.

Usage:
    python collect_task_poses.py --task pastaTransfer4 --episodes 001 030 059
    python collect_task_poses.py --task pastaTransfer4 --episodes 001 030 059 --execute
"""
import os, sys, time, argparse
import numpy as np

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)

from calibrate_firmware_gravity import get_q, capture_visit
from calibrate_gravity_residual import (
    ridge_fit, report_identifiability, before_after_report, repeatability_spotcheck,
    load_session_records, save_session_records, LINK_NAMES,
)
# NOTE: load_api/connect/send_cartesian_pose/TrajectoryPoint must all come
# from replay_episode.py, NOT calibrate_firmware_gravity.py — even though
# both modules define a same-named TrajectoryPoint ctypes.Structure with
# identical fields, they are different Python classes. If SendBasicTrajectory's
# argtypes get set (by one module's load_api) to one class while the other
# module's send_cartesian_pose constructs an instance of the OTHER class,
# ctypes raises "expected TrajectoryPoint instance instead of TrajectoryPoint"
# (found the hard way). get_q/capture_visit above are safe to mix in because
# GetAngularPosition/GetAngularForceGravityFree never have argtypes set in
# either module, so no cross-module class-identity check applies to them.
from replay_episode import (
    connect, load_tool_trajectory, load_T_tool_eef,
    kinova_pose_to_matrix, matrix_to_kinova_pose, send_cartesian_pose, get_cartesian_pose,
)


def pick_frames(traj, n):
    """n evenly-spaced frames spanning [10%, 90%] of the episode -- avoids the
    very start/end where the tool may not yet be in a representative task pose."""
    n_total = len(traj)
    idx = np.linspace(0.10, 0.90, n) * (n_total - 1)
    idx = sorted(set(int(round(i)) for i in idx))
    return [traj[i] for i in idx]


def order_for_travel(targets):
    """Greedy nearest-neighbor reorder on EEF position (translation only) --
    same motivation as calibrate_gravity_residual.order_poses_for_travel
    (avoid large jumps that can exceed convergence timeouts), applied in
    Cartesian space since these are Cartesian targets."""
    remaining = list(range(len(targets)))
    order = [remaining.pop(0)]
    while remaining:
        last_xyz = targets[order[-1]][:3, 3]
        dists = [np.linalg.norm(targets[i][:3, 3] - last_xyz) for i in remaining]
        nxt = remaining.pop(int(np.argmin(dists)))
        order.append(nxt)
    return [targets[i] for i in order], order


def move_cartesian_to(api, T_target, arm_trans_speed=0.03, convergence_threshold=0.005,
                      convergence_timeout=None):
    cur = get_cartesian_pose(api)[:3]
    tgt = T_target[:3, 3]
    dist = float(np.linalg.norm(tgt - cur))
    timeout = convergence_timeout or max(5.0, dist / max(arm_trans_speed, 1e-3) + 3.0)
    send_cartesian_pose(api, matrix_to_kinova_pose(T_target), trans_speed=arm_trans_speed)
    t0 = time.time()
    converged = False
    while time.time() - t0 < timeout:
        actual = get_cartesian_pose(api)[:3]
        if np.linalg.norm(actual - tgt) < convergence_threshold:
            converged = True
            break
        time.sleep(0.02)
    return dist, converged


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', required=True, help='e.g. pastaTransfer4')
    p.add_argument('--episodes', nargs='+', required=True, help='e.g. 001 030 059')
    p.add_argument('--frames_per_episode', type=int, default=3)
    p.add_argument('--mesh', default=os.path.join(PIPELINE_DIR, 'spoon.obj'))
    p.add_argument('--T_eef_spoon', default='data/T_eef_spoon.npy')
    p.add_argument('--robot_extrinsics', default='data/robot_extrinsics.npy')
    p.add_argument('--n_samples', type=int, default=200)
    p.add_argument('--hz', type=float, default=100.0)
    p.add_argument('--arm_trans_speed', type=float, default=0.03)
    p.add_argument('--ridge_lambda', type=float, default=0.1)
    p.add_argument('--n_repeat_check', type=int, default=3)
    p.add_argument('--session_records_path', default='data/gravity_calibration_records.npz')
    p.add_argument('--output', default='data/gravity_phi.npy')
    p.add_argument('--execute', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    session_path = os.path.join(PIPELINE_DIR, args.session_records_path)
    out_path = os.path.join(PIPELINE_DIR, args.output)

    print(f'Deriving T_tool_eef from {args.mesh} + {args.T_eef_spoon} ...')
    T_tool_eef = load_T_tool_eef(args.mesh, args.T_eef_spoon)

    all_targets, all_labels = [], []
    for ep in args.episodes:
        ep_dir = os.path.join(PIPELINE_DIR, 'data', 'episodes', args.task, ep)
        traj = load_tool_trajectory(ep_dir, os.path.join(PIPELINE_DIR, args.robot_extrinsics))
        picked = pick_frames(traj, args.frames_per_episode)
        for fid, T_base_tool in picked:
            all_targets.append(T_base_tool @ T_tool_eef)
            all_labels.append(f'{args.task}/{ep}#f{fid}')

    print(f'\nPicked {len(all_targets)} target EEF poses from {len(args.episodes)} episode(s) '
         f'of "{args.task}":')
    ordered_targets, order = order_for_travel(all_targets)
    ordered_labels = [all_labels[i] for i in order]
    for lbl, T in zip(ordered_labels, ordered_targets):
        xyz = T[:3, 3]
        print(f'  {lbl:28s} xyz=({xyz[0]*100:.1f},{xyz[1]*100:.1f},{xyz[2]*100:.1f})cm')

    est_minutes = len(ordered_targets) * 35.0 / 60.0
    print(f'\nEstimated time: ~{est_minutes:.1f} min')

    if not args.execute:
        print('\n*** DRY RUN — no hardware connection, no motion. Add --execute to run for real. ***')
        return

    api = connect()   # replay_episode.connect(): sets Cartesian control + reapplies gravity params
    try:
        new_records = []
        for i, (lbl, T_target) in enumerate(zip(ordered_labels, ordered_targets)):
            print(f'\n[{i+1}/{len(ordered_targets)}] {lbl}', end='  ')
            dist, converged = move_cartesian_to(api, T_target, arm_trans_speed=args.arm_trans_speed)
            if not converged:
                print(f'(dist={dist*100:.1f}cm) WARNING: convergence timed out — sampling anyway', end='  ')
            tau_gf = capture_visit(api, n_samples=args.n_samples, hz=args.hz)
            q_actual = get_q(api)
            print(f'q={np.round(q_actual, 1)}')
            new_records.append((q_actual, tau_gf))

        prior_records = load_session_records(session_path)
        print(f'\n{len(prior_records)} prior records in {session_path}')
        records = prior_records + new_records
        save_session_records(records, session_path)
        print(f'Saved {len(records)} total accumulated records -> {session_path} '
             f'({len(new_records)} new task-pose records)')

        phi, cond_number, S, poorly_identified, Vt = ridge_fit(records, args.ridge_lambda)
        print(f'\nFitted phi (24 values, ridge_lambda={args.ridge_lambda}):')
        print(np.round(phi, 4))
        print(f'\ncond(Y_g_stacked) [unregularized] = {cond_number}')
        report_identifiability(S, poorly_identified, Vt, LINK_NAMES)
        before_after_report(records, phi)
        # repeatability_spotcheck moves via calibrate_firmware_gravity.move_and_settle
        # (JOINT-space SendBasicTrajectory), but `api` was connected via replay_episode's
        # load_api() (CARTESIAN-space) which registered SendBasicTrajectory.argtypes
        # against replay_episode's OWN TrajectoryPoint class -- a different Python class
        # than calibrate_firmware_gravity's, despite identical fields. Re-point argtypes
        # at the class move_and_settle will actually construct before calling it.
        from calibrate_firmware_gravity import TrajectoryPoint as _CFG_TrajectoryPoint
        api.SendBasicTrajectory.argtypes = [_CFG_TrajectoryPoint]
        repeatability_spotcheck(api, records, phi, args.n_repeat_check, args.n_samples, args.hz)

        np.save(out_path, phi)
        meta_path = out_path.replace('.npy', '_meta.npz')
        np.savez(meta_path, date=time.strftime('%Y-%m-%d %H:%M:%S'),
                 n_poses=len(records), ridge_lambda=args.ridge_lambda,
                 cond_number=cond_number, tool_description=f'task poses from {args.task}')
        print(f'\nSaved phi -> {out_path}')
        print(f'Saved metadata -> {meta_path}')

    finally:
        api.CloseAPI()
        print('\nAPI closed.')


if __name__ == '__main__':
    main()
