#!/usr/bin/env python3
"""
End-to-end: record a human demonstration, track it, pause for you to move
the tool onto the robot gripper, then replay it on the robot while saving
images and calibrated force data.

Chains the already-tested individual scripts as subprocesses (each stage
keeps its own hardware/CUDA setup isolated -- 01_record.py's camera+FP
session is a completely different process context than replay_episode.py's
Kinova SDK connection, so merging them in-process would be fragile). Aborts
the whole run if any stage fails, rather than plowing ahead with bad data.

Stages:
  1. 01_record.py --auto_track       (record + track + visualize the demo)
  2. [PAUSE] -- attach the tool to the robot gripper, press ENTER
  3. 04c_to_base_frame.py            (task frame -> robot base frame)
  4. replay_episode.py (dry run)     (sanity-check the planned motion)
  5. [CONFIRM] -- press ENTER to actually move the robot, or Ctrl+C to abort
  6. replay_episode.py --execute --capture_camera  (real motion + images + raw torque)
  7. analyze_replay_full.py          (calibrated force data + per-frame image tagging)

Usage:
    python record_and_replay_episode.py --task PastaTransfer_force \\
        --mesh newspoon1.obj --tool_prompt "spoon"

    # explicit episode id, skip the recording stage (episode already recorded/tracked)
    python record_and_replay_episode.py --task PastaTransfer_force --episode 003 \\
        --mesh newspoon1.obj --tool_prompt "spoon" --skip_record
"""
import os, sys, argparse, subprocess

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))


def next_episode_id(task_dir):
    if not os.path.isdir(task_dir):
        return '001'
    existing = [d for d in os.listdir(task_dir)
               if os.path.isdir(os.path.join(task_dir, d)) and d.isdigit()]
    if not existing:
        return '001'
    return f'{max(int(d) for d in existing) + 1:03d}'


def check_cameras(min_count):
    import pyrealsense2 as rs
    n = len(list(rs.context().devices))
    if n < min_count:
        print(f'ERROR: need at least {min_count} camera(s), found {n}.')
        sys.exit(1)
    print(f'  {n} camera(s) detected.')


def check_kinova():
    r = subprocess.run(['lsusb'], capture_output=True, text=True)
    if 'Kinova' not in r.stdout:
        print('ERROR: no Kinova device found on USB. Check power + cable, then retry.')
        sys.exit(1)
    print('  Kinova arm detected on USB.')


def run_stage(label, cmd):
    print(f'\n{"="*70}\n{label}\n{"="*70}')
    print('  $ ' + ' '.join(cmd))
    result = subprocess.run(cmd, cwd=PIPELINE_DIR)
    if result.returncode != 0:
        print(f'\nERROR: "{label}" failed (exit {result.returncode}). Aborting the rest of the pipeline.')
        sys.exit(result.returncode)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', required=True)
    p.add_argument('--episode', default=None, help='Default: auto-increment from existing episodes')
    p.add_argument('--duration', type=float, default=15.0)
    p.add_argument('--mesh', required=True)
    p.add_argument('--tool_prompt', required=True)
    p.add_argument('--track_cam', type=int, default=1)
    p.add_argument('--task_frame', default=None,
                   help='Camera extrinsics for stage-1 tracking, passed through to 01_record.py. '
                        'Defaults to 01_record.py\'s own default (data/cam{track_cam}_extrinsics.npy) '
                        'when omitted -- pass this explicitly if this rig/task uses different camera '
                        'calibration files (e.g. a different physical camera position per task).')
    p.add_argument('--box_threshold', type=float, default=None,
                   help='GroundingDINO box threshold, passed through to 01_record.py. Defaults to '
                        '01_record.py\'s own default (0.3) when omitted.')
    p.add_argument('--text_threshold', type=float, default=None,
                   help='GroundingDINO text threshold, passed through to 01_record.py. Defaults to '
                        '01_record.py\'s own default (0.25) when omitted.')
    p.add_argument('--robot_extrinsics', default='data/robot_extrinsics_stick_corrected_zmeasured.npy')
    p.add_argument('--T_eef_spoon', default='data/T_eef_spoon.npy')
    p.add_argument('--subsample', type=int, default=3)
    p.add_argument('--speed_scale', type=float, default=0.3)
    p.add_argument('--arm_trans_speed', type=float, default=0.05)
    p.add_argument('--auto_track_refine_iter', type=int, default=5)
    p.add_argument('--auto_track_est_refine_iter', type=int, default=8)
    p.add_argument('--skip_record', action='store_true',
                   help='Skip stage 1 -- use an already-recorded/tracked episode '
                        '(requires --episode).')
    p.add_argument('--skip_visualize', action='store_true',
                   help='Skip the auto-visualize sanity check after recording.')
    p.add_argument('--skip_confirm', action='store_true',
                   help='Do not pause for confirmation before real robot motion. '
                        'Only use this once you trust the pipeline -- default is to '
                        'always show the dry run and wait for ENTER first.')
    return p.parse_args()


def main():
    args = parse_args()
    if args.skip_record and args.episode is None:
        print('ERROR: --skip_record requires --episode (which existing episode to replay).')
        sys.exit(1)

    task_dir = os.path.join('data', 'episodes', args.task)

    print('Checking cameras...')
    check_cameras(args.track_cam + 1)

    if not args.skip_record:
        episode = args.episode or next_episode_id(task_dir)
        print(f'\nRecording episode: {args.task}/{episode}')
        run_stage('STAGE 1 — Record + auto-track + visualize', [
            sys.executable, '01_record.py',
            '--task', args.task, '--episode', episode,
            '--duration', str(args.duration),
            '--mesh', args.mesh, '--tool_prompt', args.tool_prompt,
            '--track_cam', str(args.track_cam),
            '--auto_track',
            '--auto_track_refine_iter', str(args.auto_track_refine_iter),
            '--auto_track_est_refine_iter', str(args.auto_track_est_refine_iter),
        ] + (['--skip_auto_visualize'] if args.skip_visualize else [])
          + (['--task_frame', args.task_frame] if args.task_frame else [])
          + (['--box_threshold', str(args.box_threshold)] if args.box_threshold is not None else [])
          + (['--text_threshold', str(args.text_threshold)] if args.text_threshold is not None else []))
    else:
        episode = args.episode

    episode_dir = os.path.join(task_dir, episode)
    if not args.skip_visualize and not args.skip_record:
        viz_dir = os.path.join(episode_dir, 'augmented', 'viz_poses')
        print(f'\nSpot-check the tracking before continuing: eog {viz_dir}/*_cam{args.track_cam}_pose.jpg')

    print(f'\n{"="*70}')
    print(f'PAUSE — attach {args.tool_prompt!r} to the robot gripper now.')
    print(f'{"="*70}')
    input('Press ENTER once the tool is gripped and ready for replay (Ctrl+C to abort)... ')

    print('\nChecking robot connection...')
    check_kinova()

    run_stage('STAGE 2 — Convert to robot base frame', [
        sys.executable, '04c_to_base_frame.py',
        '--episode_dir', episode_dir,
        '--robot_extrinsics', args.robot_extrinsics,
    ])

    dry_run_cmd = [
        sys.executable, 'replay_episode.py',
        '--episode_dir', episode_dir,
        '--mesh', args.mesh,
        '--T_eef_spoon', args.T_eef_spoon,
        '--robot_extrinsics', args.robot_extrinsics,
        '--subsample', str(args.subsample),
    ]
    run_stage('STAGE 3 — Dry run (no motion, sanity check)', dry_run_cmd)

    if not args.skip_confirm:
        print(f'\n{"="*70}')
        input('Dry run looks OK? Press ENTER to execute for REAL on the robot '
              '(Ctrl+C to abort)... ')

    run_stage('STAGE 4 — Replay for real (motion + images + raw torque)', dry_run_cmd + [
        '--execute', '--capture_camera',
        '--no_wait_convergence',
        '--speed_scale', str(args.speed_scale),
        '--arm_trans_speed', str(args.arm_trans_speed),
    ])

    run_stage('STAGE 5 — Calibrated force analysis + image/force tagging', [
        sys.executable, 'analyze_replay_full.py',
        '--episode_dir', episode_dir,
    ])

    print(f'\n{"="*70}')
    print(f'DONE — {args.task}/{episode}')
    print(f'{"="*70}')
    print(f'  Demo frames (all cameras):   {episode_dir}/cam*/')
    print(f'  Tracked poses:               {episode_dir}/augmented/tool_poses_base.npz')
    print(f'  Replay images (all cameras): {episode_dir}/replay/cam*/')
    print(f'  Raw torque (dense): {episode_dir}/replay/torque_log_dense.npz')
    print(f'  Raw torque (waypoints): {episode_dir}/replay/torque_log.npz')
    print(f'  Calibrated forces (dense):    {episode_dir}/replay/replay_full_forces.npz')
    print(f'  Calibrated forces (per-frame, tagged to images): {episode_dir}/replay/replay_forces_per_frame.npz')
    print(f'  Plot:               {episode_dir}/replay/replay_full_forces.png')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nAborted by user.')
        sys.exit(130)
