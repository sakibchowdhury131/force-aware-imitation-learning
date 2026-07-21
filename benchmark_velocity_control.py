#!/usr/bin/env python3
"""
Benchmark for Kinova Jaco2 CARTESIAN_VELOCITY streaming, before building the
full spline + admittance continuous-control architecture on top of it.

Context: TrajectoryPoint.Position.Type = CARTESIAN_VELOCITY (KinovaTypes.h
POSITION_TYPE enum, value 7) reinterprets the same CartesianInfo fields
SendBasicTrajectory already uses for position commands (07_deploy.py,
replay_episode.py) as a velocity vector (m/s, rad/s) instead. kinova-ros
(https://github.com/Kinovarobotics/kinova-ros) confirms this is a real,
working mode on this SDK, with two hard requirements: publishing must sustain
>=100Hz or "the robot will not able to achieve the requested velocity", and
motion stops the instant publishing stops (no coasting on a stale command --
a hung control loop fails safe).

This script tests exactly those two things, standalone, before committing to
the full two-thread spline/admittance architecture:
  1. Achievable SendBasicTrajectory call rate over this USB/ctypes link when
     streaming CARTESIAN_VELOCITY points at --rate_hz.
  2. Whether the arm actually moves the expected distance and stops cleanly.

Default is a TIMING-ONLY dry run: streams ZERO-velocity CARTESIAN_VELOCITY
commands at --rate_hz for --duration seconds -- exercises the exact same
SendBasicTrajectory call/USB round trip as a real test, with zero risk of
motion (commanded velocity is 0). Pass --execute to send the real --speed
value instead (the arm WILL move, --speed m/s along --axis).

Usage:
    # Timing-only (arm does not move)
    python benchmark_velocity_control.py --rate_hz 100 --duration 2.0

    # Real motion test (small, slow, short by default)
    python benchmark_velocity_control.py --rate_hz 100 --duration 2.0 \\
        --speed 0.02 --axis x --execute
"""
import os, sys, time, ctypes, argparse, importlib.util
import numpy as np

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)

CARTESIAN_VELOCITY = 7   # KinovaTypes.h POSITION_TYPE enum


def _load_deploy_module():
    """'07_deploy' starts with a digit, not a valid module name for `import` --
    load it by file path instead. Reuses its ctypes structs / connect() /
    get_cartesian_pose() unchanged -- zero duplication of the Jaco SDK
    bindings, zero modification to that file."""
    spec = importlib.util.spec_from_file_location(
        '_deploy07', os.path.join(PIPELINE_DIR, '07_deploy.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def send_cartesian_velocity(deploy, api, vx, vy, vz, wx, wy, wz):
    """One CARTESIAN_VELOCITY TrajectoryPoint (m/s, rad/s)."""
    tp = deploy.TrajectoryPoint()
    ctypes.memset(ctypes.byref(tp), 0, ctypes.sizeof(tp))
    tp.Position.Type = CARTESIAN_VELOCITY
    tp.Position.CartesianPosition.X = float(vx)
    tp.Position.CartesianPosition.Y = float(vy)
    tp.Position.CartesianPosition.Z = float(vz)
    tp.Position.CartesianPosition.ThetaX = float(wx)
    tp.Position.CartesianPosition.ThetaY = float(wy)
    tp.Position.CartesianPosition.ThetaZ = float(wz)
    tp.Position.HandMode = deploy.HAND_NOMOVEMENT
    api.SendBasicTrajectory(tp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rate_hz', type=float, default=100.0,
                   help='Target publish rate (kinova-ros says 100Hz is required for '
                        'the robot to achieve the requested velocity).')
    p.add_argument('--duration', type=float, default=2.0,
                   help='How long to stream velocity commands, seconds.')
    p.add_argument('--speed', type=float, default=0.02,
                   help='Commanded Cartesian translation velocity, m/s. Only applied '
                        'with --execute -- otherwise streams zero velocity (timing-only).')
    p.add_argument('--axis', choices=['x', 'y', 'z'], default='x',
                   help='Which axis to command --speed along (base frame).')
    p.add_argument('--execute', action='store_true',
                   help='Actually send --speed (nonzero) -- the arm WILL move. '
                        'Without this, streams zero-velocity commands: same call rate, '
                        'same SDK/USB round trip, zero motion.')
    args = p.parse_args()

    deploy = _load_deploy_module()
    api = deploy.connect()

    vx = vy = vz = 0.0
    if args.execute:
        if args.axis == 'x':
            vx = args.speed
        elif args.axis == 'y':
            vy = args.speed
        else:
            vz = args.speed
        expected_cm = args.speed * args.duration * 100
        print(f"\n*** EXECUTE: commanding {args.speed*100:.1f} cm/s along {args.axis} "
              f"for {args.duration:.1f}s (~{expected_cm:.1f} cm expected displacement) ***")
        print("Ctrl+C within the next 3 seconds to abort...")
        time.sleep(3)
    else:
        print("\n*** TIMING-ONLY dry run: streaming ZERO velocity (arm will not move). "
              "Pass --execute to command real motion. ***\n")

    pose_before = deploy.get_cartesian_pose(api)

    dt_nominal = 1.0 / args.rate_hz
    n_ticks = int(args.duration * args.rate_hz)
    call_latencies = []
    t_start = time.time()
    next_tick = t_start
    for _ in range(n_ticks):
        t0 = time.time()
        send_cartesian_velocity(deploy, api, vx, vy, vz, 0, 0, 0)
        call_latencies.append(time.time() - t0)
        next_tick += dt_nominal
        sleep_for = next_tick - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)
    t_end = time.time()

    # Explicit stop -- a few zero-velocity ticks. kinova-ros says motion stops
    # the instant publishing stops, but send a deterministic stop regardless.
    for _ in range(5):
        send_cartesian_velocity(deploy, api, 0, 0, 0, 0, 0, 0)
        time.sleep(dt_nominal)

    pose_after = deploy.get_cartesian_pose(api)

    # ── Report ──────────────────────────────────────────────────────────────
    actual_duration = t_end - t_start
    actual_rate = n_ticks / actual_duration if actual_duration > 0 else float('nan')
    lat = np.array(call_latencies)

    print(f"\n=== Timing ===")
    print(f"  Requested: {n_ticks} ticks @ {args.rate_hz:.0f}Hz over {args.duration:.2f}s")
    print(f"  Actual:    {n_ticks} ticks in {actual_duration:.3f}s -> "
          f"{actual_rate:.1f}Hz achieved")
    print(f"  SendBasicTrajectory call latency (ms): "
          f"mean={lat.mean()*1000:.2f}  max={lat.max()*1000:.2f}  "
          f"p99={np.percentile(lat, 99)*1000:.2f}")
    if actual_rate < 0.95 * args.rate_hz:
        print(f"  WARNING: achieved rate is >5% below the {args.rate_hz:.0f}Hz target -- "
              f"per kinova-ros, the robot may not reach the commanded velocity at this rate.")

    disp = pose_after[:3] - pose_before[:3]
    disp_mag = np.linalg.norm(disp)
    expected_disp = args.speed * args.duration if args.execute else 0.0
    print(f"\n=== Motion ===")
    print(f"  Pose before (cm): {np.round(pose_before[:3]*100, 2)}")
    print(f"  Pose after  (cm): {np.round(pose_after[:3]*100, 2)}")
    print(f"  Displacement: {disp_mag*100:.2f} cm  (expected ~{expected_disp*100:.2f} cm)")
    if args.execute and expected_disp > 0:
        print(f"  Achieved/expected ratio: {disp_mag/expected_disp:.2f}")


if __name__ == '__main__':
    main()
