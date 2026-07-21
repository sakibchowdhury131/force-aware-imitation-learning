#!/usr/bin/env python3
"""
Benchmark for the force/admittance side of the proposed continuous-streaming
deployment architecture, before building it. Read-only: never sends a motion
command (connects with control=False, same as diag_live_forces.py) -- zero
risk to the arm, and completely separate from 07_deploy.py / test_policy.py,
which are NOT touched by this script at all. Existing position-based
deployments are unaffected.

Reuses the already-built, already-calibrated force-recovery pipeline
(contact_detector.py, calibrate_firmware_gravity.py, fit_gravity_residual_nn.py
-- see diag_live_forces.py, which this borrows its stages from) rather than
re-deriving anything.

Tests three things the colleague's admittance design depends on:
  1. Per-stage compute cost (firmware-only / +gravity-regressor / +gravity-NN
     / +mass-Coriolis-dynamics-regressor) -- diag_live_forces.py's own
     docstring already flags the dynamics regressor as ~25ms/step, i.e. ~40Hz
     max, far short of a 100Hz admittance loop's 10ms budget. This confirms
     exactly which stages are affordable at what rate.
  2. Noise floor of the (feasible-speed) force estimate, both raw and after a
     real-time causal 2nd-order Butterworth low-pass (scipy sosfilt with
     persistent state -- NOT filtfilt, which is non-causal/offline-only and
     unusable in a real-time loop), at the colleague's proposed 5Hz cutoff.
  3. What magnitude of admittance correction x = (F_filtered - f_desired)/K
     that noise floor would produce at rest (should be small/negligible) vs.
     during a real push (should be a sensible, well-scaled few-mm-to-cm
     response) -- computed and logged only, never applied to the robot.

Usage:
    # Stage timing only (a few seconds, arm untouched)
    python benchmark_force_admittance.py --mode timing

    # Static noise floor (arm untouched, --duration seconds)
    python benchmark_force_admittance.py --mode static --duration 5

    # Interactive: push/tap the arm during the recording window
    python benchmark_force_admittance.py --mode push --duration 8
"""
import os, sys, time, argparse
import numpy as np
import torch
from scipy.signal import butter, sosfilt, sosfilt_zi

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)

from contact_detector import torque_to_wrench, gravity_regressor, full_dynamics_regressor, VelocityDifferentiator
from fit_gravity_residual_nn import GravityResidualNet, predict as predict_gravity_residual
from calibrate_firmware_gravity import load_api, connect, get_q, get_qdot, get_tau_gf, apply_saved_gravity_params


class OnlineButterworth:
    """Causal 2nd-order low-pass, one instance per scalar channel, carrying
    filter state (zi) between calls -- the real-time-loop-compatible
    counterpart to scipy.signal.filtfilt (non-causal, needs the whole
    signal, cannot be used online)."""
    def __init__(self, cutoff_hz: float, fs_hz: float, order: int = 2):
        self.sos = butter(order, cutoff_hz, btype='low', fs=fs_hz, output='sos')
        self._zi = None

    def update(self, x: float) -> float:
        if self._zi is None:
            self._zi = sosfilt_zi(self.sos) * x   # steady-state init, avoids startup transient
        y, self._zi = sosfilt(self.sos, [x], zi=self._zi)
        return float(y[0])


def load_pipeline(args):
    phi = np.load(os.path.join(PIPELINE_DIR, args.phi))
    ckpt = torch.load(os.path.join(PIPELINE_DIR, args.gravity_nn), weights_only=False)
    g_nn = GravityResidualNet(); g_nn.load_state_dict(ckpt['state_dict']); g_nn.eval()
    g_x_mean, g_x_std = ckpt['x_mean'], ckpt['x_std']
    pi = np.load(os.path.join(PIPELINE_DIR, args.dynamics_pi)) if args.with_dynamics else None
    return phi, g_nn, g_x_mean, g_x_std, pi


def compute_force_staged(q_deg, qdot_deg, tau_gf, phi, g_nn, g_x_mean, g_x_std, pi,
                         qdiff, t_now, damping, with_dynamics):
    """Returns dict of stage_name -> (F_xyz, elapsed_seconds)."""
    stages = {}

    t0 = time.time()
    F_fw = torque_to_wrench(q_deg, tau_gf, damping=damping)
    stages['firmware_only'] = (F_fw[:3].copy(), time.time() - t0)

    t0 = time.time()
    g_res_lin = gravity_regressor(q_deg) @ phi
    tau_g = tau_gf - g_res_lin
    F_g = torque_to_wrench(q_deg, tau_g, damping=damping)
    stages['+gravity_regressor'] = (F_g[:3].copy(), time.time() - t0)

    t0 = time.time()
    g_res_nn = predict_gravity_residual(g_nn, g_x_mean, g_x_std, q_deg)
    tau_gnn = tau_g - g_res_nn
    F_gnn = torque_to_wrench(q_deg, tau_gnn, damping=damping)
    stages['+gravity_nn'] = (F_gnn[:3].copy(), time.time() - t0)

    if with_dynamics:
        t0 = time.time()
        qddot_deg = qdiff.update(qdot_deg, t_now)
        dyn_pred = full_dynamics_regressor(q_deg, qdot_deg, qddot_deg) @ pi
        tau_final = tau_gnn - dyn_pred
        F_final = torque_to_wrench(q_deg, tau_final, damping=damping)
        stages['+dynamics_regressor'] = (F_final[:3].copy(), time.time() - t0)

    return stages


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['timing', 'static', 'push'], default='timing',
                   help='timing: per-stage compute cost only. static: noise floor, arm '
                        'untouched. push: interactive -- push/tap the arm during recording.')
    p.add_argument('--phi', default='data/gravity_phi_task_only.npy')
    p.add_argument('--gravity_nn', default='data/gravity_residual_nn.pt')
    p.add_argument('--dynamics_pi', default='data/dynamics_residual_pi.npy')
    p.add_argument('--with_dynamics', action='store_true',
                   help='Also time/use the mass-Coriolis dynamics regressor stage '
                        '(known slow, ~25ms/step per diag_live_forces.py -- off by default '
                        'since quasi-static admittance does not need it).')
    p.add_argument('--rate_hz', type=float, default=100.0, help='Target poll rate.')
    p.add_argument('--duration', type=float, default=5.0, help='Seconds to record (static/push).')
    p.add_argument('--tare_duration', type=float, default=1.0,
                   help='Seconds to average for the resting-bias tare, captured just before '
                        'the recording window (arm must be untouched during this).')
    p.add_argument('--cutoff_hz', type=float, default=5.0, help="Butterworth cutoff (colleague's spec).")
    p.add_argument('--damping', type=float, default=0.05, help='Jacobian-transpose damping.')
    p.add_argument('--K', type=float, default=200.0,
                   help='Admittance stiffness gain, N/m -- correction = (F_filtered-f_desired)/K.')
    p.add_argument('--f_desired', type=float, default=0.0, help='Target force magnitude, N.')
    p.add_argument('--max_correction_cm', type=float, default=2.0, help='Correction cap, cm.')
    return p.parse_args()


def main():
    args = parse_args()
    phi, g_nn, g_x_mean, g_x_std, pi = load_pipeline(args)
    print('Loaded gravity regressor + gravity-residual NN'
          + (' + dynamics regressor' if args.with_dynamics else '') + '.')

    api = load_api()
    connect(api, control=False)   # READ-ONLY: never sends a motion command
    if not apply_saved_gravity_params(api):
        print('ERROR: firmware gravity params failed to reapply -- aborting rather than '
              'showing bad force data.')
        api.CloseAPI(); sys.exit(1)
    print('Firmware gravity params reapplied: OK\n')

    qdiff = VelocityDifferentiator(smoothing=0.3)
    filt = [OnlineButterworth(args.cutoff_hz, args.rate_hz) for _ in range(3)]  # Fx, Fy, Fz

    if args.mode == 'timing':
        print(f"=== Stage timing (30 samples, arm as-is) ===")
        all_stages = {}
        t0 = time.time()
        for i in range(30):
            q = get_q(api); qd = get_qdot(api); gf = get_tau_gf(api)
            stages = compute_force_staged(q, qd, gf, phi, g_nn, g_x_mean, g_x_std, pi,
                                          qdiff, time.time() - t0, args.damping, args.with_dynamics)
            for name, (_, dt) in stages.items():
                all_stages.setdefault(name, []).append(dt)
        for name, times in all_stages.items():
            arr = np.array(times) * 1000
            hz_max = 1000.0 / arr.mean()
            print(f"  {name:22s}: mean={arr.mean():.2f}ms  max={arr.max():.2f}ms  "
                  f"-> max sustainable rate ~{hz_max:.0f}Hz")
        api.CloseAPI()
        return

    def read_F_raw(t_now):
        q = get_q(api); qd = get_qdot(api); gf = get_tau_gf(api)
        stages = compute_force_staged(q, qd, gf, phi, g_nn, g_x_mean, g_x_std, pi,
                                      qdiff, t_now, args.damping, args.with_dynamics)
        return stages['+gravity_nn'][0] if not args.with_dynamics else stages['+dynamics_regressor'][0]

    # ── Tare: capture the resting bias at THIS pose and subtract it below ──────
    # A low-pass filter only removes the fast-varying part of a signal; a
    # constant sensor bias passes straight through (DC gain 1), so filtering
    # alone cannot fix the ~3N resting offset found earlier -- this is the fix.
    print(f"*** Taring: capturing baseline for {args.tare_duration:.1f}s -- "
          f"leave the arm untouched ***")
    dt_nominal = 1.0 / args.rate_hz
    n_tare = max(1, int(args.tare_duration * args.rate_hz))
    tare_samples = []
    t0 = time.time()
    for _ in range(n_tare):
        tare_samples.append(read_F_raw(time.time() - t0))
        time.sleep(dt_nominal)
    tare_offset = np.mean(tare_samples, axis=0)
    print(f"  tare offset (N): [{tare_offset[0]:+.3f}, {tare_offset[1]:+.3f}, "
          f"{tare_offset[2]:+.3f}]  (||.||={np.linalg.norm(tare_offset):.3f}N)\n")

    # static / push modes
    if args.mode == 'push':
        print(f"*** Recording {args.duration:.0f}s -- PUSH/TAP the arm during this window! ***")
        print("(gently -- this is just measuring the force signal, nothing moves)\n")
        time.sleep(1.5)
    else:
        print(f"*** Recording {args.duration:.0f}s -- leave the arm untouched (noise floor) ***\n")

    n_ticks = int(args.duration * args.rate_hz)
    raw_hist, filt_hist, corr_hist = [], [], []
    t_start = time.time()
    next_tick = t_start
    for _ in range(n_ticks):
        t_now = time.time() - t_start
        F_raw = read_F_raw(t_now) - tare_offset
        F_filt = np.array([filt[i].update(F_raw[i]) for i in range(3)])
        correction = (F_filt - args.f_desired) / args.K
        # Cap the VECTOR MAGNITUDE, not each component independently -- clipping
        # per-axis lets the combined displacement reach cap*sqrt(3) when all
        # three axes are simultaneously near the limit (caught empirically:
        # 2cm-per-axis clipping produced a 3.46cm actual displacement).
        cap_m = args.max_correction_cm / 100
        mag = np.linalg.norm(correction)
        if mag > cap_m:
            correction = correction * (cap_m / mag)

        raw_hist.append(F_raw.copy())
        filt_hist.append(F_filt.copy())
        corr_hist.append(correction.copy())

        next_tick += dt_nominal
        sleep_for = next_tick - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)
    t_end = time.time()
    api.CloseAPI()

    raw_hist  = np.array(raw_hist)
    filt_hist = np.array(filt_hist)
    corr_hist = np.array(corr_hist)
    raw_mag  = np.linalg.norm(raw_hist, axis=1)
    filt_mag = np.linalg.norm(filt_hist, axis=1)
    corr_mag = np.linalg.norm(corr_hist, axis=1) * 100  # cm

    achieved_hz = n_ticks / (t_end - t_start)
    print(f"\n=== Timing ===")
    print(f"  Requested {args.rate_hz:.0f}Hz, achieved {achieved_hz:.1f}Hz over {n_ticks} samples")

    print(f"\n=== Force (||F||, N) ===")
    print(f"  raw:      mean={raw_mag.mean():.3f}  std={raw_mag.std():.3f}  "
          f"max={raw_mag.max():.3f}")
    print(f"  filtered: mean={filt_mag.mean():.3f}  std={filt_mag.std():.3f}  "
          f"max={filt_mag.max():.3f}   (cutoff={args.cutoff_hz:.1f}Hz)")
    print(f"  noise reduction (std): {(1 - filt_mag.std()/raw_mag.std())*100:.0f}%")

    print(f"\n=== Admittance correction (K={args.K:.0f} N/m, f_desired={args.f_desired:.1f}N, "
          f"cap={args.max_correction_cm:.1f}cm) ===")
    print(f"  ||correction||: mean={corr_mag.mean():.3f}cm  std={corr_mag.std():.3f}cm  "
          f"max={corr_mag.max():.3f}cm")
    n_clipped = np.sum(corr_mag >= args.max_correction_cm * 0.99)
    print(f"  samples at/near the {args.max_correction_cm:.1f}cm cap: {n_clipped}/{len(corr_mag)} "
          f"({100*n_clipped/len(corr_mag):.1f}%)")

    if args.mode == 'static':
        print(f"\n  At-rest correction of {corr_mag.mean():.3f}cm (mean) with this K would be "
              f"applied to the arm CONTINUOUSLY even with nothing touching it -- this is the "
              f"number to check against how much spurious motion you're willing to accept.")


if __name__ == '__main__':
    main()
