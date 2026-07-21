#!/usr/bin/env python3
"""
Deliverable 0 — Gravity-residual repeatability test. RUN THIS FIRST.

The pose-dependent gravity residual calibration (calibrate_gravity_residual.py)
assumes tau_residual(q) is a DETERMINISTIC, repeatable function of pose: the
same joint configuration must give the same gravity-free torque every time. If
it does, a static gravity model Y_g(q)@phi can capture it. If the residual
drifts run-to-run at a fixed pose (actuator thermal effects, gear hysteresis,
sensor-bias wander), a static model is the wrong tool and will silently
underperform. THIS TEST DECIDES THAT before any calibration code is run.

  Do NOT proceed to calibrate_gravity_residual.py until this reports REPEATABLE
  (or you accept the measured floor as your recovered-force accuracy limit).

Whatever spread this finds is the honest lower bound on recovered-force
accuracy: the static phi calibration can never do better than the per-pose
repeatability. So peak-to-peak here feeds the noise-floor claim directly —
smallest resolvable contact force >= max(jitter, per-pose repeatability spread).

METHOD
  - K poses (default 6) spanning the workspace (mix shoulder-heavy,
    elbow-extended, wrist-varied — gravity residual varies most with the poses
    that load the proximal joints).
  - Visit each pose R times (default 5), re-visiting in a SHUFFLED order each
    run. Interleaving matters: if the order were always identical, slow thermal
    drift would look like a fixed per-pose offset and hide the very
    non-repeatability we're testing for.
  - At each visit, hold still and average several hundred
    GetAngularForceGravityFree samples (same averaging calibration will use).
  - Log elapsed wall-clock time at every visit — this is what separates
    thermal drift (correlates with time) from random hysteresis (doesn't).

MODES
  --mode interactive  (default, READ-ONLY, no motion commanded):
      You drive the arm (joystick) to each pose and hold still; press Enter.
      On the first pass you define the K poses; they're saved to --pose_file so
      you can re-approach the same targets. The script only READS sensors, so it
      can never command motion. Re-approach quality is reported (q-spread per
      pose) so you know how well you matched — large spread confounds the torque
      comparison, and the report warns if so.

  --mode commanded --execute  (COMMANDS MOTION — angular position control):
      Automatically drives to each saved pose in shuffled order (exact
      re-visits). Requires --pose_file with predefined poses. Dry-run by default
      (prints the planned sequence, no motion) — add --execute to actually move.

USAGE
  # 1. interactive, first time — define 6 poses and run 5 shuffled passes
  python test_residual_repeatability.py --mode interactive --pose_file data/repeat_poses.npy

  # 2. commanded replay of saved poses (verify dry-run first, then --execute)
  python test_residual_repeatability.py --mode commanded --pose_file data/repeat_poses.npy
  python test_residual_repeatability.py --mode commanded --pose_file data/repeat_poses.npy --execute
"""
import os, sys, ctypes, time, argparse, csv, random
import numpy as np

_LIB_DIR = os.path.join(os.path.expanduser('~/working_dir/kinovaDrivers'), 'sdk', 'lib')
if _LIB_DIR not in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
    os.environ['LD_LIBRARY_PATH'] = _LIB_DIR + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    os.execv(sys.executable, [sys.executable] + sys.argv)

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)
from contact_detector import torque_to_wrench
from calibrate_firmware_gravity import apply_saved_gravity_params

LIB_PATH      = os.path.join(_LIB_DIR, 'USBCommandLayerUbuntu.so')
COMM_LIB_PATH = os.path.join(_LIB_DIR, 'USBCommLayerUbuntu.so')
NO_ERROR_KINOVA, SERIAL_LENGTH, MAX_KINOVA_DEVICE = 1, 20, 20
ANGULAR_POSITION, NOMOVEMENT_HAND = 2, 0


class KinovaDevice(ctypes.Structure):
    _fields_ = [('SerialNumber', ctypes.c_char * SERIAL_LENGTH),
                ('Model', ctypes.c_char * SERIAL_LENGTH),
                ('VersionMajor', ctypes.c_int), ('VersionMinor', ctypes.c_int),
                ('VersionRelease', ctypes.c_int), ('DeviceType', ctypes.c_int),
                ('DeviceID', ctypes.c_int)]

class AngularInfo(ctypes.Structure):
    _fields_ = [(f'Actuator{i}', ctypes.c_float) for i in range(1, 8)]

class CartesianInfo(ctypes.Structure):
    _fields_ = [('X', ctypes.c_float), ('Y', ctypes.c_float), ('Z', ctypes.c_float),
                ('ThetaX', ctypes.c_float), ('ThetaY', ctypes.c_float), ('ThetaZ', ctypes.c_float)]

class FingersPosition(ctypes.Structure):
    _fields_ = [('Finger1', ctypes.c_float), ('Finger2', ctypes.c_float), ('Finger3', ctypes.c_float)]

class AngularPosition(ctypes.Structure):
    _fields_ = [('Actuators', AngularInfo), ('Fingers', FingersPosition)]

class UserPosition(ctypes.Structure):
    _fields_ = [('Type', ctypes.c_int), ('Delay', ctypes.c_float),
                ('CartesianPosition', CartesianInfo), ('Actuators', AngularInfo),
                ('HandMode', ctypes.c_int), ('Fingers', FingersPosition)]

class Limitation(ctypes.Structure):
    _fields_ = [('speedParameter1', ctypes.c_float), ('speedParameter2', ctypes.c_float),
                ('speedParameter3', ctypes.c_float), ('forceParameter1', ctypes.c_float),
                ('forceParameter2', ctypes.c_float), ('forceParameter3', ctypes.c_float),
                ('accelerationParameter1', ctypes.c_float), ('accelerationParameter2', ctypes.c_float),
                ('accelerationParameter3', ctypes.c_float)]

class TrajectoryPoint(ctypes.Structure):
    _fields_ = [('Position', UserPosition), ('LimitationsActive', ctypes.c_int),
                ('SynchroType', ctypes.c_int), ('Limitations', Limitation)]


def load_api():
    ctypes.CDLL(COMM_LIB_PATH, mode=ctypes.RTLD_GLOBAL)
    api = ctypes.CDLL(LIB_PATH)
    for fn in ('InitAPI', 'CloseAPI', 'RefresDevicesList', 'GetDevices', 'SetActiveDevice',
               'StartControlAPI', 'StopControlAPI', 'SetAngularControl',
               'GetAngularPosition', 'GetAngularForceGravityFree',
               'SendBasicTrajectory', 'EraseAllTrajectories'):
        getattr(api, fn).restype = ctypes.c_int
    api.SendBasicTrajectory.argtypes = [TrajectoryPoint]
    return api


def ok(r):
    return r == NO_ERROR_KINOVA


def connect(api, control: bool):
    r = api.InitAPI()
    if not ok(r):
        raise SystemExit(f'InitAPI() failed: {r}')
    api.RefresDevicesList()
    devs = (KinovaDevice * MAX_KINOVA_DEVICE)()
    err = ctypes.c_int(NO_ERROR_KINOVA)
    if api.GetDevices(devs, ctypes.byref(err)) == 0:
        api.CloseAPI(); raise SystemExit('No Kinova device found')
    api.SetActiveDevice(devs[0])
    if control:
        api.StartControlAPI(); api.StopControlAPI(); api.StartControlAPI()
        api.SetAngularControl()
    print(f'Connected: {devs[0].Model.decode()} ({devs[0].SerialNumber.decode()})'
          + ('  [ANGULAR CONTROL]' if control else '  [READ-ONLY]'))
    grav_ok = apply_saved_gravity_params(api)
    print(f'Firmware gravity params (data/gravity_params.npy) reapplied: '
          f'{"OK" if grav_ok else "not applied — see message above"}  '
          f'(NOTE: this changes what "repeatable" is measured against — the residual '
          f'being tested is now the post-firmware-fix residual, not the raw one used '
          f'when this gate first passed on 2026-07-02)')


def read6(api, fn):
    s = AngularPosition()
    getattr(api, fn)(ctypes.byref(s))
    a = s.Actuators
    return np.array([a.Actuator1, a.Actuator2, a.Actuator3,
                     a.Actuator4, a.Actuator5, a.Actuator6], np.float64)


def get_q(api):
    return read6(api, 'GetAngularPosition')


def get_tau_gf(api):
    return read6(api, 'GetAngularForceGravityFree')


def angle_diff_deg(a, b):
    """Per-joint smallest signed difference, wraparound-safe (Kinova may report
    cumulative degrees)."""
    return (a - b + 180.0) % 360.0 - 180.0


def send_angular(api, q_deg, speed_dps):
    tp = TrajectoryPoint()
    ctypes.memset(ctypes.byref(tp), 0, ctypes.sizeof(tp))
    tp.Position.Type = ANGULAR_POSITION
    tp.Position.HandMode = NOMOVEMENT_HAND
    for i in range(6):
        setattr(tp.Position.Actuators, f'Actuator{i+1}', float(q_deg[i]))
    if speed_dps > 0:
        tp.LimitationsActive = 1
        tp.Limitations.speedParameter1 = float(speed_dps)
        tp.Limitations.speedParameter2 = float(speed_dps)
    api.SendBasicTrajectory(tp)


def capture_visit(api, n_samples, hz):
    """Average n_samples of (q, tau_gf) while stationary. Returns (q_mean,
    tau_gf_mean, q_std)."""
    qs, taus = [], []
    dt = 1.0 / hz
    for _ in range(n_samples):
        qs.append(get_q(api)); taus.append(get_tau_gf(api))
        time.sleep(dt)
    qs, taus = np.stack(qs), np.stack(taus)
    return qs.mean(axis=0), taus.mean(axis=0), qs.std(axis=0)


# ════════════════════════════════════════════════════════════════════════════
# Reporting
# ════════════════════════════════════════════════════════════════════════════

def fnorm(q_deg, tau_gf):
    return float(np.linalg.norm(torque_to_wrench(q_deg, tau_gf)[:3]))


def report(records, jitter_floor, output_csv, plot_path):
    """records: list of dicts with pose_id, visit, elapsed_s, q(6), tau_gf(6), Fnorm."""
    # ── per-visit CSV ──
    with open(output_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pose_id', 'visit', 'elapsed_s']
                   + [f'q{i+1}' for i in range(6)]
                   + [f'tau{i+1}' for i in range(6)] + ['Fnorm'])
        for r in records:
            w.writerow([r['pose_id'], r['visit'], f"{r['elapsed_s']:.1f}"]
                       + [f'{v:.3f}' for v in r['q']]
                       + [f'{v:.4f}' for v in r['tau_gf']] + [f"{r['Fnorm']:.4f}"])
    print(f'\nPer-visit data -> {output_csv}')

    pose_ids = sorted(set(r['pose_id'] for r in records))
    print('\n' + '=' * 78)
    print('PER-POSE SUMMARY')
    print('=' * 78)
    print(f"{'pose':>4} {'visits':>6} {'|F| mean':>9} {'|F| std':>8} "
          f"{'|F| p2p':>8} {'q re-approach std(deg)':>24} {'verdict':>10}")
    worst_ptp = 0.0
    any_qspread_warn = False
    pose_stats = {}
    for pid in pose_ids:
        rs = [r for r in records if r['pose_id'] == pid]
        F = np.array([r['Fnorm'] for r in rs])
        qs = np.stack([r['q'] for r in rs])
        tau = np.stack([r['tau_gf'] for r in rs])
        ptp = float(F.max() - F.min())
        q_reapproach = float(np.max(qs.std(axis=0)))   # worst joint's spread across visits
        worst_ptp = max(worst_ptp, ptp)
        flag = 'ok' if ptp <= 0.5 else 'HIGH'
        qwarn = '' if q_reapproach < 1.0 else '  <-poses differ!'
        if q_reapproach >= 1.0:
            any_qspread_warn = True
        pose_stats[pid] = dict(F=F, elapsed=np.array([r['elapsed_s'] for r in rs]),
                               ptp=ptp, tau_std=tau.std(axis=0))
        print(f"{pid:>4} {len(rs):>6} {F.mean():>9.3f} {F.std():>8.3f} "
              f"{ptp:>8.3f} {q_reapproach:>21.2f}{'':3} {flag:>10}{qwarn}")

    # ── drift vs hysteresis: correlate |F| with elapsed time, per pose ──
    corrs = []
    for pid in pose_ids:
        st = pose_stats[pid]
        if len(st['F']) >= 3 and st['F'].std() > 1e-9 and st['elapsed'].std() > 1e-9:
            corrs.append(np.corrcoef(st['elapsed'], st['F'])[0, 1])
    mean_abs_corr = float(np.mean(np.abs(corrs))) if corrs else 0.0

    print('\n' + '=' * 78)
    print('VERDICT')
    print('=' * 78)
    print(f'Known pose-agnostic jitter floor : ~{jitter_floor[0]:.1f}-{jitter_floor[1]:.1f} N')
    print(f'Worst per-pose |F| peak-to-peak  : {worst_ptp:.3f} N')
    print(f'Mean |elapsed-vs-|F|| correlation: {mean_abs_corr:.2f} '
          f'(near 1 = time-driven/thermal; near 0 = random)')
    if any_qspread_warn:
        print('WARNING: some poses had >=1 deg re-approach spread — part of the |F| spread '
              'is pose mismatch, not non-repeatability. Use --mode commanded for exact '
              're-visits, or match poses more tightly, before trusting a HIGH verdict.')

    repeatable = worst_ptp <= 0.5
    time_driven = mean_abs_corr >= 0.7
    print()
    if repeatable:
        verdict = ('REPEATABLE — proceed to static gravity-model calibration '
                   '(calibrate_gravity_residual.py).')
    elif time_driven:
        verdict = ('THERMAL DRIFT — residual moves with elapsed time. Warm the arm up '
                   '(~20-30 min of operation) before calibrating; recheck warm; consider '
                   'recalibrating per session or capturing calibration warm.')
    else:
        verdict = ('HYSTERETIC / NON-REPEATABLE — static phi will not fully capture the '
                   f'residual. Achievable recovered-force floor ~= {worst_ptp:.2f} N '
                   '(report this as the limit rather than expecting it to vanish).')
    print('OVERALL: ' + verdict)

    # ── plot |F| vs elapsed time, one line per pose ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        for pid in pose_ids:
            st = pose_stats[pid]
            order = np.argsort(st['elapsed'])
            ax.plot(st['elapsed'][order], st['F'][order], '-o', label=f'pose {pid}')
        ax.axhspan(0, jitter_floor[1], color='green', alpha=0.08,
                   label=f'jitter floor (<{jitter_floor[1]:.1f} N)')
        ax.set_xlabel('elapsed time (s)'); ax.set_ylabel('|F| (N)')
        ax.set_title('Gravity-residual repeatability: |F| vs elapsed time per pose\n'
                     '(monotonic-with-time = thermal drift; scatter = hysteresis)')
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout(); fig.savefig(plot_path, dpi=110)
        print(f'\nPlot -> {plot_path}')
    except Exception as e:
        print(f'\n(plot skipped: {e!r})')


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['interactive', 'commanded'], default='interactive')
    p.add_argument('--n_poses', type=int, default=6, help='K test poses (default 6)')
    p.add_argument('--n_runs', type=int, default=5, help='R visits per pose (default 5)')
    p.add_argument('--samples_per_visit', type=int, default=300,
                   help='Torque samples averaged per visit (beats down jitter). Default 300.')
    p.add_argument('--hz', type=float, default=100.0, help='Sample poll rate (default 100)')
    p.add_argument('--pose_file', default='data/repeat_poses.npy',
                   help='Saved (K,6) joint poses. Loaded if present; in interactive mode, '
                        'defined and saved here on the first pass if absent.')
    p.add_argument('--redefine', action='store_true',
                   help='(interactive) Re-define poses even if --pose_file exists.')
    p.add_argument('--jitter_floor', type=float, nargs=2, default=[0.1, 0.5],
                   help='Known pose-agnostic noise floor (N), for the verdict comparison.')
    p.add_argument('--seed', type=int, default=0, help='RNG seed for run-order shuffling.')
    p.add_argument('--output_csv', default='data/residual_repeatability.csv')
    p.add_argument('--plot_path', default='residual_repeatability.png')
    # commanded-mode motion params
    p.add_argument('--execute', action='store_true',
                   help='(commanded) Actually move the arm. Default: dry-run (print only).')
    p.add_argument('--speed_dps', type=float, default=15.0,
                   help='(commanded) Angular speed limit (deg/s) sent to the arm.')
    p.add_argument('--converge_deg', type=float, default=1.0,
                   help='(commanded) Per-joint convergence tolerance (deg).')
    p.add_argument('--converge_timeout', type=float, default=20.0,
                   help='(commanded) Max seconds to wait for arrival per pose.')
    p.add_argument('--settle_s', type=float, default=1.0,
                   help='Seconds to settle after arriving/being-held before averaging.')
    return p.parse_args()


def define_poses_interactive(api, k):
    poses = []
    print(f'\nDefine {k} test poses. Drive the arm (joystick) to each and hold still.')
    print('Aim for a spread: shoulder-heavy, elbow-extended, and wrist-varied configs.')
    for i in range(k):
        input(f'  Move to pose {i} and hold, then press Enter...')
        q = get_q(api)
        poses.append(q)
        print(f'    recorded pose {i}: {np.round(q, 1)}')
    return np.stack(poses)


def run_interactive(api, args, poses):
    records = []
    t0 = time.time()
    order_rng = random.Random(args.seed)
    for run in range(args.n_runs):
        order = list(range(len(poses)))
        order_rng.shuffle(order)
        print(f'\n=== Run {run+1}/{args.n_runs}  (order: {order}) ===')
        for pid in order:
            print(f'  Target pose {pid}: {np.round(poses[pid], 1)}')
            input(f'  Drive to pose {pid}, hold still, press Enter to sample...')
            if args.settle_s > 0:
                time.sleep(args.settle_s)
            q_m, tau_m, q_s = capture_visit(api, args.samples_per_visit, args.hz)
            mism = np.max(np.abs(angle_diff_deg(q_m, poses[pid])))
            F = fnorm(q_m, tau_m)
            records.append(dict(pose_id=pid, visit=run, elapsed_s=time.time() - t0,
                                q=q_m, tau_gf=tau_m, Fnorm=F))
            warn = '' if mism < 2.0 else f'  (WARN: {mism:.1f} deg off target)'
            print(f'    |F|={F:.3f} N   sample q-noise(max)={q_s.max():.3f} deg   '
                  f'off-target(max)={mism:.2f} deg{warn}')
    return records


def run_commanded(api, args, poses):
    records = []
    t0 = time.time()
    order_rng = random.Random(args.seed)
    for run in range(args.n_runs):
        order = list(range(len(poses)))
        order_rng.shuffle(order)
        print(f'\n=== Run {run+1}/{args.n_runs}  (order: {order}) ===')
        for pid in order:
            target = poses[pid]
            print(f'  -> pose {pid}: {np.round(target, 1)}', end='')
            if not args.execute:
                print('   [DRY RUN — not moving]')
                continue
            send_angular(api, target, args.speed_dps)
            tw = time.time()
            while time.time() - tw < args.converge_timeout:
                if np.max(np.abs(angle_diff_deg(get_q(api), target))) < args.converge_deg:
                    break
                time.sleep(0.05)
            arrived = np.max(np.abs(angle_diff_deg(get_q(api), target)))
            if args.settle_s > 0:
                time.sleep(args.settle_s)
            q_m, tau_m, q_s = capture_visit(api, args.samples_per_visit, args.hz)
            F = fnorm(q_m, tau_m)
            records.append(dict(pose_id=pid, visit=run, elapsed_s=time.time() - t0,
                                q=q_m, tau_gf=tau_m, Fnorm=F))
            print(f'   arrived({arrived:.2f}deg)  |F|={F:.3f} N')
    return records


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.join(PIPELINE_DIR, args.output_csv)) or '.', exist_ok=True)
    pose_path = os.path.join(PIPELINE_DIR, args.pose_file)

    commanded = (args.mode == 'commanded')
    api = load_api()
    connect(api, control=(commanded and args.execute))

    try:
        # ── obtain poses ──
        if commanded:
            if not os.path.exists(pose_path):
                raise SystemExit(f'--mode commanded needs an existing --pose_file ({pose_path}); '
                                 f'create it first with --mode interactive.')
            poses = np.load(pose_path)
            print(f'Loaded {len(poses)} poses from {pose_path}')
        else:
            if os.path.exists(pose_path) and not args.redefine:
                poses = np.load(pose_path)
                print(f'Loaded {len(poses)} poses from {pose_path} (use --redefine to redo)')
            else:
                poses = define_poses_interactive(api, args.n_poses)
                os.makedirs(os.path.dirname(pose_path) or '.', exist_ok=True)
                np.save(pose_path, poses)
                print(f'Saved {len(poses)} poses -> {pose_path}')

        # ── multi-run capture ──
        if commanded:
            if not args.execute:
                print('\n*** DRY RUN — no motion will be commanded. Add --execute to move. ***')
            records = run_commanded(api, args, poses)
        else:
            records = run_interactive(api, args, poses)

    finally:
        if commanded and args.execute:
            api.EraseAllTrajectories()
        api.CloseAPI()

    if not records or all(r['pose_id'] is None for r in records):
        print('\nNo data captured (dry run?). Nothing to report.')
        return

    report(records, args.jitter_floor,
           os.path.join(PIPELINE_DIR, args.output_csv),
           os.path.join(PIPELINE_DIR, args.plot_path))


if __name__ == '__main__':
    main()
