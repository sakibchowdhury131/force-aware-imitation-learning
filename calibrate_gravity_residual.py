#!/usr/bin/env python3
"""
Fit the software gravity-residual regressor: tau_residual(q) = Y_g(q) @ phi
(see contact_detector.py's gravity_regressor / recover_external_force).

UPDATE (2026-07-03): the firmware fix's FIRST attempt (2026-07-02) produced a
bad fit (J6 fault, joystick errors) — that data/gravity_params.npy has since
been overwritten. A SECOND run succeeded cleanly: 21.4N -> 1.69N (89-94%
reduction), confirmed fault-free on hardware, confirmed to generalize to
held-out poses (verify_gravity_params.py), and confirmed to need
apply_saved_gravity_params() reapplied every session (does not survive a
power cycle). See calibrate_firmware_gravity.py's docstring and
project_force_sensing.md memory for the full timeline.

This script now fits the RESIDUAL LEFT ON TOP of that good firmware model
(mean ~1.7-2.5N in the calibrated pose region, more outside it — see
verify_gravity_params.py's held-out results) rather than replacing the
firmware fix. It:
  - never sends the arm through an undocumented autonomous trajectory
  - never touches firmware gravity state directly (no SetGravityOptimalZParam/
    SetGravityType calls of its own — it only calls apply_saved_gravity_params()
    to make sure the SAME already-fitted firmware model is active before
    collecting, so training data has a consistent baseline)
  - only reads GetAngularForceGravityFree and commands ordinary, dry-run-checked
    angular position moves (the same safe motion primitives already used and
    validated in calibrate_firmware_gravity.py / test_residual_repeatability.py)
  - explicitly ridge-regularizes the fit, since gravity data alone cannot
    identify every inertial-parameter direction (rank-deficient system)

IMPORTANT: any data collected here is only valid together with whatever
firmware gravity state was active at collection time. The archived
data/archive_pre_firmware_fix/ records were collected against the RAW/
uncalibrated firmware baseline (before the good fit existed) — do not mix
them with new data collected now.

METHOD
  1. Collect stationary, no-contact GetAngularForceGravityFree readings at
     N poses (default 40) spanning the workspace, each averaged over several
     hundred samples to beat down the ~0.1-0.5 N jitter. Default mode is
     COMMANDED (angular position control, well-conditioned poses generated and
     filtered for cond(J)) — NOT hand-guided: a prior repeatability test in
     this session found the joystick is not precise enough for exact
     pose logging (see test_residual_repeatability.py). --mode interactive is
     still available if you'd rather hand-guide and press Enter per pose.
  2. Stack Y_g(q_i) (6x24 each) and tau_gf_i (6,) across all N poses into
     tau_gf_stacked = Y_g_stacked @ phi (6N x 24 system).
  3. Solve for phi via RIDGE (Tikhonov) least squares, not plain lstsq — some
     parameter directions never affect static torque (e.g. a link's COM offset
     along its own joint axis contributes zero moment about that axis), so the
     system is rank-deficient and plain lstsq would fit noise into those
     directions. Reports cond(Y_g_stacked) and flags poorly-identified
     directions (small singular values) before deciding lambda mattered.
  4. Saves phi -> data/gravity_phi.npy + metadata (date, N poses, condition
     number, tool description).
  5. Before/after report: ||F|| at the calibration poses using raw tau_gf vs
     tau_gf - Y_g(q)@phi (via contact_detector.recover_external_force).
  6. Repeatability spot-check: revisits a few poses after fitting and confirms
     the residual (post-subtraction) stays near the jitter floor — a quick
     consistency check, distinct from (but consistent with) the full
     test_residual_repeatability.py gate this session already passed.

USAGE
  python calibrate_gravity_residual.py                      # dry run, prints plan
  python calibrate_gravity_residual.py --execute             # commanded, real motion
  python calibrate_gravity_residual.py --mode interactive --execute  # hand-guided instead
"""
import os, sys, time, argparse, random
import numpy as np

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)

# Reuse the already-validated safe primitives (connect, motion+settle, torque
# sampling, joint-limit clamping) from calibrate_firmware_gravity.py. NOT
# importing run_gravity_estimation / apply_saved_gravity_params / TEST_POSES_DEG
# — those are the parts flagged unsafe in that file's docstring; everything
# imported here is the generic, already-safe motion/reading layer.
from calibrate_firmware_gravity import (
    load_api, connect, get_q, get_qdot, get_tau_gf, clamp_pose,
    move_and_settle, capture_visit, apply_saved_gravity_params,
    JOINT_LIMITS_DEG, SAFE_SPEED_DPS,
)
from contact_detector import (
    gravity_regressor, torque_to_wrench, recover_external_force, compute_jacobian,
)


# Same real observed working poses used for test_residual_repeatability.py's
# data/repeat_poses.npy (already confirmed REPEATABLE on hardware) — reused
# here as anchors so the regressor is trained on the same real workspace
# region, not an arbitrary generic pose set.
ANCHOR_POSES_DEG = np.array([
    [146.7, 211.3, 65.0, 408.4, 220.1, 120.4],
    [146.5, 193.7, 122.9, 389.7, 283.4, 168.9],
    [172.3, 212.6, 71.4, 313.5, 242.7, 232.1],
])


def generate_poses(n: int, seed: int = 0, max_cond: float = 50.0,
                   min_spread_deg: float = 8.0, perturb_scale=(20, 20, 25, 30, 25, 30)):
    """Generate n candidate poses by perturbing around ANCHOR_POSES_DEG, kept
    within real joint limits, filtered to well-conditioned Jacobian (avoids
    the J5=180deg-type singularity this session hit on hardware), and spread
    out from each other (min_spread_deg) for better phi identifiability."""
    rng = np.random.default_rng(seed)
    scale = np.array(perturb_scale, dtype=np.float64)
    pool = []
    attempts = 0
    while len(pool) < n and attempts < n * 200:
        attempts += 1
        anchor = ANCHOR_POSES_DEG[rng.integers(len(ANCHOR_POSES_DEG))]
        q = clamp_pose(anchor + rng.uniform(-1, 1, 6) * scale)
        cond = np.linalg.cond(compute_jacobian(q))
        if cond > max_cond:
            continue
        if pool and min(np.max(np.abs(q - qc)) for qc in pool) < min_spread_deg:
            continue
        pool.append(q)
    if len(pool) < n:
        print(f'WARNING: only generated {len(pool)}/{n} poses meeting the cond(J)<{max_cond} '
             f'and spread>{min_spread_deg}deg constraints — consider loosening them.')
    return order_poses_for_travel(np.stack(pool))


def order_poses_for_travel(poses: np.ndarray) -> np.ndarray:
    """Greedy nearest-neighbor reorder so consecutive visits are close in
    joint space, minimizing large multi-joint jumps between poses (one such
    jump — 88deg on J3 + 62deg on J5 between two poses generated in
    unsorted/random order — exceeded move_and_settle's convergence timeout on
    real hardware and triggered a false-positive abort)."""
    remaining = list(range(len(poses)))
    order = [remaining.pop(0)]
    while remaining:
        last = poses[order[-1]]
        dists = [np.max(np.abs(poses[i] - last)) for i in remaining]
        order.append(remaining.pop(int(np.argmin(dists))))
    return poses[order]


# ════════════════════════════════════════════════════════════════════════════
# Pose collection
# ════════════════════════════════════════════════════════════════════════════

def collect_commanded(api, poses, n_samples, hz):
    records = []
    for i, target in enumerate(poses):
        # Adaptive convergence timeout, scaled to the actual distance this
        # move needs to travel (poses are now travel-ordered via
        # order_poses_for_travel, so this should rarely bind — but it's a
        # second, independent safety net). A prior run hit an 88deg
        # single-joint jump that exceeded the old fixed 25s timeout despite a
        # naive distance/speed estimate of only ~6s for that joint alone —
        # real coordinated multi-joint moves + settle time clearly take
        # longer than that naive estimate, by an amount we don't have enough
        # data to fit precisely. Erring generous (floor raised to 45s, and a
        # steep per-degree term) rather than guessing a tight formula.
        q_now = get_q(api)
        max_delta = float(np.max(np.abs(target - q_now)))
        adaptive_timeout = max(45.0, max_delta / SAFE_SPEED_DPS * 4.0 + 20.0)

        print(f'[{i+1}/{len(poses)}] -> {np.round(target, 1)}  '
             f'(delta={max_delta:.0f}deg, timeout={adaptive_timeout:.0f}s)', end='  ')
        err, settled = move_and_settle(api, target, converge_timeout=adaptive_timeout)
        if err > 5.0:
            print(f'ABORT: arrived {err:.1f} deg off target — hardware not tracking '
                 f'commands correctly (see the J6-fault incident earlier this session). '
                 f'Stopping rather than sampling at the wrong pose.')
            return None
        if not settled:
            print(f'(WARNING: velocity never settled, sampling anyway)', end='  ')
        tau_gf = capture_visit(api, n_samples=n_samples, hz=hz)
        q_actual = get_q(api)
        print(f'arrived({err:.2f}deg)')
        records.append((q_actual, tau_gf))
    return records


def collect_interactive(api, n_poses, n_samples, hz):
    records = []
    print(f'\nDefine/visit {n_poses} poses. Drive the arm (joystick) to each, hold still.')
    for i in range(n_poses):
        input(f'  Move to pose {i+1}/{n_poses} and hold, then press Enter...')
        tau_gf = capture_visit(api, n_samples=n_samples, hz=hz)
        q_actual = get_q(api)
        print(f'    recorded: q={np.round(q_actual, 1)}')
        records.append((q_actual, tau_gf))
    return records


# ════════════════════════════════════════════════════════════════════════════
# Ridge fit
# ════════════════════════════════════════════════════════════════════════════

def ridge_fit(records, ridge_lambda: float):
    """Stacks Y_g(q_i)/tau_gf_i across records and solves the Tikhonov-
    regularized system via an augmented least-squares system (numerically
    stable — avoids explicitly forming the squared-condition-number Y^T Y):

        [Y_stacked      ] @ phi = [tau_stacked]
        [sqrt(lambda)*I ]         [0          ]

    Returns (phi, cond_number, singular_values, poorly_identified_mask).
    """
    Y_blocks = [gravity_regressor(q) for q, _ in records]
    tau_blocks = [tau for _, tau in records]
    Y_stacked = np.vstack(Y_blocks)          # (6N, 4*n_links)
    tau_stacked = np.concatenate(tau_blocks)  # (6N,)
    n_params = Y_stacked.shape[1]

    # Condition number / singular-value diagnostics on the RAW (unregularized)
    # stacked system — this is what tells us which parameter directions are
    # weakly identified by the data itself, before regularization masks it.
    U, S, Vt = np.linalg.svd(Y_stacked, full_matrices=False)
    cond_number = float(S[0] / S[-1]) if S[-1] > 1e-12 else float('inf')
    poorly_identified = S < (0.01 * S[0])   # <1% of the largest singular value

    Y_aug = np.vstack([Y_stacked, np.sqrt(ridge_lambda) * np.eye(n_params)])
    tau_aug = np.concatenate([tau_stacked, np.zeros(n_params)])
    phi, *_ = np.linalg.lstsq(Y_aug, tau_aug, rcond=None)

    return phi, cond_number, S, poorly_identified, Vt


def report_identifiability(S, poorly_identified, Vt, link_names):
    print('\nSingular-value spectrum of the stacked gravity regressor:')
    param_labels = []
    for name in link_names:
        param_labels += [f'{name}.m', f'{name}.mcx', f'{name}.mcy', f'{name}.mcz']
    for i, (s, bad) in enumerate(zip(S, poorly_identified)):
        flag = '  <- POORLY IDENTIFIED (<1% of largest)' if bad else ''
        print(f'  sigma[{i:2d}] = {s:9.4f}{flag}')
    if poorly_identified.any():
        print('\nPoorly-identified parameter directions (dominant components of the '
             'corresponding right-singular vectors):')
        for i in np.where(poorly_identified)[0]:
            v = Vt[i]
            top = np.argsort(-np.abs(v))[:3]
            desc = ', '.join(f'{param_labels[j]}({v[j]:+.2f})' for j in top)
            print(f'  direction {i}: dominated by {desc}')
        print('This is EXPECTED (not a bug) — e.g. a link\'s COM offset along its own '
             'joint axis contributes zero static torque about that axis, so gravity '
             'data alone can never identify it. The ridge term prevents these directions '
             'from fitting noise into large, meaningless phi values.')


# ════════════════════════════════════════════════════════════════════════════
# Before/after + repeatability check
# ════════════════════════════════════════════════════════════════════════════

def before_after_report(records, phi):
    print('\n' + '=' * 78)
    print('BEFORE / AFTER (at calibration poses)')
    print('=' * 78)
    print(f'{"pose":>4} {"||F|| before (N)":>17} {"||F|| after (N)":>16} {"reduction (%)":>14}')
    befores, afters = [], []
    for i, (q, tau_gf) in enumerate(records):
        F_before = float(np.linalg.norm(torque_to_wrench(q, tau_gf)[:3]))
        _tau_ext, F_ext = recover_external_force(q, tau_gf, phi)
        F_after = float(np.linalg.norm(F_ext[:3]))
        pct = (F_before - F_after) / F_before * 100.0 if F_before > 1e-9 else 0.0
        befores.append(F_before); afters.append(F_after)
        print(f'{i:>4} {F_before:>17.3f} {F_after:>16.3f} {pct:>13.1f}%')
    mb, ma = float(np.mean(befores)), float(np.mean(afters))
    print(f'\nMean ||F|| before: {mb:.3f} N')
    print(f'Mean ||F|| after : {ma:.3f} N  ({(mb-ma)/mb*100:.1f}% average reduction)')
    return mb, ma


def repeatability_spotcheck(api, records, phi, n_check, n_samples, hz):
    """Revisit a few calibration poses AFTER fitting and confirm the
    post-subtraction residual stays near the jitter floor — a quick
    generalization/consistency check (per the original spec's requirement),
    distinct from but consistent with test_residual_repeatability.py's full
    gate earlier this session."""
    print('\n' + '=' * 78)
    print(f'REPEATABILITY SPOT-CHECK — revisiting {n_check} poses post-fit')
    print('=' * 78)
    idx = random.sample(range(len(records)), min(n_check, len(records)))
    for i in idx:
        q_target, _ = records[i]
        err, settled = move_and_settle(api, q_target)
        tau_gf = capture_visit(api, n_samples=n_samples, hz=hz)
        q_actual = get_q(api)
        _tau_ext, F_ext = recover_external_force(q_actual, tau_gf, phi)
        F = float(np.linalg.norm(F_ext[:3]))
        print(f'  pose {i}: revisit ||F|| (post-subtraction) = {F:.3f} N  '
             f'(arrived {err:.2f} deg off, settled={settled})')


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

LINK_NAMES = ['shoulder', 'arm', 'forearm', 'wrist1', 'wrist2', 'hand']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['commanded', 'interactive'], default='commanded')
    p.add_argument('--n_poses', type=int, default=8,
                   help='Poses to collect THIS session (default 8, ~4.7 min estimated, '
                        'comfortably under the 5-min safe budget). A 40-pose single '
                        'run (~20-30 min) showed strong evidence of time-correlated drift '
                        '(before-N doubled from first 10 to last 10 poses) that a wide-region '
                        'repeatability check over a short ~3.5min run did NOT show — so the '
                        'fix is short batches, not fewer total poses. Use --append across '
                        'multiple short sessions to build up the full dataset instead of one '
                        'long run.')
    p.add_argument('--append', action='store_true',
                   help='Load previously-collected records from --session_records_path, '
                        'add this session\'s new poses to them, and fit phi on the combined '
                        'set. Without this flag, a run starts a fresh dataset (overwriting '
                        'any previous one at that path).')
    p.add_argument('--session_records_path', default='data/gravity_calibration_records.npz',
                   help='Where accumulated (q, tau_gf) pairs are stored across sessions.')
    p.add_argument('--n_samples', type=int, default=200,
                   help='Torque samples averaged per pose (default 200, matches '
                        'calibrate_firmware_gravity.py\'s convention)')
    p.add_argument('--hz', type=float, default=100.0)
    p.add_argument('--ridge_lambda', type=float, default=0.1,
                   help='Tikhonov regularization strength. Larger = more conservative '
                        '(shrinks poorly-identified phi directions harder toward zero).')
    p.add_argument('--n_repeat_check', type=int, default=3)
    p.add_argument('--tool_description', default='',
                   help='Free-text note on what tool is attached (phi is tool-specific '
                        'the same way the fixed bias is — recalibrate on tool change).')
    p.add_argument('--output', default='data/gravity_phi.npy')
    p.add_argument('--seed', type=int, default=0,
                   help='Combined with the current record count when --append is set, so '
                        'repeated --append runs generate DIFFERENT poses each time rather '
                        'than repeating the same batch.')
    p.add_argument('--execute', action='store_true',
                   help='Actually connect + move the arm. Default: dry run (prints the '
                        'planned pose set and exits, no hardware).')
    return p.parse_args()


def load_session_records(path):
    if not os.path.exists(path):
        return []
    d = np.load(path)
    return list(zip(d['q'], d['tau_gf']))


def save_session_records(records, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    q_arr = np.stack([q for q, _ in records])
    tau_arr = np.stack([tau for _, tau in records])
    np.savez(path, q=q_arr, tau_gf=tau_arr)


SAFE_SESSION_MINUTES = 5.0   # stay well under the ~16-20min window where we've
                             # seen real drift/faults on this hardware — the
                             # 6-pose x 5-run repeatability check (~3.5 min
                             # total) showed no drift at all, so this leaves margin.
OBSERVED_SEC_PER_POSE = 35.0  # ~200s/6 poses from the repeatability check


def main():
    args = parse_args()
    out_path = os.path.join(PIPELINE_DIR, args.output)
    session_path = os.path.join(PIPELINE_DIR, args.session_records_path)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    prior_records = load_session_records(session_path) if args.append else []
    if args.append:
        print(f'--append: {len(prior_records)} records already in {session_path}')

    est_minutes = args.n_poses * OBSERVED_SEC_PER_POSE / 60.0
    if est_minutes > SAFE_SESSION_MINUTES:
        print(f'WARNING: --n_poses={args.n_poses} is estimated to take ~{est_minutes:.1f} min, '
             f'above the {SAFE_SESSION_MINUTES:.0f}-min budget we\'ve validated as drift-free. '
             f'Consider a smaller --n_poses and --append across more sessions instead.')

    if args.mode == 'commanded':
        # Seed varies with how many records already exist, so repeated
        # --append runs sample DIFFERENT poses each time instead of the same batch.
        poses = generate_poses(args.n_poses, seed=args.seed + len(prior_records))
        print(f'Generated {len(poses)} well-conditioned poses '
             f'(cond(J)<50, >=8deg apart) around real observed working configs. '
             f'Estimated time: ~{est_minutes:.1f} min.')
    else:
        poses = None  # collected live in interactive mode

    if not args.execute:
        print('\n*** DRY RUN — no hardware connection, no motion. Add --execute to run '
             'for real. ***')
        if poses is not None:
            for i, q in enumerate(poses):
                print(f'  [{i}] {np.round(q, 1)}  cond(J)={np.linalg.cond(compute_jacobian(q)):.1f}')
        return

    api = load_api()
    try:
        connect(api, control=True)
        grav_ok = apply_saved_gravity_params(api)
        print(f'Firmware gravity params (data/gravity_params.npy) reapplied: '
              f'{"OK" if grav_ok else "not applied — see message above"}  '
              f'(this software regressor now fits the residual ON TOP of the firmware '
              f'model, so it must be collected under the SAME firmware gravity state '
              f'every time — do not mix data collected before/after re-running this)')
        t_session_start = time.time()
        if args.mode == 'commanded':
            new_records = collect_commanded(api, poses, args.n_samples, args.hz)
        else:
            new_records = collect_interactive(api, args.n_poses, args.n_samples, args.hz)
        session_elapsed = time.time() - t_session_start
        print(f'\nThis session\'s collection took {session_elapsed/60:.1f} min.')

        if new_records is None:
            print('Collection aborted (hardware tracking error) — no fit performed. '
                 'Prior accumulated records (if any) were NOT modified.')
            return

        records = prior_records + new_records
        save_session_records(records, session_path)
        print(f'Saved {len(records)} total accumulated records -> {session_path} '
             f'({len(new_records)} new this session)')

        if len(records) < 7:
            print(f'\nOnly {len(records)} total poses accumulated — need at least 7 to '
                 f'identify the 24-parameter phi vector (even with ridge regularization, '
                 f'too few poses gives an unreliable fit). Run again with --append to add '
                 f'more before fitting.')
            return

        phi, cond_number, S, poorly_identified, Vt = ridge_fit(records, args.ridge_lambda)
        print(f'\nFitted phi (24 values, ridge_lambda={args.ridge_lambda}):')
        print(np.round(phi, 4))
        print(f'\ncond(Y_g_stacked) [unregularized] = {cond_number:.2e}')
        report_identifiability(S, poorly_identified, Vt, LINK_NAMES)

        mean_before, mean_after = before_after_report(records, phi)

        repeatability_spotcheck(api, records, phi, args.n_repeat_check,
                                args.n_samples, args.hz)

        np.save(out_path, phi)
        meta_path = out_path.replace('.npy', '_meta.npz')
        np.savez(meta_path, date=time.strftime('%Y-%m-%d %H:%M:%S'),
                 n_poses=len(records), ridge_lambda=args.ridge_lambda,
                 cond_number=cond_number, tool_description=args.tool_description,
                 mean_before=mean_before, mean_after=mean_after)
        print(f'\nSaved phi -> {out_path}')
        print(f'Saved metadata -> {meta_path}')
        print('\nLoad it with ContactDetector(baseline_mode="gravity_model") + '
             f'.load_phi("{args.output}"), or pass phi directly to '
             'contact_detector.recover_external_force().')

    finally:
        api.CloseAPI()
        print('\nAPI closed.')


if __name__ == '__main__':
    main()
