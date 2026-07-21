#!/usr/bin/env python3
"""
Extend the gravity calibration's training set using ALREADY-COLLECTED dense
logs from the dynamics-excitation replays (004/009/014/019/024/029/039/044/049),
instead of new hardware time: extracts near-static (|qdot| < threshold)
stretches from each episode's continuous motion, averages each contiguous
stretch into one representative (q, tau_gf) pair (like a real calibration
pose, not a single noisy instantaneous sample), and merges with the existing
40-pose task-only training set before refitting phi.

This directly targets the gravity-coverage gap identified by the
zero-velocity NN probe: these 9 episodes' poses were never in
gravity_phi_task_only.npy's training set, and near-static residual there
was found to be nearly as large as the fast-motion residual -- i.e. mostly
uncovered gravity error, not real dynamics.
"""
import os
import numpy as np

from dynamics_dataset import load_episode_dense
from calibrate_gravity_residual import ridge_fit, report_identifiability, LINK_NAMES, load_session_records, save_session_records
from contact_detector import torque_to_wrench, recover_external_force

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCITATION_EPISODES = ['004', '009', '014', '019', '024', '029', '039', '044', '049']


def extract_static_stretches(episode, task='pastaTransfer4', qdot_thresh=3.0, min_run_len=5):
    """Returns list of (q_mean, tau_gf_mean) for each contiguous stretch where
    |qdot| < qdot_thresh for at least min_run_len consecutive dense samples."""
    t, q_deg, qdot_deg, qddot_deg, gf = load_episode_dense(episode, task)
    qdot_norm = np.linalg.norm(qdot_deg, axis=1)
    is_static = qdot_norm < qdot_thresh

    records = []
    i = 0
    n = len(is_static)
    while i < n:
        if is_static[i]:
            j = i
            while j < n and is_static[j]:
                j += 1
            if j - i >= min_run_len:
                # drop the first/last couple samples of the run (still settling in/out)
                lo, hi = i + 1, j - 1
                if hi > lo:
                    records.append((q_deg[lo:hi].mean(axis=0), gf[lo:hi].mean(axis=0)))
            i = j
        else:
            i += 1
    return records


def main():
    print('Extracting near-static stretches from excitation episodes (no new hardware motion)...')
    new_records = []
    for ep in EXCITATION_EPISODES:
        recs = extract_static_stretches(ep)
        print(f'  {ep}: {len(recs)} static stretches')
        new_records += recs
    print(f'Total new records: {len(new_records)}')

    session_path = os.path.join(PIPELINE_DIR, 'data', 'gravity_calibration_records_task_only.npz')
    prior_records = load_session_records(session_path)
    print(f'{len(prior_records)} prior records in {session_path}')

    all_records = prior_records + new_records
    out_session_path = os.path.join(PIPELINE_DIR, 'data', 'gravity_calibration_records_task_only_v2.npz')
    save_session_records(all_records, out_session_path)
    print(f'Saved {len(all_records)} total records -> {out_session_path}')

    phi, cond_number, S, poorly_identified, Vt = ridge_fit(all_records, ridge_lambda=0.1)
    report_identifiability(S, poorly_identified, Vt, LINK_NAMES)

    out_phi_path = os.path.join(PIPELINE_DIR, 'data', 'gravity_phi_task_only_v2.npy')
    np.save(out_phi_path, phi)
    print(f'\nSaved -> {out_phi_path}')

    # Before/after: compare OLD phi vs NEW phi on (a) the original 40 training poses,
    # (b) the new near-static-extracted poses specifically.
    old_phi = np.load(os.path.join(PIPELINE_DIR, 'data', 'gravity_phi_task_only.npy'))

    def report(records, label):
        Fb_old, Fb_new = [], []
        for q, tau_gf in records:
            F_before = np.linalg.norm(torque_to_wrench(q, tau_gf)[:3])
            _t, F_old = recover_external_force(q, tau_gf, old_phi)
            _t, F_new = recover_external_force(q, tau_gf, phi)
            Fb_old.append(np.linalg.norm(F_old[:3]))
            Fb_new.append(np.linalg.norm(F_new[:3]))
        Fb_old, Fb_new = np.array(Fb_old), np.array(Fb_new)
        print(f'\n{label} (n={len(records)}):')
        print(f'  old phi (task_only):    mean ||F|| = {Fb_old.mean():.3f} N')
        print(f'  new phi (task_only_v2): mean ||F|| = {Fb_new.mean():.3f} N')

    report(prior_records, 'ORIGINAL 40 gravity-training poses')
    report(new_records, 'NEW near-static poses from excitation episodes')


if __name__ == '__main__':
    main()
