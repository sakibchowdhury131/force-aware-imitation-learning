#!/usr/bin/env python3
"""
One-off check: did the fitted gravity parameters survive whatever fix cleared
the J6 fault (power cycle / joystick fault-reset)? Kinova's SDK has no getter
for the current gravity type/params, so the only way to tell is empirically —
move to a pose we already have clean before/after numbers for and compare.

Reuses connect/move_and_settle/capture_visit from calibrate_firmware_gravity.py
directly rather than duplicating them.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibrate_firmware_gravity as G

# pose 0: before=16.811 N, after=1.872 N — clean, uncontaminated by the J6 fault
# (both measured with all 6 joints converged normally).
POSE0 = G.TEST_POSES_DEG[0]
BEFORE_REF = 16.811
AFTER_REF  = 1.872

api = G.load_api()
try:
    G.connect(api, control=True)
    print(f'\nMoving to pose 0: {np.round(POSE0, 1)} ...')
    err, settled = G.move_and_settle(api, POSE0)
    print(f'Arrived (max err {err:.2f} deg), settled={settled}')

    tau_gf = G.capture_visit(api, n_samples=200, hz=100.0)
    q_actual = G.get_q(api)
    F = float(np.linalg.norm(G.torque_to_wrench(q_actual, tau_gf)[:3]))

    print(f'\n||F|| now = {F:.3f} N')
    print(f'  reference "before" (firmware default gravity): {BEFORE_REF:.3f} N')
    print(f'  reference "after"  (fitted OPTIMAL gravity)   : {AFTER_REF:.3f} N')

    d_before = abs(F - BEFORE_REF)
    d_after  = abs(F - AFTER_REF)
    if d_after < d_before:
        print('\n=> Matches "after" — the fitted gravity parameters PERSISTED through the fix.')
    else:
        print('\n=> Matches "before" — the fit did NOT persist; call '
             'apply_saved_gravity_params(api) at the start of every session.')
finally:
    api.CloseAPI()
    print('\nAPI closed.')
