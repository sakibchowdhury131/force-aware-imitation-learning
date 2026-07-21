#!/usr/bin/env python3
"""
Continuous-streaming policy deployment: two threads (slow policy inference +
fast velocity-mode control), replacing 07_deploy.py's blocking
predict->clamp->wait_convergence loop with a spline-tracked, force-admittance
-corrected velocity stream. This is a SEPARATE script -- 07_deploy.py and
test_policy.py are imported/reused UNCHANGED, never modified, so every
existing position-based deployment command/checkpoint keeps working exactly
as before regardless of what happens here.

Validated building blocks this reuses (see benchmark_velocity_control.py and
benchmark_force_admittance.py for the standalone tests that established each
one works on this hardware before being wired in here):
  - TrajectoryPoint.Position.Type = CARTESIAN_VELOCITY (KinovaTypes.h enum 7)
    streamed via the same SendBasicTrajectory already used for position
    commands -- confirmed clean 100Hz, ~2ms call latency, ~95% velocity
    tracking accuracy.
  - Force pipeline (firmware + gravity regressor + gravity-residual NN, the
    mass/Coriolis dynamics regressor left OUT as confirmed too slow for
    100Hz) + a per-run TARE (resting bias is pose-dependent and large enough,
    ~3.3-3.6N here, to dominate an untared admittance correction) + a causal
    (online, stateful) 2nd-order Butterworth low-pass at --cutoff_hz.
  - Vector-magnitude correction capping (a bug where per-axis clipping let
    the combined displacement reach cap*sqrt(3) was caught and fixed in the
    benchmark script; this script clips the magnitude directly from the
    start).

Architecture
------------
Slow loop (policy thread): captures n_obs_steps observations spaced to match
training (subsample/record_fps seconds apart), runs policy inference
(DDPM / flow-matching / ACT -- branches on the checkpoint's train_method,
same as 07_deploy.py), timestamps each predicted pose against a shared
monotonic clock, discards any already in the past (inference latency trim),
and refits a per-axis clamped cubic spline over [current commanded position]
+ [remaining future poses], with the near-end boundary condition pinned to
the CURRENT commanded position and velocity (scipy CubicSpline
bc_type=((1, v_cur), 'not-a-knot')) so position and velocity are continuous
across chunk transitions -- no jump when a new chunk arrives.

Fast loop (100Hz, main thread): reads q/qdot/force, evaluates the spline at
the current time for (pos, vel), computes the admittance correction
x_cmd = x_spline + clip(K^-1 (F_filtered - f_desired), max_correction_cm),
tracks x_cmd via v_cmd = v_spline + kp_track*(x_cmd - x_actual) (feedforward
from the spline + proportional position-error feedback -- the spec calls for
sending "x_cmd" through a velocity-mode interface, which needs this kind of
feedforward+feedback law to convert a position target into a velocity
command; this is the one place this script fills a gap the spec left
implicit). Rotation is tracked more simply -- SLERP between buffered
orientation waypoints (not a velocity-continuous spline) with a proportional
angular-velocity command toward the SLERP target, since the spec's
spline+admittance formula is explicitly translation-only.

Safety: if the buffer runs dry (policy slower than the buffer drains), holds
the last commanded pose with ZERO velocity rather than extrapolating past
the spline's fitted domain. If filtered force exceeds --force_threshold,
freezes the position reference (holds the pose from the moment the
threshold was crossed) while the admittance term keeps applying around that
frozen point ("hold compliant"), until force drops back down. --K,
--force_threshold, --cutoff_hz, --f_desired, and --admittance (on/off) are
all config parameters, per the spec, so this one script covers both the
stiff-position-mode and blind-compliance baseline conditions.

--execute defaults to OFF (dry run): every velocity command is computed and
logged exactly as it would be, but sent as ZERO -- same call rate, same
control-flow, zero risk of motion. Given how much new, never-live-tested
logic this script has, run dry first and inspect the console/log output
before ever adding --execute.

Usage (dry run, position-only tracking, no compliance):
    python deploy_streaming.py \\
        --checkpoint data/checkpoints/PastaTransfer_force_replay_dualcam_h16_all37/policy_final.pt \\
        --no_arm_mask \\
        --robot_extrinsics data/robot_extrinsics_stick_corrected_zmeasured.npy \\
        --robot_extrinsics_proprio data/robot_extrinsics_stick_corrected_zmeasured.npy \\
        --mesh newspoon1.obj --tool_prompt "spoon" --track_cam 1 \\
        --init_episode_dir data/episodes/PastaTransfer_force/021 --init_frame 0

Usage (blind compliance, real motion):
    ... --admittance --f_desired 0 --K 200 --force_threshold 15 --execute
"""
import os, sys, time, ctypes, argparse, threading, collections, importlib.util
import numpy as np
import cv2
import torch
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp
from scipy.signal import butter, sosfilt, sosfilt_zi

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)

from policy_common import pose_matrix_to_9d
from test_policy import (load_model, make_transform, predict_action_sequence,
                         predict_action_sequence_flow, predict_action_sequence_act,
                         draw_axes_simple, draw_axes_with_horizon)
from contact_detector import torque_to_wrench, gravity_regressor, compute_jacobian
from fit_gravity_residual_nn import GravityResidualNet, predict as predict_gravity_residual

CARTESIAN_VELOCITY = 7   # KinovaTypes.h POSITION_TYPE enum


def _load_deploy_module():
    """'07_deploy' starts with a digit, not a valid module name for `import` --
    load it by file path instead. Reuses its ctypes structs, FK, masking, and
    camera helpers UNCHANGED -- this script never modifies 07_deploy.py, so
    every existing position-based deployment keeps working regardless."""
    spec = importlib.util.spec_from_file_location(
        '_deploy07', os.path.join(PIPELINE_DIR, '07_deploy.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Robot connection: superset of 07_deploy.py's + calibrate_firmware_gravity.py's
# api bindings (position+velocity commands AND force/velocity reads) -- neither
# existing load_api() binds everything this script needs on its own. ─────────────

def load_api_full(deploy):
    ctypes.CDLL(deploy.COMM_LIB_PATH, mode=ctypes.RTLD_GLOBAL)
    api = ctypes.CDLL(deploy.LIB_PATH)
    for fn in ('InitAPI', 'CloseAPI', 'RefresDevicesList', 'GetDevices', 'SetActiveDevice',
               'StartControlAPI', 'StopControlAPI', 'GetCartesianPosition', 'GetAngularPosition',
               'GetAngularVelocity', 'GetAngularForceGravityFree', 'SetCartesianControl',
               'SendBasicTrajectory', 'EraseAllTrajectories'):
        getattr(api, fn).restype = ctypes.c_int
    api.SendBasicTrajectory.argtypes = [deploy.TrajectoryPoint]
    return api


def connect_full(deploy, api):
    r = api.InitAPI()
    if not deploy.ok(r):
        raise SystemExit(f'InitAPI() failed: {r}')
    api.RefresDevicesList()
    devices = (deploy.KinovaDevice * deploy.MAX_KINOVA_DEVICE)()
    err = ctypes.c_int(deploy.NO_ERROR_KINOVA)
    n = api.GetDevices(devices, ctypes.byref(err))
    if n == 0:
        api.CloseAPI(); raise SystemExit('No Kinova device found')
    api.SetActiveDevice(devices[0])
    api.StartControlAPI(); api.StopControlAPI(); api.StartControlAPI()
    api.SetCartesianControl()
    print(f"Connected to {devices[0].Model.decode()} (serial {devices[0].SerialNumber.decode()})")
    grav_ok = deploy.apply_saved_gravity_params(api)
    print(f"Firmware gravity params reapplied: {'OK' if grav_ok else 'FAILED — aborting'}")
    if not grav_ok:
        api.CloseAPI(); sys.exit(1)
    return api


def get_qdot_deg(deploy, api):
    pos = deploy.AngularPosition()
    api.GetAngularVelocity(ctypes.byref(pos))
    a = pos.Actuators
    return np.array([a.Actuator1, a.Actuator2, a.Actuator3,
                     a.Actuator4, a.Actuator5, a.Actuator6], dtype=np.float64)


def get_tau_gf(deploy, api):
    pos = deploy.AngularPosition()
    api.GetAngularForceGravityFree(ctypes.byref(pos))
    a = pos.Actuators
    return np.array([a.Actuator1, a.Actuator2, a.Actuator3,
                     a.Actuator4, a.Actuator5, a.Actuator6], dtype=np.float64)


def get_cartesian_pose_checked(deploy, api):
    """Like deploy.get_cartesian_pose, but returns (pose, ok) instead of just
    printing a warning and returning a possibly-stale/zero struct on failure
    -- the fast loop needs to know READ FAILED so it can skip that tick
    rather than command based on garbage data."""
    pos = deploy.CartesianPosition()
    r = api.GetCartesianPosition(ctypes.byref(pos))
    c = pos.Coordinates
    pose = np.array([c.X, c.Y, c.Z, c.ThetaX, c.ThetaY, c.ThetaZ], dtype=np.float64)
    return pose, deploy.ok(r)


def send_cartesian_velocity(deploy, api, vx, vy, vz, wx, wy, wz):
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


class OnlineButterworth:
    """Causal 2nd-order low-pass with persistent state -- the real-time-loop
    counterpart to filtfilt (non-causal, needs the whole signal in advance,
    unusable online). One instance per scalar channel."""
    def __init__(self, cutoff_hz: float, fs_hz: float, order: int = 2):
        self.sos = butter(order, cutoff_hz, btype='low', fs=fs_hz, output='sos')
        self._zi = None

    def update(self, x: float) -> float:
        if self._zi is None:
            self._zi = sosfilt_zi(self.sos) * x
        y, self._zi = sosfilt(self.sos, [x], zi=self._zi)
        return float(y[0])


class SharedState:
    """Lock-protected state shared between the slow (policy) and fast
    (control) threads. The fast loop OWNS last_cmd_*  (updates it every
    tick); the slow loop READS it once per chunk, to seed the new spline's
    near-end boundary condition (continuous position+velocity across chunk
    transitions)."""
    def __init__(self):
        self.lock = threading.Lock()
        self.spline_x = self.spline_y = self.spline_z = None
        self.spline_t0 = None     # time.time() the spline's relative axis is measured from
        self.spline_tmax = None   # last valid relative time -- past this, buffer is dry
        self.rot_t_rel = None     # (N,) relative times for rotation waypoints
        self.rot_list = None      # Rotation, length N
        # GROUND TRUTH, read fresh from the robot every fast-loop tick -- the
        # slow loop anchors every new spline to THESE, never to what was
        # merely commanded. Anchoring to the commanded pose instead (an
        # earlier version of this script did exactly that) lets any gap
        # between commanded and actual position compound indefinitely across
        # chunk transitions, since nothing ever re-checks it against reality
        # -- the same root mistake --wait_convergence fixed in 07_deploy.py,
        # reintroduced here and found the same way (a live divergence).
        self.last_actual_pos = None   # (3,) measured base-frame EEF position, m
        self.last_actual_R = None     # measured orientation, scipy Rotation
        self.stop = False
        # Live diagnostics for the preview HUD -- written every fast-loop tick,
        # read a few times/sec by slow_loop's display code. Piggybacks on the
        # same lock acquisition fast_loop already does for last_actual_*, so
        # this is free (no extra locking at 100Hz).
        self.hud_step = 0
        self.hud_force = 0.0
        self.hud_v_precap = 0.0
        self.hud_rot_err_deg = 0.0
        self.hud_tracking_cm = 0.0


# ── Slow loop: observe, infer, timestamp, trim, refit spline ────────────────────

def slow_loop(shared, deploy, ctx):
    (model, normalizer, train_method, flow_time_scale, flow_ode_steps, transform, unet,
     args, device, pipe, align, depth_scale, pipe_other, align_other, depth_scale_other,
     T_task_base_prop, T_eef_spoon, to_origin, T_tool_eef, T_base_task, action_frame,
     n_obs_steps, n_views, dt_action, action_horizon, tf_world2cam, K) = ctx
    inv_to_origin = np.linalg.inv(to_origin)

    # All cv2 GUI calls (window creation + imshow/waitKey) happen on THIS
    # thread only -- OpenCV's Linux GUI backends aren't reliably thread-safe
    # across different threads, and fast_loop (the control loop) runs on the
    # main thread, so mixing window creation there with imshow here risks a
    # crash/hang. Keep every cv2 GUI call confined to slow_loop.
    cv2.namedWindow("Deploy",      cv2.WINDOW_NORMAL)
    cv2.namedWindow("Policy View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Deploy",      848, 480)
    cv2.resizeWindow("Policy View", 848, 480)

    obs_buffer = collections.deque(maxlen=n_obs_steps)
    latest_poses_base = None   # raw policy output (same frame as action_frame), for the preview
    disp_step = 0
    while not shared.stop:
        while len(obs_buffer) < n_obs_steps and not shared.stop:
            t_obs = time.time()
            q_deg = deploy.get_joint_angles_deg(deploy.api_global)
            rgb, _ = deploy.capture(pipe, align, depth_scale)

            T_base_eef_fk = deploy.joint_angles_to_eef(q_deg)
            T_base_tool = T_base_eef_fk @ T_eef_spoon @ to_origin
            T_task_tool = np.linalg.inv(T_task_base_prop) @ T_base_tool
            proprio_raw = pose_matrix_to_9d(T_task_tool if action_frame == 'task' else T_base_tool)
            proprio_norm = normalizer.normalize(proprio_raw.reshape(1, 9))[0]

            if args.no_arm_mask:
                masked_rgb = rgb
                mask = None
            else:
                mask = deploy.unet_mask(unet, rgb, args.unet_threshold, device)
                masked_rgb = deploy.apply_mask(rgb, mask)
            img_t = transform(Image.fromarray(masked_rgb))

            masked_other = None
            if n_views > 1 and pipe_other is not None:
                rgb_other, _ = deploy.capture(pipe_other, align_other, depth_scale_other)
                if args.no_arm_mask:
                    masked_other = rgb_other
                else:
                    mask_o = deploy.unet_mask(unet, rgb_other, args.unet_threshold, device)
                    masked_other = deploy.apply_mask(rgb_other, mask_o)
                view_tensor = torch.stack([img_t, transform(Image.fromarray(masked_other))])
            else:
                view_tensor = img_t.unsqueeze(0)

            # ── Preview: "Deploy" (current + predicted-horizon axes on raw cam)
            # and "Policy View" (what the policy actually sees) -- same helpers
            # 07_deploy.py uses, called here every observation tick (~10Hz). ──
            ob_in_cam = tf_world2cam @ np.linalg.inv(T_base_task) @ T_base_tool @ inv_to_origin
            vis = draw_axes_simple(rgb, ob_in_cam, K, scale=0.10)
            if mask is not None and mask.sum() > 0:
                red = np.zeros_like(vis); red[:, :, 0] = 255
                m = mask.astype(bool)
                vis[m] = cv2.addWeighted(vis, 0.5, red, 0.5, 0)[m]
            if latest_poses_base is not None:
                if action_frame == 'task':
                    cam_poses_pred = [tf_world2cam @ p.astype(np.float64) @ inv_to_origin
                                      for p in latest_poses_base]
                else:
                    cam_poses_pred = [tf_world2cam @ np.linalg.inv(T_base_task)
                                      @ p.astype(np.float64) @ inv_to_origin
                                      for p in latest_poses_base]
                vis = draw_axes_with_horizon(vis, cam_poses_pred, K, scale=0.04)

            with shared.lock:
                hud_step, hud_force = shared.hud_step, shared.hud_force
                hud_v, hud_rot, hud_track = (shared.hud_v_precap, shared.hud_rot_err_deg,
                                             shared.hud_tracking_cm)
            t_task = T_task_tool[:3, 3]
            cv2.putText(vis, f"step {hud_step}  {'EXECUTE' if args.execute else 'DRY RUN'}  "
                             f"|F|={hud_force:.1f}N  v_precap={hud_v*100:.1f}cm/s  "
                             f"rot_err={hud_rot:.1f}deg  track_err={hud_track:.1f}cm",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
            cv2.putText(vis, f"proprio(task) xyz= {t_task[0]*100:.1f},{t_task[1]*100:.1f},{t_task[2]*100:.1f} cm",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            if latest_poses_base is not None:
                p0 = latest_poses_base[0]
                cv2.putText(vis, f"pred[0] xyz= {p0[0,3]*100:.1f},{p0[1,3]*100:.1f},{p0[2,3]*100:.1f} cm",
                            (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

            pv_left = cv2.resize(cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR), (424, 480))
            if masked_other is not None:
                pv_right = cv2.resize(cv2.cvtColor(masked_other, cv2.COLOR_RGB2BGR), (424, 480))
                cv2.putText(pv_left, "cam", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                pv_right = pv_left
            cv2.imshow("Policy View", np.concatenate([pv_left, pv_right], axis=1))
            cv2.imshow("Deploy", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                shared.stop = True
            disp_step += 1

            obs_buffer.append((view_tensor, proprio_norm, t_obs))
            elapsed = time.time() - t_obs
            if elapsed < dt_action:
                time.sleep(dt_action - elapsed)

        if shared.stop:
            break

        view_tensors = [o[0] for o in obs_buffer]
        proprio_list = [o[1] for o in obs_buffer]
        t_obs_last = obs_buffer[-1][2]

        t_infer_start = time.time()
        if train_method == 'flow_matching':
            _, poses = predict_action_sequence_flow(
                model, normalizer, view_tensors, proprio_list, device,
                time_scale=flow_time_scale, ode_steps=flow_ode_steps)
        elif train_method == 'act':
            _, poses = predict_action_sequence_act(model, normalizer, view_tensors, proprio_list, device)
        else:
            _, poses = predict_action_sequence(
                model, normalizer, ctx_noise_scheduler[0], view_tensors, proprio_list, device)
        t_infer_end = time.time()
        latest_poses_base = poses

        chunk_t_abs, chunk_pos, chunk_rot = [], [], []
        for i, T_pred in enumerate(poses):
            t_pred_abs = t_obs_last + (i + 1) * dt_action
            if t_pred_abs <= t_infer_end:
                continue   # latency trim: already in the past
            T_pred64 = T_pred.astype(np.float64)
            T_base_tool_pred = (T_base_task @ T_pred64) if action_frame == 'task' else T_pred64
            T_base_eef_pred = T_base_tool_pred @ T_tool_eef
            chunk_t_abs.append(t_pred_abs)
            chunk_pos.append(T_base_eef_pred[:3, 3].copy())
            chunk_rot.append(Rotation.from_matrix(T_base_eef_pred[:3, :3]))

        infer_ms = (t_infer_end - t_infer_start) * 1000
        if not chunk_t_abs:
            print(f"  [slow loop] WARNING: entire chunk stale ({infer_ms:.0f}ms inference "
                  f"latency >= {action_horizon*dt_action*1000:.0f}ms horizon) -- discarding, "
                  f"buffer will run dry until the next chunk")
            obs_buffer.clear()
            continue

        with shared.lock:
            x_cur = shared.last_actual_pos.copy() if shared.last_actual_pos is not None else chunk_pos[0]
            R_cur = shared.last_actual_R if shared.last_actual_R is not None else chunk_rot[0]
        # v_cur = 0 (not the previous spline's commanded velocity): starting
        # every new spline from rest is a deliberate simplification, trading
        # away velocity-continuity smoothness at chunk transitions for one
        # less place a not-grounded-in-reality value could compound error.
        # Revisit once position-anchoring is proven solid in real testing.
        v_cur = np.zeros(3)

        t_now = time.time()
        t_axis_full = np.array([t_now] + chunk_t_abs) - t_now
        pos_full = np.array([x_cur] + chunk_pos)
        rot_full = [R_cur] + chunk_rot

        # Minimum spacing, not just dedup: the latency trim keeps whichever
        # 100ms-grid pose is the first to land after inference finished, and
        # that can be an arbitrarily SHORT gap after t=0 (pure luck in how
        # inference latency aligns with the fixed action grid) -- forcing a
        # cubic spline to cover a normal ~dt_action-sized position step in a
        # few ms produces a huge local derivative (confirmed: reproduced
        # 300-900cm/s velocities offline with a 5ms first gap, matching the
        # exponential blowup seen live -- 0.1cm/s to 926cm/s in ~150ms).
        # Require every kept gap to be at least a meaningful fraction of
        # dt_action, dropping points that are too close instead of just
        # near-exact duplicates.
        min_gap = dt_action * 0.3
        keep = [True]
        last_kept_t = t_axis_full[0]
        for t in t_axis_full[1:]:
            if t - last_kept_t >= min_gap:
                keep.append(True)
                last_kept_t = t
            else:
                keep.append(False)
        keep = np.array(keep)
        t_axis = t_axis_full[keep]
        pos_axis = pos_full[keep]
        rot_list = [r for r, k in zip(rot_full, keep) if k]

        if len(t_axis) < 2:
            print("  [slow loop] WARNING: <2 valid points after min-gap filtering -- skipping refit")
            obs_buffer.clear()
            continue

        try:
            sx = CubicSpline(t_axis, pos_axis[:, 0], bc_type=((1, v_cur[0]), 'not-a-knot'))
            sy = CubicSpline(t_axis, pos_axis[:, 1], bc_type=((1, v_cur[1]), 'not-a-knot'))
            sz = CubicSpline(t_axis, pos_axis[:, 2], bc_type=((1, v_cur[2]), 'not-a-knot'))
        except Exception as e:
            print(f"  [slow loop] spline fit failed: {e} -- skipping")
            obs_buffer.clear()
            continue

        with shared.lock:
            shared.spline_x, shared.spline_y, shared.spline_z = sx, sy, sz
            shared.spline_t0 = t_now
            shared.spline_tmax = t_axis[-1]
            shared.rot_t_rel = t_axis
            shared.rot_list = rot_list

        print(f"  [slow loop] inference {infer_ms:.0f}ms  kept {len(chunk_t_abs)}/{action_horizon}  "
              f"buffer valid {t_axis[-1]:.2f}s ahead")
        obs_buffer.clear()

    cv2.destroyAllWindows()


# ── Fast loop: 100Hz spline + admittance + velocity streaming ───────────────────

def fast_loop(shared, deploy, api, args, filters, tare_offset, log):
    dt_nominal = 1.0 / args.rate_hz
    frozen_pos = None
    t_start = time.time()
    next_tick = t_start
    step = 0
    print(f"\n{'*** DRY RUN' if not args.execute else '*** EXECUTE'} — "
          f"streaming at {args.rate_hz:.0f}Hz. Ctrl+C to stop. ***\n")
    try:
        while not shared.stop:
            t_now = time.time()

            q_deg = deploy.get_joint_angles_deg(api)
            qdot_deg = get_qdot_deg(deploy, api)
            tau_gf = get_tau_gf(deploy, api)
            pose_raw, pose_ok = get_cartesian_pose_checked(deploy, api)
            if not pose_ok:
                # Read failed -- don't command off possibly-stale/zero data.
                # Send zero velocity this tick and skip straight to the next.
                send_cartesian_velocity(deploy, api, 0, 0, 0, 0, 0, 0)
                step += 1
                next_tick += dt_nominal
                sleep_for = next_tick - time.time()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                continue
            actual_pose = deploy.kinova_pose_to_matrix(pose_raw)
            x_actual = actual_pose[:3, 3]
            R_actual = Rotation.from_matrix(actual_pose[:3, :3])

            F_raw = torque_to_wrench(q_deg, tau_gf, damping=args.damping)[:3] - tare_offset
            F_filt = np.array([filters[i].update(F_raw[i]) for i in range(3)])
            F_mag = float(np.linalg.norm(F_filt))

            with shared.lock:
                sx, sy, sz = shared.spline_x, shared.spline_y, shared.spline_z
                t0, tmax = shared.spline_t0, shared.spline_tmax
                rot_t_rel, rot_list = shared.rot_t_rel, shared.rot_list

            buffer_dry = (sx is None) or (t0 is None) or (t_now - t0 > tmax)

            if F_mag > args.force_threshold:
                if frozen_pos is None:
                    frozen_pos = x_actual.copy()
                    print(f"  [SAFETY] |F|={F_mag:.1f}N > threshold {args.force_threshold:.1f}N "
                          f"-- freezing reference, holding compliant")
                x_spline, v_spline = frozen_pos, np.zeros(3)
            elif buffer_dry:
                frozen_pos = None
                x_spline = x_actual   # hold at the REAL current position, not a stale command
                v_spline = np.zeros(3)
            else:
                frozen_pos = None
                trel = float(np.clip(t_now - t0, 0.0, tmax))
                x_spline = np.array([sx(trel), sy(trel), sz(trel)])
                v_spline = np.array([sx(trel, 1), sy(trel, 1), sz(trel, 1)])

            if args.admittance:
                correction = (F_filt - args.f_desired) / args.K
                cap = args.max_correction_cm / 100
                mag = np.linalg.norm(correction)
                if mag > cap:
                    correction = correction * (cap / mag)
            else:
                correction = np.zeros(3)

            x_cmd = x_spline + correction
            v_cmd = v_spline + args.kp_track * (x_cmd - x_actual)

            if rot_list is not None and len(rot_list) >= 2:
                trel_r = float(np.clip(t_now - t0, rot_t_rel[0], rot_t_rel[-1]))
                slerp = Slerp(rot_t_rel, Rotation.concatenate(rot_list))
                R_target = slerp([trel_r])[0]
            elif rot_list is not None and len(rot_list) == 1:
                R_target = rot_list[0]
            else:
                R_target = R_actual
            # Body-frame angular-velocity law (Rdot = R @ [w]x): empirically the
            # spatial-frame version (R_target * R_actual.inv()) made rot_err_deg
            # grow monotonically 0->42deg over 6s of real execution instead of
            # converging -- classic sign/frame mismatch. This is the other of
            # the two standard SO(3) proportional laws; if the firmware wants
            # body-frame Omega, this is the one that actually drives R_actual
            # toward R_target.
            R_err = R_actual.inv() * R_target
            w_cmd = args.kp_rot * R_err.as_rotvec()
            rot_err_deg = float(np.degrees(np.linalg.norm(R_err.as_rotvec())))

            # Hard safety ceiling on the FINAL commanded velocity, independent
            # of everything upstream (spline overshoot, a bad chunk, a large
            # tracking error, whatever) -- every other motion path in this
            # pipeline (07_deploy.py, replay_episode.py) clamps effective
            # speed via a per-step position clamp; this is the equivalent
            # backstop for velocity-mode streaming, which otherwise has no
            # upper bound on kp_track * (arbitrarily large) position error.
            v_mag_precap = float(np.linalg.norm(v_cmd))
            v_capped = v_mag_precap > args.max_speed
            if v_capped:
                v_cmd = v_cmd * (args.max_speed / v_mag_precap)
                print(f"  [SAFETY] step {step}: v_cmd {v_mag_precap*100:.1f}cm/s > cap "
                      f"{args.max_speed*100:.1f}cm/s -- clamped (tracking error was "
                      f"{np.linalg.norm(x_cmd - x_actual)*100:.1f}cm)")
            w_mag_precap = float(np.linalg.norm(w_cmd))
            w_capped = w_mag_precap > args.max_rot_speed
            if w_capped:
                w_cmd = w_cmd * (args.max_rot_speed / w_mag_precap)

            if args.execute:
                send_cartesian_velocity(deploy, api, *v_cmd, *w_cmd)
            else:
                send_cartesian_velocity(deploy, api, 0, 0, 0, 0, 0, 0)

            with shared.lock:
                shared.last_actual_pos = x_actual.copy()
                shared.last_actual_R = R_actual
                shared.hud_step = step
                shared.hud_force = F_mag
                shared.hud_v_precap = v_mag_precap
                shared.hud_rot_err_deg = rot_err_deg
                shared.hud_tracking_cm = float(np.linalg.norm(x_cmd - x_actual)) * 100

            log['t'].append(t_now - t_start)
            log['q'].append(q_deg.copy())
            log['qdot'].append(qdot_deg.copy())
            log['cmd_pos'].append(x_cmd.copy())
            log['v_cmd'].append(v_cmd.copy())
            log['v_mag_precap'].append(v_mag_precap)
            log['v_capped'].append(v_capped)
            log['raw_force'].append(F_raw.copy())
            log['filt_force'].append(F_filt.copy())
            log['buffer_dry'].append(buffer_dry)
            log['frozen'].append(frozen_pos is not None)
            log['R_actual_quat'].append(R_actual.as_quat())
            log['R_target_quat'].append(R_target.as_quat())
            log['rot_err_deg'].append(rot_err_deg)
            log['w_cmd'].append(w_cmd.copy())
            log['w_mag_precap'].append(w_mag_precap)
            log['w_capped'].append(w_capped)

            if step % 100 == 0:
                print(f"  step {step:5d}  t={t_now-t_start:6.2f}s  |F|={F_mag:5.2f}N  "
                      f"corr={np.linalg.norm(correction)*100:5.2f}cm  "
                      f"v_precap={v_mag_precap*100:5.1f}cm/s  "
                      f"rot_err={rot_err_deg:5.1f}deg  w_precap={w_mag_precap:5.2f}rad/s  "
                      f"{'DRY' if buffer_dry else 'OK '}  "
                      f"{'FROZEN' if frozen_pos is not None else ''}")

            step += 1
            if args.max_steps > 0 and step >= args.max_steps:
                break
            if args.duration > 0 and (t_now - t_start) >= args.duration:
                break

            next_tick += dt_nominal
            sleep_for = next_tick - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("\n  Ctrl+C -- stopping.")
    finally:
        shared.stop = True
        for _ in range(5):
            send_cartesian_velocity(deploy, api, 0, 0, 0, 0, 0, 0)
            time.sleep(0.02)
        print("  Sent explicit zero-velocity stop.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--no_arm_mask', action='store_true')
    p.add_argument('--unet_checkpoint', default='robot_segmentation_UNET/training/checkpoints/best_model.pth')
    p.add_argument('--unet_threshold', type=float, default=0.5)
    p.add_argument('--mesh', default='newspoon1.obj')
    p.add_argument('--tool_prompt', default='spoon')
    p.add_argument('--T_eef_spoon', default='data/T_eef_spoon.npy')
    p.add_argument('--robot_extrinsics', default='data/robot_extrinsics.npy')
    p.add_argument('--robot_extrinsics_proprio', default=None)
    p.add_argument('--task_frame', default=None,
                   help='tf_world2cam for the "Deploy" preview overlay. Defaults to '
                        'data/cam_extrinsics.npy (track_cam=0) or data/cam<N>_extrinsics.npy, '
                        'same as 07_deploy.py.')
    p.add_argument('--track_cam', type=int, default=1)
    p.add_argument('--init_episode_dir', default=None)
    p.add_argument('--init_frame', type=int, default=0)
    p.add_argument('--rate_hz', type=float, default=100.0, help='Fast-loop control rate.')
    p.add_argument('--damping', type=float, default=0.05, help='Jacobian-transpose damping.')
    p.add_argument('--tare_duration', type=float, default=1.0)
    p.add_argument('--cutoff_hz', type=float, default=5.0)
    p.add_argument('--admittance', action='store_true',
                   help='Enable the force-admittance correction. Without this: stiff position '
                        'mode (pure spline tracking, correction always 0) -- one baseline condition.')
    p.add_argument('--K', type=float, default=200.0, help='Admittance stiffness, N/m.')
    p.add_argument('--f_desired', type=float, default=0.0, help='Target force per axis, N.')
    p.add_argument('--max_correction_cm', type=float, default=2.0)
    p.add_argument('--force_threshold', type=float, default=15.0, help='Freeze-and-hold threshold, N.')
    p.add_argument('--kp_track', type=float, default=2.0,
                   help='Position-error feedback gain (1/s) added to the spline feedforward '
                        'velocity -- fills the position->velocity gap the spec left implicit.')
    p.add_argument('--kp_rot', type=float, default=1.5, help='Rotation-error feedback gain (1/s).')
    p.add_argument('--max_speed', type=float, default=0.05,
                   help='Hard cap on total commanded translation speed, m/s (default 5cm/s) -- '
                        'independent backstop on top of everything else, since kp_track * '
                        '(position error) otherwise has no upper bound.')
    p.add_argument('--max_rot_speed', type=float, default=0.5,
                   help='Hard cap on total commanded angular speed, rad/s.')
    p.add_argument('--subsample', type=int, default=3)
    p.add_argument('--record_fps', type=float, default=30.0)
    p.add_argument('--max_steps', type=int, default=0)
    p.add_argument('--duration', type=float, default=0.0)
    p.add_argument('--execute', action='store_true',
                   help='Send real (nonzero) velocity commands. Without this: dry run -- '
                        'everything computed and logged, but zero velocity is always sent.')
    p.add_argument('--output_dir', default='/tmp/deploy_streaming')
    p.add_argument('--device', default='cuda')
    return p.parse_args()


ctx_noise_scheduler = [None]   # slow_loop reads this; set in main() after loading the checkpoint


def main():
    args = parse_args()
    if args.task_frame is None:
        args.task_frame = ('data/cam_extrinsics.npy' if args.track_cam == 0
                           else f'data/cam{args.track_cam}_extrinsics.npy')
    os.makedirs(args.output_dir, exist_ok=True)
    deploy = _load_deploy_module()
    device = torch.device(args.device)

    # ── Load policy (same train_method branching as 07_deploy.py) ────────────
    print(f"Loading policy checkpoint: {args.checkpoint}")
    model, normalizer, ckpt = load_model(args.checkpoint, device)
    n_obs_steps  = ckpt.get('n_obs_steps', 2)
    n_views      = ckpt.get('n_views', 1)
    action_frame = ckpt.get('action_frame', 'task')
    action_horizon = model.action_horizon
    train_method = ckpt.get('train_method', 'ddpm')
    flow_time_scale = flow_ode_steps = None
    if train_method == 'ddpm':
        ctx_noise_scheduler[0] = ckpt['noise_scheduler']
        ctx_noise_scheduler[0].set_timesteps(ctx_noise_scheduler[0].config.num_train_timesteps)
    elif train_method == 'flow_matching':
        flow_time_scale = ckpt.get('time_scale', 999.0)
        flow_ode_steps = ckpt.get('ode_steps', 50)
    elif train_method != 'act':
        raise SystemExit(f"Unsupported train_method '{train_method}'")
    transform = make_transform(ckpt.get('image_size', 128), ckpt.get('crop_size', 115))
    dt_action = args.subsample / args.record_fps
    print(f"Model: train_method={train_method}  n_obs_steps={n_obs_steps}  n_views={n_views}  "
          f"action_horizon={action_horizon}  action_frame={action_frame}  dt_action={dt_action*1000:.0f}ms")

    # ── UNet arm mask ──────────────────────────────────────────────────────
    unet = None
    if not args.no_arm_mask:
        print(f"Loading UNet arm segmentor: {args.unet_checkpoint}")
        unet = deploy.load_unet(args.unet_checkpoint, device)

    # ── Calibration ────────────────────────────────────────────────────────
    T_base_task = np.load(args.robot_extrinsics).astype(np.float64)
    prop_path = args.robot_extrinsics_proprio or args.robot_extrinsics
    T_base_task_prop = np.load(prop_path).astype(np.float64)
    tf_world2cam = np.load(args.task_frame).astype(np.float64)
    print(f"Loaded tf_world2cam from {args.task_frame}")

    import trimesh
    mesh = trimesh.load(args.mesh, force='mesh')
    to_origin, _ = trimesh.bounds.oriented_bounds(mesh)
    inv_to_origin = np.linalg.inv(to_origin)
    T_eef_spoon = np.load(args.T_eef_spoon).astype(np.float64)
    T_tool_eef = inv_to_origin @ np.linalg.inv(T_eef_spoon)

    # ── Cameras ────────────────────────────────────────────────────────────
    pipe, align, K, depth_scale = deploy.start_realsense(args.track_cam)
    pipe_other = align_other = depth_scale_other = None
    if n_views > 1:
        other_cam = 0 if args.track_cam != 0 else 1
        print(f"Dual-cam checkpoint (n_views={n_views}) -- starting cam{other_cam} too.")
        pipe_other, align_other, _, depth_scale_other = deploy.start_realsense(other_cam)

    # ── Robot connection (full bindings: position + velocity + force) ────────
    api = load_api_full(deploy)
    connect_full(deploy, api)
    deploy.api_global = api   # slow_loop reads joint angles through the same shared connection

    # ── Pre-position to a known in-distribution pose (same as 07_deploy.py) ──
    if args.init_episode_dir:
        init_pose_path = os.path.join(args.init_episode_dir, 'augmented', 'tool_poses_base.npz')
        init_poses = np.load(init_pose_path)
        T_base_tool_init = init_poses[str(args.init_frame)].astype(np.float64)
        T_base_eef_init = T_base_tool_init @ T_tool_eef
        T_base_eef_now = deploy.kinova_pose_to_matrix(deploy.get_cartesian_pose(api))
        init_dist = np.linalg.norm(T_base_eef_init[:3, 3] - T_base_eef_now[:3, 3])
        print(f"\nPre-positioning to {args.init_episode_dir} frame {args.init_frame} "
              f"({init_dist*100:.1f}cm away)...")
        if args.execute:
            deploy.send_cartesian_pose(api, deploy.matrix_to_kinova_pose(T_base_eef_init))
            t_wait = time.time()
            while time.time() - t_wait < max(2.0, init_dist / 0.03 + 1.0):
                actual = deploy.get_cartesian_pose(api)[:3]
                if np.linalg.norm(actual - T_base_eef_init[:3, 3]) < 0.01:
                    break
                time.sleep(0.05)
            print("  Pre-position done.")
        else:
            print("  (dry run — not moving)")

    # ── Force pipeline: tare, then start streaming ────────────────────────────
    phi = np.load('data/gravity_phi_task_only.npy')
    ckpt_nn = torch.load('data/gravity_residual_nn.pt', weights_only=False)
    g_nn = GravityResidualNet(); g_nn.load_state_dict(ckpt_nn['state_dict']); g_nn.eval()
    g_x_mean, g_x_std = ckpt_nn['x_mean'], ckpt_nn['x_std']

    def read_F_raw():
        q = deploy.get_joint_angles_deg(api)
        gf = get_tau_gf(deploy, api)
        g_res_lin = gravity_regressor(q) @ phi
        g_res_nn = predict_gravity_residual(g_nn, g_x_mean, g_x_std, q)
        tau = gf - g_res_lin - g_res_nn
        return torque_to_wrench(q, tau, damping=args.damping)[:3]

    # ── Singularity check: torque_to_wrench prints a one-line warning buried in
    # scrolling per-tick output, easy to miss. Check once, loudly, up front --
    # force/admittance numbers are meaningless at a near-singular pose (see
    # contact_detector.py: a real ~1N*m torque error can become a reported
    # multi-million-Newton "force" here). ─────────────────────────────────────
    q_check = deploy.get_joint_angles_deg(api)
    cond_J = np.linalg.cond(compute_jacobian(q_check))
    if cond_J > 1e4:
        print(f"\n{'!'*70}\n  WARNING: cond(J) = {cond_J:.2e} at the CURRENT pose -- this is at or\n"
              f"  near a kinematic singularity (e.g. J5=180deg spherical-wrist).\n"
              f"  Force/admittance readings will be UNRELIABLE (possibly wildly wrong)\n"
              f"  until the arm moves away from here. If this dry run never actually\n"
              f"  moves (no --execute), it will STAY at this pose the whole run.\n"
              f"  Recommended: move the arm via joystick to a non-singular pose (e.g.\n"
              f"  near the --init_episode_dir target) before re-running.\n{'!'*70}\n")

    print(f"\nTaring: capturing baseline for {args.tare_duration:.1f}s -- leave the arm untouched")
    n_tare = max(1, int(args.tare_duration * args.rate_hz))
    tare_samples = [read_F_raw() for _ in range(n_tare)]
    tare_offset = np.mean(tare_samples, axis=0)
    print(f"  tare offset (N): [{tare_offset[0]:+.3f}, {tare_offset[1]:+.3f}, {tare_offset[2]:+.3f}]")

    filters = [OnlineButterworth(args.cutoff_hz, args.rate_hz) for _ in range(3)]

    T_task_base_prop = T_base_task_prop   # naming matches 07_deploy.py's proprio computation

    shared = SharedState()
    _init_pose = deploy.kinova_pose_to_matrix(deploy.get_cartesian_pose(api))
    with shared.lock:
        shared.last_actual_pos = _init_pose[:3, 3]
        shared.last_actual_R = Rotation.from_matrix(_init_pose[:3, :3])

    ctx = (model, normalizer, train_method, flow_time_scale, flow_ode_steps, transform, unet,
          args, device, pipe, align, depth_scale, pipe_other, align_other, depth_scale_other,
          T_task_base_prop, T_eef_spoon, to_origin, T_tool_eef, T_base_task, action_frame,
          n_obs_steps, n_views, dt_action, action_horizon, tf_world2cam, K)

    slow_thread = threading.Thread(target=slow_loop, args=(shared, deploy, ctx), daemon=True)
    slow_thread.start()

    log = {'t': [], 'q': [], 'qdot': [], 'cmd_pos': [], 'v_cmd': [], 'v_mag_precap': [],
          'v_capped': [], 'raw_force': [], 'filt_force': [], 'buffer_dry': [], 'frozen': [],
          'R_actual_quat': [], 'R_target_quat': [], 'rot_err_deg': [], 'w_cmd': [],
          'w_mag_precap': [], 'w_capped': []}
    try:
        fast_loop(shared, deploy, api, args, filters, tare_offset, log)
    finally:
        shared.stop = True
        slow_thread.join(timeout=2.0)
        api.CloseAPI()
        out_path = os.path.join(args.output_dir, 'log.npz')
        np.savez(out_path, **{k: np.array(v) for k, v in log.items()})
        print(f"Log saved -> {out_path}")


if __name__ == '__main__':
    main()
