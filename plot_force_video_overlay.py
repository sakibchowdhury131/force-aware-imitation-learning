#!/usr/bin/env python3
"""
Overlay the live external-force trace onto the recaptured replay camera
footage and export a real .mp4 video (via OpenCV's own VideoWriter --
no ffmpeg required). Composites each waypoint's recaptured RGB frame with a
growing Fx/Fy/Fz/|F| plot strip underneath, plus a live numeric readout
burned into the corner of the camera image.

Standalone, additive script. Does not modify plot_episode_forces.py,
plot_force_gif.py, or analyze_replay_full.py.

Usage:
    python plot_force_video_overlay.py --episode_dir data/episodes/PastaTransfer_force/046
    python plot_force_video_overlay.py --episode_dir ... --camera 1 --fps 8
"""
import os, argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--episode_dir', required=True)
    p.add_argument('--camera', type=int, default=0, help='Which recaptured camera to use (0 or 1)')
    p.add_argument('--save_path', default=None)
    p.add_argument('--fps', type=float, default=8.0, help='Output video frame rate')
    p.add_argument('--strip_height', type=int, default=220, help='Height (px) of the force-plot strip')
    return p.parse_args()


def render_plot_strip(fig, canvas, ax, lines, cursor, t, F, Fmag, i, w, h):
    line_fx, line_fy, line_fz, line_mag = lines
    line_fx.set_data(t[:i+1], F[:i+1, 0])
    line_fy.set_data(t[:i+1], F[:i+1, 1])
    line_fz.set_data(t[:i+1], F[:i+1, 2])
    line_mag.set_data(t[:i+1], Fmag[:i+1])
    cursor.set_xdata([t[i], t[i]])
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    rgb = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    return cv2.resize(rgb, (w, h))


def main():
    args = parse_args()
    replay_dir = os.path.join(args.episode_dir, 'replay')
    forces_path = os.path.join(replay_dir, 'replay_forces_per_frame.npz')
    cam_dir = os.path.join(replay_dir, f'cam{args.camera}')
    save_path = args.save_path or os.path.join(replay_dir, f'force_overlay_cam{args.camera}.mp4')

    fd = np.load(forces_path)
    frame_id = fd['frame_id']
    t = fd['t']
    F = fd['external_force_xyz']
    Fmag = np.linalg.norm(F, axis=1)
    n = len(frame_id)

    frame_files = [f'{fid:06d}.jpg' for fid in frame_id]
    missing = [f for f in frame_files if not os.path.exists(os.path.join(cam_dir, f))]
    if missing:
        raise SystemExit(f"{len(missing)} recaptured frames missing in {cam_dir}, e.g. {missing[:3]}")

    sample = cv2.imread(os.path.join(cam_dir, frame_files[0]))
    cam_h, cam_w = sample.shape[:2]

    # ── Build the plot strip figure once, update in place per frame ──
    dpi = 100
    fig_w, fig_h = cam_w / dpi, args.strip_height / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0.08, 0.22, 0.9, 0.7])
    ax.set_xlim(t[0], t[-1])
    ymin = min(F.min(), 0) - 0.5
    ymax = max(F.max(), Fmag.max()) + 0.5
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('time (s)', fontsize=8)
    ax.set_ylabel('force (N)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    (line_fx,) = ax.plot([], [], color='#d62728', lw=1.0, label='Fx')
    (line_fy,) = ax.plot([], [], color='#2ca02c', lw=1.0, label='Fy')
    (line_fz,) = ax.plot([], [], color='#1f77b4', lw=1.0, label='Fz')
    (line_mag,) = ax.plot([], [], color='k', lw=1.8, label='|F|')
    cursor = ax.axvline(t[0], color='gray', lw=1.0, ls='--', alpha=0.8)
    ax.legend(loc='upper right', fontsize=7, ncol=4)
    lines = (line_fx, line_fy, line_fz, line_mag)

    out_h = cam_h + args.strip_height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(save_path, fourcc, args.fps, (cam_w, out_h))

    for i in range(n):
        cam_img = cv2.imread(os.path.join(cam_dir, frame_files[i]))
        readout = [f't={t[i]:5.2f}s', f'|F|={Fmag[i]:5.2f} N',
                   f'Fx={F[i,0]:+5.2f} Fy={F[i,1]:+5.2f} Fz={F[i,2]:+5.2f}']
        y0 = 24
        for j, line in enumerate(readout):
            y = y0 + j * 22
            cv2.putText(cam_img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(cam_img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)

        strip = render_plot_strip(fig, canvas, ax, lines, cursor, t, F, Fmag, i, cam_w, args.strip_height)
        composite = np.vstack([cam_img, strip])
        vw.write(composite)

    vw.release()
    plt.close(fig)
    print(f"Saved -> {save_path}  ({n} frames @ {args.fps}fps = {n/args.fps:.1f}s, "
          f"{cam_w}x{out_h}, {os.path.getsize(save_path)/1e6:.1f} MB)")


if __name__ == '__main__':
    main()
