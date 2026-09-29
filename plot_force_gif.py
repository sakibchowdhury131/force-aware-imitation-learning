#!/usr/bin/env python3
"""
Animate an episode's replay_full_forces.npz as a "live view" GIF -- the
external force trace draws in over time with a moving cursor, like watching
a live force-sensor readout during the replay.

Standalone, additive script. Does not modify plot_episode_forces.py or
analyze_replay_full.py.

Usage:
    python plot_force_gif.py --episode_dir data/episodes/PastaTransfer_force/046
    python plot_force_gif.py --log_path <path>/replay_full_forces.npz --save_path out.gif
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--episode_dir', default=None,
                   help='Reads <episode_dir>/replay/replay_full_forces.npz')
    p.add_argument('--log_path', default=None,
                   help='Explicit path to replay_full_forces.npz (overrides --episode_dir)')
    p.add_argument('--save_path', default=None, help='Output .gif path')
    p.add_argument('--fps', type=int, default=15)
    p.add_argument('--n_frames', type=int, default=180,
                   help='Number of animation frames (data is subsampled to this many points)')
    p.add_argument('--speed', type=float, default=1.0,
                   help='Playback speed multiplier vs. real time (2.0 = 2x speed)')
    return p.parse_args()


def main():
    args = parse_args()
    if args.log_path:
        log_path = args.log_path
        default_save = os.path.join(os.path.dirname(log_path), 'force_live.gif')
    elif args.episode_dir:
        log_path = os.path.join(args.episode_dir, 'replay', 'replay_full_forces.npz')
        default_save = os.path.join(args.episode_dir, 'replay', 'force_live.gif')
    else:
        raise SystemExit('Pass --episode_dir or --log_path')
    save_path = args.save_path or default_save

    d = np.load(log_path)
    t = d['t']
    F = d['external_force_xyz']  # (N, 3)
    Fmag = np.linalg.norm(F, axis=1)

    N = len(t)
    n_frames = min(args.n_frames, N)
    frame_idx = np.linspace(0, N - 1, n_frames).astype(int)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=110)
    ax.set_xlim(t[0], t[-1])
    ymin = min(F.min(), 0) - 0.5
    ymax = max(F.max(), Fmag.max()) + 0.5
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('external force (N)')
    ax.set_title('Live external force — replay')
    ax.grid(alpha=0.3)

    (line_fx,) = ax.plot([], [], color='#d62728', lw=1.2, label='Fx')
    (line_fy,) = ax.plot([], [], color='#2ca02c', lw=1.2, label='Fy')
    (line_fz,) = ax.plot([], [], color='#1f77b4', lw=1.2, label='Fz')
    (line_mag,) = ax.plot([], [], color='k', lw=2.2, label='|F|')
    cursor = ax.axvline(t[0], color='gray', lw=1.0, ls='--', alpha=0.8)
    readout = ax.text(0.02, 0.95, '', transform=ax.transAxes, va='top',
                       fontfamily='monospace', fontsize=10,
                       bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    ax.legend(loc='upper right')

    def update(frame_num):
        i = frame_idx[frame_num]
        line_fx.set_data(t[:i+1], F[:i+1, 0])
        line_fy.set_data(t[:i+1], F[:i+1, 1])
        line_fz.set_data(t[:i+1], F[:i+1, 2])
        line_mag.set_data(t[:i+1], Fmag[:i+1])
        cursor.set_xdata([t[i], t[i]])
        readout.set_text(f't={t[i]:5.2f}s\n|F|={Fmag[i]:5.2f} N\n'
                          f'Fx={F[i,0]:+5.2f}  Fy={F[i,1]:+5.2f}  Fz={F[i,2]:+5.2f}')
        return line_fx, line_fy, line_fz, line_mag, cursor, readout

    interval_ms = 1000.0 / args.fps
    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                    interval=interval_ms, blit=True)

    real_duration = t[-1] - t[0]
    playback_fps = args.fps * args.speed * (n_frames / (real_duration * args.fps + 1e-9))
    writer = animation.PillowWriter(fps=max(1, int(round(args.fps * args.speed))))
    anim.save(save_path, writer=writer)
    plt.close(fig)
    print(f"Saved -> {save_path}  ({n_frames} frames, {real_duration:.1f}s of data, "
          f"{os.path.getsize(save_path)/1e6:.1f} MB)")


if __name__ == '__main__':
    main()
