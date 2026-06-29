"""
Interactive 3D plot of tool trajectories from all episodes.

Loads tool_poses_task.npz (or tool_poses_cam{N}.npz as fallback) from every
episode and plots XYZ translation as coloured lines in an HTML file viewable
in any browser.

Usage:
    python plot_trajectories_3d.py --task_dir data/episodes/pastaTransfer4
    python plot_trajectories_3d.py --task_dir data/episodes/pastaTransfer4 \\
        --subsample 3 --output tool_trajectories_3d.html
"""

import os, argparse, json
import numpy as np

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task_dir',  required=True)
    p.add_argument('--track_cam', type=int, default=0)
    p.add_argument('--subsample', type=int, default=1,
                   help='Plot every N-th pose (default 1 = all)')
    p.add_argument('--output', default=os.path.join(PIPELINE_DIR, 'tool_trajectories_3d.html'))
    return p.parse_args()


def load_trajectory(episode_dir, track_cam):
    aug = os.path.join(episode_dir, 'augmented')
    for fname in (f'tool_poses_task.npz',
                  f'tool_poses_base.npz',
                  f'tool_poses_cam{track_cam}.npz'):
        path = os.path.join(aug, fname)
        if os.path.exists(path):
            data = dict(np.load(path))
            fids = sorted(data.keys(), key=int)
            xyz  = np.array([data[f][:3, 3] for f in fids])
            return xyz, fname
    return None, None


def main():
    args = parse_args()

    episodes = sorted(
        os.path.join(args.task_dir, n)
        for n in os.listdir(args.task_dir)
        if os.path.isdir(os.path.join(args.task_dir, n))
        and os.path.exists(os.path.join(args.task_dir, n, 'meta.json'))
    )
    print(f"Found {len(episodes)} episodes.")

    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not found — installing...")
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'plotly'])
        import plotly.graph_objects as go

    fig    = go.Figure()
    colors = [f'hsl({int(360 * i / max(len(episodes), 1))}, 70%, 50%)'
              for i in range(len(episodes))]

    n_loaded = 0
    pose_src = None
    for i, ep_dir in enumerate(episodes):
        xyz, src = load_trajectory(ep_dir, args.track_cam)
        if xyz is None or len(xyz) == 0:
            print(f"  Skipping {os.path.basename(ep_dir)} — no poses found")
            continue
        pose_src = pose_src or src
        if args.subsample > 1:
            xyz = xyz[::args.subsample]

        ep_name = os.path.basename(ep_dir)
        fig.add_trace(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            mode='lines+markers',
            line=dict(color=colors[i], width=3),
            marker=dict(size=2, color=colors[i]),
            name=ep_name,
            hovertemplate=f'<b>{ep_name}</b><br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>z=%{{z:.3f}}<extra></extra>',
        ))
        # Mark start and end
        fig.add_trace(go.Scatter3d(
            x=[xyz[0, 0]], y=[xyz[0, 1]], z=[xyz[0, 2]],
            mode='markers',
            marker=dict(size=6, color=colors[i], symbol='circle'),
            name=f'{ep_name} start',
            showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=[xyz[-1, 0]], y=[xyz[-1, 1]], z=[xyz[-1, 2]],
            mode='markers',
            marker=dict(size=6, color=colors[i], symbol='x'),
            name=f'{ep_name} end',
            showlegend=False,
        ))
        n_loaded += 1

    task_name = os.path.basename(os.path.abspath(args.task_dir))
    frame_label = pose_src.replace('.npz', '') if pose_src else 'unknown'
    fig.update_layout(
        title=dict(text=f'{task_name} — Tool Trajectories ({n_loaded} episodes, {frame_label})',
                   font=dict(size=16)),
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
        ),
        legend=dict(itemsizing='constant', font=dict(size=10)),
        margin=dict(l=0, r=0, t=40, b=0),
        height=750,
    )

    fig.write_html(args.output)
    print(f"Saved → {args.output}  ({n_loaded} trajectories, frame: {frame_label})")


if __name__ == '__main__':
    main()
