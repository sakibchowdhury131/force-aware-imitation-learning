#!/usr/bin/env python3
"""
Step 5 variant — train the policy with a conditional flow-matching objective
instead of DDPM diffusion. Does NOT modify 05_train.py; reuses its dataset
class (EpisodeWindowDataset, masked human-demo images), CLI parser, and
DiffusionPolicyNet architecture completely unchanged. Only the training
objective and sampling procedure differ.

WHY FLOW MATCHING: instead of learning to denoise a fixed DDPM forward
process (add Gaussian noise over N discrete steps, predict the noise), flow
matching learns a velocity field along a straight-line path between a noise
sample x0 ~ N(0,I) and the ground-truth action sequence x1:

    x_t = (1 - t) * x0 + t * x1,   t ~ Uniform(0, 1)
    target velocity: u_t = x1 - x0
    loss: MSE(v_theta(x_t, t, obs), u_t)

Sampling integrates the learned ODE dx/dt = v_theta(x_t, t, obs) from x0 via
Euler steps from t=0 to t=1 (--ode_steps steps, default 50) -- no discrete
noise schedule needed, and typically far fewer function evaluations than
DDPM's 100-step denoising.

The same DiffusionPolicyNet (ResNet-18 encoder + ConditionalUNet1D) is reused
as the velocity-prediction network v_theta -- its timestep embedding
(_SinusoidalPosEmb) accepts any real-valued input, so continuous t in [0,1]
is scaled by --time_scale (default 999, matching the DDPM timestep range the
architecture was originally sized for) before being fed in. This is a fresh
model (not fine-tuned from a DDPM checkpoint), so the exact scale only
matters for consistency between training and inference, which this script
guarantees by saving --time_scale into the checkpoint.

Usage:
    python 05_train_flow.py --data_dir data/episodes/PastaTransfer_force \\
        --output_dir data/checkpoints/PastaTransfer_force_flow_dualcam_h16_all37 \\
        --n_views 2 --track_cam 1 --action_horizon 16 --subsample 3 --action_frame task
"""
import os, sys, argparse, importlib.util
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers.optimization import get_cosine_schedule_with_warmup

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)

from policy_common import DiffusionPolicyNet, PROPRIO_DIM


def _load_train05_module():
    """'05_train' starts with a digit, not a valid module name for `import` --
    load it by file path instead. Zero modification to that file."""
    spec = importlib.util.spec_from_file_location(
        '_train05', os.path.join(PIPELINE_DIR, '05_train.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_epoch(model, dataloader, device, time_scale, optimizer=None, lr_sched=None):
    """One flow-matching pass over dataloader. Trains (with optimizer step) if
    optimizer is given, otherwise runs in eval/no_grad mode."""
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = 0.0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for obs_imgs, proprio, actions in dataloader:
            obs_imgs = obs_imgs.to(device)
            proprio  = proprio.to(device)
            actions  = actions.to(device)
            B = obs_imgs.shape[0]

            x1 = actions.reshape(B, -1)             # ground-truth action chunk
            x0 = torch.randn_like(x1)                # noise endpoint

            t  = torch.rand(B, device=device)        # (B,) uniform in [0, 1)
            t_ = t.view(-1, 1)
            xt = (1 - t_) * x0 + t_ * x1
            target_v = x1 - x0

            obs_emb = model.encode_obs(obs_imgs, proprio)
            pred_v  = model(xt, t * time_scale, obs_emb)
            loss = F.mse_loss(pred_v, target_v)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_sched.step()

            total_loss += loss.item()
    return total_loss / len(dataloader)


def _save(args, epoch, model, optimizer, dataset, name=None):
    name = name or f'policy_epoch{epoch:04d}.pt'
    ckpt = {
        'epoch':            epoch,
        'model':            model.state_dict(),
        'train_method':     'flow_matching',
        'time_scale':       args.time_scale,
        'ode_steps':        args.ode_steps,
        'action_horizon':   args.action_horizon,
        'n_obs_steps':      args.n_obs_steps,
        'n_views':          args.n_views,
        'proprio_dim':      PROPRIO_DIM,
        'image_size':       args.image_size,
        'crop_size':        args.crop_size,
        'normalizer':       dataset.normalizer.state_dict(),
        'unet_dims':        list(args.unet_dims),
        'unet_kernel':      args.unet_kernel,
        'action_frame':     args.action_frame,
    }
    if optimizer is not None:
        ckpt['optimizer'] = optimizer.state_dict()
    path = os.path.join(args.output_dir, name)
    torch.save(ckpt, path)
    print(f"  Saved checkpoint: {path}")


def train(args, train05):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    train_ds = train05.EpisodeWindowDataset(
        data_dir=args.data_dir, n_obs_steps=args.n_obs_steps,
        action_horizon=args.action_horizon, image_size=args.image_size,
        crop_size=args.crop_size, training=True, cam0_only=args.cam0_only,
        track_cam=args.track_cam, n_views=args.n_views, subsample=args.subsample,
        action_frame=args.action_frame,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    model = DiffusionPolicyNet(
        action_dim=9, action_horizon=args.action_horizon, n_obs_steps=args.n_obs_steps,
        n_views=args.n_views, unet_dims=tuple(args.unet_dims), unet_kernel=args.unet_kernel,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.95, 0.999),
                                  eps=1e-8, weight_decay=1e-6)
    lr_sched = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500,
                                               num_training_steps=args.num_epochs * len(train_loader))

    print(f"Flow-matching training: {args.num_epochs} epochs  |  batch {args.batch_size}  |  "
          f"{len(train_loader)} steps/epoch  |  time_scale={args.time_scale}  ode_steps={args.ode_steps}")

    train_ema = None
    ema_alpha = 0.05
    for epoch in range(args.num_epochs):
        train_avg = _run_epoch(model, train_loader, device, args.time_scale,
                               optimizer=optimizer, lr_sched=lr_sched)
        train_ema = train_avg if train_ema is None else ema_alpha * train_avg + (1 - ema_alpha) * train_ema
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1:4d}/{args.num_epochs}]  train={train_avg:.4f} (ema={train_ema:.4f})")
        if (epoch + 1) % args.checkpoint_every == 0:
            _save(args, epoch + 1, model, optimizer, train_ds)

    _save(args, args.num_epochs, model, None, train_ds, name='policy_final.pt')
    print(f"\nTraining complete -> {args.output_dir}/policy_final.pt")


def main():
    # Pull the flow-matching-only flags out of argv before handing the rest
    # to 05_train.py's own parse_args() -- that parser doesn't know them, and
    # it isn't being modified to add them.
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--time_scale', type=float, default=999.0,
                   help='Scale applied to continuous t in [0,1] before feeding the '
                        'sinusoidal timestep embedding (matches the DDPM range the '
                        'architecture was originally sized for).')
    p.add_argument('--ode_steps', type=int, default=50,
                   help='Euler integration steps for sampling (t=0 -> t=1).')
    flow_args, remaining = p.parse_known_args(sys.argv[1:])

    train05 = _load_train05_module()
    sys.argv = [sys.argv[0]] + remaining
    args = train05.parse_args()
    args.time_scale = flow_args.time_scale
    args.ode_steps  = flow_args.ode_steps

    train(args, train05)


if __name__ == '__main__':
    main()
