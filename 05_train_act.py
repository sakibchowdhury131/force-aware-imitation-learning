#!/usr/bin/env python3
"""
Step 5 variant — train an ACT (Action Chunking Transformer) policy instead of
the DDPM diffusion policy. Does NOT modify 05_train.py; reuses its dataset
class (EpisodeWindowDataset, masked human-demo images) and CLI parser
completely unchanged. Only the model (act_common.ACTPolicy) and training
objective differ -- ACT predicts the whole action_horizon chunk in a single
transformer forward pass (DETR-style learned queries), trained as a
conditional VAE (L1 reconstruction + KL on a latent style variable), instead
of iterative DDPM denoising. See act_common.py for architecture details.

Usage:
    python 05_train_act.py --data_dir data/episodes/PastaTransfer_force \\
        --output_dir data/checkpoints/PastaTransfer_force_act_dualcam_h16_all37 \\
        --n_views 2 --track_cam 1 --action_horizon 16 --subsample 3 --action_frame task
"""
import os, sys, argparse, importlib.util
import torch
from torch.utils.data import DataLoader
from diffusers.optimization import get_cosine_schedule_with_warmup

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)

from policy_common import PROPRIO_DIM
from act_common import ACTPolicy, act_loss


def _load_train05_module():
    """'05_train' starts with a digit, not a valid module name for `import` --
    load it by file path instead. Zero modification to that file."""
    spec = importlib.util.spec_from_file_location(
        '_train05', os.path.join(PIPELINE_DIR, '05_train.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_epoch(model, dataloader, device, kl_weight, optimizer=None, lr_sched=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total, total_l1, total_kl = 0.0, 0.0, 0.0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for obs_imgs, proprio, actions in dataloader:
            obs_imgs = obs_imgs.to(device)
            proprio  = proprio.to(device)
            actions  = actions.to(device)

            pred_actions, mu, logvar = model(obs_imgs, proprio, actions)
            loss, l1, kl = act_loss(pred_actions, actions, mu, logvar, kl_weight)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_sched.step()

            total    += loss.item()
            total_l1 += l1.item()
            total_kl += kl.item()
    n = len(dataloader)
    return total / n, total_l1 / n, total_kl / n


def _save(args, epoch, model, optimizer, dataset, name=None):
    name = name or f'policy_epoch{epoch:04d}.pt'
    ckpt = {
        'epoch':            epoch,
        'model':            model.state_dict(),
        'train_method':     'act',
        'hidden_dim':       args.hidden_dim,
        'latent_dim':       args.latent_dim,
        'n_heads':          args.n_heads,
        'n_enc_layers':     args.n_enc_layers,
        'n_dec_layers':     args.n_dec_layers,
        'kl_weight':        args.kl_weight,
        'action_horizon':   args.action_horizon,
        'n_obs_steps':      args.n_obs_steps,
        'n_views':          args.n_views,
        'proprio_dim':      PROPRIO_DIM,
        'image_size':       args.image_size,
        'crop_size':        args.crop_size,
        'normalizer':       dataset.normalizer.state_dict(),
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

    model = ACTPolicy(
        action_dim=9, proprio_dim=PROPRIO_DIM, action_horizon=args.action_horizon,
        n_obs_steps=args.n_obs_steps, n_views=args.n_views, hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim, n_heads=args.n_heads, n_enc_layers=args.n_enc_layers,
        n_dec_layers=args.n_dec_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_sched = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500,
                                               num_training_steps=args.num_epochs * len(train_loader))

    print(f"ACT training: {args.num_epochs} epochs  |  batch {args.batch_size}  |  "
          f"{len(train_loader)} steps/epoch  |  hidden_dim={args.hidden_dim}  "
          f"latent_dim={args.latent_dim}  kl_weight={args.kl_weight}")

    train_ema = None
    ema_alpha = 0.05
    for epoch in range(args.num_epochs):
        avg, l1, kl = _run_epoch(model, train_loader, device, args.kl_weight,
                                 optimizer=optimizer, lr_sched=lr_sched)
        train_ema = avg if train_ema is None else ema_alpha * avg + (1 - ema_alpha) * train_ema
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1:4d}/{args.num_epochs}]  loss={avg:.4f} (ema={train_ema:.4f})  "
                  f"l1={l1:.4f}  kl={kl:.4f}")
        if (epoch + 1) % args.checkpoint_every == 0:
            _save(args, epoch + 1, model, optimizer, train_ds)

    _save(args, args.num_epochs, model, None, train_ds, name='policy_final.pt')
    print(f"\nTraining complete -> {args.output_dir}/policy_final.pt")


def main():
    # Pull the ACT-only flags out of argv before handing the rest to
    # 05_train.py's own parse_args() -- that parser doesn't know them, and
    # it isn't being modified to add them.
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--hidden_dim',   type=int, default=256)
    p.add_argument('--latent_dim',   type=int, default=32)
    p.add_argument('--n_heads',      type=int, default=8)
    p.add_argument('--n_enc_layers', type=int, default=4)
    p.add_argument('--n_dec_layers', type=int, default=7)
    p.add_argument('--kl_weight',    type=float, default=10.0)
    act_args, remaining = p.parse_known_args(sys.argv[1:])

    train05 = _load_train05_module()
    sys.argv = [sys.argv[0]] + remaining
    args = train05.parse_args()
    for k, v in vars(act_args).items():
        setattr(args, k, v)

    train(args, train05)


if __name__ == '__main__':
    main()
