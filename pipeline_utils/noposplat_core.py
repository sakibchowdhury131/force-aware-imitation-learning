"""
noposplat_core.py
=================
Core NoPoSplat novel-view functions, vendored into this pipeline so the scripts
here (live_novel_views.py, 02_augment_noposplat.py) depend only on the directory
itself -- no import of an external render_novel_views.py from another checkout.

These are the same functions as the standalone `render_novel_views.py`:
letterbox/aspect input geometry + intrinsics handling, model loading, per-pixel
Gaussian extraction and cleanup, pose-free camera recovery (PnP + Slerp), alpha
rendering via gsplat, and the alpha-aware crop that removes black disocclusion
borders. The CLI/main lives in the scripts that import this module.

Still requires a clone of the official NoPoSplat repo as the model backend (no pip
package exists for the model); point `load_noposplat_encoder(..., noposplat_root)`
at that clone. gsplat (CUDA) is needed for rendering.
"""

import os
import sys

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp


# NoPoSplat checkpoints live on HF (botaoye/NoPoSplat). The 512x512 mixed model
# generalises best to in-the-wild / lab scenes; the 256 one is faster.
HF_REPO = "botaoye/NoPoSplat"
CKPT_512 = "mixRe10kDl3dv_512x512.ckpt"
CKPT_256 = "mixRe10kDl3dv.ckpt"

SH_C0 = 0.28209479177387814   # degree-0 SH constant
PAD_VALUE = 0.5               # neutral grey fill for letterbox pad (normalises to 0)


# --------------------------------------------------------------------------- #
# Intrinsics handling                                                          #
# --------------------------------------------------------------------------- #
def K_from_fov(W, H, fov_deg):
    f = 0.5 * W / np.tan(0.5 * np.deg2rad(fov_deg))
    return np.array([[f, 0, W / 2.0],
                     [0, f, H / 2.0],
                     [0, 0, 1.0]], dtype=np.float64)


def _orig_size(meta, fallback):
    """Original (width, height) of the captured frames from meta.json."""
    if isinstance(meta.get('resolution'), (list, tuple)) and len(meta['resolution']) == 2:
        return int(meta['resolution'][0]), int(meta['resolution'][1])
    ow = meta.get('width') or meta.get('orig_width')
    oh = meta.get('height') or meta.get('orig_height')
    if ow and oh:
        return int(ow), int(oh)
    return fallback


def load_orig_intrinsics(meta, cam0, cam1):
    """Return (K0, K1) as 3x3 matrices in the ORIGINAL capture pixels, or
    (None, None) if meta.json carries no usable intrinsics. Handles a list form
    (meta["intrinsics"] = [{"K": ...}, {"K": ...}]), a dict form keyed by cam
    name, and flat fx/fy/cx/cy."""
    intr = meta.get('intrinsics')

    def _as_K(e):
        K = e['K'] if isinstance(e, dict) and 'K' in e else e
        return np.asarray(K, dtype=np.float64).reshape(3, 3).copy()

    if isinstance(intr, list) and max(cam0, cam1) < len(intr):
        return _as_K(intr[cam0]), _as_K(intr[cam1])
    if isinstance(intr, dict) and f'cam{cam0}' in intr and f'cam{cam1}' in intr:
        return _as_K(intr[f'cam{cam0}']), _as_K(intr[f'cam{cam1}'])
    if all(k in meta for k in ('fx', 'fy', 'cx', 'cy')):
        K = np.array([[meta['fx'], 0, meta['cx']],
                      [0, meta['fy'], meta['cy']],
                      [0, 0, 1]], dtype=np.float64)
        return K, K.copy()
    return None, None


def normalize_K(K, W, H):
    """NoPoSplat convention: row0 / W, row1 / H -> resolution-independent 3x3."""
    Kn = K.astype(np.float64).copy()
    Kn[0, :] /= float(W)
    Kn[1, :] /= float(H)
    return Kn


def full_intrinsics(meta, cam0, cam1, ow, oh, fov_deg):
    """Return (K0_full, K1_full, have_intr) in NATIVE capture pixels, falling back
    to a horizontal-FoV guess over the full native frame."""
    K0, K1 = load_orig_intrinsics(meta, cam0, cam1)
    if K0 is not None:
        return K0.astype(np.float64), K1.astype(np.float64), True
    K = K_from_fov(ow, oh, fov_deg)
    return K, K.copy(), False


# --------------------------------------------------------------------------- #
# Input geometry (letterbox pad-to-square, or aspect-preserving resize)        #
# A unified geometry dict serves both modes. Keys:                             #
#   Win, Hin : model-input tensor size (both multiples of 16)                  #
#   cw, ch   : resized-content size placed inside the input                    #
#   px, py   : content offset inside the input (0,0 in aspect; centred in lb)  #
#   sx, sy   : native->input per-axis scale (equal in letterbox)               #
# --------------------------------------------------------------------------- #
def round16(x, lo=16):
    """Round to the nearest positive multiple of 16 (the ViT patch size), floored
    at `lo`. NoPoSplat's PatchEmbedDust3R requires H, W to be multiples of 16."""
    return max(int(lo), int(round(float(x) / 16.0)) * 16)


def aspect_geometry(ow, oh, long_side, min_short=256):
    """Aspect-preserving resize (no pad): the long side maps to `long_side` and the
    short side to the proportional length, both snapped to multiples of 16."""
    long_side = round16(long_side)
    if ow >= oh:                                     # landscape
        Win = long_side
        Hin = round16(oh * long_side / ow, lo=min(min_short, long_side))
    else:                                            # portrait
        Hin = long_side
        Win = round16(ow * long_side / oh, lo=min(min_short, long_side))
    return {"Win": Win, "Hin": Hin, "cw": Win, "ch": Hin, "px": 0, "py": 0,
            "sx": Win / float(ow), "sy": Hin / float(oh)}


def letterbox_geometry(ow, oh, res):
    """Letterbox an (ow x oh) frame into a square res x res input: resize so the
    LONG side maps to res, then pad the short side symmetrically with neutral grey."""
    L = float(max(ow, oh))
    s = res / L                                      # uniform scale (no stretch)
    cw = min(res, int(round(ow * s)))
    ch = min(res, int(round(oh * s)))
    px = (res - cw) // 2
    py = (res - ch) // 2
    return {"Win": res, "Hin": res, "cw": cw, "ch": ch, "px": px, "py": py,
            "sx": s, "sy": s}


def input_K(K, geo):
    """Map a native-pixel intrinsic K into the model-input grid: per-axis resize by
    (sx, sy) then a pad shift of the principal point (OpenCV pixel-centre
    convention -- the exact inverse of input_active_mask)."""
    sx, sy, px, py = geo["sx"], geo["sy"], geo["px"], geo["py"]
    Kp = K.astype(np.float64).copy()
    Kp[0, 0] *= sx
    Kp[1, 1] *= sy
    Kp[0, 1] *= sx                                    # skew (usually 0)
    Kp[0, 2] = (K[0, 2] + 0.5) * sx - 0.5 + px
    Kp[1, 2] = (K[1, 2] + 0.5) * sy - 0.5 + py
    return Kp


# --------------------------------------------------------------------------- #
# Image loading (resize into the input grid -> mean/std 0.5 normalise)         #
# --------------------------------------------------------------------------- #
def read_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_input_tensor(rgb_full, geo, device):
    """Resize an RGB frame to the content size (cw, ch), place it in the (Hin, Win)
    input grid, and normalise (mean=std=0.5) for NoPoSplat. Aspect mode fills the
    whole grid; letterbox mode pads the surround with neutral grey. Returns
    (3, Hin, Win)."""
    Win, Hin = geo["Win"], geo["Hin"]
    cw, ch, px, py = geo["cw"], geo["ch"], geo["px"], geo["py"]
    content = cv2.resize(rgb_full, (cw, ch), interpolation=cv2.INTER_AREA)
    if cw == Win and ch == Hin:                              # aspect mode: no pad
        canvas = content.astype(np.float32)
    else:
        canvas = np.full((Hin, Win, 3), PAD_VALUE * 255.0, dtype=np.float32)
        canvas[py:py + ch, px:px + cw] = content.astype(np.float32)
    t = torch.from_numpy(canvas).to(device) / 255.0          # (Hin,Win,3) [0,1]
    t = t.permute(2, 0, 1)                                    # (3,Hin,Win)
    t = (t - 0.5) / 0.5
    return t


def input_active_mask(geo):
    """Boolean (Hin*Win,) mask (row-major) of the content pixels, plus (Hin*Win, 2)
    NATIVE pixel coords for every input pixel. Pad pixels (letterbox mode) are
    False; the native mapping is the exact inverse of input_K's pixel convention."""
    Win, Hin = geo["Win"], geo["Hin"]
    sx, sy, cw, ch, px, py = (geo["sx"], geo["sy"], geo["cw"], geo["ch"],
                              geo["px"], geo["py"])
    gx, gy = np.meshgrid(np.arange(Win), np.arange(Hin))     # (Hin,Win), row-major
    nx = (gx - px + 0.5) / sx - 0.5                          # input -> native x
    ny = (gy - py + 0.5) / sy - 0.5                          # input -> native y
    mask = (gx >= px) & (gx < px + cw) & (gy >= py) & (gy < py + ch)
    pix = np.stack([nx.ravel(), ny.ravel()], axis=-1)
    return mask.ravel(), pix


def erode_active_mask(active_mask_np, Hin, Win, edge_trim):
    """Shrink the content mask inward by `edge_trim` input-grid pixels, dropping the
    border ring whose depth is unconstrained and which smears into the novel-view
    corners."""
    if edge_trim <= 0:
        return active_mask_np
    m = active_mask_np.reshape(Hin, Win).astype(np.uint8)
    k = 2 * int(edge_trim) + 1
    kernel = np.ones((k, k), np.uint8)
    m = cv2.erode(m, kernel, iterations=1)
    return m.astype(bool).ravel()


# --------------------------------------------------------------------------- #
# Camera-pose recovery (NoPoSplat is pose-free; cam1 recovered via PnP)        #
# --------------------------------------------------------------------------- #
def solve_pnp_w2c(obj, img, K, max_pts=8000):
    """obj (M,3) 3D points in cam0 frame; img (M,2) their pixel coords in the target
    camera (NATIVE pixels). Solve PnP (RANSAC + LM refine) -> world(cam0)->cam w2c."""
    obj = np.asarray(obj, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(img, dtype=np.float64).reshape(-1, 2)
    valid = np.isfinite(obj).all(axis=1) & np.isfinite(img).all(axis=1)
    obj, img = obj[valid], img[valid]
    if len(obj) < 6:
        raise RuntimeError("Not enough valid points for PnP")
    if len(obj) > max_pts:
        sel = np.random.choice(len(obj), max_pts, replace=False)
        obj, img = obj[sel], img[sel]

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj, img, K.astype(np.float64), None,
        flags=cv2.SOLVEPNP_EPNP, reprojectionError=4.0,
        iterationsCount=200, confidence=0.999)
    if not ok:
        raise RuntimeError("PnP failed to recover camera pose")
    if inliers is not None and len(inliers) >= 6:
        idx = inliers.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj[idx], img[idx],
                                          K.astype(np.float64), None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = tvec.ravel()
    return w2c


def interpolate_w2c(w2c1, w2c2, alpha):
    R1, t1 = w2c1[:3, :3], w2c1[:3, 3]
    R2, t2 = w2c2[:3, :3], w2c2[:3, 3]
    key = Rotation.from_matrix(np.stack([R1, R2]))
    R = Slerp([0, 1], key)([alpha])[0].as_matrix()
    t = (1.0 - alpha) * t1 + alpha * t2
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def generate_novel_view_w2c(w2c0, w2c1, n):
    """Beta(5,2) biased to cam0, Beta(2,5) biased to cam1."""
    n0 = (n + 1) // 2
    n1 = n // 2
    alphas = (list(np.random.beta(5, 2, n0)) + list(np.random.beta(2, 5, n1)))
    return np.stack([interpolate_w2c(w2c0, w2c1, float(a)) for a in alphas])


# --------------------------------------------------------------------------- #
# Pull render-ready Gaussians out of a NoPoSplat prediction                    #
# --------------------------------------------------------------------------- #
def decompose_covariance(cov):
    """Decompose (N,3,3) covariances into scales (N,3) and wxyz rotations (N,4).
    cov = R diag(s^2) R^T, so eigendecompose -> scales = sqrt(eigvals), rotation =
    eigvectors (forced proper via determinant sign)."""
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)          # (N,3), (N,3,3)
    scales = torch.sqrt(eigenvalues.clamp(min=1e-12))           # (N,3)
    det = torch.linalg.det(eigenvectors)                        # (N,)
    eigenvectors[:, :, 0] *= det.sign().unsqueeze(-1)           # fix improper rotations
    rot_wxyz = matrix_to_quaternion_wxyz(eigenvectors)          # (N,4)
    return scales, rot_wxyz


def matrix_to_quaternion_wxyz(R):
    """Convert (N,3,3) rotation matrices to (N,4) wxyz quaternions (Shepperd)."""
    N = R.shape[0]
    q = torch.empty((N, 4), device=R.device, dtype=R.dtype)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    s = torch.sqrt((trace + 1.0).clamp(min=1e-12)) * 2  # s = 4*w
    mask = trace > 0
    if mask.any():
        q[mask, 0] = 0.25 * s[mask]
        q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / s[mask]
        q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / s[mask]
        q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / s[mask]

    mask2 = (~mask) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if mask2.any():
        s2 = torch.sqrt((1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]).clamp(min=1e-12)) * 2
        q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s2
        q[mask2, 1] = 0.25 * s2
        q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2
        q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s2

    mask3 = (~mask) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    if mask3.any():
        s3 = torch.sqrt((1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]).clamp(min=1e-12)) * 2
        q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s3
        q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s3
        q[mask3, 2] = 0.25 * s3
        q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3

    mask4 = (~mask) & (~mask2) & (~mask3)
    if mask4.any():
        s4 = torch.sqrt((1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]).clamp(min=1e-12)) * 2
        q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s4
        q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s4
        q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s4
        q[mask4, 3] = 0.25 * s4

    q = q / (q.norm(dim=-1, keepdim=True) + 1e-12)
    return q


def extract_per_view(gauss, v, N_per_view):
    """Slice view `v` out of NoPoSplat's two-view-concatenated Gaussians and convert
    to render-ready params. The encoder returns composed covariances; we decompose
    them back into scales + rotations for gsplat, and read RGB from the SH DC band."""
    s = v * N_per_view
    e = s + N_per_view
    means = gauss.means[0, s:e]                         # (N,3)
    covs = gauss.covariances[0, s:e]                    # (N,3,3)
    opac = gauss.opacities[0, s:e].reshape(-1, 1)       # (N,1)
    sh = gauss.harmonics[0, s:e]                        # (N,3,d_sh)
    scales, rot_wxyz = decompose_covariance(covs)
    dc = sh[..., 0]                                      # (N,3)
    colors = (SH_C0 * dc + 0.5).clamp(0.0, 1.0)
    return {'means': means, 'scales': scales, 'rotations': rot_wxyz,
            'opacities': opac, 'colors': colors}


def gaussian_keep_mask(g, max_depth_pct=99.0, max_scale_pct=99.5,
                       max_anisotropy=30.0, min_depth=1e-3):
    """Boolean keep-mask dropping the splats that smear novel-view edges: non-finite
    params, points at/behind cam0, the farthest depth tail, the largest-scale tail
    (giant blobs), and needle-like splats (the comet streaks). Percentile
    thresholds are computed on the currently-valid subset."""
    means, scales = g['means'], g['scales']
    z = means[:, 2]
    keep = torch.isfinite(means).all(dim=1) & torch.isfinite(scales).all(dim=1)
    keep = keep & (z > min_depth)

    if max_depth_pct < 100.0 and keep.any():
        thr = torch.quantile(z[keep], max_depth_pct / 100.0)
        keep = keep & (z <= thr)

    smax = scales.max(dim=1).values
    if max_scale_pct < 100.0 and keep.any():
        thr = torch.quantile(smax[keep], max_scale_pct / 100.0)
        keep = keep & (smax <= thr)

    if max_anisotropy is not None and max_anisotropy > 0:
        smin = scales.min(dim=1).values.clamp(min=1e-12)
        keep = keep & ((smax / smin) <= max_anisotropy)

    return keep


# --------------------------------------------------------------------------- #
# Rendering with the alpha (coverage) channel                                  #
# --------------------------------------------------------------------------- #
def render_view_alpha(g, w2c, K, H, W, antialias=True):
    """Render an RGB novel view AND its per-pixel alpha (coverage). Holes in the
    novel view have alpha ~= 0 (the black background shows through)."""
    import gsplat
    colors, alphas, _ = gsplat.rasterization(
        means=g['means'],
        quats=g['rotations'],                         # wxyz; gsplat normalises internally
        scales=g['scales'],
        opacities=g['opacities'].squeeze(-1),
        colors=g['colors'],
        viewmats=w2c[None],                           # (1,4,4) world(cam0)->cam
        Ks=K[None],                                   # (1,3,3)
        width=W, height=H,
        render_mode='RGB',
        rasterize_mode='antialiased' if antialias else 'classic',
    )
    rgb = colors[0].clamp(0.0, 1.0)        # (H,W,3)
    alpha = alphas[0, ..., 0]              # (H,W)
    return rgb, alpha


# --------------------------------------------------------------------------- #
# Crop computation (border-connected holes -> hole-free rectangle)             #
# --------------------------------------------------------------------------- #
def compute_crop(alpha_np, thresh, pad, max_frac):
    """Return (left, top, right, bottom) of the largest axis-aligned rectangle that
    contains no BORDER-CONNECTED hole. Each hole pixel pushes in the margin of its
    nearest image border; interior specks and high-alpha dark objects are ignored.
    The per-side margin is capped at `max_frac` of the dimension."""
    H, W = alpha_np.shape
    left, top, right, bottom = 0, 0, W, H

    holes = (alpha_np < thresh).astype(np.uint8)
    if holes.any():
        # keep only holes touching an image border (the disocclusion regions)
        n_lab, labels = cv2.connectedComponents(holes, connectivity=4)
        border_lab = (set(labels[0, :]) | set(labels[-1, :]) |
                      set(labels[:, 0]) | set(labels[:, -1]))
        border_lab.discard(0)
        if border_lab:
            bh = np.isin(labels, list(border_lab))
            rr, cc = np.nonzero(bh)
            # nearest border for each hole pixel: 0=left,1=top,2=right,3=bottom
            dist = np.stack([cc, rr, (W - 1 - cc), (H - 1 - rr)], axis=1)
            near = dist.argmin(axis=1)
            if np.any(near == 0): left = int(cc[near == 0].max()) + 1
            if np.any(near == 1): top = int(rr[near == 1].max()) + 1
            if np.any(near == 2): right = int(cc[near == 2].min())
            if np.any(near == 3): bottom = int(rr[near == 3].min())

    # inward safety pad
    left += pad; top += pad; right -= pad; bottom -= pad

    # cap the zoom: never crop more than max_frac of a side
    maxL, maxT = int(max_frac * W), int(max_frac * H)
    left = min(left, maxL); top = min(top, maxT)
    right = max(right, W - maxL); bottom = max(bottom, H - maxT)

    # degenerate guard: if a span collapsed, drop the crop on that axis
    if right - left < W // 4:
        left, right = 0, W
    if bottom - top < H // 4:
        top, bottom = 0, H
    return left, top, right, bottom


def crop_rescale_K(K, left, top, cw, ch, outW, outH):
    """Intrinsics after cropping to (cw x ch) at offset (left, top) then resizing to
    (outW x outH). Principal point shifts by the crop offset; focal + principal
    point scale by the resize (OpenCV pixel-centre convention)."""
    Kc = K.astype(np.float64).copy()
    Kc[0, 2] -= left
    Kc[1, 2] -= top
    sx, sy = outW / float(cw), outH / float(ch)
    Kc[0, 0] *= sx
    Kc[0, 1] *= sx
    Kc[0, 2] = (Kc[0, 2] + 0.5) * sx - 0.5
    Kc[1, 1] *= sy
    Kc[1, 2] = (Kc[1, 2] + 0.5) * sy - 0.5
    return Kc


# --------------------------------------------------------------------------- #
# Model loading                                                                #
# --------------------------------------------------------------------------- #
def load_noposplat_encoder(noposplat_root, ckpt_path, res, device):
    """Instantiate the NoPoSplat encoder via the repo's own hydra config, then load
    the encoder weights out of the lightning checkpoint (keys prefixed 'encoder.')."""
    sys.path.insert(0, noposplat_root)
    from hydra import initialize_config_dir, compose

    cfg_dir = os.path.join(noposplat_root, 'config')
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg_dict = compose(config_name='main',
                           overrides=['+experiment=re10k', 'mode=test',
                                      'model.encoder.pretrained_weights='])

    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg
    from src.model.encoder import get_encoder

    set_cfg(cfg_dict)
    cfg = load_typed_root_config(cfg_dict)
    encoder, _ = get_encoder(cfg.model.encoder)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    enc_state = {k[len('encoder.'):]: v for k, v in state.items()
                 if k.startswith('encoder.')}
    if not enc_state:                       # some ckpts store encoder keys directly
        enc_state = state
    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    if missing:
        print(f"  [load] {len(missing)} missing keys (first few): {missing[:5]}")
    if unexpected:
        print(f"  [load] {len(unexpected)} unexpected keys (first few): {unexpected[:5]}")

    encoder = encoder.to(device).eval()
    return encoder
