"""Camera pose interpolation utilities (Slerp + Beta distribution sampling)."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp


def interpolate_w2c(w2c1: torch.Tensor, w2c2: torch.Tensor, alpha: float) -> torch.Tensor:
    """Slerp rotation + linear translation between two world-to-camera matrices."""
    R1, t1 = w2c1[:3, :3], w2c1[:3, 3]
    R2, t2 = w2c2[:3, :3], w2c2[:3, 3]
    q1 = Rotation.from_matrix(R1.cpu().numpy()).as_quat()
    q2 = Rotation.from_matrix(R2.cpu().numpy()).as_quat()
    slerp   = Slerp([0, 1], Rotation.from_quat([q1, q2]))
    R_interp = torch.tensor(slerp([alpha])[0].as_matrix(),
                            device=w2c1.device, dtype=torch.float32)
    t_interp = (1.0 - alpha) * t1 + alpha * t2
    mat = torch.eye(4, device=w2c1.device)
    mat[:3, :3] = R_interp
    mat[:3, 3]  = t_interp
    return mat


def generate_novel_view_w2c(w2c1: torch.Tensor, w2c2: torch.Tensor,
                             num_matrices: int = 6) -> torch.Tensor:
    """
    Generate novel-view w2c matrices by Slerp-interpolating between two cameras.
    Uses Beta(5,2) alphas biased toward cam1 and Beta(2,5) biased toward cam2,
    matching the Tool-as-Interface paper's generate_intermediate_matrices().

    Returns: (num_matrices, 4, 4)
    """
    num_cam1 = (num_matrices + 1) // 2
    num_cam2 = num_matrices // 2
    alphas = (torch.distributions.Beta(5, 2).sample([num_cam1]).tolist() +
              torch.distributions.Beta(2, 5).sample([num_cam2]).tolist())
    return torch.stack([interpolate_w2c(w2c1, w2c2, a) for a in alphas])
