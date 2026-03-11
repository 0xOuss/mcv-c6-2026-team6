"""
RAFT Optical Flow Wrapper.

Teed & Deng, "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow" (ECCV 2020)
https://github.com/princeton-vl/RAFT

Architecture intuition:
  1. Feature Encoder:   CNN extracts features from both frames → feature maps
  2. Context Network:   CNN on frame 1 only → captures scene context (helps with occlusions)
  3. 4D Cost Volume:    Computes correlation between EVERY pair of pixels across both frames
                        → allows comparing any pixel in frame 1 with any pixel in frame 2
                        → unlike FlowNet which only computes local correlations
  4. GRU Update:        Recurrently refines flow estimate (12 steps by default)
                        → each iteration sees the cost volume and current flow estimate
                        → "warms up" the estimate gradually, like gradient descent

Why RAFT is better than Farneback:
  - Learned features: understands semantics, not just raw pixel values
  - 4D cost volume: global matching → no stuck-at-local-minimum problem
  - Iterative refinement: each step adds detail, handles large displacements gracefully

Setup:
  git clone https://github.com/princeton-vl/RAFT
  cd RAFT && pip install -e .
  # Download weights:
  bash download_models.sh
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional


def load_raft_model(model_path: str, small: bool = False,
                    mixed_precision: bool = False, device: str = "auto"):
    """
    Load RAFT model from checkpoint.

    Args:
        model_path:      Path to .pth checkpoint (raft-sintel.pth, raft-kitti.pth, etc.)
        small:           Use RAFT-Small (faster, less accurate)
        mixed_precision: Use FP16 (faster on modern GPUs)
        device:          "cuda", "cpu", or "auto"
    Returns:
        model, device
    """
    import sys
    # Add RAFT to path — adjust if repo is at different location
    raft_dir = Path(__file__).parent.parent.parent / "external" / "RAFT"
    if str(raft_dir / "core") not in sys.path:
        sys.path.insert(0, str(raft_dir / "core"))

    from raft import RAFT
    from argparse import Namespace

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    args = Namespace(
        small=small,
        mixed_precision=mixed_precision,
        alternate_corr=False
    )
    model = RAFT(args)
    checkpoint = torch.load(model_path, map_location=device)
    # Handle DataParallel checkpoints
    if any(k.startswith("module.") for k in checkpoint.keys()):
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model, device


def run_raft(model, device: str, img1: np.ndarray, img2: np.ndarray,
             iters: int = 20) -> np.ndarray:
    """
    Run RAFT on an image pair.

    Args:
        model, device: from load_raft_model
        img1, img2:    (H, W, 3) RGB float32 [0,1] or uint8 [0,255]
        iters:         Number of GRU refinement iterations
                       More iters = better quality but slower
                       5 iters ≈ fast, 20 iters = standard, 32 = max quality

    Returns:
        flow: (H, W, 2) float32 [u, v] in pixels

    Note on image preprocessing:
        RAFT expects uint8 tensors (0-255) of shape (1, 3, H, W)
        We pad to multiples of 8 for the feature pyramid
    """
    import torch.nn.functional as F

    # Convert to uint8 [0, 255] if needed
    if img1.dtype != np.uint8:
        img1 = (img1 * 255).clip(0, 255).astype(np.uint8)
        img2 = (img2 * 255).clip(0, 255).astype(np.uint8)

    H, W = img1.shape[:2]

    # Pad to multiple of 8 (required by RAFT's feature pyramid)
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8

    def to_tensor(img):
        t = torch.from_numpy(img).permute(2, 0, 1).float()
        t = F.pad(t, [0, pad_w, 0, pad_h])
        return t.unsqueeze(0).to(device)  # (1, 3, H', W')

    t1 = to_tensor(img1)
    t2 = to_tensor(img2)

    with torch.no_grad():
        _, flow_up = model(t1, t2, iters=iters, test_mode=True)

    flow = flow_up[0].permute(1, 2, 0).cpu().numpy()  # (H', W', 2)
    flow = flow[:H, :W]  # Remove padding
    return flow


def run_raft_with_backward(model, device: str,
                            img1: np.ndarray, img2: np.ndarray,
                            iters: int = 20):
    """
    Run RAFT in both directions for forward-backward consistency.

    Returns:
        flow_fwd: (H, W, 2) flow from img1 → img2
        flow_bwd: (H, W, 2) flow from img2 → img1
    """
    flow_fwd = run_raft(model, device, img1, img2, iters)
    flow_bwd = run_raft(model, device, img2, img1, iters)
    return flow_fwd, flow_bwd
