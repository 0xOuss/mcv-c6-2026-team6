"""
PyFlow (Coarse-to-Fine Optical Flow) Wrapper.

Implements the Coarse2Fine approach from:
Brox, Bruhn, Papenberg, Weickert. "High accuracy optical flow estimation
based on a theory for warping." ECCV 2004.

Algorithm intuition:
  1. Build image pyramid (multiple scales: half size each level)
  2. At coarsest scale: estimate flow (small displacements are easy to handle here
     because the image is tiny and global structure is preserved)
  3. Warp image 2 toward image 1 using estimated flow at current scale
  4. Estimate residual flow on warped image
  5. Upsample flow estimate to next finer scale, repeat

Why this is better than naive approaches:
  - "Coarse-to-fine" handles LARGE displacements that would violate the
    linearization assumption of optical flow (∂I/∂t ≈ ∇I · flow)
  - Without the pyramid: only works for displacements < ~5px
  - With 5 pyramid levels at 0.75 ratio: handles displacements up to ~50px

Install: pip install pyflow
         (May need: apt-get install build-essential; requires Cython)
"""

import numpy as np
from typing import Optional
import time


def run_pyflow(img1: np.ndarray, img2: np.ndarray,
               alpha: float = 0.012,
               ratio: float = 0.75,
               minWidth: int = 20,
               nOuterFPIterations: int = 7,
               nInnerFPIterations: int = 1,
               nSORIterations: int = 30,
               colType: int = 0) -> np.ndarray:
    """
    Run PyFlow (Coarse2Fine) optical flow.

    Args:
        img1, img2:            (H, W, 3) or (H, W) float64 [0,1]
        alpha:                 Regularization weight (smoothness prior strength)
                               Larger α → smoother flow (less detail, fewer artifacts)
                               Smaller α → sharper boundaries (more noise)
                               Default: 0.012 (from original Brox paper)
        ratio:                 Pyramid downsampling ratio (0.5 or 0.75)
                               0.75 → more levels → better for large displacements
        minWidth:              Coarsest pyramid level width in pixels
        nOuterFPIterations:    Outer fixed-point iterations (for nonlinear terms)
        nInnerFPIterations:    Inner fixed-point iterations
        nSORIterations:        SOR (Successive Over-Relaxation) iterations for linear solve
        colType:               0 = grayscale, 1 = color

    Returns:
        flow: (H, W, 2) float32 [u, v] in pixels

    Note: PyFlow expects float64 images in [0, 1] range.
    """
    try:
        import pyflow
    except ImportError:
        raise ImportError(
            "PyFlow not installed. Install with: pip install pyflow\n"
            "May need: apt-get install build-essential python3-dev"
        )

    # PyFlow requires float64
    if img1.dtype != np.float64:
        img1 = img1.astype(np.float64)
    if img2.dtype != np.float64:
        img2 = img2.astype(np.float64)

    # Ensure [0, 1] range
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)

    u, v, _ = pyflow.coarse2fine_flow(
        img1, img2,
        alpha, ratio, minWidth,
        nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType
    )
    flow = np.stack([u, v], axis=-1).astype(np.float32)
    return flow
