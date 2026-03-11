"""
Farneback Optical Flow (OpenCV implementation).

Gunnar Farneback, "Two-Frame Motion Estimation Based on Polynomial Expansion" (SCIA 2003)

Algorithm intuition:
  - Approximates each neighborhood as a polynomial (quadratic)
  - Flow estimated by comparing polynomial coefficients between frames
  - Multi-scale pyramid to handle large displacements

Why it's in OpenCV and fast:
  - Pure C++ implementation with SSE/AVX optimizations
  - No neural network overhead
  - Well-suited for real-time applications (0.1s/frame)

Limitations:
  - Assumes brightness constancy (fails under lighting changes)
  - Aperture problem: can't determine flow perpendicular to edges on textureless surfaces
  - Dense flow means every pixel gets a flow vector (including background)
"""

import cv2
import numpy as np
import time
from typing import Optional


def run_farneback(img1: np.ndarray, img2: np.ndarray,
                  pyr_scale: float = 0.5,
                  levels: int = 3,
                  winsize: int = 15,
                  iterations: int = 3,
                  poly_n: int = 5,
                  poly_sigma: float = 1.2,
                  flags: int = 0) -> np.ndarray:
    """
    Run Farneback optical flow on an image pair.

    Args:
        img1, img2:  Input images — (H, W) grayscale or (H, W, 3) RGB float32 [0,1]
        pyr_scale:   Scale between pyramid levels (0.5 = each level is half size)
                     → smaller = more levels needed for same displacement range
        levels:      Number of pyramid levels
                     → more levels = handles larger displacements
        winsize:     Neighborhood window size for polynomial fit
                     → larger = smoother but less precise at boundaries
        iterations:  Iterations per pyramid level (more = more accurate, slower)
        poly_n:      Size of pixel neighborhood for polynomial expansion
                     → 5 or 7 recommended; larger = more robust to noise
        poly_sigma:  Gaussian std for polynomial weights
                     → larger = smoother polynomial fit
        flags:       cv2.OPTFLOW_USE_INITIAL_FLOW to use previous flow as init

    Returns:
        flow: (H, W, 2) float32 [u, v] in pixel units
    """
    # Convert to grayscale if needed
    if img1.ndim == 3:
        if img1.dtype != np.uint8:
            img1_gray = (img1 * 255).astype(np.uint8)
            img2_gray = (img2 * 255).astype(np.uint8)
        else:
            img1_gray = img1
            img2_gray = img2
        img1_gray = cv2.cvtColor(img1_gray, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2_gray, cv2.COLOR_RGB2GRAY)
    else:
        img1_gray = (img1 * 255).astype(np.uint8) if img1.dtype != np.uint8 else img1
        img2_gray = (img2 * 255).astype(np.uint8) if img2.dtype != np.uint8 else img2

    flow = cv2.calcOpticalFlowFarneback(
        img1_gray, img2_gray,
        flow=None,
        pyr_scale=pyr_scale,
        levels=levels,
        winsize=winsize,
        iterations=iterations,
        poly_n=poly_n,
        poly_sigma=poly_sigma,
        flags=flags
    )
    return flow  # (H, W, 2)


def farneback_ablation_winsize(img1: np.ndarray, img2: np.ndarray,
                               winsizes: list = [5, 10, 15, 21]) -> dict:
    """
    Ablation: effect of window size on flow quality.
    Larger window → smoother but less precise at object boundaries.
    Returns dict: {winsize: flow}

    Intuition: winsize is the most important HP for Farneback.
    Large window averages over many pixels → motion boundary blur
    Small window is noisy on textureless regions
    """
    results = {}
    for ws in winsizes:
        t0 = time.perf_counter()
        flow = run_farneback(img1, img2, winsize=ws)
        dt = time.perf_counter() - t0
        results[ws] = {"flow": flow, "runtime": dt}
    return results
