"""
Optical Flow Evaluation Metrics

Implements:
  - MSEN: Mean Square Error in Non-occluded areas
  - PEPN: Percentage of Erroneous Pixels in Non-occluded areas
  - EPE:  End-Point Error (Euclidean distance)
  - Runtime measurement

Intuition:
  - MSEN is sensitive to large errors (squares them) → penalizes bad outliers
  - PEPN tells you how often you're "wrong enough to matter" (>3px threshold)
  - EPE is the standard metric in the Sintel/KITTI leaderboards
  - A method can have low PEPN but high MSEN if a few pixels have huge errors
"""

import time
import numpy as np
from typing import Optional, Tuple


def compute_epe(flow_pred: np.ndarray, flow_gt: np.ndarray,
                mask: Optional[np.ndarray] = None) -> float:
    """
    End-Point Error: mean Euclidean distance between predicted and GT flow vectors.
    EPE = mean( sqrt( (u_pred - u_gt)^2 + (v_pred - v_gt)^2 ) )

    Args:
        flow_pred: (H, W, 2) predicted flow [u, v]
        flow_gt:   (H, W, 2) ground truth flow [u, v]
        mask:      (H, W) bool — if provided, only evaluate masked pixels
    Returns:
        scalar EPE in pixels
    """
    diff = flow_pred - flow_gt
    epe_map = np.sqrt(diff[:, :, 0]**2 + diff[:, :, 1]**2)  # (H, W)

    if mask is not None:
        return float(epe_map[mask].mean())
    return float(epe_map.mean())


def compute_msen(flow_pred: np.ndarray, flow_gt: np.ndarray,
                 mask: Optional[np.ndarray] = None) -> float:
    """
    Mean Square Error in Non-occluded areas.
    MSEN = mean( (u_pred - u_gt)^2 + (v_pred - v_gt)^2 )

    Note: MSEN = EPE^2 on average (not exactly, since mean of squares != square of means)
    MSEN emphasizes large errors more than EPE.

    Args:
        flow_pred: (H, W, 2)
        flow_gt:   (H, W, 2)
        mask:      (H, W) bool — non-occluded mask
    """
    diff = flow_pred - flow_gt
    se_map = diff[:, :, 0]**2 + diff[:, :, 1]**2  # squared error per pixel

    if mask is not None:
        return float(se_map[mask].mean())
    return float(se_map.mean())


def compute_pepn(flow_pred: np.ndarray, flow_gt: np.ndarray,
                 mask: Optional[np.ndarray] = None,
                 threshold: float = 3.0) -> float:
    """
    Percentage of Erroneous Pixels in Non-occluded areas.
    A pixel is "erroneous" if EPE > threshold (default: 3.0 pixels).

    PEPN = 100 * (count of pixels where EPE > threshold) / (total valid pixels)

    Args:
        flow_pred:  (H, W, 2)
        flow_gt:    (H, W, 2)
        mask:       (H, W) bool — non-occluded mask
        threshold:  EPE threshold in pixels (KITTI uses 3.0)
    Returns:
        percentage in [0, 100]
    """
    diff = flow_pred - flow_gt
    epe_map = np.sqrt(diff[:, :, 0]**2 + diff[:, :, 1]**2)
    erroneous = epe_map > threshold

    if mask is not None:
        total = mask.sum()
        if total == 0:
            return 0.0
        return 100.0 * erroneous[mask].sum() / total

    return 100.0 * erroneous.mean()


def compute_all_metrics(flow_pred: np.ndarray, flow_gt: np.ndarray,
                        noc_mask: Optional[np.ndarray] = None,
                        pepn_threshold: float = 3.0) -> dict:
    """
    Compute all optical flow metrics at once.

    Returns dict with keys: msen, pepn, epe, epe_all (without mask)
    """
    results = {
        "msen":    compute_msen(flow_pred, flow_gt, noc_mask),
        "pepn":    compute_pepn(flow_pred, flow_gt, noc_mask, pepn_threshold),
        "epe_noc": compute_epe(flow_pred, flow_gt, noc_mask),
        "epe_all": compute_epe(flow_pred, flow_gt, None),
    }
    return results


class RuntimeTimer:
    """Context manager for measuring runtime."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start

    @property
    def seconds(self) -> float:
        return self.elapsed


def evaluate_method(name: str, flow_func, img1: np.ndarray, img2: np.ndarray,
                    flow_gt: np.ndarray, noc_mask: Optional[np.ndarray] = None,
                    n_runs: int = 3) -> dict:
    """
    Evaluate a single optical flow method.

    Args:
        name:      Method name for logging
        flow_func: Callable(img1, img2) -> flow (H, W, 2)
        img1, img2: Input images (H, W, 3) float32
        flow_gt:   Ground truth flow (H, W, 2)
        noc_mask:  Non-occluded validity mask (H, W) bool
        n_runs:    Number of timing runs (take median)
    Returns:
        dict with all metrics + runtime
    """
    # Warmup
    flow_pred = flow_func(img1, img2)

    # Timing: run n_runs times, take median
    times = []
    for _ in range(n_runs):
        with RuntimeTimer() as t:
            flow_pred = flow_func(img1, img2)
        times.append(t.seconds)
    runtime = float(np.median(times))

    metrics = compute_all_metrics(flow_pred, flow_gt, noc_mask)
    metrics["runtime"] = runtime
    metrics["method"] = name

    print(f"[{name}]  MSEN={metrics['msen']:.4f}  PEPN={metrics['pepn']:.2f}%"
          f"  EPE(noc)={metrics['epe_noc']:.4f}  t={runtime:.3f}s")

    return metrics, flow_pred
