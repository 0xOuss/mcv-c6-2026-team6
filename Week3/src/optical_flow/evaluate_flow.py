"""
src/optical_flow/evaluate_flow.py

Evaluation metrics for optical flow:
  - MSEN: Mean Squared Error in Non-occluded areas
  - PEPN: Percentage of Erroneous Pixels in Non-occluded areas (threshold = 3px)
  - Runtime

Intuition:
  MSEN measures average magnitude of flow error (sensitive to large errors).
  PEPN measures how many pixels have error > 3px (robust to outliers).
  Together they give complementary views: MSEN punishes large mistakes more.
"""

import numpy as np
import time
import cv2
from typing import Callable, Dict, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.kitti_io import load_image_pair, load_kitti_gt


def endpoint_error(flow_pred: np.ndarray, flow_gt: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel endpoint error (EPE).
    EPE = sqrt((u_pred - u_gt)^2 + (v_pred - v_gt)^2)

    Args:
        flow_pred: (H, W, 2)
        flow_gt:   (H, W, 2)
    Returns:
        epe: (H, W) float32
    """
    diff = flow_pred - flow_gt
    epe = np.sqrt((diff[:, :, 0] ** 2) + (diff[:, :, 1] ** 2))
    return epe.astype(np.float32)


def compute_msen(flow_pred: np.ndarray, flow_gt: np.ndarray,
                 valid_mask: np.ndarray) -> float:
    """
    MSEN: Mean Squared Error in Non-occluded areas.

    INTUITION: Takes the mean of squared endpoint errors.
    Squaring penalizes large errors heavily — a single 10px error contributes
    100x more than a 1px error. Good for detecting catastrophic failures.

    Args:
        flow_pred: (H, W, 2) predicted flow
        flow_gt:   (H, W, 2) ground truth flow
        valid_mask: (H, W) bool — True in non-occluded, valid GT pixels
    Returns:
        MSEN scalar
    """
    epe = endpoint_error(flow_pred, flow_gt)
    return float(np.mean(epe[valid_mask] ** 2))


def compute_pepn(flow_pred: np.ndarray, flow_gt: np.ndarray,
                 valid_mask: np.ndarray, threshold: float = 3.0) -> float:
    """
    PEPN: Percentage of Erroneous Pixels in Non-occluded areas.

    INTUITION: Counts the fraction of pixels where EPE > threshold (3px).
    More robust to outliers than MSEN — doesn't get overwhelmed by a few
    very large errors. Measures how often the method is 'wrong enough to matter'.

    Args:
        flow_pred:  (H, W, 2)
        flow_gt:    (H, W, 2)
        valid_mask: (H, W) bool
        threshold:  EPE threshold in pixels (KITTI standard: 3px)
    Returns:
        PEPN in [0, 1]
    """
    epe = endpoint_error(flow_pred, flow_gt)
    erroneous = epe[valid_mask] > threshold
    return float(np.mean(erroneous))


def evaluate_method(method_fn: Callable,
                    img1: np.ndarray,
                    img2: np.ndarray,
                    flow_gt: np.ndarray,
                    valid_mask: np.ndarray,
                    n_runs: int = 3,
                    pepn_threshold: float = 3.0) -> Dict:
    """
    Evaluate an optical flow method on a single frame pair.

    Args:
        method_fn: callable (img1, img2) → flow (H,W,2)
        img1, img2: uint8 BGR images
        flow_gt, valid_mask: GT from KITTI
        n_runs: average runtime over this many runs
        pepn_threshold: error threshold for PEPN

    Returns:
        dict with: msen, pepn, runtime_mean, runtime_std, flow_pred, epe_map
    """
    # Warm up + measure runtime
    runtimes = []
    flow_pred = None
    for i in range(n_runs):
        t0 = time.perf_counter()
        flow_pred = method_fn(img1, img2)
        t1 = time.perf_counter()
        runtimes.append(t1 - t0)

    # Ensure flow matches GT size
    if flow_pred.shape[:2] != flow_gt.shape[:2]:
        flow_pred = cv2.resize(flow_pred, (flow_gt.shape[1], flow_gt.shape[0]),
                               interpolation=cv2.INTER_LINEAR)

    msen = compute_msen(flow_pred, flow_gt, valid_mask)
    pepn = compute_pepn(flow_pred, flow_gt, valid_mask, pepn_threshold)
    epe_map = endpoint_error(flow_pred, flow_gt)

    return {
        "msen": msen,
        "pepn": pepn,
        "pepn_pct": pepn * 100,
        "runtime_mean": float(np.mean(runtimes)),
        "runtime_std": float(np.std(runtimes)),
        "flow_pred": flow_pred,
        "epe_map": epe_map,
    }


def evaluate_all_methods(methods: dict,
                         img1: np.ndarray,
                         img2: np.ndarray,
                         flow_gt: np.ndarray,
                         valid_mask: np.ndarray) -> Dict:
    """
    Evaluate multiple OF methods and return a results dict.

    Args:
        methods: dict[name → callable]
    Returns:
        results: dict[name → metrics dict]
    """
    results = {}
    for name, fn in methods.items():
        print(f"  Evaluating {name}...", end=" ", flush=True)
        try:
            res = evaluate_method(fn, img1, img2, flow_gt, valid_mask)
            results[name] = res
            print(f"MSEN={res['msen']:.4f}  PEPN={res['pepn_pct']:.2f}%  "
                  f"t={res['runtime_mean']:.3f}s")
        except Exception as e:
            print(f"FAILED: {e}")
            results[name] = None
    return results


def print_results_table(results: dict):
    """Pretty-print results as a markdown table."""
    print("\n" + "=" * 70)
    print(f"{'Method':<20} {'MSEN':>8} {'PEPN%':>8} {'Runtime(s)':>12}")
    print("=" * 70)
    for name, res in results.items():
        if res is None:
            print(f"{name:<20} {'FAILED':>8}")
            continue
        print(f"{name:<20} {res['msen']:>8.4f} {res['pepn_pct']:>8.2f}% "
              f"{res['runtime_mean']:>10.3f}s")
    print("=" * 70)


def compute_error_histogram(epe_map: np.ndarray,
                            valid_mask: np.ndarray,
                            bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute histogram of endpoint errors (useful for understanding error distribution).
    Returns bin_edges, counts.
    """
    errors = epe_map[valid_mask].ravel()
    counts, bin_edges = np.histogram(errors, bins=bins, range=(0, 20))
    return bin_edges, counts
