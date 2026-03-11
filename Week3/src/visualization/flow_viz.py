"""
Optical Flow Visualization.

Standard visualizations used in the optical flow community:
  1. HSV color wheel:  hue = direction, saturation = magnitude
                       → Most common in papers; allows seeing both direction and speed
  2. Quiver plot:      Arrow field subsampled every N pixels
                       → Good for understanding overall motion patterns
  3. Error map:        Per-pixel error magnitude (EPE) colored by severity
                       → Shows WHERE each method fails
  4. Side-by-side:     Compare GT vs predictions in same figure
  5. FB consistency:   Show which pixels have reliable flow estimates
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Optional, List, Dict


def flow_to_hsv(flow: np.ndarray, max_magnitude: Optional[float] = None) -> np.ndarray:
    """
    Convert flow field to HSV color image.

    Encoding:
        Hue (0-180 in OpenCV):   flow direction (angle)
        Saturation (0-255):      constant 255 (full color)
        Value (0-255):           flow magnitude (normalized)

    Args:
        flow:          (H, W, 2) float32 [u, v]
        max_magnitude: clip magnitude at this value (None = auto from 99th percentile)
    Returns:
        rgb: (H, W, 3) uint8
    """
    u, v = flow[:, :, 0], flow[:, :, 1]
    magnitude = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u)  # radians [-π, π]

    # Normalize angle to [0, 180] for OpenCV HSV (hue range 0-180)
    hue = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)

    # Normalize magnitude
    if max_magnitude is None:
        max_magnitude = np.percentile(magnitude, 99)
    if max_magnitude == 0:
        max_magnitude = 1.0
    value = np.clip(magnitude / max_magnitude * 255, 0, 255).astype(np.uint8)

    saturation = np.full_like(hue, 255)
    hsv = np.stack([hue, saturation, value], axis=-1)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def draw_flow_wheel(size: int = 256) -> np.ndarray:
    """
    Draw the HSV color wheel legend for optical flow.
    Shows: which color corresponds to which direction.
    """
    center = size // 2
    y, x = np.mgrid[-center:center, -center:center]
    angle = np.arctan2(y.astype(float), x.astype(float))
    magnitude = np.sqrt(x**2 + y**2).astype(float)
    radius = center * 0.9
    mask = magnitude <= radius

    hue = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
    value = np.clip(magnitude / radius * 255, 0, 255).astype(np.uint8)
    saturation = np.full_like(hue, 255)
    saturation[~mask] = 0
    value[~mask] = 255

    hsv = np.stack([hue, saturation, value], axis=-1)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def plot_quiver(flow: np.ndarray, img: Optional[np.ndarray] = None,
                step: int = 12, scale: float = 1.0,
                title: str = "Optical Flow") -> plt.Figure:
    """
    Quiver (arrow) plot of flow field.

    Args:
        flow:  (H, W, 2)
        img:   background image (optional)
        step:  subsample every `step` pixels
        scale: arrow scale multiplier
        title: plot title
    Returns:
        matplotlib Figure
    """
    H, W = flow.shape[:2]
    y, x = np.mgrid[step//2:H:step, step//2:W:step]
    u = flow[y, x, 0]
    v = flow[y, x, 1]

    fig, ax = plt.subplots(figsize=(10, 6))
    if img is not None:
        ax.imshow(img if img.max() <= 1.0 else img / 255.0)
    else:
        ax.set_facecolor('#1a1a2e')

    magnitude = np.sqrt(u**2 + v**2)
    q = ax.quiver(x, y, u, -v, magnitude, cmap='plasma',
                  scale=W / (scale * step * 2), width=0.001)
    plt.colorbar(q, ax=ax, label='Flow magnitude (px)')
    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.axis('off')
    plt.tight_layout()
    return fig


def plot_error_map(flow_pred: np.ndarray, flow_gt: np.ndarray,
                   valid_mask: Optional[np.ndarray] = None,
                   vmax: float = 10.0, title: str = "EPE Error Map") -> plt.Figure:
    """
    Plot per-pixel End-Point Error as a heatmap.

    Color: blue = low error, red = high error.
    Occluded regions (where valid_mask=False) shown in gray.

    Args:
        flow_pred:  (H, W, 2)
        flow_gt:    (H, W, 2)
        valid_mask: (H, W) bool — valid pixels
        vmax:       clip errors above this value (pixels)
    """
    diff = flow_pred - flow_gt
    epe = np.sqrt(diff[:, :, 0]**2 + diff[:, :, 1]**2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Error map
    masked_epe = np.ma.masked_where(~valid_mask, epe) if valid_mask is not None else epe
    im = axes[0].imshow(masked_epe, cmap='RdYlBu_r', vmin=0, vmax=vmax)
    axes[0].set_title(f'{title}\n(valid pixels only)', fontsize=11)
    plt.colorbar(im, ax=axes[0], label='EPE (pixels)')
    axes[0].axis('off')

    # Error histogram
    if valid_mask is not None:
        vals = epe[valid_mask]
    else:
        vals = epe.ravel()
    axes[1].hist(vals, bins=50, color='#0891B2', edgecolor='white', linewidth=0.5)
    axes[1].axvline(np.mean(vals), color='red', linestyle='--', label=f'Mean={np.mean(vals):.2f}px')
    axes[1].axvline(np.median(vals), color='orange', linestyle='--', label=f'Median={np.median(vals):.2f}px')
    axes[1].set_xlabel('EPE (pixels)'); axes[1].set_ylabel('Pixel count')
    axes[1].set_title('Error Distribution'); axes[1].legend()
    axes[1].set_xlim(0, vmax)

    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_methods_comparison(flows: Dict[str, np.ndarray],
                             img1: np.ndarray,
                             flow_gt: Optional[np.ndarray] = None,
                             valid_mask: Optional[np.ndarray] = None) -> plt.Figure:
    """
    Side-by-side comparison of multiple flow methods.
    Top row: HSV flow visualization
    Bottom row: Error maps (if GT provided)

    Args:
        flows:      {"method_name": flow_array, ...}
        img1:       First input image
        flow_gt:    Ground truth flow (optional)
        valid_mask: Valid pixels mask (optional)
    """
    n_methods = len(flows)
    has_gt = flow_gt is not None

    n_rows = 3 if has_gt else 2  # HSV + quiver [+ error]
    n_cols = n_methods + (1 if has_gt else 0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    # Column 0: GT (if available)
    col_start = 0
    if has_gt:
        gt_hsv = flow_to_hsv(flow_gt)
        axes[0, 0].imshow(gt_hsv); axes[0, 0].set_title("Ground Truth\n(HSV)", fontsize=10)
        axes[0, 0].axis('off')
        axes[1, 0].imshow(img1 if img1.max() > 1 else img1)
        axes[1, 0].set_title("Input Frame", fontsize=10); axes[1, 0].axis('off')
        if n_rows > 2:
            axes[2, 0].axis('off')
        col_start = 1

    # Method columns
    for i, (name, flow) in enumerate(flows.items()):
        col = i + col_start

        # Row 0: HSV
        hsv = flow_to_hsv(flow)
        axes[0, col].imshow(hsv)
        axes[0, col].set_title(f"{name}\n(HSV)", fontsize=10)
        axes[0, col].axis('off')

        # Row 1: Quiver overlay on img1
        H, W = flow.shape[:2]
        step = max(H, W) // 25
        y, x = np.mgrid[step//2:H:step, step//2:W:step]
        u, v = flow[y, x, 0], flow[y, x, 1]
        mag = np.sqrt(u**2 + v**2)
        axes[1, col].imshow(img1 if img1.max() > 1 else img1)
        axes[1, col].quiver(x, y, u, -v, mag, cmap='plasma',
                             scale=W/(step*2), width=0.0015, alpha=0.9)
        axes[1, col].set_title(f"{name}\n(Quiver)", fontsize=10)
        axes[1, col].axis('off')

        # Row 2: Error map
        if n_rows > 2 and has_gt:
            diff = flow - flow_gt
            epe = np.sqrt(diff[:, :, 0]**2 + diff[:, :, 1]**2)
            masked = np.ma.masked_where(~valid_mask, epe) if valid_mask is not None else epe
            im = axes[2, col].imshow(masked, cmap='hot', vmin=0, vmax=10)
            msen = float(((diff**2).sum(-1)[valid_mask] if valid_mask is not None
                          else (diff**2).sum(-1)).mean())
            axes[2, col].set_title(f"{name}\nError Map (MSEN={msen:.2f})", fontsize=10)
            axes[2, col].axis('off')

    plt.tight_layout()
    return fig


def plot_fb_consistency(flow_fwd: np.ndarray, flow_bwd: np.ndarray,
                         threshold: float = 1.0) -> plt.Figure:
    """
    Visualize forward-backward consistency.

    Shows:
      - Forward flow (HSV)
      - FB consistency score (lower = more reliable)
      - Reliable mask

    Intuition:
      If a pixel truly moves from p to p', then running flow backward from p'
      should bring us back to p. If not, that pixel is either occluded,
      on a boundary, or the flow estimate is wrong there.
    """
    consistency = np.sqrt(
        (flow_fwd[:, :, 0] + flow_bwd[:, :, 0])**2 +
        (flow_fwd[:, :, 1] + flow_bwd[:, :, 1])**2
    )
    mask = consistency < threshold

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(flow_to_hsv(flow_fwd))
    axes[0].set_title("Forward Flow (HSV)", fontsize=11); axes[0].axis('off')

    im = axes[1].imshow(consistency, cmap='hot', vmin=0, vmax=5)
    axes[1].set_title(f"FB Inconsistency\n(lower = more reliable)", fontsize=11)
    axes[1].axis('off'); plt.colorbar(im, ax=axes[1], label='px')

    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title(f"Reliable Mask\n({mask.mean()*100:.1f}% reliable @ {threshold}px)", fontsize=11)
    axes[2].axis('off')

    pct = mask.mean() * 100
    fig.suptitle(f'Forward-Backward Consistency Analysis — {pct:.1f}% of pixels reliable',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig
