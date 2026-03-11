"""
Tracking Visualization.

Key visualizations:
  1. BBox + ID overlay on video frame
  2. Side-by-side same-frame comparison (W/o vs W/ optical flow)
     → Directly addresses Week 2 feedback
  3. Trajectory traces (showing full path of each track)
  4. ID switch heatmap (where do switches occur?)
  5. Flow warping visualization (showing predicted vs actual positions)
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import colorsys


# Deterministic color palette for track IDs
def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Return a consistent BGR color for a given track ID."""
    np.random.seed(track_id * 137 + 42)  # deterministic
    hue = (track_id * 0.618033988749895) % 1.0  # golden ratio for spread
    rgb = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))  # BGR for cv2


def draw_tracks_on_frame(frame: np.ndarray, tracks: np.ndarray,
                          show_id: bool = True,
                          show_conf: bool = False,
                          thickness: int = 2) -> np.ndarray:
    """
    Draw tracking bounding boxes on a frame.

    Args:
        frame:    (H, W, 3) BGR image
        tracks:   (N, 6) [x1, y1, x2, y2, track_id, conf]
        show_id:  Draw track ID above bbox
        show_conf: Draw confidence score
        thickness: Bbox line thickness
    Returns:
        annotated frame (copy)
    """
    out = frame.copy()
    for row in tracks:
        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        tid = int(row[4])
        conf = float(row[5]) if len(row) > 5 else 1.0

        color = get_track_color(tid)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        # Label
        label = f"ID:{tid}"
        if show_conf:
            label += f" {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(out, label, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_of_predictions(frame: np.ndarray, tracks: np.ndarray,
                          predictions: np.ndarray) -> np.ndarray:
    """
    Visualize OF-predicted bbox positions as dashed boxes with arrows.

    Args:
        frame:       (H, W, 3) BGR
        tracks:      (N, 6) current confirmed bboxes
        predictions: (N, 4) predicted bboxes after OF warp
    Returns:
        annotated frame
    """
    out = frame.copy()
    for i, (track_row, pred_box) in enumerate(zip(tracks, predictions)):
        x1, y1, x2, y2 = int(track_row[0]), int(track_row[1]), int(track_row[2]), int(track_row[3])
        px1, py1, px2, py2 = int(pred_box[0]), int(pred_box[1]), int(pred_box[2]), int(pred_box[3])
        tid = int(track_row[4])
        color = get_track_color(tid)

        # Current bbox (solid)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Predicted bbox (dashed)
        _draw_dashed_rect(out, (px1, py1), (px2, py2), color, dash_len=8)

        # Arrow from current center to predicted center
        cx, cy = (x1+x2)//2, (y1+y2)//2
        pcx, pcy = (px1+px2)//2, (py1+py2)//2
        if abs(pcx-cx) > 1 or abs(pcy-cy) > 1:
            cv2.arrowedLine(out, (cx, cy), (pcx, pcy), (0, 255, 0), 2, tipLength=0.3)
    return out


def _draw_dashed_rect(img, pt1, pt2, color, dash_len=10, thickness=2):
    """Draw a dashed rectangle."""
    x1, y1 = pt1; x2, y2 = pt2
    for (a, b, c, d) in [(x1,y1,x2,y1), (x2,y1,x2,y2), (x2,y2,x1,y2), (x1,y2,x1,y1)]:
        pts = list(zip(
            np.linspace(a, c, abs(c-a)+abs(d-b) if (abs(c-a)+abs(d-b))>0 else 1, dtype=int),
            np.linspace(b, d, abs(c-a)+abs(d-b) if (abs(c-a)+abs(d-b))>0 else 1, dtype=int)
        ))
        for i in range(0, len(pts)-1, dash_len*2):
            end_i = min(i + dash_len, len(pts)-1)
            cv2.line(img, pts[i], pts[end_i], color, thickness)


def plot_same_frame_comparison(frame: np.ndarray,
                                 tracks_no_of: np.ndarray,
                                 tracks_with_of: np.ndarray,
                                 title: str = "Qualitative Comparison – Same Frame") -> plt.Figure:
    """
    Side-by-side comparison on the SAME frame — directly addressing Week 2 feedback.

    Left:  Tracking WITHOUT optical flow (shows ID switches as mismatched colors)
    Right: Tracking WITH optical flow (shows correct ID maintenance)

    ID switches appear as unexpected color changes — very visible in this format.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw both
    img_no_of    = draw_tracks_on_frame(frame, tracks_no_of)
    img_with_of  = draw_tracks_on_frame(frame, tracks_with_of)
    img_no_of    = cv2.cvtColor(img_no_of, cv2.COLOR_BGR2RGB)
    img_with_of  = cv2.cvtColor(img_with_of, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(img_no_of)
    axes[0].set_title("Without Optical Flow\n(ID switches visible as color jumps)",
                       fontsize=12, fontweight='bold', color='#DC2626')
    axes[0].axis('off')

    axes[1].imshow(img_with_of)
    axes[1].set_title("With Optical Flow (RAFT)\n(Correct IDs maintained through occlusion)",
                       fontsize=12, fontweight='bold', color='#0F766E')
    axes[1].axis('off')

    # Highlight differences
    for ax, side in zip(axes, ['no_of', 'with_of']):
        color = '#DC2626' if side == 'no_of' else '#0F766E'
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_trajectories(frames_tracks: Dict[int, np.ndarray],
                       background: np.ndarray,
                       title: str = "Track Trajectories") -> plt.Figure:
    """
    Plot full trajectories of all tracks on a background image.

    Args:
        frames_tracks: {frame_id: (N, 6) tracks} for all frames
        background:    Background image to overlay on
    """
    # Collect per-track centroids
    track_paths: Dict[int, List[Tuple]] = {}
    for frame_id in sorted(frames_tracks.keys()):
        for row in frames_tracks[frame_id]:
            tid = int(row[4])
            cx = (row[0] + row[2]) / 2
            cy = (row[1] + row[3]) / 2
            if tid not in track_paths:
                track_paths[tid] = []
            track_paths[tid].append((frame_id, cx, cy))

    fig, ax = plt.subplots(figsize=(12, 8))
    bg_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    ax.imshow(bg_rgb, alpha=0.6)

    for tid, path in track_paths.items():
        if len(path) < 3:
            continue
        xs = [p[1] for p in path]
        ys = [p[2] for p in path]
        color_bgr = get_track_color(tid)
        color_rgb = (color_bgr[2]/255, color_bgr[1]/255, color_bgr[0]/255)
        ax.plot(xs, ys, '-', color=color_rgb, linewidth=1.5, alpha=0.8)
        ax.plot(xs[-1], ys[-1], 'o', color=color_rgb, markersize=4)

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    return fig


def plot_id_switch_heatmap(id_switches: List[Tuple[int, float, float]],
                            frame_shape: Tuple[int, int],
                            title: str = "ID Switch Spatial Distribution") -> plt.Figure:
    """
    Heatmap showing where in the frame ID switches occur.

    Args:
        id_switches: list of (frame_id, cx, cy) where a switch occurred
        frame_shape: (H, W)
    """
    H, W = frame_shape
    heatmap = np.zeros((H, W), dtype=np.float32)

    for _, cx, cy in id_switches:
        x, y = int(cx), int(cy)
        if 0 <= x < W and 0 <= y < H:
            # Gaussian splat
            r = 30
            x1, y1 = max(0, x-r), max(0, y-r)
            x2, y2 = min(W, x+r), min(H, y+r)
            yy, xx = np.mgrid[y1:y2, x1:x2]
            d2 = (xx - x)**2 + (yy - y)**2
            heatmap[y1:y2, x1:x2] += np.exp(-d2 / (2 * (r/2)**2))

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(np.zeros((H, W, 3), dtype=np.uint8))
    if heatmap.max() > 0:
        im = ax.imshow(heatmap, cmap='hot', alpha=0.8,
                       vmin=0, vmax=np.percentile(heatmap[heatmap > 0], 95))
        plt.colorbar(im, ax=ax, label='ID switch density')
    ax.set_title(f"{title}\n({len(id_switches)} total switches)", fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    return fig
