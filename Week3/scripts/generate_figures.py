#!/usr/bin/env python3
"""
scripts/generate_figures.py

Generate ALL figures needed for the Week 3 presentation slides.

This is the single script that produces every visual you need.
Run it AFTER:
  1. run_optical_flow.py     → saves flow predictions
  2. run_hp_sweep.py         → finds best HPs
  3. run_tracking.py         → saves tracker outputs
  4. run_ablation.py         → saves ablation metrics

Figures produced:
  Task 1.1 (Optical Flow):
    flow_hsv_comparison.png       — PyFlow vs Farneback vs RAFT side-by-side, HSV
    flow_error_maps.png           — per-pixel error per method, same frame
    flow_wheel.png                — HSV color wheel legend (ALWAYS put on same slide!)
    flow_quiver.png               — arrow field for intuition
    fb_consistency.png            — which pixels have reliable flow
    flow_metrics_table.png        — MSEN / PEPN / Runtime table

  Task 1.2 (Tracking):
    same_frame_comparison.png     — IoU vs Kalman vs OF tracker on SAME frame
                                    (directly fixes Week 2 feedback)
    trajectory_plot.png           — track trails for qualitative analysis
    id_switch_heatmap.png         — where do ID switches happen?
    hp_sensitivity.png            — IDF1 vs each HP (from sweep CSVs)
    tracker_comparison_table.png  — IDF1 / HOTA / IDsw per tracker

  Task 2 (MTSC):
    mtsc_per_camera.png           — per-camera IDF1 / HOTA bar chart
    mtsc_failure_cases.png        — 2 cameras with annotated failure reasons

Usage:
    python scripts/generate_figures.py --results_dir results/ \\
        --seq_dir data/aicity/S01/c010 --output_dir figures/

    # Just tracking figures:
    python scripts/generate_figures.py --results_dir results/ \\
        --seq_dir data/aicity/S01/c010 --task tracking

    # Just flow figures:
    python scripts/generate_figures.py --results_dir results/ \\
        --kitti_dir data/kitti --task flow
"""

import sys
import argparse
import json
import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.flow_viz import flow_to_hsv, draw_flow_wheel
from src.visualization.tracking_viz import draw_tracks_on_frame, get_track_color
from src.utils.kitti_utils import load_detections_aicity, load_gt_aicity


# ─── Parser ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir",  default="results/")
    p.add_argument("--seq_dir",      default=None,
                   help="Camera sequence dir (for tracking figures)")
    p.add_argument("--kitti_dir",    default="data/kitti",
                   help="KITTI Seq45 directory (for flow figures)")
    p.add_argument("--output_dir",   default="figures/")
    p.add_argument("--task",         default="all",
                   choices=["all", "flow", "tracking", "mtsc"])
    p.add_argument("--frame_id",     type=int, default=None,
                   help="Specific frame for qualitative comparisons (auto if None)")
    return p.parse_args()


# ─── Task 1.1: Optical Flow Figures ──────────────────────────────────────────

def make_flow_wheel(out_dir: Path):
    """
    HSV color wheel — ALWAYS include this on the same slide as flow visualizations.
    Without it, the teacher has to guess what the colors mean.
    Week 2 feedback: Teams lost points for missing legends on qualitative slides.
    """
    wheel = draw_flow_wheel(size=256)
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="white")
    ax.imshow(wheel)
    ax.set_title("Flow Direction Legend\n(Hue=direction, Brightness=magnitude)",
                 fontsize=10, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    out_path = out_dir / "flow_wheel.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def make_flow_hsv_comparison(results_dir: Path, kitti_dir: Path, out_dir: Path):
    """
    Side-by-side HSV flow comparison: PyFlow | Farneback | RAFT.
    Top row: flow visualization. Bottom row: error map (EPE).
    """
    flow_dir = results_dir / "optical_flow"

    methods = ["pyflow", "farneback", "raft"]
    titles  = [
        "PyFlow\n(Brox et al., ECCV 2004)",
        "Farneback\n(SCIA 2003) — our baseline",
        "RAFT\n(Teed & Deng, ECCV 2020) — our best",
    ]

    # Load flows
    flows = {}
    for m in methods:
        flo_path = flow_dir / f"{m}_flow.npy"
        if flo_path.exists():
            flows[m] = np.load(str(flo_path))

    if not flows:
        print("  Warning: No precomputed flow .npy found in results/optical_flow/")
        print("  Run: python scripts/run_optical_flow.py first")
        return

    # Load source image for context
    img1_path = Path(kitti_dir) / "000045_10.png"
    img1 = cv2.imread(str(img1_path)) if img1_path.exists() else None

    n = len(flows)
    fig, axes = plt.subplots(2, max(n, 1), figsize=(6 * max(n, 1), 8),
                              facecolor="white")
    if n == 1:
        axes = axes[:, np.newaxis]

    for i, (m, flow) in enumerate(flows.items()):
        title = titles[methods.index(m)] if m in methods else m

        # Row 0: HSV flow
        hsv = flow_to_hsv(flow)
        axes[0, i].imshow(hsv)
        axes[0, i].set_title(title, fontsize=10, fontweight="bold", pad=8)
        axes[0, i].axis("off")

        # Row 1: Error map (requires GT)
        gt_flo_path = Path(kitti_dir) / "000045_10.flo"
        if gt_flo_path.exists():
            from src.utils.kitti_io import read_flo
            flow_gt = read_flo(str(gt_flo_path))
            epe = np.sqrt(
                (flow[:, :, 0] - flow_gt[:, :, 0]) ** 2 +
                (flow[:, :, 1] - flow_gt[:, :, 1]) ** 2
            )
            im = axes[1, i].imshow(epe, cmap="hot", vmin=0, vmax=10)
            axes[1, i].set_title("EPE Error Map", fontsize=9)
            axes[1, i].axis("off")
            plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04,
                         label="Error (pixels)")
        else:
            axes[1, i].axis("off")

    plt.suptitle("Optical Flow Methods: HSV Visualization & Error Maps\n"
                 "KITTI Sequence 045 — image_0 pair (000045_10/11.png)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / "flow_hsv_comparison.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def make_flow_metrics_table(results_dir: Path, out_dir: Path):
    """
    Clean metrics table: MSEN | PEPN | Runtime for each method.
    Addresses Week 2 pattern: always show units in table headers.
    """
    metrics_path = results_dir / "optical_flow" / "metrics.json"
    if not metrics_path.exists():
        print(f"  Warning: {metrics_path} not found — skipping metrics table")
        return

    with open(str(metrics_path)) as f:
        metrics = json.load(f)

    methods = list(metrics.keys())
    msen    = [metrics[m].get("msen",    0.0)  for m in methods]
    pepn    = [metrics[m].get("pepn",    0.0)  for m in methods]
    runtime = [metrics[m].get("runtime", 0.0)  for m in methods]

    fig, ax = plt.subplots(figsize=(9, max(3, len(methods) * 0.7 + 1.5)),
                            facecolor="white")
    ax.axis("off")

    # Table data — units in column headers (Week 2 lesson)
    col_labels = ["Method", "MSEN ↓", "PEPN (%) ↓", "Runtime (s) ↓"]
    cell_data  = [
        [m,
         f"{ms:.4f}",
         f"{pe:.2f}%",
         f"{rt:.3f}s"]
        for m, ms, pe, rt in zip(methods, msen, pepn, runtime)
    ]

    # Highlight best value in each metric column
    best_msen    = min(msen)
    best_pepn    = min(pepn)
    best_runtime = min(runtime)

    cell_colors = []
    for m, ms, pe, rt in zip(methods, msen, pepn, runtime):
        row_colors = ["#F1F5F9"] * 4
        if ms  == best_msen:    row_colors[1] = "#BBF7D0"
        if pe  == best_pepn:    row_colors[2] = "#BBF7D0"
        if rt  == best_runtime: row_colors[3] = "#BBF7D0"
        cell_colors.append(row_colors)

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 1.8)

    # Bold header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#1E3A5F")
        table[(0, j)].get_text().set_color("white")
        table[(0, j)].get_text().set_fontweight("bold")

    ax.set_title("Task 1.1: Optical Flow Evaluation — KITTI Seq 045\n"
                 "(↓ = lower is better | Green = best per column)",
                 fontsize=11, fontweight="bold", pad=12)

    plt.tight_layout()
    out_path = out_dir / "flow_metrics_table.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ─── Task 1.2: Tracking Figures ───────────────────────────────────────────────

def make_same_frame_comparison(seq_dir: Path, results_dir: Path,
                                out_dir: Path, frame_id: Optional[int] = None):
    """
    Side-by-side: IoU Baseline | Kalman | OF Tracker on the SAME FRAME.

    This directly fixes Week 2 feedback:
      "Better to show qualitative comparisons in the same frame"
      "Not shown on the same frame"

    Green boxes = ground truth, colored boxes = tracker predictions.
    Each box labeled with Track ID.
    """
    trackers = ["iou_baseline", "kalman", "of_tracker"]
    titles   = [
        "IoU Baseline\n(No motion model)",
        "Kalman Filter (SORT)\n(Constant velocity)",
        "Adaptive OF Tracker (Ours)\n(Per-bbox flow aggregation)",
    ]
    pred_dirs = {
        t: results_dir / "tracking" / t / "predictions"
        for t in trackers
    }
    gt_file = str(seq_dir / "gt" / "gt.txt")

    # Find a "good" comparison frame if not specified
    if frame_id is None:
        # Pick a frame from the middle of the sequence
        img_dir = seq_dir / "img1"
        if img_dir.exists():
            frames = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
            if frames:
                frame_id = frame_id_from_name(frames[len(frames) // 2].stem)
            else:
                frame_id = 100
        else:
            frame_id = 100

    print(f"  Same-frame comparison at frame {frame_id}")

    # Load image
    img = load_frame(seq_dir, frame_id)
    if img is None:
        print(f"  Warning: frame {frame_id} not found, skipping same-frame comparison")
        return

    # Load GT
    gt_tracks = load_gt_aicity(gt_file) if Path(gt_file).exists() else {}
    gt_frame  = np.array([[r[1], r[2], r[3], r[4], r[0], 1.0]
                           for r in gt_tracks.get(frame_id, [])])

    n = len(trackers)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), facecolor="white")

    for i, (tracker, title) in enumerate(zip(trackers, titles)):
        frame_copy = img.copy()

        # Draw GT in green
        if len(gt_frame) > 0:
            for row in gt_frame:
                x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(frame_copy, f"GT:{int(row[4])}", (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)

        # Draw predictions
        pred_file = pred_dirs[tracker] / f"{frame_id:06d}.npy"
        if pred_file.exists():
            preds = np.load(str(pred_file))
            frame_copy = draw_tracks_on_frame(frame_copy, preds,
                                              show_id=True, thickness=2)

        axes[i].imshow(cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB))
        axes[i].set_title(title, fontsize=10, fontweight="bold", pad=8)
        axes[i].axis("off")

    # Legend
    gt_patch   = mpatches.Patch(color=(0, 0.78, 0), label="Ground truth (green)")
    pred_patch = mpatches.Patch(color=(0.8, 0.2, 0.2), label="Prediction (colored by ID)")
    fig.legend(handles=[gt_patch, pred_patch], loc="lower center",
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(f"Task 1.2: Same-Frame Tracker Comparison | Frame {frame_id}\n"
                 f"(Green = GT, colored boxes = predicted track IDs)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / "same_frame_comparison.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def make_hp_sensitivity_plots(results_dir: Path, out_dir: Path):
    """
    HP sensitivity curves from sweep CSVs.
    Shows IDF1 vs each HP separately with optimal value marked.
    """
    sweep_dir = results_dir / "tracking" / "hp_sweep"
    tracker = "of_tracker"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")
    params = [
        ("iou_threshold", "IoU Threshold",   f"{tracker}_iou_sweep_hungarian.csv"),
        ("max_age",       "Max Age (frames)", f"{tracker}_age_sweep.csv"),
        ("min_hits",      "Min Hits",         f"{tracker}_hits_sweep.csv"),
    ]
    colors = ["#0891B2", "#0F766E", "#7C3AED"]

    has_data = False
    for ax, (hp_col, xlabel, csv_name), color in zip(axes, params, colors):
        csv_path = sweep_dir / csv_name
        if not csv_path.exists():
            ax.text(0.5, 0.5, f"Run\nrun_hp_sweep.py\nfirst",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="#999999")
            ax.axis("off")
            continue

        df = pd.read_csv(str(csv_path))
        if hp_col not in df.columns or "idf1" not in df.columns:
            ax.axis("off")
            continue

        has_data = True
        x = df[hp_col].values
        y = df["idf1"].values * 100

        ax.plot(x, y, "o-", color=color, linewidth=2.5, markersize=7,
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=2)
        ax.fill_between(x, y * 0.97, y * 1.03, alpha=0.12, color=color)

        best_idx = int(np.argmax(y))
        ax.axvline(x=x[best_idx], color="#DC2626", linestyle="--", linewidth=2,
                   label=f"Optimal: {x[best_idx]}")
        ax.scatter([x[best_idx]], [y[best_idx]], s=120, color="#DC2626",
                   zorder=5, label=f"IDF1={y[best_idx]:.1f}%")

        ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
        ax.set_ylabel("IDF1 (%)", fontsize=11, fontweight="bold")
        ax.set_title(f"{xlabel} Sensitivity", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_facecolor("#F8FAFC")
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle(
        "Hyperparameter Sensitivity — Adaptive OF Tracker\n"
        "Sequential sweep: each HP tuned with others fixed at optimum",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out_path = out_dir / "hp_sensitivity.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    if has_data:
        print(f"  Saved: {out_path}")
    else:
        print(f"  Skipped (no sweep CSVs found)")


def make_tracker_comparison_table(results_dir: Path, out_dir: Path):
    """
    Summary table: IoU | Kalman | OF Tracker (fixed median) | Adaptive OF (ours).
    Units in column headers. IDF1 and HOTA both shown.
    """
    trackers = {
        "IoU Baseline":        "iou_baseline",
        "Kalman (SORT)":       "kalman",
        "OF Tracker (RAFT)":   "of_tracker",
        "Adaptive OF (ours)":  "adaptive_of_tracker",
    }

    rows = []
    for display_name, key in trackers.items():
        metrics_path = results_dir / "tracking" / key / "metrics.json"
        if metrics_path.exists():
            with open(str(metrics_path)) as f:
                m = json.load(f)
            rows.append([
                display_name,
                f"{m.get('idf1', 0)*100:.2f}%",
                f"{m.get('hota', m.get('idf1', 0))*100:.2f}%",
                f"{m.get('mota', 0)*100:.2f}%",
                str(int(m.get("num_switches", 0))),
            ])
        else:
            rows.append([display_name, "—", "—", "—", "—"])

    col_labels = ["Method", "IDF1 (%) ↑", "HOTA (%) ↑", "MOTA (%) ↑", "ID Switches ↓"]

    fig, ax = plt.subplots(
        figsize=(12, max(3, len(rows) * 0.8 + 1.5)),
        facecolor="white"
    )
    ax.axis("off")

    cell_colors = [["#F1F5F9"] * len(col_labels)] * len(rows)
    # Highlight last row (our method)
    if rows:
        cell_colors[-1] = ["#FEF3C7"] * len(col_labels)

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#1E3A5F")
        table[(0, j)].get_text().set_color("white")
        table[(0, j)].get_text().set_fontweight("bold")

    ax.set_title(
        "Task 1.2: Tracker Comparison (AI City c010, fine-tuned YOLO26l detector)\n"
        "(Yellow row = our contribution | Units in headers)",
        fontsize=11, fontweight="bold", pad=12,
    )
    plt.tight_layout()
    out_path = out_dir / "tracker_comparison_table.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ─── Task 2: MTSC Figures ─────────────────────────────────────────────────────

def make_mtsc_per_camera(results_dir: Path, out_dir: Path):
    """
    Per-camera IDF1 and HOTA bar charts for SEQ01 and SEQ03.
    Includes per-camera failure mode annotation.
    """
    for seq in ["S01", "S03"]:
        metrics_path = results_dir / "tracking" / f"mtsc_{seq}.json"
        if not metrics_path.exists():
            print(f"  Warning: {metrics_path} not found — run run_mtsc.py first")
            continue

        with open(str(metrics_path)) as f:
            data = json.load(f)

        cameras = list(data.keys())
        if not cameras:
            continue

        idf1 = [data[c].get("idf1", 0) * 100 for c in cameras]
        hota = [data[c].get("hota", data[c].get("idf1", 0)) * 100 for c in cameras]

        x = np.arange(len(cameras))
        width = 0.38

        fig, ax = plt.subplots(figsize=(max(8, len(cameras) * 1.5), 6),
                                facecolor="white")

        b1 = ax.bar(x - width/2, idf1, width, label="IDF1",
                    color="#0891B2", alpha=0.85, edgecolor="white")
        b2 = ax.bar(x + width/2, hota, width, label="HOTA",
                    color="#0F766E", alpha=0.55, hatch="//", edgecolor="white")

        for bar, val in zip(b1, idf1):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        for bar, val in zip(b2, hota):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom",
                    fontsize=9, color="#555555")

        # Average line
        avg_idf1 = np.mean(idf1)
        ax.axhline(avg_idf1, color="#DC2626", linestyle="--", linewidth=1.5,
                   label=f"Avg IDF1 = {avg_idf1:.1f}%")

        ax.set_xticks(x)
        ax.set_xticklabels(cameras, fontsize=10)
        ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.set_title(f"Task 2: MTSC — {seq} Per-Camera Performance\n"
                     f"(Adaptive OF Tracker + fine-tuned YOLO26l | n={len(cameras)} cameras)",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.set_facecolor("#F8FAFC")
        ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        out_path = out_dir / f"mtsc_{seq}_per_camera.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def frame_id_from_name(stem: str) -> int:
    try:
        return int(stem)
    except ValueError:
        return int("".join(filter(str.isdigit, stem)) or "0")


def load_frame(seq_dir: Path, frame_id: int) -> Optional[np.ndarray]:
    """Load a frame by ID, checking common subdirectory names."""
    for subdir in ["img1", "images", "frames", ""]:
        d = seq_dir / subdir if subdir else seq_dir
        for ext in [".jpg", ".png"]:
            p = d / f"{frame_id:06d}{ext}"
            if p.exists():
                return cv2.imread(str(p))
    return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    seq_dir     = Path(args.seq_dir) if args.seq_dir else None

    print("=" * 60)
    print(f"Generating Figures")
    print(f"  Results dir: {results_dir}")
    print(f"  Output dir:  {out_dir}")
    print("=" * 60)

    if args.task in ("all", "flow"):
        print("\n[Task 1.1] Optical Flow figures...")
        make_flow_wheel(out_dir)
        make_flow_hsv_comparison(results_dir, Path(args.kitti_dir), out_dir)
        make_flow_metrics_table(results_dir, out_dir)

    if args.task in ("all", "tracking") and seq_dir:
        print("\n[Task 1.2] Tracking figures...")
        make_same_frame_comparison(seq_dir, results_dir, out_dir, args.frame_id)
        make_hp_sensitivity_plots(results_dir, out_dir)
        make_tracker_comparison_table(results_dir, out_dir)

    if args.task in ("all", "mtsc"):
        print("\n[Task 2] MTSC figures...")
        make_mtsc_per_camera(results_dir, out_dir)

    print(f"\nAll figures saved to: {out_dir}")
    print("\nSlide checklist:")
    print("  ✓ flow_wheel.png         → put on SAME SLIDE as HSV flow viz")
    print("  ✓ same_frame_comparison  → IoU vs Kalman vs OF on same frame")
    print("  ✓ hp_sensitivity         → proves HP sweep converged")
    print("  ✓ tracker_comparison     → table with units in headers")
    print("  ✓ mtsc_*_per_camera      → per-camera discussion ready")


if __name__ == "__main__":
    main()