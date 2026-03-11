#!/usr/bin/env python3
"""
Task 1.2: Object Tracking with Optical Flow

Runs all four trackers on a single camera sequence and:
  - Computes IDF1 / MOTA metrics
  - Generates side-by-side qualitative comparison
  - Produces ablation study results
  - Identifies where ID switches occur (spatial analysis)

Usage:
    python scripts/run_tracking.py --seq_dir data/aicity/S03/c010 \
        --hp_file results/tracking/hp_sweep/best_hyperparameters.json
    python scripts/run_tracking.py --seq_dir data/aicity/S03/c010 --ablation
"""

import sys
import argparse
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.kitti_utils import load_detections_aicity, load_gt_aicity
from src.tracking.iou_tracker import IoUTracker
from src.tracking.kalman_tracker import KalmanTracker
from src.tracking.of_tracker import OFTracker
from src.tracking.adaptive_of_tracker import AdaptiveOFTracker
from src.evaluation.tracking_metrics import (
    compute_mot_metrics, write_mot_results, compute_id_switches
)
from src.visualization.tracking_viz import (
    draw_tracks_on_frame, plot_same_frame_comparison, plot_trajectories,
    plot_id_switch_heatmap, draw_of_predictions
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq_dir", required=True)
    p.add_argument("--config", default="configs/tracker.yaml")
    p.add_argument("--hp_file", default=None,
                   help="JSON file from run_hp_sweep.py with best HPs")
    p.add_argument("--iou_thresh", type=float, default=0.45)
    p.add_argument("--max_age", type=int, default=8)
    p.add_argument("--min_hits", type=int, default=3)
    p.add_argument("--matching", default="hungarian")
    p.add_argument("--output_dir", default="results/tracking")
    p.add_argument("--ablation", action="store_true",
                   help="Run ablation study (all components isolated)")
    p.add_argument("--save_video", action="store_true",
                   help="Save annotated video for qualitative inspection")
    return p.parse_args()


def load_config(path): return yaml.safe_load(open(path))


def _load_flow(flow_dir, frame_id):
    if flow_dir is None:
        return None, None
    fp  = Path(flow_dir) / f"flow_{frame_id:06d}.npy"
    fbp = Path(flow_dir) / f"flow_bwd_{frame_id:06d}.npy"
    # Cast float16→float32 to avoid overflow in downstream math
    fwd = np.load(str(fp)).astype(np.float32)  if fp.exists()  else None
    bwd = np.load(str(fbp)).astype(np.float32) if fbp.exists() else None
    return fwd, bwd


def run_tracker(tracker, detections: dict, gt_tracks: dict,
                flow_dir: Optional[str] = None) -> tuple:
    """
    Run tracker for all frames in a sequence.

    FIX: Previously only checked isinstance(tracker, OFTracker) for flow loading.
    AdaptiveOFTracker is a separate class so it was never getting flow data.
    Now checks for both OFTracker and AdaptiveOFTracker.

    Returns:
        pred_tracks: {frame_id: [[tid, x1, y1, x2, y2], ...]}
        raw_results: {frame_id: np.array (N,6)}
    """
    tracker.reset()
    pred_tracks = {}
    raw_results = {}

    uses_flow = isinstance(tracker, (OFTracker, AdaptiveOFTracker))
    all_frames = sorted(gt_tracks.keys()) if gt_tracks else sorted(detections.keys())

    for frame_id in all_frames:
        dets_np = np.array(detections.get(frame_id, [])) \
                  if detections.get(frame_id) else np.empty((0, 5))

        if uses_flow and flow_dir:
            flow, flow_bwd = _load_flow(flow_dir, frame_id)
            result = tracker.update(dets_np, frame_id, flow=flow, flow_bwd=flow_bwd)
        else:
            result = tracker.update(dets_np, frame_id)

        raw_results[frame_id] = result
        if len(result) > 0:
            pred_tracks[frame_id] = [[int(r[4]), r[0], r[1], r[2], r[3]] for r in result]

    return pred_tracks, raw_results


def _get_flow_dir(seq_dir: Path, method: str) -> Optional[str]:
    """
    Return the flow directory for a specific method.
    Prefers method-specific dir (flow_unimatch/, flow_raft/ etc.),
    falls back to the flow/ symlink if specific dir doesn't exist.
    """
    specific = seq_dir / f"flow_{method}"
    if specific.exists():
        return str(specific)
    symlink = seq_dir / "flow"
    if symlink.exists():
        return str(symlink)
    return None


def run_ablation(seq_dir: Path, det_file: str, gt_file: str,
                 best_hps: dict, output_dir: Path):
    """
    Ablation study: add one component at a time and measure IDF1 improvement.

    FIX: Now uses correct flow directories per ablation step:
      - "Farneback" step  → flow_farneback/
      - "RAFT" step       → flow_raft/
      - All other OF steps → flow/ symlink (best method = UniMatch)
      - Full pipeline      → flow/ symlink (best method = UniMatch)

    Components:
      1. IoU baseline (no motion model, no OF)
      2. + Kalman filter prediction
      3. + OF Warping (Farneback)   ← uses flow_farneback/
      4. + OF Warping (RAFT)        ← uses flow_raft/
      5. + OF Warping (UniMatch)    ← uses flow/ → flow_unimatch/
      6. + Flow threshold gating
      7. + FB consistency masking
      8. Full pipeline (all + tuned HPs, UniMatch flow)
    """
    detections = load_detections_aicity(
        det_file,
        roi_file=str(Path(det_file).parent.parent / "roi.jpg"),
        nms_iou=0.5
    )
    gt_tracks = load_gt_aicity(gt_file)

    # Flow dirs for each ablation step
    flow_farn     = _get_flow_dir(seq_dir, "farneback")
    flow_raft     = _get_flow_dir(seq_dir, "raft")
    flow_best     = str(seq_dir / "flow") if (seq_dir / "flow").exists() else None

    configs = [
        # (label, TrackerClass, hps, flow_dir_to_use)
        ("Baseline (IoU only)",
            IoUTracker,   {'iou_threshold': 0.5, 'max_age': 1, 'min_hits': 1},
            None),
        ("+ Kalman Filter",
            KalmanTracker, {'iou_threshold': 0.5, 'max_age': 5, 'min_hits': 2},
            None),
        ("+ OF Warping (Farneback)",
            OFTracker,    {'iou_threshold': 0.5, 'max_age': 5, 'min_hits': 2,
                           'flow_aggregation': 'median', 'flow_threshold': 0},
            flow_farn),
        ("+ OF Warping (RAFT)",
            OFTracker,    {'iou_threshold': 0.5, 'max_age': 5, 'min_hits': 2,
                           'flow_aggregation': 'median', 'flow_threshold': 0},
            flow_raft),
        ("+ OF Warping (UniMatch)",
            OFTracker,    {'iou_threshold': 0.5, 'max_age': 5, 'min_hits': 2,
                           'flow_aggregation': 'median', 'flow_threshold': 0},
            flow_best),
        ("+ Flow Threshold Gate",
            OFTracker,    {'iou_threshold': 0.5, 'max_age': 5, 'min_hits': 2,
                           'flow_threshold': 1.5},
            flow_best),
        ("+ FB Consistency Mask",
            OFTracker,    {'iou_threshold': 0.5, 'max_age': 5, 'min_hits': 2,
                           'flow_threshold': 1.5, 'use_fb_consistency': True},
            flow_best),
        ("Full Pipeline (AdaptiveOF + tuned HPs)",
            AdaptiveOFTracker, best_hps,
            flow_best),
    ]

    rows = []
    for name, TrackerClass, hps, f_dir in configs:
        if f_dir is None and TrackerClass in (OFTracker, AdaptiveOFTracker):
            # Flow dir missing — skip with warning
            print(f"  {name:45s}  [SKIP — flow dir not found]")
            continue
        tracker = TrackerClass(**hps)
        pred_tracks, _ = run_tracker(tracker, detections, gt_tracks, f_dir)
        metrics = compute_mot_metrics(gt_tracks, pred_tracks)
        row = {'config': name, **metrics}
        rows.append(row)
        print(f"  {name:45s}  IDF1={metrics['idf1']*100:.1f}%  "
              f"Switches={metrics['num_switches']}")

    df = pd.DataFrame(rows)
    df.to_csv(str(output_dir / "ablation_study.csv"), index=False)

    # ── Plot ablation bar chart ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(df)))

    for ax, col, ylabel in zip(axes,
                                ['idf1', 'num_switches'],
                                ['IDF1 (%)', 'ID Switches']):
        vals = df[col].values * (100 if col == 'idf1' else 1)
        bars = ax.bar(range(len(df)), vals, color=colors,
                      edgecolor='white', linewidth=0.5, zorder=3)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['config'].tolist(), rotation=45,
                           ha='right', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(f'Ablation: {ylabel}', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.set_facecolor('#F8FAFC')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f'{val:.1f}' + ('%' if col == 'idf1' else ''),
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.suptitle('Ablation Study: Component-wise Contribution',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(output_dir / "ablation_study.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Ablation saved: {output_dir}/ablation_study.png")
    return df


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    seq_dir  = Path(args.seq_dir)
    det_file = str(seq_dir / "det" / "det.txt")
    gt_file  = str(seq_dir / "gt" / "gt.txt")
    img_dir  = seq_dir / "img1"

    # Use flow/ symlink which points to best method (UniMatch after T1.1)
    flow_dir = str(seq_dir / "flow") if (seq_dir / "flow").exists() else None

    out_dir = Path(args.output_dir) / seq_dir.parent.name / seq_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Task 1.2: Tracking with Optical Flow")
    print(f"Sequence: {seq_dir}")
    print(f"Flow dir: {flow_dir} "
          f"({'→ ' + str(Path(flow_dir).resolve().name) if flow_dir and Path(flow_dir).is_symlink() else ''})")
    print("=" * 60)

    # ── Load best HPs ─────────────────────────────────────────────
    if args.hp_file and Path(args.hp_file).exists():
        with open(args.hp_file) as f:
            all_hps = json.load(f)
        best_hps = all_hps.get('of_tracker', {
            'iou_threshold': args.iou_thresh,
            'max_age': args.max_age,
            'min_hits': args.min_hits,
            'matching': args.matching
        })
    else:
        best_hps = {
            'iou_threshold': args.iou_thresh,
            'max_age': args.max_age,
            'min_hits': args.min_hits,
            'matching': args.matching
        }
    print(f"HPs: {best_hps}")

    detections = load_detections_aicity(
        det_file,
        roi_file=str(seq_dir / "roi.jpg"),
        nms_iou=0.5
    )
    gt_tracks = load_gt_aicity(gt_file)

    # ── Ablation mode ─────────────────────────────────────────────
    if args.ablation:
        print("\nRunning ablation study...")
        run_ablation(seq_dir, det_file, gt_file, best_hps, out_dir)
        return

    # ── Run all four trackers ─────────────────────────────────────
    print("\nRunning trackers...")
    trackers = {
        "iou_baseline": IoUTracker(iou_threshold=0.5, max_age=1, min_hits=1),
        "kalman":       KalmanTracker(iou_threshold=0.5, max_age=5, min_hits=2),
        "of_tracker":   OFTracker(**best_hps),
        "adaptive_of":  AdaptiveOFTracker(**best_hps),
    }

    all_metrics = {}
    all_pred_tracks = {}

    for name, tracker in trackers.items():
        print(f"\n  Running: {name}")
        pred_tracks, raw = run_tracker(tracker, detections, gt_tracks, flow_dir)
        metrics = compute_mot_metrics(gt_tracks, pred_tracks)
        all_metrics[name] = metrics
        all_pred_tracks[name] = (pred_tracks, raw)
        write_mot_results(
            {fid: r for fid, r in raw.items() if len(r) > 0},
            str(out_dir / f"results_{name}.txt")
        )
        print(f"    IDF1={metrics['idf1']*100:.1f}%  MOTA={metrics['mota']*100:.1f}%  "
              f"Switches={metrics['num_switches']}")

    # ── Save metrics table ────────────────────────────────────────
    df = pd.DataFrame([
        {'tracker': k,
         **{m: round(v * 100 if m in ['idf1', 'mota', 'motp', 'precision', 'recall']
                     else v, 2)
            for m, v in met.items() if m != 'tracker'}}
        for k, met in all_metrics.items()
    ])
    df.to_csv(str(out_dir / "metrics_comparison.csv"), index=False)
    print(f"\nMetrics saved: {out_dir}/metrics_comparison.csv")

    # ── Qualitative: same-frame comparison ───────────────────────
    # Use best OF tracker for comparison vs baseline
    _best_of_key = 'adaptive_of' if 'adaptive_of' in all_pred_tracks else 'of_tracker'

    id_switches_no_of = compute_id_switches(
        gt_tracks, all_pred_tracks['iou_baseline'][0]
    )
    id_switches_of = compute_id_switches(
        gt_tracks, all_pred_tracks[_best_of_key][0]
    )

    switch_frames_no_of = {s['frame'] for s in id_switches_no_of}
    switch_frames_of    = {s['frame'] for s in id_switches_of}
    interesting_frames  = switch_frames_no_of - switch_frames_of

    if interesting_frames and img_dir.exists():
        target_frame = sorted(interesting_frames)[len(interesting_frames) // 2]
        print(f"\nGenerating same-frame comparison at frame {target_frame}...")

        img_path = img_dir / f"{target_frame:06d}.jpg"
        if not img_path.exists():
            img_path = img_dir / f"{target_frame:06d}.png"
        if img_path.exists():
            frame = cv2.imread(str(img_path))
            t_no_of = np.array(
                all_pred_tracks['iou_baseline'][1].get(target_frame, []))
            t_of    = np.array(
                all_pred_tracks[_best_of_key][1].get(target_frame, []))

            if len(t_no_of) > 0 and len(t_of) > 0:
                fig = plot_same_frame_comparison(frame, t_no_of, t_of)
                fig.savefig(str(out_dir / "qualitative_comparison.png"),
                            dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  Saved: {out_dir}/qualitative_comparison.png")

    # ── ID switch spatial heatmaps ────────────────────────────────
    if img_dir.exists():
        imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        if imgs:
            bg = cv2.imread(str(imgs[0]))
            H, W = bg.shape[:2]
            for hmap_name, switches in [
                ('iou_baseline', id_switches_no_of),
                (_best_of_key,   id_switches_of)
            ]:
                fig = plot_id_switch_heatmap(
                    [(s['frame'], s['cx'], s['cy']) for s in switches],
                    (H, W),
                    title=f"ID Switch Heatmap – {hmap_name} ({len(switches)} switches)"
                )
                fig.savefig(str(out_dir / f"id_switch_heatmap_{hmap_name}.png"),
                            dpi=150, bbox_inches='tight')
                plt.close()

    print(f"\nAll outputs saved to: {out_dir}")
    print("Done! ✓")


if __name__ == "__main__":
    main()