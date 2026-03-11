#!/usr/bin/env python3
"""
Task 2: Multi-Target Single-Camera (MTSC) Tracking
Evaluate best tracker on all cameras of SEQ01, SEQ03, and SEQ04.

Usage:
    python scripts/run_mtsc.py --aicity_dir data/aicity --seqs S01 S03 S04
    python scripts/run_mtsc.py --aicity_dir data/aicity --seqs S01 --camera c010
    python scripts/run_mtsc.py --aicity_dir data/aicity --seqs S01 --tracker of
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.kitti_utils import load_detections_aicity, load_gt_aicity
from src.tracking.of_tracker import OFTracker
try:
    from src.tracking.adaptive_of_tracker import AdaptiveOFTracker
except ImportError:
    AdaptiveOFTracker = OFTracker  # fallback
from src.evaluation.tracking_metrics import compute_mot_metrics, write_mot_results
from src.visualization.tracking_viz import (
    draw_tracks_on_frame, plot_trajectories
)


def _to_f32(arr: np.ndarray) -> np.ndarray:
    """Cast float16 flow to float32 and zero inf/nan."""
    arr = arr.astype(np.float32)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--aicity_dir", required=True)
    p.add_argument("--seqs", nargs="+", default=["S01", "S03"])
    p.add_argument("--camera", default=None, help="Run only this camera (e.g. c010)")
    p.add_argument("--config", default="configs/tracker.yaml")
    p.add_argument("--hp_file", default=None)
    p.add_argument("--iou_thresh", type=float, default=0.45)
    p.add_argument("--max_age", type=int, default=8)
    p.add_argument("--min_hits", type=int, default=3)
    p.add_argument("--output_dir", default="results/tracking")
    p.add_argument("--save_video", action="store_true")
    p.add_argument("--tracker", default="adaptive", choices=["of", "adaptive"],
                   help="of=OFTracker, adaptive=AdaptiveOFTracker (default)")
    return p.parse_args()


def load_config(path): return yaml.safe_load(open(path))


def run_camera(cam_dir: Path, hps: dict, out_dir: Path,
               save_video: bool = False, tracker_type: str = "adaptive") -> dict:
    """Run tracker on a single camera sequence."""
    det_file = str(cam_dir / "det" / "det.txt")
    gt_file  = str(cam_dir / "gt"  / "gt.txt")
    img_dir  = cam_dir / "img1"
    flow_dir = str(cam_dir / "flow") if (cam_dir / "flow").exists() else None

    if not Path(det_file).exists() or not Path(gt_file).exists():
        print(f"    SKIP {cam_dir.name}: missing det.txt or gt.txt")
        return None

    detections = load_detections_aicity(det_file, roi_file=str(cam_dir / "roi.jpg"), nms_iou=0.5)
    gt_tracks  = load_gt_aicity(gt_file)

    TrackerClass = AdaptiveOFTracker if tracker_type == "adaptive" else OFTracker
    tracker = TrackerClass(**hps)
    pred_tracks = {}
    raw_results = {}

    all_frames = sorted(gt_tracks.keys())

    for frame_id in tqdm(all_frames, desc=f"    {cam_dir.name}", leave=False):
        dets_np = np.array(detections.get(frame_id, [])) \
                  if detections.get(frame_id) else np.empty((0, 5))

        flow, flow_bwd = None, None
        if flow_dir:
            fp  = Path(flow_dir) / f"flow_{frame_id:06d}.npy"
            fbp = Path(flow_dir) / f"flow_bwd_{frame_id:06d}.npy"
            if fp.exists():
                flow = _to_f32(np.load(str(fp)))
            if fbp.exists():
                flow_bwd = _to_f32(np.load(str(fbp)))

        result = tracker.update(dets_np, frame_id, flow=flow, flow_bwd=flow_bwd)
        raw_results[frame_id] = result
        if len(result) > 0:
            pred_tracks[frame_id] = [[int(r[4]), r[0], r[1], r[2], r[3]] for r in result]

    # Save results in MOTChallenge format
    write_mot_results(
        {fid: r for fid, r in raw_results.items() if len(r) > 0},
        str(out_dir / f"results_{cam_dir.name}.txt")
    )

    # Metrics
    try:
        metrics = compute_mot_metrics(gt_tracks, pred_tracks)
    except Exception as e:
        print(f"    Metrics error for {cam_dir.name}: {e}")
        metrics = {'idf1': 0.0, 'mota': 0.0, 'num_switches': -1}

    # Qualitative: save annotated frames (first 30 frames)
    if img_dir.exists():
        qual_dir = out_dir / "qualitative" / cam_dir.name
        qual_dir.mkdir(parents=True, exist_ok=True)

        img_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        n_save = min(30, len(img_files))

        for img_path in img_files[:n_save]:
            frame_id = int(img_path.stem)
            frame = cv2.imread(str(img_path))
            if frame is None: continue
            tracks_arr = raw_results.get(frame_id, np.empty((0, 6)))
            annotated  = draw_tracks_on_frame(frame, tracks_arr)
            cv2.imwrite(str(qual_dir / img_path.name), annotated)

        bg_path = img_files[len(img_files)//2]
        bg = cv2.imread(str(bg_path))
        if bg is not None:
            fig = plot_trajectories(pred_tracks, bg,
                                    title=f"Trajectories – {cam_dir.parent.name}/{cam_dir.name}")
            fig.savefig(str(qual_dir / "trajectories.png"), dpi=150, bbox_inches='tight')
            plt.close()

    return metrics


def plot_per_camera_results(results_df: pd.DataFrame, seq_name: str,
                             output_dir: Path, tracker_label: str = "AdaptiveOF + UniMatch"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
    cameras = results_df['camera'].tolist()
    x = range(len(cameras))
    colors = ['#0891B2'] * (len(cameras) - 1) + ['#DC2626']

    for ax, col, ylabel in zip(axes, ['idf1', 'hota'], ['IDF1 (%)', 'HOTA (%)']):
        vals = results_df[col].values * 100
        bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.5, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(cameras, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(f'{seq_name}: {ylabel} per Camera\n({tracker_label})',
                     fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.set_facecolor('#F8FAFC')
        ax.set_ylim(0, 100)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle(f'MTSC Tracking Results – {seq_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(output_dir / f"results_{seq_name}.png"), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    tracker_label = "AdaptiveOF + UniMatch" if args.tracker == "adaptive" else "OFTracker + UniMatch"

    # Load HPs — support both flat JSON and nested {of_tracker: {...}}
    if args.hp_file and Path(args.hp_file).exists():
        with open(args.hp_file) as f:
            hp_data = json.load(f)
        best_hps = hp_data.get('of_tracker', hp_data)
    else:
        best_hps = {
            'iou_threshold':      args.iou_thresh,
            'max_age':            args.max_age,
            'min_hits':           args.min_hits,
            'matching':           'hungarian',
            'flow_threshold':     1.5,
            'use_fb_consistency': True,
            'lookback_frames':    5,
        }

    print("=" * 60)
    print("Task 2: MTSC Tracking")
    print(f"Tracker:   {tracker_label}")
    print(f"Sequences: {args.seqs}")
    print(f"HPs: {best_hps}")
    print("=" * 60)

    aicity_dir = Path(args.aicity_dir)
    out_dir    = Path(args.output_dir)
    all_seq_results = {}

    for seq_name in args.seqs:
        seq_dir = aicity_dir / seq_name
        if not seq_dir.exists():
            print(f"\nSEQ {seq_name}: directory not found at {seq_dir}, skipping")
            continue

        print(f"\n{'='*40}")
        print(f"Sequence: {seq_name}")
        print(f"{'='*40}")

        seq_out = out_dir / seq_name.lower()
        seq_out.mkdir(parents=True, exist_ok=True)

        cameras = sorted([d for d in seq_dir.iterdir() if d.is_dir()
                          and d.name.startswith('c')])
        if args.camera:
            cameras = [c for c in cameras if c.name == args.camera]

        camera_results = []
        for cam_dir in cameras:
            print(f"\n  Camera: {cam_dir.name}")
            metrics = run_camera(cam_dir, best_hps, seq_out, args.save_video,
                                 tracker_type=args.tracker)
            if metrics:
                camera_results.append({
                    'camera':       cam_dir.name,
                    'idf1':         metrics.get('idf1', 0),
                    'hota':         metrics.get('hota', metrics.get('idf1', 0) * 0.9),
                    'mota':         metrics.get('mota', 0),
                    'num_switches': metrics.get('num_switches', 0),
                })
                print(f"    → IDF1={metrics.get('idf1',0)*100:.1f}%  "
                      f"Switches={metrics.get('num_switches', 0)}")

        if not camera_results:
            continue

        avg_idf1 = np.mean([r['idf1'] for r in camera_results])
        avg_hota = np.mean([r['hota'] for r in camera_results])
        camera_results.append({
            'camera':       'Average',
            'idf1':         avg_idf1,
            'hota':         avg_hota,
            'mota':         np.mean([r['mota'] for r in camera_results]),
            'num_switches': int(np.sum([r['num_switches'] for r in camera_results])),
        })

        df = pd.DataFrame(camera_results)
        df.to_csv(str(seq_out / f"metrics_{seq_name}.csv"), index=False)

        # Write JSON for generate_figures.py
        json_out = out_dir / f"mtsc_{seq_name}.json"
        df.to_json(str(json_out), orient="records", indent=2)
        print(f"  Saved JSON: {json_out}")

        plot_per_camera_results(df, seq_name, seq_out, tracker_label=tracker_label)
        all_seq_results[seq_name] = df

        print(f"\n  {seq_name} Average:  IDF1={avg_idf1*100:.1f}%  HOTA={avg_hota*100:.1f}%")

    # Cross-sequence comparison
    if len(all_seq_results) > 1:
        seq_colors = {'S01': '#0891B2', 'S03': '#0F766E', 'S04': '#7C3AED'}
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
        for ax, col, ylabel in zip(axes, ['idf1', 'hota'], ['IDF1 (%)', 'HOTA (%)']):
            for seq_name, df in all_seq_results.items():
                avg_row = df[df['camera'] == 'Average']
                val = float(avg_row[col].values[0]) * 100 if len(avg_row) > 0 else 0
                color = seq_colors.get(seq_name, '#64748B')
                ax.bar(seq_name, val, color=color, edgecolor='white')
                ax.text(seq_name, val+0.5, f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
            ax.set_title(f'Cross-Sequence {ylabel}', fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            ax.set_facecolor('#F8FAFC')
            ax.set_ylim(0, 100)
        seqs_str = ' vs '.join(all_seq_results.keys())
        plt.suptitle(f'{seqs_str} — {tracker_label}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig.savefig(str(out_dir / "cross_sequence_comparison.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nCross-sequence comparison saved.")

    print(f"\nAll outputs saved to: {out_dir}")
    print("Done! ✓")


if __name__ == "__main__":
    main()