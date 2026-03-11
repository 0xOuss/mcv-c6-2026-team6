#!/usr/bin/env python3
"""
scripts/run_hp_sweep.py

Hyperparameter sweep for all trackers.

Sweeps in order (sequential: each HP tuned with others at current best):
  1. iou_threshold  → find best
  2. max_age        → find best (with best iou)
  3. min_hits       → find best (with best iou + age)

Also compares Hungarian vs Greedy matching.

Outputs (for slides):
  results/tracking/hp_sweep/{tracker}_iou_sweep_hungarian.csv
  results/tracking/hp_sweep/{tracker}_iou_sweep_greedy.csv
  results/tracking/hp_sweep/{tracker}_age_sweep.csv
  results/tracking/hp_sweep/{tracker}_hits_sweep.csv
  results/tracking/hp_sweep/hp_sensitivity_{tracker}.png
  results/tracking/hp_sweep/matching_comparison.png
  results/tracking/hp_sweep/best_hyperparameters.json  ← load in run_tracking.py

Usage:
    python scripts/run_hp_sweep.py --seq_dir data/aicity/S01/c010
    python scripts/run_hp_sweep.py --seq_dir data/aicity/S01/c010 --tracker adaptive_of
    python scripts/run_hp_sweep.py --seq_dir data/aicity/S01/c010 --fast
"""

import sys
import argparse
import json
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.kitti_utils import load_detections_aicity, load_gt_aicity
from src.evaluation.tracking_metrics  import compute_mot_metrics
from src.tracking.iou_tracker          import IoUTracker
from src.tracking.kalman_tracker        import KalmanTracker
from src.tracking.of_tracker            import OFTracker
from src.tracking.adaptive_of_tracker   import AdaptiveOFTracker


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq_dir", required=True)
    p.add_argument("--tracker", default="all",
                   choices=["iou", "kalman", "of", "adaptive_of", "all"])
    p.add_argument("--output_dir", default="results/tracking/hp_sweep")
    p.add_argument("--fast", action="store_true",
                   help="Use coarse grid (for testing)")
    return p.parse_args()


TRACKER_MAP = {
    "iou":         IoUTracker,
    "kalman":      KalmanTracker,
    "of":          OFTracker,
    "adaptive_of": AdaptiveOFTracker,
}

TRACKER_EXTRA_HPS = {
    "of":          {"flow_aggregation": "median", "flow_threshold": 1.5,
                    "use_fb_consistency": False},
    "adaptive_of": {"use_fb_consistency": True, "fb_threshold": 1.0,
                    "lookback_frames": 5},
}


def run_one(TrackerClass, hps: dict, detections: dict, gt_tracks: dict,
             flow_dir: Optional[Path] = None) -> dict:
    """Run tracker with given HPs and return metrics."""
    tracker = TrackerClass(**hps)
    pred_tracks = {}

    for frame_id in sorted(set(detections.keys()) | set(gt_tracks.keys())):
        dets = detections.get(frame_id, [])
        dets_np = np.array(dets) if dets else np.empty((0, 5))

        flow, flow_bwd = None, None
        if flow_dir and flow_dir.exists():
            fwd = flow_dir / f"flow_fwd_{frame_id:06d}.npy"
            bwd = flow_dir / f"flow_bwd_{frame_id:06d}.npy"
            if fwd.exists(): flow     = np.load(str(fwd))
            if bwd.exists(): flow_bwd = np.load(str(bwd))

        if isinstance(tracker, (OFTracker, AdaptiveOFTracker)):
            result = tracker.update(dets_np, frame_id, flow=flow, flow_bwd=flow_bwd)
        else:
            result = tracker.update(dets_np, frame_id)

        if len(result) > 0:
            pred_tracks[frame_id] = [
                [int(r[4]), r[0], r[1], r[2], r[3]] for r in result
            ]

    try:
        return compute_mot_metrics(gt_tracks, pred_tracks)
    except Exception:
        return {"idf1": 0.0, "mota": 0.0, "num_switches": 9999}


def sweep_param(TrackerClass, param_name: str, values: list,
                fixed_hps: dict, detections: dict, gt_tracks: dict,
                flow_dir: Optional[Path]) -> pd.DataFrame:
    rows = []
    for val in tqdm(values, desc=f"  {param_name}", leave=False):
        hps = {**fixed_hps, param_name: val}
        m = run_one(TrackerClass, hps, detections, gt_tracks, flow_dir)
        rows.append({param_name: val, "idf1": m.get("idf1", 0),
                     "mota": m.get("mota", 0),
                     "num_switches": m.get("num_switches", 0)})
    return pd.DataFrame(rows)


def plot_sensitivity(dfs: dict, out_dir: Path, tracker_name: str):
    """One subplot per HP, all on same figure."""
    items = [(k, v) for k, v in dfs.items() if v is not None]
    n = len(items)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), facecolor="white")
    if n == 1:
        axes = [axes]

    colors = ["#0891B2", "#0F766E", "#7C3AED"]
    xlabels = {"iou_threshold": "IoU Threshold",
               "max_age": "Max Age (frames)",
               "min_hits": "Min Hits"}

    for ax, (param, df), color in zip(axes, items, colors):
        if param not in df.columns:
            ax.axis("off")
            continue

        x = df[param].values
        y = df["idf1"].values * 100

        ax.plot(x, y, "o-", color=color, linewidth=2.5, markersize=7,
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=2)
        ax.fill_between(x, y * 0.97, y * 1.03, alpha=0.12, color=color)

        best_i = int(np.argmax(y))
        ax.axvline(x[best_i], color="#DC2626", linestyle="--", linewidth=2,
                   label=f"Best: {x[best_i]}")
        ax.scatter([x[best_i]], [y[best_i]], s=120, color="#DC2626", zorder=5)

        ax.set_xlabel(xlabels.get(param, param), fontsize=11, fontweight="bold")
        ax.set_ylabel("IDF1 (%)", fontsize=11, fontweight="bold")
        ax.set_title(f"{xlabels.get(param, param)} Sensitivity", fontsize=11,
                     fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_facecolor("#F8FAFC")
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle(f"HP Sensitivity — {tracker_name}\n"
                 f"(Each HP swept with others fixed at current best)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / f"hp_sensitivity_{tracker_name}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_matching_comparison(df_hungarian: pd.DataFrame,
                              df_greedy: pd.DataFrame,
                              out_dir: Path):
    """IDF1 and ID switches: Hungarian vs Greedy across IoU thresholds."""
    if df_hungarian is None or df_greedy is None:
        return
    if "iou_threshold" not in df_hungarian.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
    for ax, metric, ylabel, mult in zip(
        axes,
        ["idf1", "num_switches"],
        ["IDF1 (%)", "ID Switches"],
        [100, 1],
    ):
        x  = df_hungarian["iou_threshold"].values
        yh = df_hungarian[metric].values * mult
        yg = df_greedy[metric].values * mult

        ax.plot(x, yh, "o-", color="#0891B2", label="Hungarian", lw=2.5, ms=6)
        ax.plot(x, yg, "s--", color="#DC2626", label="Greedy", lw=2.5, ms=6)
        delta = yh - yg
        ax.fill_between(x, yh, yg, where=delta > 0, alpha=0.1, color="#0891B2")
        ax.fill_between(x, yh, yg, where=delta < 0, alpha=0.1, color="#DC2626")
        ax.set_xlabel("IoU Threshold", fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_title(f"Hungarian vs Greedy: {ylabel}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_facecolor("#F8FAFC")
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle(
        "Matching Strategy: Hungarian (globally optimal) vs Greedy (locally greedy)\n"
        "Hungarian = scipy.optimize.linear_sum_assignment  |  Greedy = highest-IoU-first",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    out_path = out_dir / "matching_comparison.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    args    = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_dir = Path(args.seq_dir)

    det_file = str(seq_dir / "det" / "det.txt")
    gt_file  = str(seq_dir / "gt"  / "gt.txt")
    flow_dir = seq_dir / "flow_raft"

    detections = load_detections_aicity(det_file)
    gt_tracks  = load_gt_aicity(gt_file)

    # Grid values
    if args.fast:
        iou_vals  = [0.2, 0.35, 0.5, 0.65, 0.8]
        age_vals  = [1, 3, 5, 10, 20]
        hits_vals = [1, 2, 3, 5]
    else:
        iou_vals  = [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]
        age_vals  = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30]
        hits_vals = [1, 2, 3, 4, 5]

    trackers_to_sweep = (
        list(TRACKER_MAP.keys()) if args.tracker == "all"
        else [args.tracker]
    )

    best_hps_all = {}

    for tname in trackers_to_sweep:
        TrackerClass = TRACKER_MAP[tname]
        extra = TRACKER_EXTRA_HPS.get(tname, {})
        print(f"\n{'='*50}\nSweeping: {tname}\n{'='*50}")

        # Skip flow-dependent trackers if no flow available
        if tname in ("of", "adaptive_of") and not flow_dir.exists():
            print(f"  WARNING: {flow_dir} not found.")
            print("  Run: python scripts/precompute_flow.py first")
            print("  Skipping flow-based trackers.")
            continue

        fd = flow_dir if tname in ("of", "adaptive_of") else None

        # Base HPs
        base_hps = {"iou_threshold": 0.5, "max_age": 5, "min_hits": 2,
                    "matching": "hungarian", **extra}

        # 1. IoU sweep (both matching strategies)
        print("[1/3] IoU threshold sweep (Hungarian)...")
        df_iou_h = sweep_param(TrackerClass, "iou_threshold", iou_vals,
                                base_hps, detections, gt_tracks, fd)
        df_iou_h.to_csv(str(out_dir / f"{tname}_iou_sweep_hungarian.csv"), index=False)

        print("[1/3] IoU threshold sweep (Greedy)...")
        base_greedy = {**base_hps, "matching": "greedy"}
        df_iou_g = sweep_param(TrackerClass, "iou_threshold", iou_vals,
                                base_greedy, detections, gt_tracks, fd)
        df_iou_g.to_csv(str(out_dir / f"{tname}_iou_sweep_greedy.csv"), index=False)

        best_iou = float(df_iou_h.loc[df_iou_h["idf1"].idxmax(), "iou_threshold"])
        print(f"  Best IoU: {best_iou}")

        # 2. max_age sweep
        print("[2/3] max_age sweep...")
        hps2 = {**base_hps, "iou_threshold": best_iou}
        df_age = sweep_param(TrackerClass, "max_age", age_vals,
                             hps2, detections, gt_tracks, fd)
        df_age.to_csv(str(out_dir / f"{tname}_age_sweep.csv"), index=False)
        best_age = int(df_age.loc[df_age["idf1"].idxmax(), "max_age"])
        print(f"  Best max_age: {best_age}")

        # 3. min_hits sweep
        print("[3/3] min_hits sweep...")
        hps3 = {**base_hps, "iou_threshold": best_iou, "max_age": best_age}
        df_hits = sweep_param(TrackerClass, "min_hits", hits_vals,
                              hps3, detections, gt_tracks, fd)
        df_hits.to_csv(str(out_dir / f"{tname}_hits_sweep.csv"), index=False)
        best_hits = int(df_hits.loc[df_hits["idf1"].idxmax(), "min_hits"])
        print(f"  Best min_hits: {best_hits}")

        best_hps = {
            "iou_threshold": best_iou,
            "max_age":       best_age,
            "min_hits":      best_hits,
            "matching":      "hungarian",
            **extra,
        }
        best_hps_all[f"{tname}_tracker" if tname not in ("iou",) else "iou_baseline"] = best_hps
        best_hps_all[tname] = best_hps
        print(f"  Best HPs: {best_hps}")

        # Plots
        plot_sensitivity(
            {"iou_threshold": df_iou_h, "max_age": df_age, "min_hits": df_hits},
            out_dir, tname
        )
        plot_matching_comparison(df_iou_h, df_iou_g, out_dir)

    # Save best HPs
    best_path = out_dir / "best_hyperparameters.json"
    with open(str(best_path), "w") as f:
        json.dump(best_hps_all, f, indent=2)
    print(f"\nBest HPs saved: {best_path}")
    print("\nUse in run_tracking.py:")
    print(f"  python scripts/run_tracking.py --seq_dir {args.seq_dir} "
          f"--hp_file {best_path}")


if __name__ == "__main__":
    main()