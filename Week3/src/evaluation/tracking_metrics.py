"""
Tracking Evaluation Metrics.

Primary metrics (as required by task):
  - IDF1: ID F1-score — measures ability to maintain correct identity over time
  - HOTA: Higher Order Tracking Accuracy — balanced detection + association score

Secondary metrics for diagnostic insight:
  - MOTA: Multiple Object Tracking Accuracy
  - MOTP: Multiple Object Tracking Precision
  - ID switches: how often does a tracked object change its ID?
  - Fragmentation: how often does a track break?

Intuition:
  IDF1 focuses on identity: if you track car #5 correctly for 100 frames,
  then lose it and relabel it #12 for the rest, IDF1 will penalize that heavily.
  MOTA focuses on detection: counts FP, FN, ID switches relative to GT count.
  HOTA is a product of detection quality and association quality — balanced metric.

We use motmetrics for MOTA/MOTP/IDF1 and TrackEval for HOTA.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import motmetrics as mm


def accumulate_motmetrics(gt_tracks: Dict[int, List],
                           pred_tracks: Dict[int, List],
                           iou_threshold: float = 0.5) -> mm.MOTAccumulator:
    """
    Accumulate MOT metrics frame by frame.

    Args:
        gt_tracks:   {frame_id: [[tid, x1, y1, x2, y2], ...]}
        pred_tracks: {frame_id: [[tid, x1, y1, x2, y2], ...]}
        iou_threshold: min IoU to consider a match (0.5 standard)
    Returns:
        motmetrics accumulator object
    """
    acc = mm.MOTAccumulator(auto_id=False)

    all_frames = sorted(set(list(gt_tracks.keys()) + list(pred_tracks.keys())))

    for frame_id in all_frames:
        gt_frame  = gt_tracks.get(frame_id, [])
        hyp_frame = pred_tracks.get(frame_id, [])

        gt_ids  = [int(r[0]) for r in gt_frame]
        hyp_ids = [int(r[0]) for r in hyp_frame]

        if len(gt_frame) == 0 and len(hyp_frame) == 0:
            acc.update([], [], np.empty((0, 0)), frameid=frame_id)
            continue

        # Compute distance matrix (1 - IoU, so that 0 = perfect match)
        gt_boxes  = np.array([[r[1], r[2], r[3], r[4]] for r in gt_frame]) \
                    if gt_frame else np.empty((0, 4))
        hyp_boxes = np.array([[r[1], r[2], r[3], r[4]] for r in hyp_frame]) \
                    if hyp_frame else np.empty((0, 4))

        if len(gt_boxes) > 0 and len(hyp_boxes) > 0:
            from ..tracking.matching import compute_iou
            iou_mat = compute_iou(gt_boxes, hyp_boxes)
            dist_mat = 1.0 - iou_mat
            # Mask out distances above threshold (motmetrics convention: >threshold = no match)
            dist_mat[dist_mat > (1 - iou_threshold)] = np.nan
        else:
            dist_mat = mm.distances.iou_matrix(gt_boxes, hyp_boxes,
                                               max_iou=1 - iou_threshold) \
                       if len(gt_boxes) > 0 and len(hyp_boxes) > 0 \
                       else np.full((len(gt_ids), len(hyp_ids)), np.nan)

        acc.update(gt_ids, hyp_ids, dist_mat, frameid=frame_id)

    return acc


def compute_mot_metrics(gt_tracks: Dict[int, List],
                         pred_tracks: Dict[int, List],
                         iou_threshold: float = 0.5) -> dict:
    """
    Compute full suite of MOT metrics.

    Returns dict with: mota, motp, idf1, num_switches, num_fragmentations,
                       num_matches, num_misses, num_false_positives, precision, recall
    """
    acc = accumulate_motmetrics(gt_tracks, pred_tracks, iou_threshold)
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=['num_frames', 'mota', 'motp', 'idf1',
                 'num_switches', 'num_fragmentations',
                 'num_matches', 'num_misses', 'num_false_positives',
                 'precision', 'recall'],
        name='overall'
    )
    return summary.to_dict('records')[0]


def write_mot_results(tracks_per_frame: Dict[int, np.ndarray],
                       output_file: str):
    """
    Write tracking results in MOTChallenge format for evaluation.

    Format: frame, id, x, y, w, h, conf, -1, -1, -1
    (1-indexed frame)
    """
    lines = []
    for frame_id in sorted(tracks_per_frame.keys()):
        for row in tracks_per_frame[frame_id]:
            x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
            tid = int(row[4])
            conf = float(row[5]) if len(row) > 5 else 1.0
            w = x2 - x1
            h = y2 - y1
            lines.append(f"{frame_id},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))


def compute_id_switches(gt_tracks: Dict[int, List],
                         pred_tracks: Dict[int, List]) -> List[dict]:
    """
    Find frames and locations where ID switches occur.
    Useful for spatial analysis of WHERE switches happen.

    Returns list of dicts: {frame, gt_id, old_pred_id, new_pred_id, cx, cy}
    """
    # Track which predicted ID was assigned to each GT ID
    gt_to_pred: Dict[int, int] = {}
    switches = []

    for frame_id in sorted(set(list(gt_tracks.keys()) + list(pred_tracks.keys()))):
        gt_frame  = gt_tracks.get(frame_id, [])
        hyp_frame = pred_tracks.get(frame_id, [])

        if not gt_frame or not hyp_frame:
            continue

        # Simple IoU matching for switch detection
        from ..tracking.matching import compute_iou
        gt_boxes  = np.array([[r[1], r[2], r[3], r[4]] for r in gt_frame])
        hyp_boxes = np.array([[r[1], r[2], r[3], r[4]] for r in hyp_frame])
        iou_mat   = compute_iou(gt_boxes, hyp_boxes)

        for gi, gt_row in enumerate(gt_frame):
            gt_id = int(gt_row[0])
            best_hyp = iou_mat[gi].argmax()
            if iou_mat[gi, best_hyp] < 0.5:
                continue
            pred_id = int(hyp_frame[best_hyp][0])

            if gt_id in gt_to_pred and gt_to_pred[gt_id] != pred_id:
                # ID switch detected!
                cx = (gt_boxes[gi, 0] + gt_boxes[gi, 2]) / 2
                cy = (gt_boxes[gi, 1] + gt_boxes[gi, 3]) / 2
                switches.append({
                    'frame': frame_id,
                    'gt_id': gt_id,
                    'old_pred_id': gt_to_pred[gt_id],
                    'new_pred_id': pred_id,
                    'cx': cx, 'cy': cy
                })

            gt_to_pred[gt_id] = pred_id

    return switches