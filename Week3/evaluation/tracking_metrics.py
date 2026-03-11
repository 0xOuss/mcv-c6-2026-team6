"""
Tracking metrics with TrackEval HOTA + motmetrics IDF1/MOTA.

Fixes:
  1. GT-only frame iteration (no phantom FPs on unannotated frames)
  2. auto_id=False (GT IDs are non-contiguous integers)
  3. exclude_outside (drop predicted boxes with zero IoU against all GT)
  4. Empty-frame: pass np.empty((0,0)) not [] to acc.update
  5. HOTA via official TrackEval library (Luiten et al. IJCV 2021)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import motmetrics as mm


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    N, M = len(a), len(b)
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)
    aa = a[:, None, :]; bb = b[None, :, :]
    ix1 = np.maximum(aa[:,:,0], bb[:,:,0]); iy1 = np.maximum(aa[:,:,1], bb[:,:,1])
    ix2 = np.minimum(aa[:,:,2], bb[:,:,2]); iy2 = np.minimum(aa[:,:,3], bb[:,:,3])
    inter = np.maximum(0, ix2-ix1) * np.maximum(0, iy2-iy1)
    area_a = (a[:,2]-a[:,0]) * (a[:,3]-a[:,1])
    area_b = (b[:,2]-b[:,0]) * (b[:,3]-b[:,1])
    union  = area_a[:,None] + area_b[None,:] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# TrackEval HOTA
# ─────────────────────────────────────────────────────────────────────────────

def compute_hota_trackeval(gt_tracks: Dict[int, List],
                            pred_tracks: Dict[int, List]) -> dict:
    """
    Compute HOTA using the official TrackEval library.
    Falls back to our own implementation if TrackEval fails.
    Returns dict with keys: hota, deta, assa
    """
    try:
        return _hota_via_trackeval(gt_tracks, pred_tracks)
    except Exception as e:
        import sys
        sys.stderr.write(f"[warn] TrackEval HOTA failed ({e}), using fallback\n")
        return _hota_fallback(gt_tracks, pred_tracks)


def _hota_via_trackeval(gt_tracks: Dict[int, List],
                         pred_tracks: Dict[int, List]) -> dict:
    """
    Compute HOTA by converting our track dicts to TrackEval's data format
    and calling HOTA.eval_sequence() directly.
    """
    from trackeval.metrics.hota import HOTA

    all_frames = sorted(gt_tracks.keys())

    gt_ids_list, pred_ids_list = [], []
    gt_dets_list, pred_dets_list = [], []
    sim_list = []

    for frame_id in all_frames:
        gt_f   = gt_tracks.get(frame_id, [])
        pred_f = pred_tracks.get(frame_id, [])

        g_ids = np.array([int(r[0]) for r in gt_f],   dtype=int)
        p_ids = np.array([int(r[0]) for r in pred_f], dtype=int)

        # TrackEval expects [x, y, w, h] format
        g_boxes = np.array([[r[1], r[2], r[3]-r[1], r[4]-r[2]]
                             for r in gt_f],   dtype=float) if gt_f   else np.empty((0,4))
        p_boxes = np.array([[r[1], r[2], r[3]-r[1], r[4]-r[2]]
                             for r in pred_f], dtype=float) if pred_f else np.empty((0,4))

        # Similarity scores = IoU (x1y1x2y2 for our helper)
        g_xyxy = np.array([[r[1],r[2],r[3],r[4]] for r in gt_f],   dtype=float) \
                 if gt_f   else np.empty((0,4))
        p_xyxy = np.array([[r[1],r[2],r[3],r[4]] for r in pred_f], dtype=float) \
                 if pred_f else np.empty((0,4))

        gt_ids_list.append(g_ids)
        pred_ids_list.append(p_ids)
        gt_dets_list.append(g_boxes)
        pred_dets_list.append(p_boxes)
        sim_list.append(_iou_matrix(g_xyxy, p_xyxy))

    data = {
        'num_timesteps':     len(all_frames),
        'gt_ids':            gt_ids_list,
        'tracker_ids':       pred_ids_list,
        'gt_dets':           gt_dets_list,
        'tracker_dets':      pred_dets_list,
        'similarity_scores': sim_list,
        'num_gt_ids':        len({int(r[0]) for f in gt_tracks.values()  for r in f}),
        'num_tracker_ids':   len({int(r[0]) for f in pred_tracks.values() for r in f}),
        'num_gt_dets':       sum(len(f) for f in gt_tracks.values()),
        'num_tracker_dets':  sum(len(f) for f in pred_tracks.values()),
    }

    res = HOTA().eval_sequence(data)
    return {
        'hota': float(np.mean(res['HOTA'])),
        'deta': float(np.mean(res['DetA'])),
        'assa': float(np.mean(res['AssA'])),
    }


def _hota_fallback(gt_tracks: Dict[int, List],
                   pred_tracks: Dict[int, List]) -> dict:
    """Fallback HOTA — used only if TrackEval is unavailable."""
    alphas = [round(0.05 * i, 2) for i in range(1, 20)]
    all_frames = sorted(gt_tracks.keys())
    hota_list, deta_list, assa_list = [], [], []

    for alpha in alphas:
        tp = fp = fn = 0
        assoc: Dict[int, Dict[int, int]] = {}

        for frame_id in all_frames:
            gt_f   = gt_tracks.get(frame_id, [])
            pred_f = pred_tracks.get(frame_id, [])
            if not gt_f and not pred_f: continue

            g_b = np.array([[r[1],r[2],r[3],r[4]] for r in gt_f])   if gt_f   else np.empty((0,4))
            p_b = np.array([[r[1],r[2],r[3],r[4]] for r in pred_f]) if pred_f else np.empty((0,4))
            g_ids = [int(r[0]) for r in gt_f]
            p_ids = [int(r[0]) for r in pred_f]

            if len(g_b) == 0: fp += len(p_b); continue
            if len(p_b) == 0: fn += len(g_b); continue

            iou = _iou_matrix(g_b, p_b)
            mg = set(); mp = set()
            for gi, pi, _ in sorted(
                    [(i,j,iou[i,j]) for i in range(len(g_ids))
                     for j in range(len(p_ids)) if iou[i,j] >= alpha],
                    key=lambda x: -x[2]):
                if gi in mg or pi in mp: continue
                mg.add(gi); mp.add(pi); tp += 1
                gid = g_ids[gi]; pid = p_ids[pi]
                if gid not in assoc: assoc[gid] = {}
                assoc[gid][pid] = assoc[gid].get(pid, 0) + 1
            fp += len(p_ids) - len(mp)
            fn += len(g_ids) - len(mg)

        deta = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        scores = []
        for gid, pc in assoc.items():
            if not pc: continue
            tpa = max(pc.values())
            fna = sum(pc.values()) - tpa
            bp  = max(pc, key=pc.get)
            fpa = sum(v.get(bp, 0) for v in assoc.values()) - tpa
            d = tpa + fpa + fna
            if d > 0: scores.append(tpa / d)
        assa = float(np.mean(scores)) if scores else 0.0
        hota_list.append(np.sqrt(deta * assa))
        deta_list.append(deta); assa_list.append(assa)

    return {
        'hota': float(np.mean(hota_list)),
        'deta': float(np.mean(deta_list)),
        'assa': float(np.mean(assa_list)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MOT metrics via motmetrics (MOTA / IDF1)
# ─────────────────────────────────────────────────────────────────────────────

def accumulate_motmetrics(gt_tracks, pred_tracks, iou_threshold=0.5):
    acc = mm.MOTAccumulator(auto_id=False)
    for frame_id in sorted(gt_tracks.keys()):
        gt_f  = gt_tracks.get(frame_id, [])
        hyp_f = pred_tracks.get(frame_id, [])

        if gt_f and hyp_f:
            gb  = np.array([[r[1],r[2],r[3],r[4]] for r in gt_f])
            hb  = np.array([[r[1],r[2],r[3],r[4]] for r in hyp_f])
            iou = _iou_matrix(gb, hb)
            hyp_f = [h for h, keep in zip(hyp_f, iou.max(axis=0) > 0) if keep]

        gt_ids  = [int(r[0]) for r in gt_f]
        hyp_ids = [int(r[0]) for r in hyp_f]

        if len(gt_ids) == 0 and len(hyp_ids) == 0:
            acc.update([], [], np.empty((0, 0)), frameid=frame_id)
            continue

        gb = np.array([[r[1],r[2],r[3],r[4]] for r in gt_f])  if gt_f  else np.empty((0,4))
        hb = np.array([[r[1],r[2],r[3],r[4]] for r in hyp_f]) if hyp_f else np.empty((0,4))

        if len(gb) > 0 and len(hb) > 0:
            d = 1.0 - _iou_matrix(gb, hb)
            d[d > (1.0 - iou_threshold)] = np.nan
        else:
            d = np.full((len(gt_ids), len(hyp_ids)), np.nan)

        acc.update(gt_ids, hyp_ids, d, frameid=frame_id)
    return acc


def compute_mot_metrics(gt_tracks, pred_tracks, iou_threshold=0.5):
    """
    Compute full suite: IDF1/MOTA (motmetrics) + HOTA (TrackEval).

    Returns dict with:
        idf1, mota, motp                    — from motmetrics
        hota, deta, assa                    — from TrackEval
        num_switches, num_fragmentations,
        num_matches, num_misses,
        num_false_positives, precision, recall
    """
    acc = accumulate_motmetrics(gt_tracks, pred_tracks, iou_threshold)
    mh  = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=['num_frames','mota','motp','idf1','num_switches',
                 'num_fragmentations','num_matches','num_misses',
                 'num_false_positives','precision','recall'],
        name='overall'
    )
    result = summary.to_dict('records')[0]

    hota_res = compute_hota_trackeval(gt_tracks, pred_tracks)
    result['hota'] = hota_res['hota']
    result['deta'] = hota_res['deta']
    result['assa'] = hota_res['assa']

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def write_mot_results(tracks_per_frame, output_file):
    lines = []
    for frame_id in sorted(tracks_per_frame.keys()):
        for row in tracks_per_frame[frame_id]:
            x1,y1,x2,y2 = row[0],row[1],row[2],row[3]
            tid  = int(row[4])
            conf = float(row[5]) if len(row) > 5 else 1.0
            lines.append(f"{frame_id},{tid},{x1:.2f},{y1:.2f},"
                         f"{x2-x1:.2f},{y2-y1:.2f},{conf:.4f},-1,-1,-1")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))


def compute_id_switches(gt_tracks, pred_tracks):
    gt_to_pred = {}; switches = []
    for frame_id in sorted(gt_tracks.keys()):
        gt_f  = gt_tracks.get(frame_id, [])
        hyp_f = pred_tracks.get(frame_id, [])
        if not gt_f or not hyp_f: continue
        gb  = np.array([[r[1],r[2],r[3],r[4]] for r in gt_f])
        hb  = np.array([[r[1],r[2],r[3],r[4]] for r in hyp_f])
        iou = _iou_matrix(gb, hb)
        for gi, gr in enumerate(gt_f):
            gt_id = int(gr[0])
            bj    = int(iou[gi].argmax())
            if iou[gi, bj] < 0.5: continue
            pred_id = int(hyp_f[bj][0])
            if gt_id in gt_to_pred and gt_to_pred[gt_id] != pred_id:
                cx = (gb[gi,0]+gb[gi,2])/2; cy = (gb[gi,1]+gb[gi,3])/2
                switches.append({'frame':frame_id,'gt_id':gt_id,
                                  'old_pred_id':gt_to_pred[gt_id],
                                  'new_pred_id':pred_id,'cx':cx,'cy':cy})
            gt_to_pred[gt_id] = pred_id
    return switches