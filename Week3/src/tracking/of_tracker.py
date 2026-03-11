"""
Optical Flow Tracker with motion filter, age-decay IoU relaxation,
FB consistency masking, and occlusion recovery.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from .track import SimpleTrack, TrackState
from .matching import match, compute_iou


def _to_f32(arr):
    """Cast float16 flow array to float32 and zero out inf/nan."""
    arr = arr.astype(np.float32)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def get_bbox_flow(flow, bbox, aggregation="median", fb_mask=None):
    H, W = flow.shape[:2]
    x1,y1,x2,y2 = [int(round(v)) for v in bbox]
    x1,y1 = max(0,x1), max(0,y1); x2,y2 = min(W,x2), min(H,y2)
    if x2 <= x1 or y2 <= y1: return 0.0, 0.0
    ru = flow[y1:y2, x1:x2, 0].copy().astype(np.float32)
    rv = flow[y1:y2, x1:x2, 1].copy().astype(np.float32)
    ru[~np.isfinite(ru)] = 0.0
    rv[~np.isfinite(rv)] = 0.0
    if fb_mask is not None:
        mr = fb_mask[y1:y2, x1:x2]
        if mr.sum() > 10: ru = ru[mr]; rv = rv[mr]
    if ru.size == 0: return 0.0, 0.0
    if aggregation == "median":
        return float(np.median(ru)), float(np.median(rv))
    elif aggregation == "mean":
        return float(ru.mean()), float(rv.mean())
    elif aggregation == "trimmed_mean":
        from scipy.stats import trim_mean
        return float(trim_mean(ru.ravel(), 0.1)), float(trim_mean(rv.ravel(), 0.1))
    raise ValueError(aggregation)


def compute_fb_mask(fwd, bwd, threshold=1.0):
    fwd = _to_f32(fwd)
    bwd = _to_f32(bwd)
    c = np.sqrt((fwd[:,:,0]+bwd[:,:,0])**2 + (fwd[:,:,1]+bwd[:,:,1])**2)
    return c < threshold


def is_stationary(flow, bbox, thresh=0.8):
    H, W = flow.shape[:2]
    x1,y1,x2,y2 = [int(round(v)) for v in bbox]
    x1,y1 = max(0,x1), max(0,y1); x2,y2 = min(W,x2), min(H,y2)
    if x2 <= x1 or y2 <= y1: return True
    region = flow[y1:y2, x1:x2].astype(np.float32)
    region[~np.isfinite(region)] = 0.0
    mag = np.sqrt(region[:,:,0]**2 + region[:,:,1]**2)
    return (mag < thresh).mean() > 0.70


class OFTracker:
    def __init__(self, iou_threshold=0.45, max_age=8, min_hits=3,
                 matching="hungarian", flow_aggregation="median",
                 flow_threshold=1.5, use_fb_consistency=True,
                 fb_threshold=1.0, lookback_frames=5,
                 use_motion_filter=True, static_flow_thresh=0.8,
                 age_iou_decay=0.05):
        self.iou_threshold      = iou_threshold
        self.max_age            = max_age
        self.min_hits           = min_hits
        self.matching           = matching
        self.flow_aggregation   = flow_aggregation
        self.flow_threshold     = flow_threshold
        self.use_fb_consistency = use_fb_consistency
        self.fb_threshold       = fb_threshold
        self.lookback_frames    = lookback_frames
        self.use_motion_filter  = use_motion_filter
        self.static_flow_thresh = static_flow_thresh
        self.age_iou_decay      = age_iou_decay
        self.tracks = []; self.frame_count = 0
        self._flow_buf = {}; self._fb_buf = {}
        SimpleTrack.count = 0

    def reset(self):
        self.tracks = []; self.frame_count = 0
        self._flow_buf = {}; self._fb_buf = {}
        SimpleTrack.count = 0

    def update(self, detections, frame_id, flow=None, flow_bwd=None):
        self.frame_count += 1
        detections = np.array(detections) if len(detections) > 0 else np.empty((0,5))
        det_boxes  = detections[:, :4] if len(detections) > 0 else np.empty((0,4))

        # Sanitize flow arrays: float16 → float32, zero inf/nan
        if flow is not None:
            flow = _to_f32(flow)
        if flow_bwd is not None:
            flow_bwd = _to_f32(flow_bwd)

        fb_mask = None
        if flow is not None and flow_bwd is not None and self.use_fb_consistency:
            fb_mask = compute_fb_mask(flow, flow_bwd, self.fb_threshold)
        if flow is not None: self._flow_buf[frame_id] = flow

        # Predict
        for trk in self.tracks:
            use_of = False
            if flow is not None:
                if self.use_motion_filter and is_stationary(flow, trk.bbox,
                                                             self.static_flow_thresh):
                    pass  # stationary: use last bbox
                else:
                    fu, fv = get_bbox_flow(flow, trk.bbox, self.flow_aggregation, fb_mask)
                    if np.sqrt(fu**2+fv**2) >= self.flow_threshold:
                        trk.warp_prediction(fu, fv); use_of = True
            if not use_of:
                trk.pred_bbox = trk.bbox.copy()
            trk.predict()

        pred_boxes = np.array([t.pred_bbox for t in self.tracks]) \
                     if self.tracks else np.empty((0,4))

        matches, unmatched_t, unmatched_d = match(
            pred_boxes, det_boxes, self.iou_threshold, self.matching)

        for ti, di in matches:
            self.tracks[ti].update(det_boxes[di], frame_id)

        # Age-decay second-chance matching
        still_ud = list(unmatched_d)
        newly_d  = set()
        for ti in unmatched_t:
            trk = self.tracks[ti]
            age = trk.time_since_update
            if age == 0 or age > self.lookback_frames: continue
            relaxed = max(self.iou_threshold - age * self.age_iou_decay,
                          self.iou_threshold * 0.5)
            best_d, best_iou = None, 0.0
            for di in still_ud:
                if di in newly_d: continue
                iv = compute_iou(trk.bbox[None,:], det_boxes[di][None,:])[0,0]
                if iv > best_iou: best_iou = iv; best_d = di
            if best_d is not None and best_iou >= relaxed:
                self._interpolate(trk, det_boxes[best_d], frame_id)
                trk.update(det_boxes[best_d], frame_id)
                newly_d.add(best_d)
        still_ud = [d for d in still_ud if d not in newly_d]

        for di in still_ud:
            self.tracks.append(SimpleTrack(det_boxes[di], frame_id))

        results = []; surviving = []
        for trk in self.tracks:
            if trk.hits >= self.min_hits: trk.state = TrackState.CONFIRMED
            if trk.time_since_update > self.max_age:
                trk.state = TrackState.DELETED; continue
            surviving.append(trk)
            if trk.state == TrackState.CONFIRMED:
                results.append([*trk.bbox, trk.id, 1.0])
        self.tracks = surviving

        for k in [k for k in self._flow_buf if k < frame_id - self.lookback_frames - 2]:
            del self._flow_buf[k]; self._fb_buf.pop(k, None)

        return np.array(results) if results else np.empty((0,6))

    def _interpolate(self, trk, new_bbox, new_frame):
        if not trk.history: return
        last_frame, last_bbox = trk.history[-1]
        gap = new_frame - last_frame
        if gap <= 1: return
        for i in range(1, gap):
            alpha = i / gap
            trk.history.append((last_frame + i, last_bbox*(1-alpha) + new_bbox*alpha))