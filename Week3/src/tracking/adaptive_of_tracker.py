"""
Adaptive Optical Flow Tracker.

FIXES applied vs original:
  Bug 1 (critical): Replaced class-level SimpleTrack.count with a
    per-instance `_id_counter`.  Same root cause as the other trackers.

  Bug 3 (moderate): Applied the same predict()-order fix as of_tracker.py.
    warp_prediction() is called before matching; age increment happens
    only for unmatched tracks, after matching.

  Bug 4 (float16 overflow): Sanitize flow/flow_bwd to float32 at entry,
    zeroing inf/nan before any arithmetic.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from collections import defaultdict

from .track import SimpleTrack, TrackState
from .matching import match, compute_iou
from ..optical_flow.adaptive_flow import (
    get_bbox_flow_adaptive,
    compute_fb_consistency_mask,
    bbox_flow_method_stats,
)


def _to_f32(arr: np.ndarray) -> np.ndarray:
    """Cast float16 flow array to float32 and zero out inf/nan."""
    arr = arr.astype(np.float32)
    arr[~np.isfinite(arr)] = 0.0
    return arr


class AdaptiveOFTracker:
    """
    Optical Flow tracker with per-bbox adaptive aggregation.
    Drop-in replacement for OFTracker.
    """

    def __init__(
        self,
        iou_threshold: float = 0.45,
        max_age: int = 8,
        min_hits: int = 3,
        matching: str = "hungarian",
        flow_threshold: float = 1.0,
        use_fb_consistency: bool = True,
        fb_threshold: float = 1.0,
        lookback_frames: int = 5,
    ):
        self.iou_threshold      = iou_threshold
        self.max_age            = max_age
        self.min_hits           = min_hits
        self.matching           = matching
        self.flow_threshold     = flow_threshold
        self.use_fb_consistency = use_fb_consistency
        self.fb_threshold       = fb_threshold
        self.lookback_frames    = lookback_frames

        self.tracks: List[SimpleTrack] = []
        self.frame_count = 0

        # FIX Bug 1: instance-level counter
        self._id_counter: int = 0

        self._method_counts: Dict[str, int] = defaultdict(int)
        self._flow_buffer:    Dict[int, np.ndarray] = {}
        self._fb_mask_buffer: Dict[int, np.ndarray] = {}

    def reset(self):
        self.tracks           = []
        self.frame_count      = 0
        self._id_counter      = 0
        self._method_counts   = defaultdict(int)
        self._flow_buffer     = {}
        self._fb_mask_buffer  = {}

    def _new_track(self, bbox: np.ndarray, frame_id: int) -> SimpleTrack:
        """Create a new track with a unique ID from the instance counter."""
        self._id_counter += 1
        trk = SimpleTrack(bbox, frame_id)
        trk.id = self._id_counter
        return trk

    @property
    def aggregation_stats(self) -> Dict[str, int]:
        return dict(self._method_counts)

    def update(
        self,
        detections: np.ndarray,
        frame_id: int,
        flow: Optional[np.ndarray] = None,
        flow_bwd: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self.frame_count += 1
        detections = (np.array(detections) if len(detections) > 0
                      else np.empty((0, 5)))
        det_boxes = (detections[:, :4] if len(detections) > 0
                     else np.empty((0, 4)))

        # FIX Bug 4: sanitize float16 → float32, zero inf/nan
        if flow is not None:
            flow = _to_f32(flow)
        if flow_bwd is not None:
            flow_bwd = _to_f32(flow_bwd)

        # ── FB consistency mask ────────────────────────────────────────
        fb_mask = None
        if (flow is not None and flow_bwd is not None
                and self.use_fb_consistency):
            fb_mask = compute_fb_consistency_mask(
                flow, flow_bwd, self.fb_threshold
            )
            self._fb_mask_buffer[frame_id] = fb_mask

        if flow is not None:
            self._flow_buffer[frame_id] = flow

        # ── Step 1: Warp predictions ───────────────────────────────────
        for trk in self.tracks:
            if flow is not None:
                fu, fv, method = get_bbox_flow_adaptive(
                    flow, trk.bbox, fb_mask
                )
                self._method_counts[method] += 1
                if method not in ("skip", "skip_empty"):
                    trk.warp_prediction(fu, fv)
                else:
                    trk.pred_bbox = trk.bbox.copy()
            else:
                trk.pred_bbox = trk.bbox.copy()

        pred_boxes = (
            np.array([trk.pred_bbox for trk in self.tracks])
            if self.tracks else np.empty((0, 4))
        )

        # ── Step 2: Primary matching ───────────────────────────────────
        matches, unmatched_tracks, unmatched_dets = match(
            pred_boxes, det_boxes, self.iou_threshold, self.matching
        )

        # ── Step 3: Update matched tracks ─────────────────────────────
        matched_track_idxs = set()
        for t_idx, d_idx in matches:
            self.tracks[t_idx].update(det_boxes[d_idx], frame_id)
            matched_track_idxs.add(t_idx)

        # ── Step 3b: Increment age for unmatched tracks only ──────────
        for t_idx, trk in enumerate(self.tracks):
            if t_idx not in matched_track_idxs:
                trk.age += 1
                trk.time_since_update += 1

        # ── Step 4: Occlusion recovery ─────────────────────────────────
        still_unmatched_dets = list(unmatched_dets)

        for t_idx in unmatched_tracks:
            trk = self.tracks[t_idx]
            if trk.time_since_update > self.lookback_frames:
                continue

            for d_idx in list(still_unmatched_dets):
                iou_val = compute_iou(
                    trk.bbox[None, :], det_boxes[d_idx][None, :]
                )[0, 0]
                if iou_val >= self.iou_threshold * 0.8:
                    self._interpolate_track(trk, det_boxes[d_idx], frame_id)
                    trk.update(det_boxes[d_idx], frame_id)
                    still_unmatched_dets.remove(d_idx)
                    break

        # ── Step 5: New tracks ─────────────────────────────────────────
        for d_idx in still_unmatched_dets:
            self.tracks.append(self._new_track(det_boxes[d_idx], frame_id))

        # ── Step 6: Prune + output ─────────────────────────────────────
        results = []
        surviving = []
        for trk in self.tracks:
            if trk.hits >= self.min_hits:
                trk.state = TrackState.CONFIRMED
            if trk.time_since_update > self.max_age:
                trk.state = TrackState.DELETED
                continue

            surviving.append(trk)
            if trk.state == TrackState.CONFIRMED:
                results.append([*trk.bbox, trk.id, 1.0])

        self.tracks = surviving

        # Clean old buffers
        old_keys = [k for k in self._flow_buffer
                    if k < frame_id - self.lookback_frames - 2]
        for k in old_keys:
            self._flow_buffer.pop(k, None)
            self._fb_mask_buffer.pop(k, None)

        return np.array(results) if results else np.empty((0, 6))

    def _interpolate_track(
        self,
        trk: SimpleTrack,
        new_bbox: np.ndarray,
        new_frame: int,
    ):
        """Linear interpolation of missing detections for recovered tracks."""
        if not trk.history:
            return
        last_frame, last_bbox = trk.history[-1]
        gap = new_frame - last_frame
        if gap <= 1:
            return
        for i in range(1, gap):
            alpha = i / gap
            interp_bbox = last_bbox * (1 - alpha) + new_bbox * alpha
            trk.history.append((last_frame + i, interp_bbox))