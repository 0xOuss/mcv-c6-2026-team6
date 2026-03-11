"""
Kalman Filter Tracker.

FIXES applied vs original:
  Bug 1 (critical): Replaced class-level KalmanBoxTracker.count with a
    per-instance `_id_counter` in KalmanTracker.  Same root cause as in
    iou_tracker.py — the class-level counter was silently zeroed when any
    other tracker was instantiated, causing ID collisions across cameras.
"""

import numpy as np
from typing import List
from .track import KalmanBoxTracker, TrackState, state_to_bbox
from .matching import match


class KalmanTracker:
    """
    SORT-style tracker: Kalman filter + IoU association.
    Based on: Bewley et al., "Simple Online and Realtime Tracking" (ICIP 2016)
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5,
                 min_hits: int = 2, matching: str = "hungarian"):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.matching = matching
        self.tracks: List[KalmanBoxTracker] = []
        self.frame_count = 0

        # FIX Bug 1: instance-level counter — never shared with other trackers
        self._id_counter = 0

    def reset(self):
        self.tracks = []
        self.frame_count = 0
        # FIX Bug 1: reset only THIS instance's counter
        self._id_counter = 0

    def _new_track(self, bbox: np.ndarray) -> KalmanBoxTracker:
        """Create a new Kalman track with a unique ID from the instance counter."""
        self._id_counter += 1
        trk = KalmanBoxTracker(bbox)
        trk.id = self._id_counter
        return trk

    def update(self, detections: np.ndarray, frame_id: int) -> np.ndarray:
        """Same interface as IoUTracker."""
        self.frame_count += 1

        detections = np.array(detections) if len(detections) > 0 else np.empty((0, 5))
        det_boxes = detections[:, :4] if len(detections) > 0 else np.empty((0, 4))

        # ── Step 1: Kalman Predict ─────────────────────────────────────
        pred_boxes = np.array([trk.predict() for trk in self.tracks]) \
                     if self.tracks else np.empty((0, 4))

        # ── Step 2: Match ──────────────────────────────────────────────
        matches, unmatched_tracks, unmatched_dets = match(
            pred_boxes, det_boxes, self.iou_threshold, self.matching
        )

        # ── Step 3: Kalman Update for matched ─────────────────────────
        for t_idx, d_idx in matches:
            self.tracks[t_idx].update(det_boxes[d_idx])

        # ── Step 4: New tracks ─────────────────────────────────────────
        for d_idx in unmatched_dets:
            self.tracks.append(self._new_track(det_boxes[d_idx]))

        # ── Step 5: Prune + output ─────────────────────────────────────
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
                results.append([*trk.get_state(), trk.id, 1.0])

        self.tracks = surviving
        return np.array(results) if results else np.empty((0, 6))