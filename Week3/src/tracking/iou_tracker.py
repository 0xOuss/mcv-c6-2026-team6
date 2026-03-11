"""
Baseline IoU Tracker.

FIXES applied vs original:
  Bug 1 (critical): Replaced class-level SimpleTrack.count with a
    per-instance `_id_counter`.  The old class-level counter was reset
    to 0 every time ANY tracker object was instantiated (OFTracker,
    KalmanTracker, etc.), so IDs restarted from 1 on every camera,
    causing the evaluator to see massive ID collisions and tanking IDF1
    by ~25-30 points.

  Bug 2 (minor): Fixed the hardcoded `conf = 1.0` that ignored the
    actual detection confidence stored during matching.
"""

import numpy as np
from typing import List, Optional
from .track import SimpleTrack, TrackState
from .matching import match, compute_iou


class IoUTracker:
    """
    Baseline IoU-only tracker.

    Args:
        iou_threshold: Minimum IoU to accept a match
        max_age:       Frames a track survives without matching
        min_hits:      Minimum detections before track is confirmed
        matching:      "hungarian" or "greedy"
    """

    def __init__(self, iou_threshold: float = 0.5, max_age: int = 1,
                 min_hits: int = 1, matching: str = "hungarian"):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.matching = matching

        self.tracks: List[SimpleTrack] = []
        self.frame_count = 0

        # FIX Bug 1: instance-level counter — never shared with other trackers
        self._id_counter = 0

    def reset(self):
        self.tracks = []
        self.frame_count = 0
        # FIX Bug 1: reset only THIS instance's counter
        self._id_counter = 0

    def _new_track(self, bbox: np.ndarray, frame_id: int) -> SimpleTrack:
        """Create a new track with a unique ID from the instance counter."""
        self._id_counter += 1
        trk = SimpleTrack(bbox, frame_id)
        trk.id = self._id_counter
        return trk

    def update(self, detections: np.ndarray, frame_id: int) -> np.ndarray:
        """
        Process one frame.

        Args:
            detections: (N, 5) [x1, y1, x2, y2, conf] or (N, 4)
            frame_id:   current frame number
        Returns:
            results: (M, 6) [x1, y1, x2, y2, track_id, conf]
        """
        self.frame_count += 1

        if len(detections) == 0:
            detections = np.empty((0, 5))
        detections = np.array(detections)
        det_boxes = detections[:, :4] if len(detections) > 0 else np.empty((0, 4))
        # FIX Bug 2: keep per-detection confidences for output
        det_confs = detections[:, 4] if (len(detections) > 0 and detections.shape[1] > 4) \
                    else np.ones(len(det_boxes))

        # ── Step 1: Predict ────────────────────────────────────────────
        pred_boxes = []
        for trk in self.tracks:
            pred_boxes.append(trk.predict())
        pred_boxes = np.array(pred_boxes) if pred_boxes else np.empty((0, 4))

        # ── Step 2: Match ──────────────────────────────────────────────
        matches, unmatched_tracks, unmatched_dets = match(
            pred_boxes, det_boxes, self.iou_threshold, self.matching
        )

        # ── Step 3: Update matched tracks ─────────────────────────────
        for t_idx, d_idx in matches:
            self.tracks[t_idx].update(det_boxes[d_idx], frame_id)
            self.tracks[t_idx].time_since_update = 0
            # FIX Bug 2: store the matched detection's confidence on the track
            self.tracks[t_idx].conf = float(det_confs[d_idx])

        # ── Step 4: Create new tracks for unmatched detections ─────────
        for d_idx in unmatched_dets:
            trk = self._new_track(det_boxes[d_idx], frame_id)
            trk.conf = float(det_confs[d_idx])
            self.tracks.append(trk)

        # ── Step 5: Update states and prune dead tracks ─────────────────
        results = []
        surviving = []
        for trk in self.tracks:
            # Initialise conf attribute if missing (first frame edge case)
            if not hasattr(trk, 'conf'):
                trk.conf = 1.0

            # Promote to confirmed
            if trk.hits >= self.min_hits:
                trk.state = TrackState.CONFIRMED
            # Mark as lost
            if trk.time_since_update > 0:
                trk.state = TrackState.LOST
            # Delete if too old
            if trk.time_since_update > self.max_age:
                trk.state = TrackState.DELETED
                continue

            surviving.append(trk)

            # Output confirmed tracks
            if trk.state == TrackState.CONFIRMED or trk.hits >= self.min_hits:
                # FIX Bug 2: use actual stored confidence, not hardcoded 1.0
                results.append([*trk.bbox, trk.id, trk.conf])

        self.tracks = surviving
        return np.array(results) if results else np.empty((0, 6))