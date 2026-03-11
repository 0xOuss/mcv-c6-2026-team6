"""
Track: single-object track state machine.

States:
  - TENTATIVE: just created, not yet confirmed (min_hits not reached)
  - CONFIRMED: seen enough times to be shown
  - LOST:      not matched for 1..max_age frames (can be re-linked)
  - DELETED:   beyond max_age, should be removed

Design choices:
  - We separate state machine from matching logic for testability
  - Kalman state: [cx, cy, w, h, vcx, vcy, vw, vh] (center-x, center-y, width, height + velocities)
  - OF-predicted position is an alternative to Kalman for prediction step
"""

import numpy as np
from enum import Enum, auto
from typing import Optional, List


class TrackState(Enum):
    TENTATIVE = auto()
    CONFIRMED  = auto()
    LOST       = auto()
    DELETED    = auto()


def bbox_to_state(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [cx, cy, w, h]."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w  = x2 - x1
    h  = y2 - y1
    return np.array([cx, cy, w, h], dtype=np.float32)


def state_to_bbox(state: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2].

    Enforce positivity on w and h to prevent NaN in IoU computation.
    Kalman filter is purely linear and can predict negative w or h,
    which would cause sqrt(w*h) → NaN → Hungarian algorithm crash.
    Fix: clamp w, h to a small positive value (same approach as Team 5).
    """
    cx, cy, w, h = state[:4]
    if w <= 0: w = 1e-6  # prevent NaN in IoU
    if h <= 0: h = 1e-6  # prevent NaN in IoU
    return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dtype=np.float32)


class KalmanBoxTracker:
    """
    Kalman filter for a single bounding box.

    State vector: [cx, cy, w, h, vcx, vcy, vw, vh]
    Observation:  [cx, cy, w, h]

    Constant velocity model:
      cx(t+1) = cx(t) + vcx(t)
      cy(t+1) = cy(t) + vcy(t)
      ...

    Why Kalman?
      - Handles missing detections gracefully by predicting forward
      - Uncertainty grows the longer we haven't seen the object
      - Natural way to smooth noisy detections

    Why might OF be better?
      - Kalman assumes constant velocity — fails for accelerating/decelerating vehicles
      - OF directly measures pixel displacement — no assumption needed
      - But OF is noisier near boundaries and for small/slow objects
    """

    count = 0  # class-level counter for unique IDs

    def __init__(self, bbox: np.ndarray):
        from scipy.linalg import block_diag

        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.state = TrackState.TENTATIVE
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.time_since_update = 0
        self.history: List[np.ndarray] = []  # for occlusion recovery / interpolation

        # Kalman filter setup (using scipy / manual implementation)
        # We implement manually to avoid filterpy dependency
        dt = 1  # 1 frame

        # State transition matrix F (8x8)
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i+4] = dt  # position += velocity * dt

        # Observation matrix H (4x8): we observe [cx, cy, w, h] only
        self.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.H[i, i] = 1.0

        # Process noise covariance Q
        # Large for velocity (we don't trust the velocity model much)
        self.Q = np.diag([1., 1., 1., 1., 0.01, 0.01, 0.0001, 0.0001]).astype(np.float32)

        # Measurement noise covariance R
        # Larger for w, h than cx, cy (detectors are less precise on size)
        init_box = bbox_to_state(bbox)
        self.R = np.diag([1., 1., 10., 10.]).astype(np.float32)

        # State covariance P (initial uncertainty is high for velocity)
        self.P = np.diag([10., 10., 10., 10., 1000., 1000., 1000., 1000.]).astype(np.float32)

        # Initial state: position from bbox, zero velocity
        self.x = np.zeros(8, dtype=np.float32)
        self.x[:4] = init_box

        self.history.append(bbox.copy())

    def predict(self) -> np.ndarray:
        """Kalman predict step. Returns predicted bbox [x1, y1, x2, y2]."""
        # x = F * x
        self.x = self.F @ self.x
        # P = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        self.hit_streak = 0
        return state_to_bbox(self.x)

    def update(self, bbox: np.ndarray):
        """Kalman update step with a new matched detection."""
        z = bbox_to_state(bbox)  # measurement

        # Innovation: y = z - H*x
        y = z - self.H @ self.x

        # Innovation covariance: S = H*P*H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain: K = P*H^T * S^{-1}
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update: x = x + K*y
        self.x = self.x + K @ y

        # Covariance update: P = (I - K*H)*P
        I_KH = np.eye(8) - K @ self.H
        self.P = I_KH @ self.P

        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        self.history.append(bbox.copy())

        # Update state
        if self.state == TrackState.TENTATIVE:
            # Promote to CONFIRMED once we see it enough times
            pass  # done externally based on min_hits

    def get_state(self) -> np.ndarray:
        """Return current bbox estimate [x1, y1, x2, y2]."""
        return state_to_bbox(self.x)


class SimpleTrack:
    """
    Lightweight track for IoU-only tracker (no Kalman).
    Stores last known bbox, uses that directly for matching.
    """

    count = 0

    def __init__(self, bbox: np.ndarray, frame_id: int):
        SimpleTrack.count += 1
        self.id = SimpleTrack.count
        self.state = TrackState.TENTATIVE
        self.bbox = bbox.copy()          # last confirmed bbox
        self.pred_bbox = bbox.copy()     # predicted bbox (may be warped by OF)
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.time_since_update = 0
        self.history: List[tuple] = [(frame_id, bbox.copy())]  # (frame, bbox)

    def predict(self) -> np.ndarray:
        """For simple tracker: prediction = last bbox (no motion model)."""
        self.age += 1
        self.time_since_update += 1
        return self.pred_bbox.copy()

    def update(self, bbox: np.ndarray, frame_id: int):
        self.bbox = bbox.copy()
        self.pred_bbox = bbox.copy()
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        self.history.append((frame_id, bbox.copy()))

    def warp_prediction(self, flow_u: float, flow_v: float):
        """
        Warp the predicted bbox by (flow_u, flow_v) pixel displacement.

        Why warp, not just shift center?
        - Shifting only the center keeps width/height fixed — usually correct for rigid vehicles
        - For very fast-moving objects, w/h might also change due to perspective, but
          this effect is small for highway cameras → we only shift [x1, y1, x2, y2]

        Args:
            flow_u: horizontal displacement (pixels)
            flow_v: vertical displacement (pixels)
        """
        self.pred_bbox = self.bbox.copy()
        self.pred_bbox[0] += flow_u   # x1
        self.pred_bbox[1] += flow_v   # y1
        self.pred_bbox[2] += flow_u   # x2
        self.pred_bbox[3] += flow_v   # y2