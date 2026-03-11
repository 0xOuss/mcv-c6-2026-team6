"""
Matching utilities for multi-object tracking.

Implements:
  - IoU computation (vectorized)
  - Greedy matching (fastest, suboptimal)
  - Hungarian matching (optimal bipartite assignment)
  - Cost matrix construction

Comparison:
  - Greedy: O(N*M) — processes highest-IoU pairs first, doesn't globally optimize
    → Can produce suboptimal assignments when detections are close together
    → Fast, simple, often works nearly as well in practice
  - Hungarian: O(N^3) — minimizes total cost globally
    → Always finds the globally optimal assignment
    → Worth using when tracks are dense (crowded intersections)

The difference matters most when:
  - Two detections are equidistant from two tracks (Greedy picks arbitrarily; Hungarian picks optimally)
  - Two vehicles are very close and IoU with both each other's track predictions is similar
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple


def compute_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute IoU between all pairs of boxes.

    Args:
        boxes_a: (N, 4) [x1, y1, x2, y2]
        boxes_b: (M, 4) [x1, y1, x2, y2]
    Returns:
        iou_matrix: (N, M) float32

    Intuition:
        IoU = intersection_area / union_area
        Ranges from 0 (no overlap) to 1 (perfect overlap)
        We use IoU as a proxy for "are these the same object across frames"
        because position + size together give a unique signature
    """
    N = len(boxes_a)
    M = len(boxes_b)
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)

    # Broadcast to (N, M, 4)
    a = boxes_a[:, None, :]  # (N, 1, 4)
    b = boxes_b[None, :, :]  # (1, M, 4)

    # Intersection
    inter_x1 = np.maximum(a[:, :, 0], b[:, :, 0])
    inter_y1 = np.maximum(a[:, :, 1], b[:, :, 1])
    inter_x2 = np.minimum(a[:, :, 2], b[:, :, 2])
    inter_y2 = np.minimum(a[:, :, 3], b[:, :, 3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Union
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union_area = area_a[:, None] + area_b[None, :] - inter_area

    iou = np.where(union_area > 0, inter_area / union_area, 0.0)
    return iou.astype(np.float32)


def greedy_matching(iou_matrix: np.ndarray,
                    iou_threshold: float = 0.5
                    ) -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
    """
    Greedy matching: iteratively pick the highest-IoU pair.

    Algorithm:
        1. Find max IoU in matrix
        2. If >= threshold, record match, zero out that row and column
        3. Repeat until no valid IoU remains

    Args:
        iou_matrix: (N, M) — N tracks, M detections
        iou_threshold: minimum IoU to accept a match
    Returns:
        matches:         list of (track_idx, det_idx)
        unmatched_tracks: list of track indices with no match
        unmatched_dets:   list of detection indices with no match
    """
    if iou_matrix.size == 0:
        return [], list(range(iou_matrix.shape[0])), list(range(iou_matrix.shape[1]))

    mat = iou_matrix.copy()
    matches = []
    matched_tracks = set()
    matched_dets   = set()

    while True:
        if mat.max() < iou_threshold:
            break
        idx = np.unravel_index(mat.argmax(), mat.shape)
        t_idx, d_idx = idx
        matches.append((int(t_idx), int(d_idx)))
        matched_tracks.add(t_idx)
        matched_dets.add(d_idx)
        mat[t_idx, :] = 0.0  # zero out row (track already matched)
        mat[:, d_idx] = 0.0  # zero out col (detection already matched)

    unmatched_tracks = [i for i in range(iou_matrix.shape[0]) if i not in matched_tracks]
    unmatched_dets   = [i for i in range(iou_matrix.shape[1]) if i not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets


def hungarian_matching(iou_matrix: np.ndarray,
                       iou_threshold: float = 0.5
                       ) -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
    """
    Hungarian (linear sum assignment) matching.

    We maximize IoU ↔ minimize cost = 1 - IoU.

    scipy.optimize.linear_sum_assignment finds the globally optimal assignment
    in O(N^3) using the Hungarian algorithm.

    After optimal assignment, we filter out pairs below the IoU threshold.

    Args:
        iou_matrix:    (N, M)
        iou_threshold: minimum IoU to accept match
    Returns:
        matches, unmatched_tracks, unmatched_dets
    """
    if iou_matrix.size == 0:
        return [], list(range(iou_matrix.shape[0])), list(range(iou_matrix.shape[1]))

    cost_matrix = 1.0 - iou_matrix  # Hungarian minimizes cost
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    matched_tracks = set()
    matched_dets   = set()

    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            matches.append((int(r), int(c)))
            matched_tracks.add(r)
            matched_dets.add(c)

    unmatched_tracks = [i for i in range(iou_matrix.shape[0]) if i not in matched_tracks]
    unmatched_dets   = [i for i in range(iou_matrix.shape[1]) if i not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets


def match(track_boxes: np.ndarray, det_boxes: np.ndarray,
          iou_threshold: float = 0.5, strategy: str = "hungarian"
          ) -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
    """
    Unified matching interface.

    Args:
        track_boxes: (N, 4) predicted track bboxes
        det_boxes:   (M, 4) detection bboxes
        iou_threshold: IoU cutoff
        strategy: "hungarian" or "greedy"
    """
    iou_mat = compute_iou(track_boxes, det_boxes)
    if strategy == "hungarian":
        return hungarian_matching(iou_mat, iou_threshold)
    elif strategy == "greedy":
        return greedy_matching(iou_mat, iou_threshold)
    else:
        raise ValueError(f"Unknown matching strategy: {strategy}")
