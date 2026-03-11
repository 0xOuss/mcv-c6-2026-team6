"""
src/utils/kitti_io.py
Read/write KITTI optical flow format (.flo files, KITTI .png flow)
"""

import numpy as np
import cv2
import struct
import os


# ─── .flo file format (Middlebury) ────────────────────────────────

def read_flo(path: str) -> np.ndarray:
    """
    Read Middlebury .flo optical flow file.
    Returns flow: (H, W, 2) float32, channels [u, v].
    """
    with open(path, "rb") as f:
        magic = struct.unpack("f", f.read(4))[0]
        assert magic == 202021.25, f"Invalid .flo magic number: {magic}"
        w = struct.unpack("i", f.read(4))[0]
        h = struct.unpack("i", f.read(4))[0]
        flow = np.frombuffer(f.read(), dtype=np.float32).reshape(h, w, 2)
    return flow


def write_flo(path: str, flow: np.ndarray):
    """Write optical flow to Middlebury .flo format."""
    h, w = flow.shape[:2]
    with open(path, "wb") as f:
        f.write(struct.pack("f", 202021.25))
        f.write(struct.pack("i", w))
        f.write(struct.pack("i", h))
        f.write(flow.astype(np.float32).tobytes())


# ─── KITTI flow format ────────────────────────────────────────────

def read_kitti_flow(path: str):
    """
    Read KITTI optical flow ground truth from a 16-bit PNG.
    KITTI encodes: u = (R + G*256 - 2^15) / 64, v = (B + A*256 - 2^15) / 64
    Also returns a validity mask (valid=1 where flow is defined).

    Returns:
        flow: (H, W, 2) float32 [u, v]
        valid: (H, W) bool
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # 16-bit 3-channel PNG
    assert img is not None, f"Could not read flow file: {path}"
    assert img.dtype == np.uint16, f"Expected uint16, got {img.dtype}"

    # KITTI flow encoding:
    u = (img[:, :, 2].astype(np.float64) - 2**15) / 64.0
    v = (img[:, :, 1].astype(np.float64) - 2**15) / 64.0
    valid = img[:, :, 0] > 0

    flow = np.stack([u, v], axis=-1).astype(np.float32)
    return flow, valid.astype(bool)


def load_image_pair(data_dir: str, seq: int, cam: str = "image_0"):
    """
    Load the KITTI frame pair for a given sequence index.
    KITTI naming: {seq:06d}_10.png (frame t), {seq:06d}_11.png (frame t+1)

    Args:
        data_dir: root of KITTI flow 2012 (contains image_0/, flow_noc/, flow_occ/)
        seq: sequence number (e.g. 45)
        cam: "image_0" (left gray) or "image_2" (left color)

    Returns:
        img1, img2: (H, W, 3) uint8 BGR images
    """
    name = f"{seq:06d}"
    img1_path = os.path.join(data_dir, cam, f"{name}_10.png")
    img2_path = os.path.join(data_dir, cam, f"{name}_11.png")

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    assert img1 is not None, f"Cannot read {img1_path}"
    assert img2 is not None, f"Cannot read {img2_path}"
    return img1, img2


def load_kitti_gt(data_dir: str, seq: int, occluded: bool = False):
    """
    Load KITTI ground truth flow.

    Args:
        data_dir: KITTI root
        seq: sequence number
        occluded: if True, load flow_occ (all pixels); else flow_noc (non-occluded only)

    Returns:
        gt_flow: (H, W, 2) float32
        valid_mask: (H, W) bool — True where GT is defined
    """
    subdir = "flow_occ" if occluded else "flow_noc"
    gt_path = os.path.join(data_dir, subdir, f"{seq:06d}_10.png")
    return read_kitti_flow(gt_path)


# ─── MOT / AI City format ─────────────────────────────────────────

def read_mot_gt(gt_path: str):
    """
    Read MOT-format ground truth.
    Format: frame, id, x, y, w, h, conf, class, visibility
    Returns list of dicts.
    """
    detections = []
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            detections.append({
                "frame": int(parts[0]),
                "id": int(parts[1]),
                "x": float(parts[2]),
                "y": float(parts[3]),
                "w": float(parts[4]),
                "h": float(parts[5]),
                "conf": float(parts[6]) if len(parts) > 6 else 1.0,
                "class": int(parts[7]) if len(parts) > 7 else 1,
                "visibility": float(parts[8]) if len(parts) > 8 else 1.0,
            })
    return detections


def write_mot_result(result_path: str, tracks: list):
    """
    Write tracking results in MOT format.
    tracks: list of dicts with keys: frame, id, x, y, w, h, conf
    """
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        for t in sorted(tracks, key=lambda x: (x["frame"], x["id"])):
            f.write(f"{t['frame']},{t['id']},{t['x']:.2f},{t['y']:.2f},"
                    f"{t['w']:.2f},{t['h']:.2f},{t.get('conf', 1.0):.4f},-1,-1,-1\n")
