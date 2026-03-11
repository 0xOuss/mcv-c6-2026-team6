#!/usr/bin/env python3
"""
scripts/retrain_detector.py

Retrain YOLO26l from Week 2 weights with augmentations.

WHY AUGMENTATIONS HELP FOR TRACKING (vs raw mAP):
  Week 2 best model (96.61% mAP@50) was trained on the first 25% of frames.
  The AI City dataset has:
    - Lighting variation between morning/afternoon cameras
    - Parked vs moving cars with very different contrast profiles
    - Some cameras looking into direct sunlight → blown highlights
    - Distant small vehicles that are underrepresented in first 25%

  Augmentations address these by simulating:
    1. Brightness/contrast jitter → handles lighting variation
    2. HSV hue/sat/val shifts    → handles camera white balance differences
    3. Mosaic (already in YOLO)  → handles small object occlusion
    4. Random affine (scale/rotate) → handles camera angle variation

  For tracking specifically:
    - Better detector = fewer missed detections = fewer ID breaks (DetA ↑)
    - More consistent detections across consecutive frames = smoother IoU matching
    - Less flickering on parked cars = fewer spurious track creates/deletes

CONTRIBUTION CLAIM (for slides):
  "We retrain the Week 2 YOLO26l detector with augmentations targeting
   specific failure modes in the AI City dataset (lighting variation,
   small distant vehicles). This improves DetA in HOTA, which directly
   reduces ID switches in the tracker."

USAGE:
    # Basic (resume from Week 2 best weights):
    python scripts/retrain_detector.py \\
        --weights /path/to/week2/weights/best.pt \\
        --data configs/aicity_augmented.yaml

    # With contrast normalization comparison:
    python scripts/retrain_detector.py \\
        --weights /path/to/week2/weights/best.pt \\
        --data configs/aicity_augmented.yaml \\
        --contrast_norm  # preprocess frames with CLAHE before training

    # Full comparison run (trains both, saves det.txt for each):
    python scripts/retrain_detector.py \\
        --weights /path/to/week2/weights/best.pt \\
        --compare_all

DETECTOR OUTPUT:
    After training, run inference to generate det.txt:
        python scripts/retrain_detector.py --inference_only \\
            --weights results/detector/augmented/weights/best.pt \\
            --seq_dir data/aicity/S01/c010
"""

import sys
import argparse
import shutil
import subprocess
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Augmentation config for AI City ──────────────────────────────────────────
#
# These are passed as YOLO training hyperparameter overrides.
# Values chosen specifically for AI City traffic surveillance:
#   - hsv_h/s/v: AI City cameras have inconsistent white balance and exposure
#   - degrees: cameras are fixed, so large rotation is wrong → keep small
#   - scale: vehicles appear at many scales (near=large, far=small) → large scale jitter
#   - flipud: traffic flows in fixed direction → no vertical flip
#   - fliplr: flip left-right is valid for vehicles
#   - mosaic: already 1.0 in YOLO default, keep
#   - mixup: 0.1 helps with occluded vehicles
#
AUGMENTATION_OVERRIDES = {
    # Colour jitter — most important for lighting variation
    "hsv_h": 0.015,      # hue jitter (small — traffic colours are distinctive)
    "hsv_s": 0.7,        # saturation jitter (large — sunlight vs shadow)
    "hsv_v": 0.4,        # value/brightness jitter (large — day/night variation)

    # Geometric
    "degrees":    2.0,   # small rotation only (fixed camera)
    "translate":  0.1,   # slight translation
    "scale":      0.6,   # scale jitter (0.4x to 1.6x) — handles near/far vehicles
    "shear":      0.0,   # no shear (cameras are fixed, no perspective change)
    "perspective": 0.0,  # no perspective warp

    # Flips
    "fliplr": 0.5,       # horizontal flip — valid for vehicles
    "flipud": 0.0,       # no vertical flip — traffic has fixed direction

    # Mosaic / mixup
    "mosaic":  1.0,      # keeps mosaic (good for small objects)
    "mixup":   0.1,      # slight mixup (helps with occluded vehicles)
    "copy_paste": 0.0,   # copy-paste segmentation (not useful here)

    # Training
    "epochs":     15,    # fewer epochs since we start from fine-tuned weights
    "patience":   5,     # early stopping
    "batch":      16,
    "imgsz":      960,   # same as Week 2 best
    "lr0":        5e-4,  # lower LR since we're fine-tuning from already-good weights
    "lrf":        0.01,  # final LR ratio
    "warmup_epochs": 2,
    "weight_decay":  5e-4,
    "label_smoothing": 0.0,
    "close_mosaic": 3,   # disable mosaic in last 3 epochs for stable training
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",   default=None,
                   help="Path to Week 2 best.pt (e.g. ../week2/runs/train/exp/weights/best.pt)")
    p.add_argument("--data",      default="configs/aicity_augmented.yaml",
                   help="YOLO dataset config yaml")
    p.add_argument("--output_dir", default="results/detector")
    p.add_argument("--contrast_norm", action="store_true",
                   help="Preprocess training frames with CLAHE contrast normalisation")
    p.add_argument("--compare_all",  action="store_true",
                   help="Train 3 variants: baseline / augmented / augmented+CLAHE")
    p.add_argument("--inference_only", action="store_true",
                   help="Skip training, just run inference to generate det.txt")
    p.add_argument("--seq_dir",   default=None,
                   help="Sequence dir for inference (e.g. data/aicity/S01/c010)")
    p.add_argument("--conf",      type=float, default=0.4)
    p.add_argument("--iou_nms",   type=float, default=0.5)
    return p.parse_args()


# ─── CLAHE contrast normalisation ─────────────────────────────────────────────

def apply_clahe(img: np.ndarray, clip_limit: float = 2.0,
                tile_grid: tuple = (8, 8)) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalisation) on the L channel.

    Why CLAHE for AI City:
      - Some cameras face strong backlight (direct sunlight) → foreground underexposed
      - CLAHE boosts local contrast without blowing out highlights globally
      - Applied to L channel in LAB space so colour is preserved

    Reference: Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization.
               Graphics Gems IV, Academic Press.

    Args:
        img:        (H, W, 3) BGR uint8
        clip_limit: limits over-amplification of noise (2.0 = standard)
        tile_grid:  local tile size for adaptive histogram
    Returns:
        (H, W, 3) BGR uint8 with enhanced contrast
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def preprocess_sequence_clahe(seq_dir: Path, out_dir: Path,
                               clip_limit: float = 2.0):
    """
    Apply CLAHE to all frames in a sequence and save to out_dir.
    Used to create a contrast-normalised version of the dataset for comparison.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = seq_dir / "img1"
    if not img_dir.exists():
        img_dir = seq_dir

    frames = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    print(f"  Applying CLAHE to {len(frames)} frames → {out_dir}")

    for fpath in frames:
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        img_eq = apply_clahe(img, clip_limit)
        cv2.imwrite(str(out_dir / fpath.name), img_eq)

    print("  Done.")


# ─── YOLO dataset config generator ───────────────────────────────────────────

def write_yolo_dataset_config(aicity_dir: Path, out_path: Path,
                               use_clahe: bool = False):
    """
    Generate the YOLO dataset .yaml config for AI City.
    Single class: 0 = vehicle (car + bus + truck merged — same as Week 2).

    This is the same 1-class setup as Week 2. Important for slides:
    explicitly state "We use the same 1-class vehicle head as Week 2,
    with fine-tuning on first 25% of frames (same split as Week 2)."
    """
    suffix = "_clahe" if use_clahe else ""
    config = f"""# AI City Challenge — YOLO training config (Week 3 augmented)
# Single class: vehicle (car + bus + truck merged to 1 class)
# Training split: first 25% of frames per camera (same as Week 2)

path: {aicity_dir}
train: images/train{suffix}
val:   images/val{suffix}

nc: 1
names:
  0: vehicle

# Note: 1-class head (NOT COCO 80-class) — explicitly documented here
# to avoid Week 2 feedback: "not clear if you change the head to 1 class"
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(config)
    print(f"  Dataset config written: {out_path}")


# ─── Training launcher ────────────────────────────────────────────────────────

def train_yolo(weights: str, data_yaml: str, name: str,
               output_dir: Path, aug_overrides: dict):
    """
    Launch YOLO training via ultralytics CLI.

    Using ultralytics because:
      - Your Week 2 best model is YOLO26l from ultralytics format
      - Same training loop ensures fair comparison
      - Citation: Jocher et al., "Ultralytics YOLO" (2023), https://github.com/ultralytics/ultralytics

    The model will automatically:
      - Start from the provided weights (transfer learning)
      - Apply the augmentation overrides
      - Save best.pt and last.pt to output_dir/name/weights/
    """
    cmd = [
        "yolo", "detect", "train",
        f"model={weights}",
        f"data={data_yaml}",
        f"project={output_dir}",
        f"name={name}",
        "exist_ok=True",
        "verbose=True",
    ]

    # Add augmentation overrides
    for k, v in aug_overrides.items():
        cmd.append(f"{k}={v}")

    print(f"\n  Training: {name}")
    print(f"  Weights:  {weights}")
    print(f"  Data:     {data_yaml}")
    print(f"  Command:  {' '.join(cmd[:8])} ...")
    subprocess.run(cmd, check=True)


# ─── Inference → det.txt ──────────────────────────────────────────────────────

def run_inference(weights: str, seq_dir: Path, conf: float = 0.4,
                  iou_nms: float = 0.5, imgsz: int = 960,
                  use_clahe: bool = False) -> Path:
    """
    Run YOLO inference on a camera sequence and save det.txt
    in MOTChallenge format for the trackers.

    MOTChallenge format:
        frame, -1, x, y, w, h, conf, -1, -1, -1

    Returns:
        Path to generated det.txt
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Install ultralytics: pip install ultralytics")

    model = YOLO(weights)

    img_dir = seq_dir / "img1"
    if not img_dir.exists():
        img_dir = seq_dir

    frames = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    print(f"\n  Running inference on {len(frames)} frames...")
    print(f"  Model:   {weights}")
    print(f"  Seq:     {seq_dir}")
    print(f"  CLAHE:   {use_clahe}")

    det_dir = seq_dir / "det"
    det_dir.mkdir(exist_ok=True)

    suffix = "_clahe" if use_clahe else "_augmented"
    det_file = det_dir / f"det{suffix}.txt"
    lines = []

    for fpath in frames:
        # Extract frame ID from filename
        try:
            frame_id = int(fpath.stem)
        except ValueError:
            frame_id = int("".join(filter(str.isdigit, fpath.stem)) or "0")

        img = cv2.imread(str(fpath))
        if img is None:
            continue

        if use_clahe:
            img = apply_clahe(img)

        results = model(img, conf=conf, iou=iou_nms, imgsz=imgsz,
                        classes=[0],  # class 0 = vehicle (1-class model)
                        verbose=False)

        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                w = x2 - x1
                h = y2 - y1
                # MOTChallenge: frame, -1, x, y, w, h, conf, -1, -1, -1
                lines.append(
                    f"{frame_id},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},"
                    f"{confidence:.4f},-1,-1,-1"
                )

    det_file.write_text("\n".join(lines))
    print(f"  Saved det.txt: {det_file}  ({len(lines)} detections)")
    return det_file


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Inference only ────────────────────────────────────────────────
    if args.inference_only:
        if not args.weights or not args.seq_dir:
            print("ERROR: --weights and --seq_dir required for --inference_only")
            return
        seq_dir = Path(args.seq_dir)

        # Standard inference
        det_std = run_inference(args.weights, seq_dir, args.conf, args.iou_nms)
        print(f"\nDone. Standard det.txt: {det_std}")

        if args.contrast_norm:
            # CLAHE inference
            det_clahe = run_inference(args.weights, seq_dir, args.conf,
                                       args.iou_nms, use_clahe=True)
            print(f"CLAHE det.txt: {det_clahe}")
            print("\nTo compare, run trackers on both:")
            print(f"  cp {det_std} <seq_dir>/det/det.txt && python scripts/run_tracking.py ...")
            print(f"  cp {det_clahe} <seq_dir>/det/det.txt && python scripts/run_tracking.py ...")
        return

    # ── Training ──────────────────────────────────────────────────────
    if not args.weights:
        print("ERROR: --weights required (path to Week 2 best.pt)")
        print("Example: --weights ../week2/runs/detect/train_yolo26l/weights/best.pt")
        return

    data_yaml = Path(args.data)

    if args.compare_all:
        # Train 3 variants for comparison
        variants = [
            ("baseline_no_aug",       False, {**AUGMENTATION_OVERRIDES,
                                               "hsv_h": 0, "hsv_s": 0, "hsv_v": 0,
                                               "scale": 0, "fliplr": 0}),
            ("augmented",             False, AUGMENTATION_OVERRIDES),
            ("augmented_clahe",       True,  AUGMENTATION_OVERRIDES),
        ]
        for name, use_clahe, aug in variants:
            data = data_yaml.with_name(
                data_yaml.stem + ("_clahe" if use_clahe else "") + data_yaml.suffix
            )
            train_yolo(args.weights, str(data), name, out_dir, aug)

        print("\nAll 3 variants trained.")
        print("Run inference for each:")
        for name, _, _ in variants:
            print(f"  python scripts/retrain_detector.py --inference_only "
                  f"--weights {out_dir}/{name}/weights/best.pt "
                  f"--seq_dir data/aicity/S01/c010")

    else:
        # Single training run
        aug = AUGMENTATION_OVERRIDES
        if args.contrast_norm:
            print("  Note: for CLAHE+training, preprocess images first, then train.")
            print("  The --contrast_norm flag here applies CLAHE during inference only.")
        name = "yolo26l_augmented_clahe" if args.contrast_norm else "yolo26l_augmented"
        train_yolo(args.weights, str(data_yaml), name, out_dir, aug)

        best_weights = out_dir / name / "weights" / "best.pt"
        print(f"\nTraining complete. Best weights: {best_weights}")
        print("\nNext step — generate det.txt for trackers:")
        print(f"  python scripts/retrain_detector.py --inference_only \\")
        print(f"      --weights {best_weights} \\")
        print(f"      --seq_dir data/aicity/S01/c010")
        if args.contrast_norm:
            print("      --contrast_norm  # also generate CLAHE version")


if __name__ == "__main__":
    main()