#!/usr/bin/env python3
"""
Pre-compute optical flow for all frames in a camera sequence.
Save as .npy files for fast loading during tracking.

This is important because:
  - RAFT takes ~0.2s/frame — running it inside the tracking loop is slow
  - Pre-computing allows fast experimentation with different tracker HPs
  - We compute both forward (t→t+1) and backward (t+1→t) for FB consistency

Supported methods:
  raft        — ECCV 2020, raft-kitti.pth (best on KITTI for our setup)
  farneback   — classical OpenCV baseline
  sea_raft    — ECCV 2024, sea_raft_m.pth (faster + better than RAFT)
  unimatch    — CVPR 2023, gmflow-scale2-regrefine6-kitti15-25b554d7.pth
  flowformer  — CVPR 2023, flowformerpp-kitti.pth

Usage:
    python scripts/precompute_flow.py --seq_dir data/aicity/S03/c010 --method raft
    python scripts/precompute_flow.py --seq_dir data/aicity/S03/c010 --method sea_raft \
        --weights external/sea_raft/models/sea_raft_m.pth
    python scripts/precompute_flow.py --seq_dir data/aicity/S01 --all_cameras --method unimatch \
        --weights external/unimatch/pretrained_models/gmflow-scale2-regrefine6-kitti15-25b554d7.pth
"""

import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

SUPPORTED_METHODS = ["raft", "farneback", "sea_raft", "unimatch", "flowformer"]

# Default weight paths per method
DEFAULT_WEIGHTS = {
    "raft":       "external/RAFT/models/raft-kitti.pth",
    "sea_raft":   "external/sea_raft/models/sea_raft_m.pth",
    "unimatch":   "external/unimatch/pretrained_models/gmflow-scale2-regrefine6-kitti15-25b554d7.pth",
    "flowformer": "external/flowformer/checkpoints/flowformerpp-kitti.pth",
    "farneback":  None,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq_dir", required=True,
                   help="Camera dir (data/aicity/S03/c010) or sequence dir with --all_cameras")
    p.add_argument("--method", default="raft", choices=SUPPORTED_METHODS)
    p.add_argument("--weights", default=None,
                   help="Model weights path. Defaults to the standard path for each method. "
                        "Also accepted as --raft_weights for backwards compatibility.")
    p.add_argument("--raft_weights", default=None,
                   help="Alias for --weights (backwards compatibility)")
    p.add_argument("--raft_iters", type=int, default=20,
                   help="Iterations for RAFT. Ignored for other methods.")
    p.add_argument("--all_cameras", action="store_true",
                   help="Process all cameras in sequence directory")
    p.add_argument("--compute_backward", action="store_true", default=True,
                   help="Also compute backward flow (for FB consistency check)")
    return p.parse_args()


def get_image_paths(cam_dir: Path):
    img_dir = cam_dir / "img1"
    if not img_dir.exists():
        return []
    paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    return paths


def load_image(path: Path) -> np.ndarray:
    """Load image as RGB float32 [0,1]."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def build_flow_func(method: str, weights_path: str, args):
    """
    Load the model and return a callable:
        flow_func(img1_rgb_float32, img2_rgb_float32) -> flow_np (H, W, 2) float32
    """
    if method == "raft":
        from src.optical_flow.raft_wrapper import load_raft_model, run_raft
        model, device = load_raft_model(weights_path)
        def flow_func(i1, i2):
            return run_raft(model, device, i1, i2, iters=args.raft_iters)

    elif method == "farneback":
        from src.optical_flow.farneback import run_farneback
        def flow_func(i1, i2):
            return run_farneback(i1, i2)

    elif method == "sea_raft":
        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"SEA-RAFT weights not found: {weights_path}\n"
                f"Download from https://github.com/princeton-vl/SEA-RAFT/releases"
            )
        from src.optical_flow.sea_raft_wrapper import load_sea_raft_model, run_sea_raft
        model, device = load_sea_raft_model(weights_path)
        def flow_func(i1, i2):
            return run_sea_raft(model, device, i1, i2)

    elif method == "unimatch":
        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"UniMatch weights not found: {weights_path}\n"
                f"Download: wget https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/"
                f"pretrained/gmflow-scale2-regrefine6-kitti15-25b554d7.pth"
                f" -O {weights_path}"
            )
        from src.optical_flow.unimatch_wrapper import load_unimatch_model, run_unimatch
        model, device = load_unimatch_model(weights_path)
        def flow_func(i1, i2):
            return run_unimatch(model, device, i1, i2)

    elif method == "flowformer":
        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"FlowFormer weights not found: {weights_path}\n"
                f"Download from https://github.com/drinkingcoder/FlowFormer-Official/releases"
            )
        from src.optical_flow.flowformer_wrapper import load_flowformer_model, run_flowformer
        model, device = load_flowformer_model(weights_path)
        def flow_func(i1, i2):
            return run_flowformer(model, device, i1, i2)

    else:
        raise ValueError(f"Unknown method: {method}")

    return flow_func


def _clear_gpu_cache(cam_name: str = ""):
    """Release GPU memory between cameras to prevent OOM accumulation."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            if cam_name:
                print(f"  GPU cache cleared after {cam_name}")
    except Exception:
        pass


def process_camera(cam_dir: Path, method: str, weights_path: str, args):
    img_paths = get_image_paths(cam_dir)
    if len(img_paths) < 2:
        print(f"  [skip] Not enough images in {cam_dir}")
        return

    subdir_name = f"flow_{method}"
    flow_dir = cam_dir / subdir_name
    flow_dir.mkdir(exist_ok=True)

    existing = len(list(flow_dir.glob("flow_[0-9]*.npy")))
    total    = len(img_paths) - 1
    if existing >= total:
        print(f"  [skip] {cam_dir.name}: flow already complete ({existing}/{total} frames)")
        _update_symlink(cam_dir, subdir_name)
        return

    print(f"  {cam_dir.name}: {len(img_paths)} frames, {existing}/{total} already done")

    try:
        flow_func = build_flow_func(method, weights_path, args)
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        if method != "raft":
            fallback_weights = DEFAULT_WEIGHTS["raft"]
            print(f"  Falling back to RAFT ({fallback_weights})")
            flow_func = build_flow_func("raft", fallback_weights, args)
            subdir_name = "flow_raft"
            flow_dir = cam_dir / subdir_name
            flow_dir.mkdir(exist_ok=True)
        else:
            return

    for i in tqdm(range(len(img_paths) - 1), desc=f"    {cam_dir.name}"):
        frame_id = int(img_paths[i+1].stem)

        fwd_path = flow_dir / f"flow_{frame_id:06d}.npy"
        bwd_path = flow_dir / f"flow_bwd_{frame_id:06d}.npy"

        if fwd_path.exists() and (not args.compute_backward or bwd_path.exists()):
            continue

        img1 = load_image(img_paths[i])
        img2 = load_image(img_paths[i+1])

        if not fwd_path.exists():
            # Downscale large frames to prevent OOM (>1080p → scale to 1080p)
            h, w = img1.shape[:2]
            if h > 1080 or w > 1920:
                scale = min(1080/h, 1920/w)
                nh, nw = int(h*scale)//32*32, int(w*scale)//32*32
                import cv2 as _cv2
                i1s = _cv2.resize(img1, (nw, nh))
                i2s = _cv2.resize(img2, (nw, nh))
                flow_fwd_small = flow_func(i1s, i2s)
                # Scale flow vectors back to original resolution
                flow_fwd = _cv2.resize(flow_fwd_small, (w, h))
                flow_fwd[:,:,0] *= w / nw
                flow_fwd[:,:,1] *= h / nh
            else:
                flow_fwd = flow_func(img1, img2)
            np.save(str(fwd_path), flow_fwd.astype(np.float16))

        if args.compute_backward and not bwd_path.exists():
            h, w = img2.shape[:2]
            if h > 1080 or w > 1920:
                scale = min(1080/h, 1920/w)
                nh, nw = int(h*scale)//32*32, int(w*scale)//32*32
                import cv2 as _cv2
                i1s = _cv2.resize(img2, (nw, nh))
                i2s = _cv2.resize(img1, (nw, nh))
                flow_bwd_small = flow_func(i1s, i2s)
                flow_bwd = _cv2.resize(flow_bwd_small, (w, h))
                flow_bwd[:,:,0] *= w / nw
                flow_bwd[:,:,1] *= h / nh
            else:
                flow_bwd = flow_func(img2, img1)
            np.save(str(bwd_path), flow_bwd.astype(np.float16))

    print(f"  Saved to: {flow_dir}")
    _update_symlink(cam_dir, subdir_name)


def _update_symlink(cam_dir: Path, subdir_name: str):
    """Update flow/ symlink to point to the given subdir."""
    symlink = cam_dir / "flow"
    if symlink.is_symlink():
        symlink.unlink()
    if not symlink.exists():
        symlink.symlink_to(subdir_name)
        print(f"  Symlink: {symlink.name} → {subdir_name}")


def main():
    args = parse_args()
    weights = args.weights or args.raft_weights or DEFAULT_WEIGHTS.get(args.method)
    cam_dir = Path(args.seq_dir)

    if args.all_cameras:
        cam_dirs = sorted([d for d in cam_dir.iterdir()
                           if d.is_dir() and d.name.startswith('c')])
        print(f"Processing {len(cam_dirs)} cameras in {cam_dir} with method={args.method}")
        for cd in cam_dirs:
            process_camera(cd, args.method, weights, args)
            # Clear GPU cache between cameras to prevent OOM accumulation
            _clear_gpu_cache(cd.name)
    else:
        process_camera(cam_dir, args.method, weights, args)

    print("\nFlow pre-computation complete! ✓")


if __name__ == "__main__":
    main()