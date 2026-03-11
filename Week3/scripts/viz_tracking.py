#!/usr/bin/env python3
"""
Tracking Visualization — renders pred + GT bounding boxes on video frames
and exports side-by-side or overlay GIFs via FFmpeg.

Usage examples:
    # Side-by-side pred vs GT
    python viz_tracking.py \
        --video   data/aicity/S01/c010/img1 \
        --pred    results/tracking/s01/results_c010.txt \
        --gt      data/aicity/S01/c010/gt/gt.txt \
        --output  output/c010_compare.gif \
        --mode    side

    # Overlay both on single video
    python viz_tracking.py \
        --video   data/aicity/S01/c010/img1 \
        --pred    results/tracking/s01/results_c010.txt \
        --gt      data/aicity/S01/c010/gt/gt.txt \
        --output  output/c010_overlay.gif \
        --mode    overlay

    # Pred only (no GT)
    python viz_tracking.py \
        --video   path/to/video.mp4 \
        --pred    results/tracking/s01/results_c010.txt \
        --output  output/c010_pred.gif \
        --mode    pred_only

    # GT only
    python viz_tracking.py \
        --video   path/to/video.mp4 \
        --gt      data/aicity/S01/c010/gt/gt.txt \
        --output  output/c010_gt.gif \
        --mode    gt_only
"""

import argparse
import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

# ─── Colour palettes ──────────────────────────────────────────────────────────
# Pred: distinct colours per ID (BGR)
PRED_PALETTE = [
    (200,  80, 255),   # magenta-purple
    (  0, 200, 255),   # gold/yellow
    (100, 255,  80),   # lime green
    (128,   0, 200),   # purple
    (  0, 180, 220),   # amber
    (160,   0, 255),   # deep purple
    (  0, 255, 200),   # cyan-green
    (255, 140,   0),   # orange
    (180,   0, 180),   # violet
    ( 20, 200, 255),   # yellow
]
GT_COLOR   = (0, 255, 0)    # pure green for GT
BOX_THICK  = 2
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_THICK = 2


def pred_color(track_id: int):
    return PRED_PALETTE[int(track_id) % len(PRED_PALETTE)]


# ─── Parsers ──────────────────────────────────────────────────────────────────

def parse_mot(path: str) -> dict:
    """
    Parse MOTChallenge / AICity format:
        frame, id, x, y, w, h, conf, -1, -1, -1
    Returns {frame_id: [(id, x, y, w, h), ...]}
    """
    data = defaultdict(list)
    if not path or not Path(path).exists():
        return data
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 6:
                parts = line.split()
            if len(parts) < 6:
                continue
            try:
                frame = int(float(parts[0]))
                tid   = int(float(parts[1]))
                x     = float(parts[2])
                y     = float(parts[3])
                w     = float(parts[4])
                h     = float(parts[5])
                # Skip invalid / occluded entries in GT (conf == 0)
                if len(parts) >= 7:
                    conf = float(parts[6])
                    if conf == 0:
                        continue
                data[frame].append((tid, x, y, w, h))
            except ValueError:
                continue
    return data


# ─── Drawing helpers ──────────────────────────────────────────────────────────

def draw_box(frame, x, y, w, h, color, label, img_w, img_h):
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(img_w - 1, int(x + w))
    y2 = min(img_h - 1, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICK)
    if label:
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
        text_color = (0, 0, 0) if color == GT_COLOR else (255, 255, 255)
        cv2.putText(frame, label, (x1 + 2, ty - 2), FONT, FONT_SCALE, text_color, FONT_THICK)


def render_pred(frame, detections, img_w, img_h):
    for (tid, x, y, w, h) in detections:
        draw_box(frame, x, y, w, h, pred_color(tid), f"ID:{tid}", img_w, img_h)


def render_gt(frame, detections, img_w, img_h):
    for (tid, x, y, w, h) in detections:
        draw_box(frame, x, y, w, h, GT_COLOR, f"GT:{tid}", img_w, img_h)


def add_banner(frame, text, color):
    """Small top-left banner showing panel type."""
    cv2.rectangle(frame, (0, 0), (len(text) * 10 + 16, 26), color, -1)
    cv2.putText(frame, text, (8, 18), FONT, 0.6,
                (0, 0, 0) if color == GT_COLOR else (255, 255, 255), 2)


# ─── Frame source ─────────────────────────────────────────────────────────────

class FrameSource:
    """Yields (frame_id, bgr_frame) pairs from either a video file or img dir."""

    def __init__(self, path: str):
        p = Path(path)
        if p.is_dir():
            exts = ('.jpg', '.jpeg', '.png', '.bmp')
            self.files = sorted([f for f in p.iterdir() if f.suffix.lower() in exts])
            self.mode  = 'dir'
            self.cap   = None
            if not self.files:
                sys.exit(f"[ERROR] No image files found in {path}")
        elif p.is_file():
            self.cap  = cv2.VideoCapture(str(p))
            self.mode = 'video'
            if not self.cap.isOpened():
                sys.exit(f"[ERROR] Cannot open video: {path}")
        else:
            sys.exit(f"[ERROR] Video/image path not found: {path}")

    def __iter__(self):
        if self.mode == 'dir':
            for f in self.files:
                frame_id = int(f.stem) if f.stem.isdigit() else self.files.index(f) + 1
                img = cv2.imread(str(f))
                if img is not None:
                    yield frame_id, img
        else:
            frame_id = 1
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                yield frame_id, frame
                frame_id += 1

    def release(self):
        if self.cap:
            self.cap.release()

    @property
    def total(self):
        if self.mode == 'dir':
            return len(self.files)
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))


# ─── Main rendering ───────────────────────────────────────────────────────────

def render_frames(video_path, pred_data, gt_data, mode, max_frames, frame_skip, start_frame=1):
    """
    Yields rendered BGR frames.
    mode: 'side' | 'overlay' | 'pred_only' | 'gt_only'
    """
    src = FrameSource(video_path)
    count = 0

    for frame_id, frame in src:
        if frame_id < start_frame:
            continue
        if count >= max_frames:
            break
        if (count % (frame_skip + 1)) != 0:
            count += 1
            continue

        h, w = frame.shape[:2]
        p_dets = pred_data.get(frame_id, [])
        g_dets = gt_data.get(frame_id, [])

        if mode == 'side':
            left  = frame.copy()
            right = frame.copy()
            render_pred(left,  p_dets, w, h)
            render_gt(right,   g_dets, w, h)
            add_banner(left,  "PRED", PRED_PALETTE[0])
            add_banner(right, "GT",   GT_COLOR)
            # Add frame number
            for img in (left, right):
                cv2.putText(img, f"f{frame_id}", (w - 70, h - 8),
                            FONT, 0.45, (200, 200, 200), 1)
            out = np.concatenate([left, right], axis=1)

        elif mode == 'overlay':
            out = frame.copy()
            render_gt(out,   g_dets, w, h)   # GT first (underneath)
            render_pred(out, p_dets, w, h)   # Pred on top
            add_banner(out, "PRED + GT", (60, 60, 200))
            cv2.putText(out, f"f{frame_id}", (w - 70, h - 8),
                        FONT, 0.45, (200, 200, 200), 1)

        elif mode == 'pred_only':
            out = frame.copy()
            render_pred(out, p_dets, w, h)
            add_banner(out, "PRED", PRED_PALETTE[0])
            cv2.putText(out, f"f{frame_id}", (w - 70, h - 8),
                        FONT, 0.45, (200, 200, 200), 1)

        elif mode == 'gt_only':
            out = frame.copy()
            render_gt(out, g_dets, w, h)
            add_banner(out, "GT", GT_COLOR)
            cv2.putText(out, f"f{frame_id}", (w - 70, h - 8),
                        FONT, 0.45, (200, 200, 200), 1)

        yield out
        count += 1

    src.release()


# ─── GIF export via FFmpeg ─────────────────────────────────────────────────────

def export_gif(frames, output_path: str, fps: float, scale: int, optimize: bool):
    if not frames:
        sys.exit("[ERROR] No frames rendered.")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit("[ERROR] ffmpeg not found. Please install ffmpeg and ensure it is on PATH.")

    output_path = str(Path(output_path).with_suffix('.gif'))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Writing {len(frames)} frames → {output_path}")

    with tempfile.TemporaryDirectory() as tmp:
        # Write frames as PNG sequence
        for i, frame in enumerate(frames):
            cv2.imwrite(os.path.join(tmp, f"frame_{i:05d}.png"), frame)

        palette = os.path.join(tmp, "palette.png")

        # Step 1: Generate palette for better GIF quality
        scale_filter = f"scale={scale}:-1:flags=lanczos" if scale > 0 else "scale=iw:ih:flags=lanczos"
        palette_cmd = [
            ffmpeg, "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmp, "frame_%05d.png"),
            "-vf", f"{scale_filter},palettegen=stats_mode=diff",
            palette
        ]
        result = subprocess.run(palette_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[WARN] Palette generation failed:\n{result.stderr}")
            # Fallback: direct export without palette
            cmd = [
                ffmpeg, "-y",
                "-framerate", str(fps),
                "-i", os.path.join(tmp, "frame_%05d.png"),
                "-vf", scale_filter,
                "-loop", "0",
                output_path
            ]
        else:
            # Step 2: Use palette for high-quality GIF
            gif_filter = f"{scale_filter} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5"
            cmd = [
                ffmpeg, "-y",
                "-framerate", str(fps),
                "-i", os.path.join(tmp, "frame_%05d.png"),
                "-i", palette,
                "-lavfi", gif_filter,
                "-loop", "0",
                output_path
            ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ERROR] FFmpeg failed:\n{result.stderr}")
            sys.exit(1)

    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"[OK] GIF saved: {output_path}  ({size_mb:.1f} MB, {len(frames)} frames @ {fps}fps)")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualise MTSC tracking: pred + GT bounding boxes → GIF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument("--video",  required=True,
                   help="Path to video file (.mp4 etc.) OR image directory (img1/)")
    p.add_argument("--pred",   default=None,
                   help="Path to predicted detections in MOTChallenge format")
    p.add_argument("--gt",     default=None,
                   help="Path to ground truth in MOTChallenge format")
    p.add_argument("--output", default="output/tracking.gif",
                   help="Output GIF path (default: output/tracking.gif)")
    p.add_argument("--mode",   default="side",
                   choices=["side", "overlay", "pred_only", "gt_only"],
                   help=(
                       "side       = pred left | GT right (default)\n"
                       "overlay    = both drawn on same frame\n"
                       "pred_only  = only predictions\n"
                       "gt_only    = only ground truth"
                   ))
    p.add_argument("--fps",        type=float, default=8.0,
                   help="GIF playback speed in fps (default: 8)")
    p.add_argument("--max_frames", type=int,   default=200,
                   help="Max number of frames to render (default: 200)")
    p.add_argument("--frame_skip", type=int,   default=0,
                   help="Skip every N frames, e.g. 1 = render every 2nd frame (default: 0)")
    p.add_argument("--start_frame", type=int,  default=None,
                   help="First frame to render. If omitted, auto-detects first frame "
                        "that has at least one detection in pred or GT.")
    p.add_argument("--scale",      type=int,   default=640,
                   help="Rescale width in pixels, 0 = no rescale (default: 640). "
                        "For side-by-side this is per-panel width.")
    p.add_argument("--no_optimize", action="store_true",
                   help="Skip palette optimisation step (faster but lower quality)")
    return p.parse_args()


def main():
    args = parse_args()

    # Validate mode vs provided files
    if args.mode in ("side", "overlay") and (not args.pred or not args.gt):
        sys.exit(f"[ERROR] mode='{args.mode}' requires both --pred and --gt")
    if args.mode == "pred_only" and not args.pred:
        sys.exit("[ERROR] mode='pred_only' requires --pred")
    if args.mode == "gt_only" and not args.gt:
        sys.exit("[ERROR] mode='gt_only' requires --gt")

    print(f"[INFO] Loading detections...")
    pred_data = parse_mot(args.pred) if args.pred else {}
    gt_data   = parse_mot(args.gt)   if args.gt   else {}

    if args.pred:
        total_pred = sum(len(v) for v in pred_data.values())
        print(f"       Pred: {len(pred_data)} frames, {total_pred} boxes")
    if args.gt:
        total_gt = sum(len(v) for v in gt_data.values())
        print(f"       GT:   {len(gt_data)} frames, {total_gt} boxes")

    # Auto-detect start frame if not specified
    if args.start_frame is not None:
        start_frame = args.start_frame
        print(f"[INFO] Starting at frame {start_frame} (user specified)")
    else:
        all_active = sorted(set(pred_data.keys()) | set(gt_data.keys()))
        start_frame = all_active[0] if all_active else 1
        print(f"[INFO] Auto-detected first active frame: {start_frame}")

    print(f"[INFO] Rendering frames (mode={args.mode}, start={start_frame}, max={args.max_frames})...")
    frames = list(render_frames(
        args.video, pred_data, gt_data,
        mode=args.mode,
        max_frames=args.max_frames,
        frame_skip=args.frame_skip,
        start_frame=start_frame
    ))
    print(f"[INFO] Rendered {len(frames)} frames.")

    # For side-by-side, scale applies per panel — resize each panel then concat
    if args.mode == "side" and args.scale > 0 and frames:
        resized = []
        for f in frames:
            fh, fw = f.shape[:2]
            panel_w = fw // 2
            scale_h = int(fh * args.scale / panel_w)
            left  = cv2.resize(f[:, :panel_w],     (args.scale, scale_h))
            right = cv2.resize(f[:, panel_w:],      (args.scale, scale_h))
            resized.append(np.concatenate([left, right], axis=1))
        frames = resized
        export_gif(frames, args.output, args.fps, scale=0,
                   optimize=not args.no_optimize)
    else:
        export_gif(frames, args.output, args.fps, scale=args.scale,
                   optimize=not args.no_optimize)


if __name__ == "__main__":
    main()