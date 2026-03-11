"""
UniMatch Optical Flow Wrapper.

Xu et al., "Unifying Flow, Stereo and Depth Estimation" (CVPR 2023)
https://github.com/autonomousvision/unimatch
ArXiv: https://arxiv.org/abs/2211.05783

WHY UNIMATCH BEATS RAFT ON KITTI:
──────────────────────────────────
RAFT builds a 4D cost volume between ALL pixel pairs using a CNN feature
encoder. Two fundamental limitations:
  1. CNN features are local — each feature vector only "sees" a small
     receptive field around each pixel
  2. The 4D cost volume is computed once; the GRU refinement only refines
     the flow estimate, it cannot revisit the matching problem

UniMatch addresses both with a transformer-based architecture:
  1. Global self-attention in the feature encoder — every pixel attends
     to every other pixel → features encode global context
  2. Cross-frame cross-attention for matching — instead of exhaustive
     4D correlation, transformer attention finds correspondences globally
  3. Dual-softmax filter — suppresses ambiguous matches before they
     contaminate the flow estimate (key contribution for textureless regions)
  4. Self-attention on the cost volume itself — refines matches using
     neighborhood context

In numbers on KITTI 2015 (F1-all, lower is better):
  RAFT:       5.10%
  UniMatch:   3.60%   ← 30% relative improvement
  Runtime:    ~same on GPU (~0.1s for KITTI resolution)

ARCHITECTURE DIAGRAM (for slides):
  img1, img2
    │
    ▼
  Shared Feature Encoder (CNN + Transformer self-attention)
    │
    ▼
  Cross-frame Attention Matching
    │
  Dual-softmax Filter  ← suppresses ambiguous matches
    │
    ▼
  Cost Volume (attention-weighted, not exhaustive 4D)
    │
    ▼
  Flow Decoder (GRU-based, similar to RAFT)
    │
    ▼
  Upsampled Flow (H, W, 2)

SETUP:
  git clone https://github.com/autonomousvision/unimatch external/unimatch
  cd external/unimatch
  pip install -r requirements.txt

  # Download pretrained weights:
  # Option 1: pretrained_models/ directory in the repo
  # Option 2: download directly:
  mkdir -p external/unimatch/pretrained_models
  # Download gmflow-scale2-regrefine6-mixdata.pth from:
  # https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained_models/
  # Recommended weight: gmflow-scale2-regrefine6-mixdata.pth (best general)
  # For KITTI specifically: gmflow-scale2-regrefine6-kitti15-847b2ab.pth

AVAILABLE WEIGHTS:
  gmflow-scale1-things.pth              — FlyingThings only (synthetic)
  gmflow-scale2-things.pth              — FlyingThings, 2-scale
  gmflow-scale2-mixdata.pth             — Mixed data (recommended)
  gmflow-scale2-regrefine6-mixdata.pth  — With refinement (best quality)
  gmflow-scale2-regrefine6-kitti15-*.pth — Fine-tuned on KITTI (best for our eval)
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple


def load_unimatch_model(
    model_path: str,
    num_scales: int = 2,
    upsample_factor: int = 4,
    feature_channels: int = 128,
    num_transformer_layers: int = 6,
    num_head: int = 1,
    ffn_dim_expansion: int = 4,
    reg_refine: bool = True,
    num_reg_refine: int = 6,
    device: str = "auto",
):
    """
    Load UniMatch model from checkpoint.

    Args:
        model_path:              Path to .pth checkpoint
        num_scales:              Number of feature pyramid scales (1 or 2)
        upsample_factor:         Upsampling factor in decoder (4 or 8)
        feature_channels:        Feature dimension (128)
        num_transformer_layers:  Self-attention layers in encoder
        num_head:                Attention heads
        ffn_dim_expansion:       FFN hidden dim multiplier
        reg_refine:              Use refinement decoder (True for best quality)
        num_reg_refine:          Refinement iterations (6 = standard)
        device:                  "cuda", "cpu", or "auto"
    Returns:
        model, device
    """
    import sys
    unimatch_dir = Path(__file__).parent.parent.parent / "external" / "unimatch"
    if not unimatch_dir.exists():
        raise FileNotFoundError(
            f"UniMatch not found at {unimatch_dir}.\n"
            "Setup:\n"
            "  git clone https://github.com/autonomousvision/unimatch external/unimatch\n"
            "  cd external/unimatch && pip install -r requirements.txt\n"
            "  # Download weights to external/unimatch/pretrained_models/"
        )

    if str(unimatch_dir) not in sys.path:
        sys.path.insert(0, str(unimatch_dir))

    from unimatch.unimatch import UniMatch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UniMatch(
        feature_channels=feature_channels,
        num_scales=num_scales,
        upsample_factor=upsample_factor,
        num_head=num_head,
        ffn_dim_expansion=ffn_dim_expansion,
        num_transformer_layers=num_transformer_layers,
        reg_refine=reg_refine,
        task="flow",
    )

    checkpoint = torch.load(model_path, map_location=device)
    # Handle various checkpoint formats
    state_dict = checkpoint.get("model", checkpoint)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"  UniMatch loaded: {Path(model_path).name} | device={device}")
    return model, device


def run_unimatch(
    model,
    device: str,
    img1: np.ndarray,
    img2: np.ndarray,
    padding_factor: int = 32,
    inference_size: Optional[Tuple[int, int]] = None,
    attn_type: str = "swin",
    attn_splits_list: Tuple[int, ...] = (2, 8),
    corr_radius_list: Tuple[int, ...] = (-1, 4),
    prop_radius_list: Tuple[int, ...] = (-1, 1),
    num_reg_refine: int = 6,
) -> np.ndarray:
    """
    Run UniMatch on an image pair.

    Args:
        model, device:    from load_unimatch_model
        img1, img2:       (H, W, 3) RGB uint8 [0,255] or float32 [0,1]
        padding_factor:   Pad image dimensions to multiples of this (32)
        inference_size:   Override inference resolution (None = use original)
        attn_type:        Attention type: "swin" (window) or "self" (global)
                          "swin" is faster; "self" is more accurate for large images
        attn_splits_list: Attention window splits per scale (default: (2, 8))
        corr_radius_list: Correlation radius per scale (-1 = global)
        prop_radius_list: Propagation radius per scale (-1 = no propagation)
        num_reg_refine:   Refinement iterations (0 = no refinement, 6 = standard)
    Returns:
        flow: (H, W, 2) float32 [u, v] in pixels

    Note on attn_type:
        "swin" = Swin Transformer windowed attention — faster, handles large images
        "self" = Full self-attention — more accurate for small images like KITTI
        For KITTI (375×1242): use "swin" to avoid OOM on smaller GPUs
        For AI City (720×1280): use "swin"
    """
    # Convert to uint8
    if img1.dtype != np.uint8:
        img1 = (img1 * 255).clip(0, 255).astype(np.uint8)
        img2 = (img2 * 255).clip(0, 255).astype(np.uint8)

    H, W = img1.shape[:2]
    ori_size = (H, W)

    # To tensor: (1, 3, H, W) float32 [0, 255]
    def to_tensor(img):
        t = torch.from_numpy(img).permute(2, 0, 1).float()
        return t.unsqueeze(0).to(device)

    t1 = to_tensor(img1)
    t2 = to_tensor(img2)

    # Optionally resize for inference
    if inference_size is not None:
        t1 = F.interpolate(t1, size=inference_size, mode="bilinear", align_corners=True)
        t2 = F.interpolate(t2, size=inference_size, mode="bilinear", align_corners=True)

    # Pad to multiple of padding_factor
    H_pad, W_pad = t1.shape[2], t1.shape[3]
    pad_h = (padding_factor - H_pad % padding_factor) % padding_factor
    pad_w = (padding_factor - W_pad % padding_factor) % padding_factor

    if pad_h > 0 or pad_w > 0:
        t1 = F.pad(t1, [0, pad_w, 0, pad_h])
        t2 = F.pad(t2, [0, pad_w, 0, pad_h])

    with torch.no_grad():
        results = model(
            t1, t2,
            attn_type=attn_type,
            attn_splits_list=list(attn_splits_list),
            corr_radius_list=list(corr_radius_list),
            prop_radius_list=list(prop_radius_list),
            num_reg_refine=num_reg_refine,
            task="flow",
        )

    flow_pr = results["flow_preds"][-1]  # (1, 2, H', W') — finest scale prediction

    # Remove padding and resize back to original
    flow_pr = flow_pr[:, :, :H_pad, :W_pad]
    if inference_size is not None or (pad_h > 0 or pad_w > 0):
        flow_pr = F.interpolate(
            flow_pr, size=ori_size, mode="bilinear", align_corners=True
        )
        # Rescale flow values to account for resolution change
        scale_h = ori_size[0] / H_pad
        scale_w = ori_size[1] / W_pad
        flow_pr[:, 0] *= scale_w  # u (horizontal)
        flow_pr[:, 1] *= scale_h  # v (vertical)

    flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
    return flow.astype(np.float32)


def run_unimatch_with_backward(
    model, device: str,
    img1: np.ndarray, img2: np.ndarray,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run UniMatch in both directions for forward-backward consistency.
    Returns: flow_fwd (H,W,2), flow_bwd (H,W,2)
    """
    flow_fwd = run_unimatch(model, device, img1, img2, **kwargs)
    flow_bwd = run_unimatch(model, device, img2, img1, **kwargs)
    return flow_fwd, flow_bwd