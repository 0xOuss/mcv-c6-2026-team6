"""
src/optical_flow/flowformer_wrapper.py

FlowFormer++ Optical Flow Wrapper.

Citations:
    FlowFormer++ (this wrapper targets):
        Shi et al., "FlowFormer++: Masked Cost Volume Autoencoding for
        Pretraining Optical Flow Estimation"
        CVPR 2023.  ArXiv: https://arxiv.org/abs/2303.01237

    FlowFormer (original, same repo):
        Huang et al., "FlowFormer: A Transformer Architecture for Optical Flow"
        ECCV 2022.  ArXiv: https://arxiv.org/abs/2203.16194

Implementation:
    https://github.com/drinkingcoder/FlowFormer-Official
    Own wrapper: this file.

WHY FLOWFORMER++ IS ARCHITECTURALLY INTERESTING:
─────────────────────────────────────────────────
FlowFormer++ is the most architecturally distinct method in this comparison.
It introduces two transformer innovations not present in RAFT or UniMatch:

  1. COST VOLUME AS TOKENS:
     RAFT and UniMatch treat the cost volume as a feature MAP and apply
     convolutions/GRU over it. FlowFormer tokenises the cost volume —
     each entry becomes a token — and applies a full Transformer ENCODER
     over the entire cost volume. This means the model can reason about
     ALL possible matches simultaneously using attention, not just local
     neighbourhoods.

  2. MASKED COST VOLUME AUTOENCODING (FlowFormer++ addition):
     Pre-trains the cost volume encoder using masked autoencoding (same
     idea as MAE for images: mask 75% of cost volume tokens, reconstruct
     them). This forces the encoder to learn robust representations of
     matching ambiguity before fine-tuning on flow. Effect: better
     generalisation to unseen scenes.

  3. MEMORY ENCODER + LATENT TOKENS:
     FlowFormer introduces "latent cost tokens" — a fixed set of learned
     tokens that summarise the cost volume across iterations. Unlike
     RAFT's GRU hidden state, these tokens are explicitly attended to
     at each refinement step, giving the model long-range memory of
     the matching history.

Benchmarks:
    Sintel Final EPE (lower = better):
        RAFT:           3.17
        FlowFormer:     2.09    ← 34% better than RAFT
        FlowFormer++:   1.07    ← 66% better than RAFT

    KITTI 2015 F1-all:
        RAFT:           5.10%
        FlowFormer:     4.68%
        FlowFormer++:   4.52%
        UniMatch:       3.60%   ← UniMatch still wins on KITTI

    → Use UniMatch as "best" on KITTI. Use FlowFormer++ to show that
      benchmark performance is dataset-dependent (best on Sintel ≠ best
      on KITTI). This is a good insight for slides.

RUNTIME:
    FlowFormer++ is slower than RAFT (~0.5–1.0s/frame on KITTI resolution).
    Not suitable as the live tracking OF backend.
    Best used for Task 1.1 evaluation only — results go in the comparison table.

SETUP:
    git clone https://github.com/drinkingcoder/FlowFormer-Official external/flowformer
    cd external/flowformer && pip install -r requirements.txt

    # Download weights and place in external/flowformer/checkpoints/
    # Available at: https://github.com/drinkingcoder/FlowFormer-Official/releases
    #   flowformerpp-sintel.pth   — best EPE on Sintel
    #   flowformerpp-kitti.pth    ← use this for KITTI evaluation
    #   flowformer-sintel.pth     — original FlowFormer, Sintel
    #   flowformer-kitti.pth      — original FlowFormer, KITTI
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple


# ─── Model loading ────────────────────────────────────────────────────────────

def load_flowformer_model(
    model_path: str,
    device: str = "auto",
):
    """
    Load FlowFormer or FlowFormer++ model from checkpoint.

    The correct config is inferred automatically from the checkpoint name:
      - "kitti" in name → uses kitti config
      - Otherwise        → uses sintel config (default)

    Args:
        model_path: Path to .pth checkpoint file.
        device:     "cuda", "cpu", or "auto".

    Returns:
        model (nn.Module), device (str)
    """
    ff_dir = Path(__file__).parent.parent.parent / "external" / "flowformer"
    if not ff_dir.exists():
        raise FileNotFoundError(
            f"FlowFormer not found at {ff_dir}\n"
            "Run setup:\n"
            "  git clone https://github.com/drinkingcoder/FlowFormer-Official "
            "external/flowformer\n"
            "  cd external/flowformer && pip install -r requirements.txt\n"
            "  # Download weights to external/flowformer/checkpoints/"
        )

    for add_path in [str(ff_dir), str(ff_dir / "core")]:
        if add_path not in sys.path:
            sys.path.insert(0, add_path)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Select config based on checkpoint filename
    model_name = Path(model_path).stem.lower()
    try:
        if "kitti" in model_name:
            from configs.kitti_ft import get_cfg
        else:
            from configs.sintel import get_cfg
        cfg = get_cfg()
    except ImportError:
        # Minimal fallback config — works with most FlowFormer++ checkpoints
        from argparse import Namespace
        cfg = Namespace(
            latentcostformer=Namespace(
                encoder_latent_dim=256,
                query_latent_dim=64,
                cost_heads_num=1,
                cost_latent_input_dim=64,
                cost_latent_token_num=8,
                cost_latent_dim=128,
                cost_scale=4,
                dropout=0.0,
                encoder_depth=3,
                decoder_depth=3,
                critical_params=["cost_heads_num"],
            )
        )

    # Import FlowFormer model class
    try:
        from core.FlowFormer import FlowFormer
    except ImportError:
        try:
            from FlowFormer import FlowFormer
        except ImportError as e:
            raise ImportError(
                f"Could not import FlowFormer model: {e}\n"
                "Check that external/flowformer is set up correctly."
            )

    model = FlowFormer(cfg.latentcostformer)

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint
    if isinstance(state_dict, dict):
        state_dict = state_dict.get("model", state_dict)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"  FlowFormer++ loaded: {Path(model_path).name} | device={device}")
    return model, device


# ─── Inference ────────────────────────────────────────────────────────────────

def run_flowformer(
    model,
    device: str,
    img1: np.ndarray,
    img2: np.ndarray,
    padding_factor: int = 32,
) -> np.ndarray:
    """
    Run FlowFormer / FlowFormer++ on an image pair.

    Args:
        model, device:   from load_flowformer_model()
        img1, img2:      (H, W, 3) RGB, uint8 [0,255] or float32 [0,1]
        padding_factor:  Pad image dims to multiples of this (32 recommended).

    Returns:
        flow: (H, W, 2) float32 array, [u, v] in pixels.

    Note on memory:
        FlowFormer++ tokenises the full cost volume. On KITTI (375×1242) this
        requires ~8GB VRAM. If you get OOM, pass a smaller image or use
        inference_size=(320, 896) to reduce resolution.
    """
    if img1.dtype != np.uint8:
        img1 = (np.clip(img1, 0, 1) * 255).astype(np.uint8)
        img2 = (np.clip(img2, 0, 1) * 255).astype(np.uint8)

    H, W = img1.shape[:2]

    def to_tensor(img):
        return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)

    t1 = to_tensor(img1)
    t2 = to_tensor(img2)

    # Pad to multiple of padding_factor
    pad_h = (padding_factor - H % padding_factor) % padding_factor
    pad_w = (padding_factor - W % padding_factor) % padding_factor
    if pad_h > 0 or pad_w > 0:
        t1 = F.pad(t1, [0, pad_w, 0, pad_h])
        t2 = F.pad(t2, [0, pad_w, 0, pad_h])

    with torch.no_grad():
        # FlowFormer returns a list of flow predictions; last = finest scale
        flow_predictions = model(t1, t2)
        if isinstance(flow_predictions, (list, tuple)):
            flow_up = flow_predictions[-1]
        else:
            flow_up = flow_predictions

    flow = flow_up[0].permute(1, 2, 0).cpu().numpy()   # (H_padded, W_padded, 2)
    flow = flow[:H, :W]   # remove padding
    return flow.astype(np.float32)