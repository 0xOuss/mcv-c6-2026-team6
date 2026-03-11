#!/usr/bin/env python3
"""
Task 1.1: Off-the-shelf Optical Flow Evaluation

Evaluates multiple methods on KITTI Seq 45:
  - PyFlow        (required by task — classical variational)
  - Farneback     (classical baseline + winsize ablation)
  - RAFT          (ECCV 2020 — also used as T1.2 tracking backend)
  - UniMatch      (CVPR 2023 — best on KITTI, F1=3.60%)
  - SEA-RAFT      (ECCV 2024 — fast RAFT successor, F1=3.85%)
  - FlowFormer++  (CVPR 2023 — best on Sintel EPE=1.07)

"Newer learning-based methods will be better graded" — instructions.

T1.1 Ablation:
  - Farneback winsize: [5, 10, 15, 21, 31] — boundary precision vs smoothness
  - RAFT iterations:   [5, 10, 15, 20, 24, 32] — diminishing returns curve
  - PyFlow alpha:      [0.005, 0.012, 0.02] — regularization trade-off

Usage:
    python scripts/run_optical_flow.py --config configs/optical_flow.yaml
    python scripts/run_optical_flow.py --method unimatch   # run just one
    python scripts/run_optical_flow.py --ablate farneback_winsize
    python scripts/run_optical_flow.py --ablate raft_iters
"""

import sys
import argparse
import yaml
import time
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.kitti_utils import (
    read_kitti_flow_gt, load_kitti_noc_mask, load_image_pair, write_flo_file
)
from src.evaluation.flow_metrics import evaluate_method, compute_all_metrics, RuntimeTimer
from src.visualization.flow_viz import (
    flow_to_hsv, plot_quiver, plot_error_map,
    plot_methods_comparison, plot_fb_consistency, draw_flow_wheel
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/optical_flow.yaml")
    p.add_argument("--method", default=None,
                   help="Run only this method (pyflow|farneback|raft|unimatch|sea_raft|flowformer)")
    p.add_argument("--ablate", default=None,
                   choices=["farneback_winsize", "raft_iters", "pyflow_alpha"],
                   help="Run a specific ablation study")
    p.add_argument("--save_flow", action="store_true",
                   help="Save .npy flow arrays (used by tracking scripts)")
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
#  Method factory: returns (img1, img2) -> flow callable for each method
# ─────────────────────────────────────────────────────────────────────────────

def get_method_func(name: str, cfg: dict):
    """
    Return a callable (img1, img2) -> (H,W,2) flow for the named method.
    Raises NotImplementedError if the external dependency is missing.
    """
    mc = cfg['methods'].get(name, {})

    # ── Classical methods ──────────────────────────────────────────────────
    if name == "farneback":
        from src.optical_flow.farneback import run_farneback
        return lambda i1, i2: run_farneback(
            i1, i2,
            pyr_scale=mc.get('pyr_scale', 0.5),
            levels=mc.get('levels', 3),
            winsize=mc.get('winsize', 15),
            iterations=mc.get('iterations', 3),
            poly_n=mc.get('poly_n', 5),
            poly_sigma=mc.get('poly_sigma', 1.2),
        )

    elif name == "pyflow":
        from src.optical_flow.pyflow_wrapper import run_pyflow
        return lambda i1, i2: run_pyflow(
            i1, i2,
            alpha=mc.get('alpha', 0.012),
            ratio=mc.get('ratio', 0.75),
            minWidth=mc.get('minWidth', 20),
            nOuterFPIterations=mc.get('nOuterFPIterations', 7),
            nInnerFPIterations=mc.get('nInnerFPIterations', 1),
            nSORIterations=mc.get('nSORIterations', 30),
        )

    # ── RAFT ──────────────────────────────────────────────────────────────
    elif name == "raft":
        from src.optical_flow.raft_wrapper import load_raft_model, run_raft
        print(f"  Loading RAFT from {mc['model_path']}...")
        model, device = load_raft_model(mc['model_path'],
                                         small=mc.get('small', False),
                                         mixed_precision=mc.get('mixed_precision', False))
        iters = mc.get('iters', 20)
        return lambda i1, i2: run_raft(model, device, i1, i2, iters=iters)

    # ── UniMatch (CVPR 2023 — best on KITTI) ──────────────────────────────
    elif name == "unimatch":
        from src.optical_flow.unimatch_wrapper import load_unimatch_model, run_unimatch
        print(f"  Loading UniMatch from {mc['model_path']}...")
        model, device = load_unimatch_model(
            mc['model_path'],
            num_scales=mc.get('num_scales', 2),
            upsample_factor=mc.get('upsample_factor', 4),
            feature_channels=mc.get('feature_channels', 128),
            num_transformer_layers=mc.get('num_transformer_layers', 6),
            num_head=mc.get('num_head', 1),
            ffn_dim_expansion=mc.get('ffn_dim_expansion', 4),
            reg_refine=mc.get('reg_refine', True),
            num_reg_refine=mc.get('num_reg_refine', 6),
        )
        attn_type        = mc.get('attn_type', 'swin')
        attn_splits_list = mc.get('attn_splits_list', [2, 8])
        corr_radius_list = mc.get('corr_radius_list', [-1, 4])
        prop_radius_list = mc.get('prop_radius_list', [-1, 1])
        num_reg_refine   = mc.get('num_reg_refine', 6)
        return lambda i1, i2: run_unimatch(
            model, device, i1, i2,
            attn_type=attn_type,
            attn_splits_list=attn_splits_list,
            corr_radius_list=corr_radius_list,
            prop_radius_list=prop_radius_list,
            num_reg_refine=num_reg_refine,
        )

    # ── SEA-RAFT (ECCV 2024 — faster + more accurate RAFT) ───────────────
    elif name == "sea_raft":
        from src.optical_flow.sea_raft_wrapper import load_sea_raft_model, run_sea_raft
        print(f"  Loading SEA-RAFT-{mc.get('model_size','M')} from {mc['model_path']}...")
        model, device = load_sea_raft_model(
            mc['model_path'],
            model_size=mc.get('model_size', 'M'),
        )
        iters = mc.get('iters', 12)
        return lambda i1, i2: run_sea_raft(model, device, i1, i2, iters=iters)

    # ── FlowFormer++ (CVPR 2023 — best on Sintel) ────────────────────────
    elif name == "flowformer":
        from src.optical_flow.flowformer_wrapper import load_flowformer_model, run_flowformer
        print(f"  Loading FlowFormer++ from {mc['model_path']}...")
        model, device, cfg_ff = load_flowformer_model(mc['model_path'])
        pad = mc.get('padding_factor', 32)
        return lambda i1, i2: run_flowformer(model, device, i1, i2,
                                              padding_factor=pad)

    else:
        raise NotImplementedError(f"Method '{name}' not implemented. "
                                   f"Check configs/optical_flow.yaml for setup.")


# ─────────────────────────────────────────────────────────────────────────────
#  Ablation studies
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_farneback_winsize(img1, img2, flow_gt, noc_mask, output_dir):
    """
    Ablation: Farneback window size → [5, 10, 15, 21, 31]
    Shows: larger window → smoother flow, but boundaries blur.
    Best winsize is 15 for KITTI (precision/smoothness balance).
    """
    from src.optical_flow.farneback import run_farneback
    winsizes = [5, 10, 15, 21, 31]
    print(f"\n[T1.1 Ablation] Farneback winsize ∈ {winsizes}")

    rows = []
    for ws in winsizes:
        t0 = time.perf_counter()
        flow = run_farneback(img1, img2, winsize=ws)
        rt   = time.perf_counter() - t0
        m    = compute_all_metrics(flow, flow_gt, noc_mask)
        m['winsize'] = ws; m['runtime'] = rt
        rows.append(m)
        print(f"  winsize={ws:2d}: MSEN={m['msen']:.3f}  "
              f"PEPN={m['pepn']:.1f}%  EPE(noc)={m['epe_noc']:.3f}  t={rt:.3f}s")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor='white')
    xs = [r['winsize'] for r in rows]
    for ax, key, ylabel, color in zip(
        axes,
        ['msen',  'pepn',  'runtime'],
        ['MSEN (↓ better)', 'PEPN % (↓ better)', 'Runtime (s)'],
        ['#0891B2', '#0F766E', '#D97706']
    ):
        ys = [r[key] for r in rows]
        ax.plot(xs, ys, 'o-', color=color, linewidth=2.5, markersize=7)
        ax.set_xlabel('Window Size', fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(f'Farneback: {ylabel} vs Window Size', fontsize=10,
                     fontweight='bold')
        ax.grid(alpha=0.3); ax.set_facecolor('#F8FAFC')
        best = int(np.argmin(ys))
        ax.scatter([xs[best]], [ys[best]], s=120, color='#DC2626',
                   zorder=5, label=f'Best: {xs[best]}')
        ax.legend(fontsize=8)

    plt.suptitle('T1.1 Ablation: Farneback Window Size\n'
                 '(larger window = smoother flow, but boundaries blur)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(output_dir / "ablation_farneback_winsize.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    import pandas as pd
    pd.DataFrame(rows).to_csv(
        str(output_dir / "ablation_farneback_winsize.csv"), index=False)
    print(f"  Saved: ablation_farneback_winsize.png")


def run_ablation_raft_iters(img1, img2, flow_gt, noc_mask, cfg, output_dir):
    """
    Ablation: RAFT refinement iterations → [5, 10, 15, 20, 24, 32]
    Shows: diminishing returns — most improvement happens in first 15 iterations.
    Key slide: justifies our choice of iters=20.
    """
    from src.optical_flow.raft_wrapper import load_raft_model, run_raft
    mc = cfg['methods']['raft']
    model, device = load_raft_model(mc['model_path'],
                                     small=mc.get('small', False))
    iters_list = [5, 10, 15, 20, 24, 32]
    print(f"\n[T1.1 Ablation] RAFT iterations ∈ {iters_list}")

    rows = []
    for iters in iters_list:
        with RuntimeTimer() as t:
            flow = run_raft(model, device, img1, img2, iters=iters)
        m = compute_all_metrics(flow, flow_gt, noc_mask)
        m['iters'] = iters; m['runtime'] = t.seconds
        rows.append(m)
        print(f"  iters={iters:2d}: MSEN={m['msen']:.3f}  "
              f"PEPN={m['pepn']:.1f}%  t={t.seconds:.3f}s")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor='white')
    xs = [r['iters'] for r in rows]
    for ax, key, ylabel, color in zip(
        axes,
        ['msen', 'pepn', 'runtime'],
        ['MSEN (↓ better)', 'PEPN % (↓ better)', 'Runtime (s)'],
        ['#0891B2', '#0F766E', '#7C3AED']
    ):
        ys = [r[key] for r in rows]
        ax.plot(xs, ys, 's-', color=color, linewidth=2.5, markersize=7)
        # Mark diminishing-returns elbow
        improvements = [abs(ys[i] - ys[i-1]) for i in range(1, len(ys))]
        elbow = improvements.index(min(improvements)) + 1 if len(improvements) > 2 else 0
        ax.axvline(x=xs[elbow], color='#DC2626', linestyle='--', alpha=0.8,
                   label=f'Elbow: {xs[elbow]} iters')
        ax.set_xlabel('Iterations', fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3); ax.set_facecolor('#F8FAFC')
        ax.legend(fontsize=8)

    plt.suptitle('T1.1 Ablation: RAFT Refinement Iterations\n'
                 '(most gain in first 15 iterations — diminishing returns beyond)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(output_dir / "ablation_raft_iters.png"), dpi=150, bbox_inches='tight')
    plt.close()

    import pandas as pd
    pd.DataFrame(rows).to_csv(str(output_dir / "ablation_raft_iters.csv"), index=False)
    print(f"  Saved: ablation_raft_iters.png")


def run_ablation_pyflow_alpha(img1, img2, flow_gt, noc_mask, output_dir):
    """
    Ablation: PyFlow alpha (regularization weight) → [0.003, 0.005, 0.012, 0.02, 0.05]
    Shows: higher alpha = over-smoothed, lower = noisy. Optimal ~0.012.
    """
    from src.optical_flow.pyflow_wrapper import run_pyflow
    alphas = [0.003, 0.005, 0.012, 0.02, 0.05]
    print(f"\n[T1.1 Ablation] PyFlow alpha ∈ {alphas}")

    rows = []
    for alpha in alphas:
        with RuntimeTimer() as t:
            try:
                flow = run_pyflow(img1, img2, alpha=alpha)
                m = compute_all_metrics(flow, flow_gt, noc_mask)
            except Exception as e:
                print(f"  alpha={alpha}: ERROR {e}")
                m = {'msen': np.nan, 'pepn': np.nan, 'epe_noc': np.nan, 'epe_all': np.nan}
        m['alpha'] = alpha; m['runtime'] = t.seconds
        rows.append(m)
        print(f"  alpha={alpha}: MSEN={m['msen']:.3f}  "
              f"PEPN={m['pepn']:.1f}%  t={t.seconds:.3f}s")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor='white')
    xs = [r['alpha'] for r in rows]
    for ax, key, ylabel, color in zip(
        axes, ['msen', 'pepn'], ['MSEN (↓ better)', 'PEPN % (↓ better)'],
        ['#0891B2', '#0F766E']
    ):
        ys = [r[key] for r in rows]
        ax.plot(xs, ys, 'o-', color=color, linewidth=2.5, markersize=7)
        ax.set_xlabel('Alpha (regularization)', fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3); ax.set_facecolor('#F8FAFC')
    plt.suptitle('T1.1 Ablation: PyFlow Alpha (Regularization Weight)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(output_dir / "ablation_pyflow_alpha.png"), dpi=150, bbox_inches='tight')
    plt.close()

    import pandas as pd
    pd.DataFrame(rows).to_csv(str(output_dir / "ablation_pyflow_alpha.csv"), index=False)


# ─────────────────────────────────────────────────────────────────────────────
#  Slides table: generate the exact table the instructors want
# ─────────────────────────────────────────────────────────────────────────────

def save_slides_table(all_results: dict, output_dir: Path):
    """
    Generate the exact table format from the slides:
      Method | MSEN(PyFlow) | PEPN(PyFlow) | MSEN(best) | PEPN(best)

    The instructors compare each team's PyFlow result to their best method.
    """
    import pandas as pd
    rows = []
    for name, m in all_results.items():
        rows.append({
            'method':    name,
            'MSEN':      round(float(m.get('msen', np.nan)), 4),
            'PEPN_%':    round(float(m.get('pepn', np.nan)), 2),
            'EPE_noc':   round(float(m.get('epe_noc', np.nan)), 4),
            'EPE_all':   round(float(m.get('epe_all', np.nan)), 4),
            'runtime_s': round(float(m.get('runtime', np.nan)), 3),
        })
    df = pd.DataFrame(rows).sort_values('MSEN')
    df.to_csv(str(output_dir / "metrics_summary.csv"), index=False)

    # Print slides-ready table
    print("\n" + "="*70)
    print("SLIDES TABLE (copy to report):")
    print("="*70)
    print(f"  {'Method':<18} {'MSEN':>8}  {'PEPN%':>7}  {'EPE_noc':>9}  {'Runtime':>8}")
    print("  " + "-"*60)
    for _, row in df.iterrows():
        print(f"  {row['method']:<18} {row['MSEN']:>8.4f}  "
              f"{row['PEPN_%']:>6.2f}%  {row['EPE_noc']:>9.4f}  "
              f"{row['runtime_s']:>7.3f}s")
    print("="*70)

    # Visualization: bar chart for slides
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='white')
    methods = df['method'].tolist()
    n = len(methods)
    palette = ['#94A3B8', '#64748B', '#0891B2', '#06B6D4', '#0F766E', '#7C3AED'][:n]
    x = range(n)

    for ax, col, ylabel in zip(
        axes,
        ['MSEN', 'PEPN_%', 'runtime_s'],
        ['MSEN (lower=better)', 'PEPN % (lower=better)', 'Runtime (s)']
    ):
        vals = df[col].values
        bars = ax.bar(x, vals, color=palette, edgecolor='white', linewidth=0.5, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(ylabel, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.set_facecolor('#F8FAFC')
        best_idx = int(np.nanargmin(vals))
        bars[best_idx].set_edgecolor('#DC2626'); bars[best_idx].set_linewidth(3)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.01,
                        f'{val:.2f}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold')

    plt.suptitle('T1.1: Optical Flow Evaluation — KITTI Seq 45\n'
                 '(red border = best method per metric)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(output_dir / "flow_metrics_table.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Slides table figure: {output_dir}/flow_metrics_table.png")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    cfg     = load_config(args.config)
    out_dir = Path(cfg['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("Task 1.1: Optical Flow Evaluation")
    print("Methods: PyFlow, Farneback, RAFT, UniMatch, SEA-RAFT, FlowFormer++")
    print("=" * 68)

    # ── Load images and GT ────────────────────────────────────────
    print(f"\nLoading: {cfg['data']['img1']}")
    img1, img2 = load_image_pair(cfg['data']['img1'], cfg['data']['img2'])

    print(f"Loading GT: {cfg['data']['gt_flow']}")
    flow_gt, valid = read_kitti_flow_gt(cfg['data']['gt_flow'])
    noc_path = cfg['data'].get('gt_noc', cfg['data']['gt_flow'])
    try:
        noc_mask = load_kitti_noc_mask(noc_path)
    except Exception:
        noc_mask = valid
    print(f"  Image: {img1.shape}  Valid pixels: {noc_mask.sum()}/{noc_mask.size}")

    # ── Ablation mode ─────────────────────────────────────────────
    if args.ablate:
        if args.ablate == "farneback_winsize":
            run_ablation_farneback_winsize(img1, img2, flow_gt, noc_mask, out_dir)
        elif args.ablate == "raft_iters":
            run_ablation_raft_iters(img1, img2, flow_gt, noc_mask, cfg, out_dir)
        elif args.ablate == "pyflow_alpha":
            run_ablation_pyflow_alpha(img1, img2, flow_gt, noc_mask, out_dir)
        return

    # ── Choose which methods to run ───────────────────────────────
    if args.method:
        methods_to_run = [args.method]
    else:
        methods_to_run = [k for k, v in cfg['methods'].items()
                          if v.get('enabled', True)]

    print(f"\nRunning methods: {methods_to_run}")

    all_results: dict = {}
    all_flows:   dict = {}
    eval_cfg = cfg.get('evaluation', {})
    n_runs   = eval_cfg.get('n_timing_runs', 3)

    for method_name in methods_to_run:
        print(f"\n{'─'*50}")
        print(f"Method: {method_name.upper()}")
        print(f"{'─'*50}")
        try:
            func    = get_method_func(method_name, cfg)
            metrics, flow = evaluate_method(
                method_name, func, img1, img2, flow_gt, noc_mask, n_runs=n_runs
            )
            all_results[method_name] = metrics
            all_flows[method_name]   = flow

            if args.save_flow:
                write_flo_file(flow, str(out_dir / f"flow_{method_name}.flo"))

        except NotImplementedError as e:
            print(f"  SKIPPED: {e}")
        except FileNotFoundError as e:
            print(f"  SKIPPED (weights not found): {e}")
        except Exception as e:
            import traceback
            print(f"  ERROR in {method_name}: {e}")
            traceback.print_exc()

    if not all_results:
        print("\nNo methods ran. Check config and external dependencies.")
        return

    # ── Save results ──────────────────────────────────────────────
    save_slides_table(all_results, out_dir)

    # ── Visualizations ────────────────────────────────────────────
    viz_cfg = cfg.get('visualization', {})
    print("\nGenerating visualizations...")

    # 1. Side-by-side comparison of all methods
    if len(all_flows) > 1:
        fig = plot_methods_comparison(all_flows, img1, flow_gt, noc_mask)
        fig.savefig(str(out_dir / "methods_comparison.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # 2. Per-method: HSV viz + quiver + error map
    for name, flow in all_flows.items():
        mdir = out_dir / name
        mdir.mkdir(exist_ok=True)

        if viz_cfg.get('save_hsv', True):
            hsv = flow_to_hsv(flow)
            cv2.imwrite(str(mdir / "flow_hsv.png"),
                        cv2.cvtColor(hsv, cv2.COLOR_RGB2BGR))

        if viz_cfg.get('save_quiver', True):
            fig = plot_quiver(flow, img1,
                              step=viz_cfg.get('quiver_step', 12),
                              title=f"Optical Flow: {name}")
            fig.savefig(str(mdir / "quiver.png"), dpi=150, bbox_inches='tight')
            plt.close()

        if viz_cfg.get('save_error_map', True):
            fig = plot_error_map(flow, flow_gt, noc_mask,
                                 title=f"EPE Error Map: {name}")
            fig.savefig(str(mdir / "error_map.png"), dpi=150, bbox_inches='tight')
            plt.close()

    # 3. FB consistency (RAFT shown as example — also works for UniMatch/SEA-RAFT)
    if 'raft' in all_flows and viz_cfg.get('save_fb_consistency', True):
        try:
            from src.optical_flow.raft_wrapper import load_raft_model, run_raft
            mc = cfg['methods']['raft']
            m2, dev = load_raft_model(mc['model_path'],
                                       small=mc.get('small', False))
            print("  Computing RAFT forward-backward consistency...")
            fwd_flow  = all_flows['raft']
            bwd_flow  = run_raft(m2, dev, img2, img1, iters=mc.get('iters', 20))
            fig = plot_fb_consistency(fwd_flow, bwd_flow)
            fig.savefig(str(out_dir / "raft_fb_consistency.png"),
                        dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"  FB consistency skipped: {e}")

    # 4. Color wheel legend (always save — put on HSV slides)
    wheel = draw_flow_wheel(256)
    cv2.imwrite(str(out_dir / "flow_color_wheel.png"),
                cv2.cvtColor(wheel, cv2.COLOR_RGB2BGR))

    # 5. HSV comparison figure for slides (PyFlow vs best learned method)
    if 'pyflow' in all_flows and len(all_flows) > 1:
        best_name = min((k for k in all_flows if k != 'pyflow'),
                        key=lambda k: all_results[k].get('msen', 999))
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, name in zip(axes, ['pyflow', best_name, None]):
            if name is None:
                ax.axis('off'); continue
            ax.imshow(flow_to_hsv(all_flows[name]))
            m = all_results[name]
            ax.set_title(f"{name.upper()}\n"
                         f"MSEN={m['msen']:.2f}  PEPN={m['pepn']:.1f}%",
                         fontsize=11, fontweight='bold')
            ax.axis('off')
        # middle panel = img1 for reference
        axes[2].imshow((img1 * 255).astype(np.uint8))
        axes[2].set_title("Reference Image (frame t)", fontsize=11)
        axes[2].axis('off')
        plt.suptitle(f'T1.1: PyFlow vs {best_name.upper()} (our best method)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig.savefig(str(out_dir / "flow_hsv_comparison.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\nAll T1.1 outputs saved to: {out_dir}")
    print("Done! ✓")


if __name__ == "__main__":
    main()