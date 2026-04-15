### Master in Computer Vision (Barcelona) 2025/26
# Project 2 (Task 2) @ C6 - Video Analysis

This repository provides the starter code for Task 2 of Project 2: Action spotting on the SoccerNet Ball Action Spotting 2025 (SN-BAS-2025) dataset.

The installation of dependencies, how to obtain the dataset, and instructions on running the spotting baseline are detailed next.

## Dependencies

You can install the required packages for the project using the following command, with `requirements.txt` specifying the versions of the various packages:

```
pip install -r requirements.txt
```

## Getting the dataset and data preparation

Refer to the README files in the [data/soccernetball](/data/soccernetball) directory for instructions on how to download the SNABS2025 dataset, preparation of directories, and extraction of the video frames.


## Running the baseline for Task 2

The `main_spotting.py` is designed to train and evaluate the baseline using the settings specified in a configuration file. You can run `main_spotting.py` using the following command:

```
python3 main_spotting.py --model <model_name>
```

Here, `<model_name>` can be chosen freely but must match the name of a configuration file (e.g. `baseline.json`) located in the config directory [config](/config/). For example, to chose the baseline model, you would run: `python3 main_spotting.py --model baseline`.

For additional details on configuration options using the configuration file, refer to the README in the [config](/config/) directory.

## Important notes

- Before running the model, ensure that you have downloaded the dataset frames and updated the directory-related configuration parameters in the relevant [config](/config/) files.
- Make sure to run the `main_spotting.py` with the `mode` parameter set to `store` at least once to generate the clips and save them. After this initial run, you can set the `mode` to `load` to reuse the same clips in subsequent executions.

## Support

For any issues related to the code, please email [aclapes@ub.edu](mailto:aclapes@ub.edu) and CC [arturxe@gmail.com](mailto:arturxe@gmail.com).



# Ablation Study — Ball Action Spotting (W6)
**MCV-C6 | Project 2 | Task 2**

---

## Overview

Your job is to **run experiments, compare results, and explain what works and why**.  
This package gives you everything to do that cleanly.

---

## File Structure

```
ablation/
├── configs/               ← One JSON per experiment
│   ├── baseline.json
│   ├── lstm.json
│   ├── lstm_2layer.json
│   ├── tcn.json
│   ├── transformer.json
│   └── focal_loss.json
├── slurm/
│   └── job_single.sh      ← SLURM job template
├── scripts/
│   ├── aggregate_results.py  ← Print comparison table from all results
│   └── count_macs.py         ← Report MACs and param count per model
├── main_spotting.py       ← Patched with W&B logging + AP10/AP12 output
├── model_spotting.py      ← Extended with LSTM / TCN / Transformer / Focal Loss
└── run_all_ablations.sh   ← Submit all jobs at once
```

---

## Step-by-Step Instructions

### 1. Setup (do once)

Copy the files into the right places in your repo:

```bash
# From your repo root:
cp ablation/main_spotting.py .
cp ablation/model_spotting.py model/model_spotting.py
cp ablation/configs/*.json config/
cp ablation/scripts/*.py scripts/
cp ablation/slurm/job_single.sh slurm/
cp ablation/run_all_ablations.sh .

mkdir -p logs
```

Edit the following before running:

| File | What to change |
|---|---|
| `configs/*.json` | `wandb_entity` → your W&B username |
| `slurm/job_single.sh` | `--partition`, `module load cuda/...`, `cd /path/to/repo`, conda env name |
| `scripts/aggregate_results.py` | `SAVE_ROOT` → same as `save_dir` in your JSONs |

---

### 2. When teammates finish their models

Once they implement LSTM / TCN / Transformer / etc., you just need to make sure:

- Their changes live in `model/model_spotting.py` (or are merged with the patched version here)
- The config key `"temporal_arch"` is used to switch between them

If they added their own new keys to the model, just add those keys to the relevant JSON config file.

---

### 3. Run experiments

**Option A — submit all at once:**
```bash
bash run_all_ablations.sh
```

**Option B — submit one at a time (useful for debugging):**
```bash
sbatch slurm/job_single.sh baseline
sbatch slurm/job_single.sh lstm
sbatch slurm/job_single.sh tcn
# etc.
```

**Monitor jobs:**
```bash
squeue -u $USER
```

**View logs:**
```bash
tail -f logs/bas_ablation_<jobid>.out
```

---

### 4. Check results

Once jobs finish, run:
```bash
python scripts/aggregate_results.py
```

This prints a table like:
```
+------------------+--------+--------+------+-------+...
| Model            |  AP10  |  AP12  | PASS | DRIVE |...
+------------------+--------+--------+------+-------+...
| transformer      | 17.45  |  9.21  | 35.1 | 26.3  |...
| lstm             | 15.21  |  8.10  | 33.2 | 24.5  |...
| baseline         | 12.34  |  6.78  | 29.9 | 21.1  |...
+------------------+--------+--------+------+-------+...

  Delta vs baseline (AP10):
  transformer   +5.11
  lstm          +2.87
```

Also check W&B for training curves: https://wandb.ai

---

### 5. Count MACs and parameters (required for report)

```bash
pip install ptflops
python scripts/count_macs.py
```

---

## Experiment Design

These are the experiments and what each tests:

| Config | What changes vs baseline | What you're testing |
|---|---|---|
| `baseline` | Nothing | Reference point |
| `lstm` | + 1-layer LSTM on top of frame embeddings | Can RNN model temporal order? |
| `lstm_2layer` | + 2-layer LSTM | Does depth help for sequential modeling? |
| `tcn` | + 3-layer dilated causal 1D conv | Can local+dilated context improve spotting? |
| `transformer` | + 2-layer Transformer encoder | Can full self-attention help? |
| `focal_loss` | Same backbone, focal loss instead of CE | Does FL help with class imbalance? |

**One variable changes at a time.** That's what makes it an ablation study.

---

## What to Write in the Report (3 slides max)

### Slide 1 — Best Model Architecture
- Diagram showing the pipeline (backbone → temporal head → FC → softmax)
- What changed vs baseline (be specific: layer sizes, number of layers)
- Why you made these choices

### Slide 2 — Per-Class AP Results (AP10 + optionally AP12)
- Table of AP per class for your best model
- Mention which classes improved and which didn't
- Note: AP10 excludes FREE KICK and GOAL (very few examples → noisy)

### Slide 3 — Ablation Table + Insights
- Table with all experiments and their AP10
- 3–5 bullet insights, e.g.:
  - "LSTM improved PASS/DRIVE (common, temporally structured actions) but not SHOT (rare, instantaneous)"
  - "Focal loss helped with rare classes like HEADER but hurt overall"
  - "2-layer LSTM didn't help over 1-layer, suggesting 50-frame clips don't require deep recurrence"
  - "Transformer overfit on small clips — possibly needs longer context to shine"

---

## AP10 vs AP12

- **AP10** = mean AP over 10 classes, **excluding FREE KICK and GOAL**
  - These two classes have very few samples → metrics are noisy/unreliable
  - AP10 is the **primary metric** used for ranking and grading
- **AP12** = all 12 classes (what the slide shows as 6.78 baseline)

---

## W&B Tracked Metrics

Each epoch logs:
- `train/loss`, `val/loss` — loss curves
- `lr` — learning rate schedule (warmup + cosine)
- `best_val_loss` — running best

At end of training:
- `test/AP10`, `test/AP12` — headline numbers
- `test/AP_<CLASS>` — per class, e.g. `test/AP_PASS`

---

## Common Issues

| Problem | Fix |
|---|---|
| `store_mode: store` runs but doesn't train | Change to `"store_mode": "load"` in config — store only needs to be done once |
| SLURM job fails immediately | Check `logs/*.err`, probably wrong partition or missing conda env |
| W&B not logging | Make sure `WANDB_API_KEY` is set in env or run `wandb login` before submitting |
| `results.json` not found by aggregator | Check `SAVE_ROOT` in `aggregate_results.py` matches `save_dir` in configs |
| OOM on GPU | Reduce `batch_size` to 2 in the config that OOMs |
