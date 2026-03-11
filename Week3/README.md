# Week3 Motion Estimator

Files added:

- `motion_estimator.py`: estimate_flow(img1,img2) using Pyflow (fallback to OpenCV).
- `evaluate.py`: find image pairs in `data/image_0/`, run estimator, report runtime and optionally MSEN/PEPN if ground-truth flows are provided.
- `utils.py`: helpers (image IO, .flo reader, MSEN/PEPN calculation).
- `requirements.txt`: dependencies for this week.

Quick start:

Install requirements (recommended in a venv):

```bash
pip install -r Week3/requirements.txt
```

Run estimator on a single pair:

```bash
python Week3/motion_estimator.py --img1 data/image_0/000000_10.png --img2 data/image_0/000000_11.png --out out_flow.npy
```

Evaluate a folder (will compute runtime for each sequence):

```bash
python Week3/evaluate.py --img_dir data/image_0 --out_csv results.csv
```

If you have ground-truth flows in `.flo` format, pass `--gt_flow_dir` to compute MSEN and PEPN.
