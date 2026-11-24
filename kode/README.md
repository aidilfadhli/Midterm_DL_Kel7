# kode/

## Environment

- **Task 1 & Task 2:** Python 3.11, PyTorch 2.11.0 (CUDA 12.8 wheels), NumPy 2.4.6, pandas 3.1.1, SciPy 1.17.1, scikit-learn 1.9.1, Matplotlib 3.11.2, seaborn 0.13.2.
- **Task 3 (full run):** Google Colab T4 GPU — Python 3.13.16, PyTorch 2.14.1+cu130, NumPy 2.4.6, pandas 2.2.3, scikit-learn 1.6.1, umap-learn 0.5.12 (also runs locally via `nbconvert`).
- All pinned packages are listed in `requirements.txt`.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
# Linux / macOS / CPU-only: install matching PyTorch first, then `pip install -r requirements.txt`
```

## Data

Extract `data_uts_dl.zip` into `../data/raw/` (git-ignored). `kamus_data.md` inside the zip describes every file. All three tasks read automatically from `../data/raw/`.

## Reproduce Everything (One Command)

Run all commands below from inside `kode/`:

```bash
python run_all.py            # all 3 tasks (reproduces every report number, figure, and CSV)
python run_all.py --task 1   # only Task 1 (multimodal defect classification)
python run_all.py --task 2   # only Task 2 (RUL regression)
python run_all.py --task 3   # only Task 3 (multimodal clustering via notebook)
```

### What each task runs and writes

| Command | Steps executed | Outputs written | Approx. runtime |
|---|---|---|---|
| `python run_all.py --task 1` | Single-batch sanity check (`train_eval.run_sanity_check`), modality & fusion comparison, mislabel check + label smoothing, per-line domain shift, missing-sensor test, 3-seed Cross-Attention ensemble | `results/task1/*.json`, `results/task1/*.csv`, `../laporan/figures/t1_*.png`, `../prediksi_klasifikasi.csv` | ~15–25 min (GPU) |
| `python run_all.py --task 2` | EDA + causal preprocessing checks, RUL labels, sliding windows + leakage test, baselines, 1D-CNN ablation + error analysis, GRU runs, per-range + 5-fold CV, split-conformal intervals, maintenance replay, shift simulation, final 5-GRU retrain + demo verification | `results/task2/*.json`, `../laporan/figures/t2_*.pdf/.png`, `task2_rul/final_weights/*.pt`, `../prediksi_rul.csv` | ~14 min (RTX 3060) |
| `python run_all.py --task 3` | Executes `task3_clustering/task3_clustering.ipynb` via `jupyter nbconvert`: image & spectrum EDA, SimCLR encoders, fusion comparison, $K$-criteria + kNN graph, 5 clustering methods (incl. DEC), UMAP/t-SNE | `results/task3/results.json`, `../laporan/figures/t3_*.png`, `../cluster.csv` | ~40–60 min (Colab T4; set `FAST = True` in notebook for a quick smoke test) |

Every number quoted in `../laporan.pdf` comes directly from `results/task1/`, `results/task2/`, and `results/task3/`.

### Fast Commands (No Training — Oral Exam / Quick Check)

```bash
# Task 1: sanity check only (~10 s) or reuse saved checkpoints in results/task1/checkpoints/
python -m task1_klasifikasi.train_eval --sanity-check
python -m task1_klasifikasi.run_experiments --resume

# Task 2: recompute 11 report numbers from saved models + verify prediksi_rul.csv (~15 s)
python -m task2_rul.demo

# Task 2: regenerate prediksi_rul.csv from saved final_weights/ (~15 s)
python -m task2_rul.predict
```

> **Task 2 note:** Verified 18 Nov 2026 on RTX 3060 Laptop GPU (exit code 0; two full runs produced identical metrics and byte-identical `prediksi_rul.csv`). Numbers may vary slightly across different GPUs/drivers.

## Structure

| Folder | Content |
|---|---|
| `common/` | Shared helpers (seeding, paths, plot style) |
| `task1_klasifikasi/` | Task 1: multimodal defect classification (`aidilfadhli`) |
| `task2_rul/` | Task 2: remaining useful life regression (`Elheims`) + saved `final_weights/` |
| `task3_clustering/` | Task 3: multimodal clustering notebook (`sandyxd18`) |
| `results/` | Saved metrics (JSON/CSV); large checkpoints git-ignored except 6 small Task 2 demo weights |

## Seeds

- **Task 1:** `seed=42` for the 80/20 stratified split and comparisons; `seeds = [42, 123, 999]` for the 3-seed Cross-Attention ensemble (`task1_klasifikasi/run_experiments.py`).
- **Task 2:** `SEEDS = (0, 1, 2)` for model comparisons (`common/seed.py`), seeds `0–4` for the 5-GRU ensemble (`task2_rul/uncertainty.py`, `task2_rul/final.py`), and `seed=0` for splits and test-like cuts.
- **Task 3:** `MAIN_SEED = 0` for SimCLR and final Agglomerative clustering (`K_OVERRIDE = 7`), and `SEEDS = (0, 1, 2)` for clustering stability (`task3_clustering/task3_clustering.ipynb`).
