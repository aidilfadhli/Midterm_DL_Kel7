# UTS Deep Learning — Group 7: Classification, Regression, and Clustering

Final submission repository for the Deep Learning midterm exam (Group 7).

## Group Members

| Name | NIM | GitHub | Task |
|---|---|---|---|
| Muhammad Aidil Fadhli Awalludin | `101032300087` | `aidilfadhli` | Task 1: Multimodal Defect Classification |
| Abdu Fattah Hidayatullah | `101032300039` | `Elheims` | Task 2: Remaining Useful Life (RUL) Regression |
| Sandy Hidayatullah | `101032300009` | `sandyxd18` | Task 3: Multimodal Material Clustering |

## Repository Layout (`main`)

| Path | Content |
|---|---|
| `laporan.pdf` | Final report (13 pages main body + references ≤ 18-page limit; 7 pages appendices) |
| `kode/` | All code, `README.md`, `requirements.txt`, saved weights, and `run_all.py` |
| `prediksi_klasifikasi.csv` | Task 1 predictions: `id_unit,kelas` (16,000 rows, 7 classes) |
| `prediksi_rul.csv` | Task 2 predictions: `id_unit,rul` (300 rows, `[0, 150]`) |
| `cluster.csv` | Task 3 predictions: `id_sampel,cluster` (36,000 rows, integers `0–6`) |
| `kontribusi.md` | Member roles, pull request links, and AI-tool usage + verification |

Supporting folders (not graded directly): `data/` (git-ignored raw dataset), `catatan/` (decision logs & presentation notes), `laporan/` (LaTeX sources and figures used to build `laporan.pdf`).

## How to Run Everything (One Command)

### 1. Set up the environment

```bash
python -m venv .venv
# Windows (PowerShell/CMD):
.venv\Scripts\python -m pip install -r kode/requirements.txt
# Linux / macOS:
# .venv/bin/python -m pip install -r kode/requirements.txt
```

> If you are on a CPU-only machine or a different CUDA version, install the matching PyTorch build from https://pytorch.org/get-started/locally/ first, then install the remaining packages from `kode/requirements.txt`.

### 2. Place the dataset

Extract `data_uts_dl.zip` into `data/raw/` so it contains:
- `klasifikasi_train.npz`, `klasifikasi_train_meta.csv`, `klasifikasi_test.npz`, `klasifikasi_test_meta.csv`
- `rul_train.csv.gz`, `rul_test.csv.gz`
- `clustering.npz`, `clustering_meta.csv`

### 3. Reproduce all numbers, figures, and CSV predictions

Run from inside `kode/`:

```bash
cd kode
python run_all.py            # all 3 tasks (Task 1 + Task 2 + Task 3)
```

Or run a single task:

```bash
python run_all.py --task 1   # Task 1: sanity check + all experiments + prediksi_klasifikasi.csv
python run_all.py --task 2   # Task 2: full pipeline (~14 min GPU) + prediksi_rul.csv + demo check
python run_all.py --task 3   # Task 3: executes task3_clustering.ipynb (~40-60 min T4 GPU) + cluster.csv
```

### 4. Fast live reproduction for the oral exam (no training)

```bash
cd kode
python -m task2_rul.demo     # ~15 s: recomputes 11 Task 2 report numbers + verifies prediksi_rul.csv
python -m task2_rul.predict  # ~15 s: regenerates prediksi_rul.csv from saved final_weights/
```

See [kode/README.md](kode/README.md) and each task's README (`kode/task1_klasifikasi/README.md`, `kode/task2_rul/README.md`, `kode/task3_clustering/README.md`) for full details, seeds, and Colab instructions.
