# Task 1: Multimodal Defect Classification

Owned by `aidilfadhli`. Output: `../prediksi_klasifikasi.csv` with columns `id_unit,kelas`
(`normal`, `gores`, `noda`, `retak`, `gelembung`, `las_dingin`, `kontaminasi`).

## Files

- `dataset.py`: `MultimodalDefectDataset`, `float64`-safe global & per-line sensor/image normalization (`compute_sensor_stats`, `compute_image_stats`), flip augmentation, missing-sensor masking, and modality dropout.
- `models.py`: `ImageEncoder` (2D ResNet), `SignalEncoder` (1D ResNet), `ImageOnlyClassifier`, `SignalOnlyClassifier`, `MultimodalLateFusion`, and `MultimodalCrossAttention`.
- `train_eval.py`: training/evaluation loops, single-batch overfitting sanity check (`run_sanity_check`), and single-model CLI.
- `run_experiments.py`: runs all Task 1 experiments (image-only, sensor-only, late fusion, cross-attention, likely-mislabel detection + label smoothing, per-line normalization, 100% & natural missing-sensor evaluation, 3-seed final ensemble + confusion matrix + learning curves) and writes `../prediksi_klasifikasi.csv`, metrics to `results/task1/`, and figures (`t1_*.png`) to `../laporan/figures/`.

## Run (from `kode/`)

All paths use `common/seed.py` (`../data/raw/` for the dataset):

```bash
python run_all.py --task 1                              # sanity check + all experiments + prediksi_klasifikasi.csv
python -m task1_klasifikasi.train_eval --sanity-check
python -m task1_klasifikasi.run_experiments             # full run
python -m task1_klasifikasi.run_experiments --resume    # reuse saved checkpoints in results/task1/checkpoints/
```

## Seeds

- Train/validation stratified split (80/20): `random_state=42`.
- Model comparisons: `seed=42`.
- Final 3-seed ensemble: `seeds = [42, 123, 999]`.
