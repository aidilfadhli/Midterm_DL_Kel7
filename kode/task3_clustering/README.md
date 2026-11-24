# Task 3: Multimodal clustering

Owned by Sandy Hidayatullah (`sandyxd18`). Output: `cluster.csv` (repo root) with columns `id_sampel,cluster`.

Everything is in one notebook: `task3_clustering.ipynb`. It covers exploration, SimCLR for images and spectra,
two fusion strategies, KMeans / GMM / Agglomerative / DEC, choosing K with several criteria, UMAP/t-SNE analysis, and writing `cluster.csv`.

## Run on Google Colab (recommended)

1. Runtime > Change runtime type > GPU (T4 is enough).
2. Upload `task3_clustering.ipynb`, and upload `data_uts_dl.zip` (or `clustering.npz` + `clustering_meta.csv`) to `/content`.
   To read the data from Drive instead, set `DATA_DIR` in the configuration cell.
3. Run all cells. About 40 to 60 minutes on a T4 with the default settings. `cluster.csv` is downloaded at the end.
4. Put `cluster.csv` in the repo root. Figures are saved to `figures/` and numbers to `results.json` under the output folder printed in the setup cell.

## Run Colab from VS Code

1. Install the **Google Colab** and **Jupyter** extensions, open the notebook, then Select Kernel > Colab > New Colab Server > GPU (T4).
2. The Colab runtime cannot see local files. When the data is not found, the notebook downloads only `data_uts_dl.zip` from the Drive folder in `DATA_GDRIVE_URL` (about 400 MB, tested: ~20 s). The same folder also holds the ML task data, which is not downloaded.
3. `files.download` may not work in this setup. The last cell therefore also prints `cluster.csv`, `results.json` and the most important figures as a base64 block in the cell output (saved inside the `.ipynb`). Pull them out on this machine:

```bash
python task3_clustering/extract_outputs.py path/to/executed_notebook.ipynb
```

This puts `cluster.csv` in the repo root, `results.json` in `kode/results/task3/` and the figures in `laporan/figures/`. It verifies a sha256 checksum first. Figures that do not fit the 4 MB payload budget are listed in the output and can be rerun separately.

The Drive download, the payload and the extraction were tested locally with `FAST = True`. The VS Code Colab extension itself was not tested.

## Run locally

```bash
python run_all.py --task 3
# or
jupyter nbconvert --to notebook --execute task3_clustering/task3_clustering.ipynb --output-dir results/task3
```

The notebook finds `../data/raw/` automatically when it is run inside the repo. Inside the repo it writes `../cluster.csv`,
`results/task3/results.json`, and figures to `../laporan/figures/`.

## Settings (first cells of the notebook)

| Name | Default | Meaning |
|---|---|---|
| `FAST` | `False` | `True` runs a tiny smoke test (3000 samples, 1 to 3 epochs). The output is meaningless and is written to a separate `_fast` folder. |
| `MAIN_SEED`, `SEEDS` | `0`, `(0, 1, 2)` | Seed of the main model; seeds used for the clustering stability check. |
| `RUN_EXTRA_SSL_SEEDS` | `False` | `True` retrains the encoders for `SEEDS[1:]` and reports how much the final labels change (about 2x the time). |
| `K_OVERRIDE` | `None` | Force the number of clusters. `None` uses the consensus of the criteria in section 7. |
| `FUSION` | `None` | `'concat'` or `'joint'`. `None` picks the one with the better silhouette. |
| `FORCE_RETRAIN` | `False` | Ignore existing checkpoints in `results/task3/checkpoints/`. |

Notes

- The full run (`FAST = False`, `K_OVERRIDE = 7`) was executed on a Colab T4. `cluster.csv`, `results.json` and the figures in `laporan/figures/` come from that run. Local checks were only `FAST = True` smoke tests (Python 3.14, torch 2.11, RTX 3050).
- Why K = 7 although the criteria in section 7 vote for 5: the kNN graph has 5 connected components, and two of them split into two islands that differ in only one modality (cluster 1 vs 5: spectrum only; cluster 3 vs 4: image only). The `graph` and `agglomerative` methods find these splits with `knn_cut` of 0.001 and 0.002. This is evidence from the data, not a check against the hidden reference.
- Mixed precision uses bf16 when the GPU supports it and fp16 with a gradient scaler otherwise (T4).
- Checkpoints are git-ignored (`kode/results/**/checkpoints/*`). On Colab they disappear with the session unless the output folder is on Drive.
