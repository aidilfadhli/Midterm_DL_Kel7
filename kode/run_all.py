"""One command to reproduce every number in the report.

Usage:
    python run_all.py            # all tasks
    python run_all.py --task 2   # one task

Each task registers its pipeline steps below as they are implemented.
"""
import argparse

from task1_klasifikasi import run_experiments as t1_run
from task1_klasifikasi import train_eval as t1_train_eval
from task2_rul import baseline as t2_baseline
from task2_rul import check_preprocess as t2_check
from task2_rul import deep as t2_deep
from task2_rul import demo as t2_demo
from task2_rul import error_analysis as t2_error
from task2_rul import evaluate as t2_evaluate
from task2_rul import eda as t2_eda
from task2_rul import final as t2_final
from task2_rul import labels as t2_labels
from task2_rul import maintenance as t2_maintenance
from task2_rul import uncertainty as t2_uncertainty
from task2_rul import validation as t2_validation
from task2_rul import windows as t2_windows

def run_task3() -> None:
    """Task 3: execute the notebook; it writes cluster.csv, results.json and figures itself.

    The full run was done on a Colab T4 (see task3_clustering/README.md). Locally this needs the
    GPU/time for SimCLR; use FAST = True in the notebook for a smoke test.
    """
    import subprocess
    import sys
    from pathlib import Path

    here = Path(__file__).resolve().parent
    out = here / "results" / "task3"
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
         str(here / "task3_clustering" / "task3_clustering.ipynb"), "--output-dir", str(out),
         "--ExecutePreprocessor.timeout=-1"],
        check=True,
    )


# Filled in step by step as each part of a task is implemented.
TASKS = {
    1: [t1_train_eval.run_sanity_check,  # (g) overfit-one-batch sanity check
        t1_run.main],                    # (b-g) modalities, fusion, label smoothing, domain shift, missing sensor, 3-seed final + prediksi_klasifikasi.csv
    2: [t2_eda.main, t2_check.main,  # (a) exploration + preprocessing checks
        t2_labels.main,              # (b) RUL labels
        t2_windows.main,             # (c) sliding windows + leakage checks
        t2_evaluate.main,            # the ruler: machine split + test-like cuts (checked against test)
        t2_baseline.main,            # (d0) non-deep baseline: Ridge + boosted trees
        t2_deep.cnn_ablation,        # (d1) 1D-CNN: original + 3 overfitting fixes, 3 seeds each (~3 min, GPU)
        t2_error.main,               # (d1) why some CNN predictions are far off (wear onset)
        t2_deep.gru_runs_isolated,   # (d2) GRU: W = 30 and 50, 3 seeds each (~2.5 min, GPU; separate process, see deep.py)
        t2_validation.main,          # (e) per-RUL-range table + 5-fold CV by machine (~15-20 min, GPU)
        t2_uncertainty.main,         # (f1) 90% intervals: MC dropout / 5-GRU ensemble / both, calibrated (~1.5 min)
        t2_maintenance.main,         # (f2) when to service: replay held-out machines, point vs cautious rule (~2.5 min)
        t2_final.main,               # final: shift by simulation, 5 GRUs on all 900 machines, prediksi_rul.csv (~3 min)
        t2_demo.main],               # check: recomputed numbers must match the report and prediksi_rul.csv
    3: [run_task3],  # (a-f) notebook: SimCLR, fusion, KMeans/GMM/Agglomerative/graph/DEC, K criteria, UMAP/t-SNE + cluster.csv
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=sorted(TASKS), default=None)
    args = parser.parse_args()

    tasks = [args.task] if args.task else sorted(TASKS)
    for t in tasks:
        if not TASKS[t]:
            print(f"Task {t}: no steps implemented yet")
            continue
        for step in TASKS[t]:
            print(f"Task {t}: running {step.__module__}")
            step()


if __name__ == "__main__":
    main()
