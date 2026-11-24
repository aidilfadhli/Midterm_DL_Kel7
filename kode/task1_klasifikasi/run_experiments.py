import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Ensure imports work whether invoked via `python run_all.py`, `python -m task1_klasifikasi.run_experiments`,
# or directly as `python task1_klasifikasi/run_experiments.py`.
KODE_DIR = Path(__file__).resolve().parents[1]
if str(KODE_DIR) not in sys.path:
    sys.path.insert(0, str(KODE_DIR))

from common.seed import DATA_DIR, FIG_DIR, RESULTS_DIR, ROOT
from task1_klasifikasi.dataset import (
    CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    MultimodalDefectDataset,
    compute_image_stats,
    compute_sensor_stats,
)
from task1_klasifikasi.models import (
    ImageOnlyClassifier,
    MultimodalCrossAttention,
    MultimodalLateFusion,
    SignalOnlyClassifier,
)
from task1_klasifikasi.train_eval import evaluate, set_seed, train_epoch

OUT_DIR = RESULTS_DIR / "task1"
CKPT_DIR = OUT_DIR / "checkpoints"
CSV_PATH = ROOT / "prediksi_klasifikasi.csv"


def save_learning_curve(history, title, path):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'Loss Curve - {title}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle=':')

    plt.subplot(1, 2, 2)
    plt.plot(history['train_f1'], label='Train Macro-F1')
    plt.plot(history['val_f1'], label='Val Macro-F1')
    plt.title(f'Macro-F1 Curve - {title}')
    plt.xlabel('Epoch')
    plt.ylabel('Macro-F1')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def train_and_eval_model(
    model_name,
    model,
    train_loader,
    val_loader,
    epochs=15,
    lr=1e-3,
    criterion=None,
    device='cpu',
    save_path=None,
    curve_path=None,
    resume=False,
):
    if criterion is None:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if resume and save_path and Path(save_path).exists():
        try:
            model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
            val_loss, val_f1, targets, preds = evaluate(model, val_loader, device, criterion)
            print(f"\n[RESUME] Loaded {model_name} from {save_path} -> Val Macro-F1: {val_f1:.4f} (Val Loss: {val_loss:.4f})")
            return float(val_f1), targets, preds, None
        except Exception as e:
            print(f"[RESUME] Could not load {save_path} ({e}); retraining from scratch...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0.0
    best_targets, best_preds = None, None
    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}

    print(f"\n--- Training {model_name} ({epochs} epochs) ---")
    start_time = time.time()
    for ep in range(1, epochs + 1):
        tr_loss, tr_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1, targets, preds = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(tr_f1)
        history['val_f1'].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_targets, best_preds = targets, preds
            if save_path:
                torch.save(model.state_dict(), save_path)

        if ep % 3 == 0 or ep == epochs:
            print(f"  Ep [{ep:2d}/{epochs:2d}] Loss: {tr_loss:.4f}/{val_loss:.4f} | F1: {tr_f1:.4f}/{val_f1:.4f}", flush=True)

    elapsed = time.time() - start_time
    print(f"Finished {model_name} in {elapsed:.1f}s. Best Val Macro-F1: {best_val_f1:.4f}", flush=True)

    if curve_path:
        save_learning_curve(history, model_name, curve_path)

    return float(best_val_f1), best_targets, best_preds, history


def detect_likely_mislabels(model, dataloader, meta_df, device, top_k=500):
    """Inspect samples where model disagrees with inspector label with high confidence."""
    model.eval()
    records = []
    offset = 0
    with torch.no_grad():
        for batch in dataloader:
            img = batch['image'].to(device)
            sig = batch['sensor'].to(device)
            labels = batch['label'].to(device)
            logits = model(img, sig)
            probs = F.softmax(logits, dim=-1)
            losses = F.cross_entropy(logits, labels, reduction='none')
            pred_cls = probs.argmax(dim=-1)
            pred_conf = probs.max(dim=-1).values
            label_prob = probs.gather(1, labels.unsqueeze(1)).squeeze(1)

            for i in range(len(labels)):
                idx = offset + i
                lbl = int(labels[i].item())
                prd = int(pred_cls[i].item())
                if prd != lbl:
                    records.append({
                        'id_unit': meta_df.loc[idx, 'id_unit'],
                        'lini': meta_df.loc[idx, 'lini'],
                        'inspector_label': IDX_TO_CLASS[lbl],
                        'model_pred': IDX_TO_CLASS[prd],
                        'pred_conf': float(pred_conf[i].item()),
                        'inspector_label_prob': float(label_prob[i].item()),
                        'ce_loss': float(losses[i].item()),
                    })
            offset += len(labels)

    df_mis = pd.DataFrame(records).sort_values('ce_loss', ascending=False)
    high_conf_mis = df_mis[df_mis['pred_conf'] >= 0.85]
    print(
        f"Total validation disagreements: {len(df_mis)} | "
        f"High-confidence (>=0.85) likely mislabels: {len(high_conf_mis)} "
        f"({len(high_conf_mis) / len(meta_df) * 100:.2f}%)"
    )
    return df_mis.head(top_k), int(len(high_conf_mis))


def main(resume: bool = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Active Device: {device} | Resume mode: {resume}", flush=True)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset from shared repo data directory (../data/raw/)
    print("Loading data...", flush=True)
    train_npz = np.load(DATA_DIR / 'klasifikasi_train.npz')
    train_meta = pd.read_csv(DATA_DIR / 'klasifikasi_train_meta.csv')
    test_npz = np.load(DATA_DIR / 'klasifikasi_test.npz')
    test_meta = pd.read_csv(DATA_DIR / 'klasifikasi_test_meta.csv')

    # Fixed Stratified Split for fair comparisons (80% train, 20% validation)
    train_idx, val_idx = train_test_split(
        np.arange(len(train_meta)),
        test_size=0.2,
        stratify=train_meta['kelas'],
        random_state=42,
    )

    train_meta_sub = train_meta.iloc[train_idx].reset_index(drop=True)
    val_meta_sub = train_meta.iloc[val_idx].reset_index(drop=True)

    # Compute sensor & image stats strictly on training split (float64 safe!)
    sensor_stats_global = compute_sensor_stats(train_npz['sensor'][train_idx], train_meta_sub, per_line=False)
    sensor_stats_perline = compute_sensor_stats(train_npz['sensor'][train_idx], train_meta_sub, per_line=True)
    image_stats_global = compute_image_stats(train_npz['gambar'][train_idx], train_meta_sub, per_line=False)
    image_stats_perline = compute_image_stats(train_npz['gambar'][train_idx], train_meta_sub, per_line=True)

    print("Verified float64 sensor_stats_global:", sensor_stats_global)
    print("Verified float64 sensor_stats_perline:", sensor_stats_perline, flush=True)

    pin = (device.type == 'cuda')

    def get_loaders(per_line=False, sensor_drop=0.15, batch_size=128, augment=True):
        s_stats = sensor_stats_perline if per_line else sensor_stats_global
        i_stats = image_stats_perline if per_line else image_stats_global
        train_ds = MultimodalDefectDataset(
            images=train_npz['gambar'][train_idx],
            sensors=train_npz['sensor'][train_idx],
            meta_df=train_meta_sub,
            sensor_stats=s_stats,
            image_stats=i_stats,
            is_train=True,
            sensor_dropout_prob=sensor_drop,
            per_line_norm=per_line,
            augment=augment,
        )
        val_ds = MultimodalDefectDataset(
            images=train_npz['gambar'][val_idx],
            sensors=train_npz['sensor'][val_idx],
            meta_df=val_meta_sub,
            sensor_stats=s_stats,
            image_stats=i_stats,
            is_train=False,
            sensor_dropout_prob=0.0,
            per_line_norm=per_line,
            augment=False,
        )
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin),
            val_ds,
        )

    results_summary = {}

    # =========================================================================
    # EXPERIMENT 1: Single Modality vs Multimodal Fusion (Poin b & c)
    # =========================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Comparing Modalities & Fusion Architectures")
    print("=" * 60, flush=True)

    train_loader, val_loader, _ = get_loaders(per_line=False, sensor_drop=0.15, augment=True)
    train_loader_sig, val_loader_sig, _ = get_loaders(per_line=False, sensor_drop=0.0, augment=False)

    # 1. Image-Only
    set_seed(42)
    m_img = ImageOnlyClassifier(num_classes=7).to(device)
    f1_img, t_img, p_img, _ = train_and_eval_model(
        'Image-Only CNN', m_img, train_loader, val_loader, epochs=15, device=device,
        save_path=CKPT_DIR / 'image_only.pt', resume=resume,
    )
    results_summary['Image-Only'] = f1_img

    # 2. Sensor-Only
    set_seed(42)
    m_sig = SignalOnlyClassifier(num_classes=7).to(device)
    f1_sig, t_sig, p_sig, _ = train_and_eval_model(
        'Sensor-Only 1D-CNN', m_sig, train_loader_sig, val_loader_sig, epochs=15, device=device,
        save_path=CKPT_DIR / 'sensor_only.pt', resume=resume,
    )
    results_summary['Sensor-Only'] = f1_sig

    # 3. Multimodal Late Fusion
    set_seed(42)
    m_late = MultimodalLateFusion(num_classes=7).to(device)
    f1_late, t_late, p_late, _ = train_and_eval_model(
        'Multimodal Late Fusion', m_late, train_loader, val_loader, epochs=15, device=device,
        save_path=CKPT_DIR / 'multimodal_late.pt',
        curve_path=FIG_DIR / 't1_curve_late_fusion.png', resume=resume,
    )
    results_summary['Late Fusion'] = f1_late

    # 4. Multimodal Cross-Attention
    set_seed(42)
    m_cross = MultimodalCrossAttention(num_classes=7).to(device)
    f1_cross, t_cross, p_cross, _ = train_and_eval_model(
        'Multimodal Cross-Attention', m_cross, train_loader, val_loader, epochs=15, device=device,
        save_path=CKPT_DIR / 'multimodal_cross.pt',
        curve_path=FIG_DIR / 't1_curve_cross_attention.png', resume=resume,
    )
    results_summary['Cross-Attention'] = f1_cross

    # Generate Per-Class F1 comparison chart (Poin c)
    per_class_f1 = {
        'Image-Only': f1_score(t_img, p_img, average=None),
        'Sensor-Only': f1_score(t_sig, p_sig, average=None),
        'Late Fusion': f1_score(t_late, p_late, average=None),
        'Cross-Attention': f1_score(t_cross, p_cross, average=None),
    }
    df_per_class = pd.DataFrame(per_class_f1, index=CLASSES)
    print("\nPer-Class F1 Score Comparison:")
    print(df_per_class.round(4), flush=True)
    df_per_class.to_csv(OUT_DIR / 'per_class_f1_comparison.csv')

    df_per_class.plot(kind='bar', figsize=(11, 5), width=0.8, colormap='viridis')
    plt.title('Per-Class F1: Nilai Tiap Modalitas vs Fusion', fontsize=12, fontweight='bold')
    plt.ylabel('F1 Score', fontsize=10)
    plt.xlabel('Kelas Cacat', fontsize=10)
    plt.xticks(rotation=15)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 't1_modality_value_comparison.png', dpi=300)
    plt.close()

    # =========================================================================
    # EXPERIMENT 2: Imperfect Labels & Label Smoothing (Poin d)
    # =========================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Robustness to Imperfect Labels (Poin d)")
    print("=" * 60, flush=True)

    m_late.load_state_dict(torch.load(CKPT_DIR / 'multimodal_late.pt', map_location=device, weights_only=True))
    df_mis, n_likely_mis = detect_likely_mislabels(m_late, val_loader, val_meta_sub, device)
    df_mis.to_csv(OUT_DIR / 'likely_mislabels.csv', index=False)
    results_summary['Likely Mislabels (Val conf>=0.85)'] = n_likely_mis

    # Train Late Fusion WITHOUT label smoothing (standard CrossEntropy)
    set_seed(42)
    m_no_ls = MultimodalLateFusion(num_classes=7).to(device)
    f1_no_ls, _, _, _ = train_and_eval_model(
        'Late Fusion (No Label Smoothing, eps=0.0)', m_no_ls, train_loader, val_loader,
        epochs=15, criterion=nn.CrossEntropyLoss(label_smoothing=0.0), device=device,
        save_path=CKPT_DIR / 'multimodal_late_no_ls.pt', resume=resume,
    )
    results_summary['Late Fusion (No Label Smoothing)'] = f1_no_ls
    print(f"F1 without Label Smoothing: {f1_no_ls:.4f} vs with Label Smoothing (0.1): {f1_late:.4f}", flush=True)

    # =========================================================================
    # EXPERIMENT 3: Domain Shift Handling (Poin e)
    # =========================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Domain Shift Mitigation (Poin e)")
    print("=" * 60, flush=True)

    val_lines = val_meta_sub['lini'].values
    print("Baseline (Global Normalization):")
    for line in ['A', 'B']:
        m = (val_lines == line)
        f1_line = float(f1_score(t_late[m], p_late[m], average='macro'))
        print(f"  Line {line}: Macro-F1 = {f1_line:.4f}")
        results_summary[f'Global Norm Line {line}'] = f1_line

    # Train with Per-Line Normalization (both image & sensor)
    train_loader_pl, val_loader_pl, _ = get_loaders(per_line=True, sensor_drop=0.15, augment=True)
    set_seed(42)
    m_pl = MultimodalLateFusion(num_classes=7).to(device)
    f1_pl, t_pl, p_pl, _ = train_and_eval_model(
        'Late Fusion + Per-Line Normalization', m_pl, train_loader_pl, val_loader_pl,
        epochs=15, device=device,
        save_path=CKPT_DIR / 'multimodal_late_perline.pt',
        curve_path=FIG_DIR / 't1_curve_late_fusion_perline.png', resume=resume,
    )
    results_summary['Late Fusion (Per-Line Norm)'] = f1_pl
    print("With Per-Line Normalization (Image + Sensor):")
    for line in ['A', 'B']:
        m = (val_lines == line)
        f1_line_pl = float(f1_score(t_pl[m], p_pl[m], average='macro'))
        print(f"  Line {line}: Macro-F1 = {f1_line_pl:.4f}")
        results_summary[f'Per-Line Norm Line {line}'] = f1_line_pl

    # Also test Cross-Attention + Per-Line Normalization
    set_seed(42)
    m_cross_pl = MultimodalCrossAttention(num_classes=7).to(device)
    f1_cross_pl, t_cross_pl, p_cross_pl, _ = train_and_eval_model(
        'Cross-Attention + Per-Line Normalization', m_cross_pl, train_loader_pl, val_loader_pl,
        epochs=15, device=device,
        save_path=CKPT_DIR / 'multimodal_cross_perline.pt', resume=resume,
    )
    results_summary['Cross-Attention (Per-Line Norm)'] = f1_cross_pl
    for line in ['A', 'B']:
        m = (val_lines == line)
        f1_line_cpl = float(f1_score(t_cross_pl[m], p_cross_pl[m], average='macro'))
        print(f"  Cross-Attn Per-Line Line {line}: Macro-F1 = {f1_line_cpl:.4f}", flush=True)
        results_summary[f'Cross-Attn Per-Line Norm Line {line}'] = f1_line_cpl

    # =========================================================================
    # EXPERIMENT 4: Missing Modality Robustness (Poin f)
    # =========================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Missing Sensor Simulation (Poin f)")
    print("=" * 60, flush=True)

    chosen_norm = bool(f1_pl >= f1_late)
    chosen_arch = 'cross_attn' if (chosen_norm and f1_cross_pl > f1_pl) or (not chosen_norm and f1_cross > f1_late) else 'late_fusion'
    print(f"Best configuration selected: arch={chosen_arch}, per_line_norm={chosen_norm}", flush=True)

    ModelCls = MultimodalCrossAttention if chosen_arch == 'cross_attn' else MultimodalLateFusion
    eval_model = ModelCls(num_classes=7).to(device)
    if chosen_arch == 'cross_attn':
        eval_ckpt = CKPT_DIR / ('multimodal_cross_perline.pt' if chosen_norm else 'multimodal_cross.pt')
        active_val_targets, active_val_preds = (t_cross_pl, p_cross_pl) if chosen_norm else (t_cross, p_cross)
    else:
        eval_ckpt = CKPT_DIR / ('multimodal_late_perline.pt' if chosen_norm else 'multimodal_late.pt')
        active_val_targets, active_val_preds = (t_pl, p_pl) if chosen_norm else (t_late, p_late)
    eval_model.load_state_dict(torch.load(eval_ckpt, map_location=device, weights_only=True))

    val_ds_missing = MultimodalDefectDataset(
        images=train_npz['gambar'][val_idx],
        sensors=np.full_like(train_npz['sensor'][val_idx], np.nan),
        meta_df=val_meta_sub,
        sensor_stats=sensor_stats_perline if chosen_norm else sensor_stats_global,
        image_stats=image_stats_perline if chosen_norm else image_stats_global,
        is_train=False,
        sensor_dropout_prob=0.0,
        per_line_norm=chosen_norm,
    )
    val_loader_missing = DataLoader(val_ds_missing, batch_size=128, shuffle=False, pin_memory=pin)
    _, f1_missing, _, _ = evaluate(eval_model, val_loader_missing, device)
    print(f"Best Model ({chosen_arch}) with 100% Missing Sensor: Macro-F1 = {f1_missing:.4f} (sensible fallback to image!)")
    results_summary['Best Model (100% Missing Sensor)'] = float(f1_missing)

    # Also evaluate on naturally missing sensor samples in validation
    val_sensors = train_npz['sensor'][val_idx]
    nat_missing_mask = np.isnan(val_sensors).all(axis=(1, 2))
    if nat_missing_mask.sum() > 0:
        f1_nat_missing = float(f1_score(active_val_targets[nat_missing_mask], active_val_preds[nat_missing_mask], average='macro'))
        print(f"Natural missing-sensor subset in Val (N={nat_missing_mask.sum()}): Macro-F1 = {f1_nat_missing:.4f}", flush=True)
        results_summary['Natural Missing Sensor Val Macro-F1'] = f1_nat_missing

    # =========================================================================
    # EXPERIMENT 5: Final Model over 3 Random Seeds (Poin g)
    # =========================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Final Model Evaluation over 3 Random Seeds (Poin g)")
    print("=" * 60, flush=True)
    seeds = [42, 123, 999]
    seed_scores = []

    active_loaders = (train_loader_pl, val_loader_pl) if chosen_norm else (train_loader, val_loader)
    val_ensemble_probs = np.zeros((len(val_meta_sub), 7), dtype=np.float32)

    for s in seeds:
        set_seed(s)
        final_model = ModelCls(num_classes=7).to(device)
        ckpt_path = CKPT_DIR / f'final_model_seed_{s}.pt'
        score, _, _, _ = train_and_eval_model(
            f'Final Model ({chosen_arch}, per_line={chosen_norm}, Seed {s})',
            final_model,
            active_loaders[0],
            active_loaders[1],
            epochs=18,
            device=device,
            save_path=ckpt_path,
            curve_path=FIG_DIR / f't1_curve_final_seed_{s}.png',
            resume=resume,
        )
        seed_scores.append(score)

        # Accumulate validation probabilities for ensemble check
        final_model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        final_model.eval()
        s_probs = []
        with torch.no_grad():
            for batch in active_loaders[1]:
                logits = final_model(batch['image'].to(device), batch['sensor'].to(device))
                s_probs.append(F.softmax(logits, dim=-1).cpu().numpy())
        val_ensemble_probs += np.vstack(s_probs) / len(seeds)

    mean_f1 = float(np.mean(seed_scores))
    std_f1 = float(np.std(seed_scores))
    val_targets_final = np.array([CLASS_TO_IDX[k] for k in val_meta_sub['kelas']])
    val_ens_preds = val_ensemble_probs.argmax(axis=-1)
    ens_val_f1 = float(f1_score(val_targets_final, val_ens_preds, average='macro'))

    print(f"\nFinal Model Results across 3 Seeds ({chosen_arch}, per_line_norm={chosen_norm}):")
    for s, sc in zip(seeds, seed_scores):
        print(f"  Seed {s:3d}: Macro-F1 = {sc:.4f}")
    print(f"Mean Macro-F1: {mean_f1:.4f} +/- {std_f1:.4f}")
    print(f"3-Seed Ensemble Val Macro-F1: {ens_val_f1:.4f}")
    print("\n3-Seed Ensemble Classification Report (Validation):")
    print(classification_report(val_targets_final, val_ens_preds, target_names=CLASSES, digits=4), flush=True)

    for line in ['A', 'B']:
        m = (val_lines == line)
        lf1 = float(f1_score(val_targets_final[m], val_ens_preds[m], average='macro'))
        print(f"  Ensemble Val Line {line}: Macro-F1 = {lf1:.4f}")
        results_summary[f'Final Ensemble Line {line}'] = lf1

    results_summary['Final Model Architecture'] = chosen_arch
    results_summary['Final Model Per-Line Norm'] = chosen_norm
    results_summary['Final Model Seeds'] = {str(s): float(sc) for s, sc in zip(seeds, seed_scores)}
    results_summary['Final Model (3 Seeds)'] = f"{mean_f1:.4f} +/- {std_f1:.4f}"
    results_summary['Final 3-Seed Ensemble Val Macro-F1'] = ens_val_f1

    # Save Confusion Matrix of Final Ensemble
    cm = confusion_matrix(val_targets_final, val_ens_preds, normalize='true')
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Normalized Confusion Matrix - Final 3-Seed Ensemble (Val)', fontsize=11, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 't1_confusion_matrix_final.png', dpi=300)
    plt.close()

    with open(OUT_DIR / 'experiment_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    # =========================================================================
    # FINAL PREDICTIONS: Generate prediksi_klasifikasi.csv
    # =========================================================================
    print("\n" + "=" * 60)
    print("GENERATING FINAL PREDICTIONS FOR HIDDEN TEST SET")
    print("=" * 60, flush=True)

    test_ds = MultimodalDefectDataset(
        images=test_npz['gambar'],
        sensors=test_npz['sensor'],
        meta_df=test_meta,
        sensor_stats=sensor_stats_perline if chosen_norm else sensor_stats_global,
        image_stats=image_stats_perline if chosen_norm else image_stats_global,
        is_train=False,
        sensor_dropout_prob=0.0,
        per_line_norm=chosen_norm,
        augment=False,
    )
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, pin_memory=pin)

    ensemble_probs = np.zeros((len(test_meta), 7), dtype=np.float32)
    for s in seeds:
        ckpt_path = CKPT_DIR / f'final_model_seed_{s}.pt'
        model = ModelCls(num_classes=7).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()

        seed_probs = []
        with torch.no_grad():
            for batch in test_loader:
                img = batch['image'].to(device)
                sig = batch['sensor'].to(device)
                logits = model(img, sig)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                seed_probs.append(probs)
        ensemble_probs += np.vstack(seed_probs) / len(seeds)

    final_preds = ensemble_probs.argmax(axis=-1)

    submission_df = pd.DataFrame({
        'id_unit': test_meta['id_unit'],
        'kelas': [IDX_TO_CLASS[p] for p in final_preds],
    })

    # Format checks
    assert list(submission_df.columns) == ['id_unit', 'kelas']
    assert len(submission_df) == len(test_meta)
    assert submission_df['id_unit'].equals(test_meta['id_unit'])
    assert submission_df['kelas'].isin(CLASSES).all()

    submission_df.to_csv(CSV_PATH, index=False)
    print(f"Saved '{CSV_PATH}' successfully!")
    print(f"Total rows: {len(submission_df)}")
    print(submission_df.head(10))
    print("\nPredicted Class Distribution in Test Set:")
    print(submission_df['kelas'].value_counts())

    # Sanity verification: ensure all 7 classes are predicted!
    predicted_classes = set(submission_df['kelas'].unique())
    missing_classes = set(CLASSES) - predicted_classes
    if missing_classes:
        print(f"WARNING: Missing predicted classes in test set: {missing_classes}")
    else:
        print("VERIFIED: All 7 classes are predicted in 'prediksi_klasifikasi.csv'!", flush=True)
    return results_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run all Task 1 experiments and generate predictions.")
    parser.add_argument('--resume', action='store_true', help="Reuse existing checkpoints where available.")
    args = parser.parse_args()
    main(resume=args.resume)
