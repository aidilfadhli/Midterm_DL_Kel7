import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CLASSES = ['normal', 'gores', 'noda', 'retak', 'gelembung', 'las_dingin', 'kontaminasi']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}


def compute_sensor_stats(sensors_array, df_meta=None, per_line=False):
    """
    Computes mean and std across valid (non-NaN) sensor readings.
    Casts valid readings to float64 before mean/std to prevent float16 overflow.
    If per_line is True, computes separate stats for Line A and Line B.
    sensors_array: (N, 3, 128)
    """
    if not per_line or df_meta is None:
        stats = {}
        for ch in range(3):
            ch_data = sensors_array[:, ch, :]
            valid = ch_data[~np.isnan(ch_data)].astype(np.float64)
            mean = float(np.mean(valid)) if len(valid) > 0 else 0.0
            std = float(np.std(valid)) if len(valid) > 0 else 1.0
            if not np.isfinite(mean) or not np.isfinite(std):
                raise ValueError(f"Non-finite sensor stats on ch {ch}: mean={mean}, std={std}")
            stats[ch] = (mean, max(std, 1e-6))
        return stats
    else:
        stats = {'A': {}, 'B': {}}
        lines = df_meta['lini'].to_numpy()
        for line in ['A', 'B']:
            mask_line = (lines == line)
            for ch in range(3):
                ch_data = sensors_array[mask_line, ch, :]
                valid = ch_data[~np.isnan(ch_data)].astype(np.float64)
                mean = float(np.mean(valid)) if len(valid) > 0 else 0.0
                std = float(np.std(valid)) if len(valid) > 0 else 1.0
                if not np.isfinite(mean) or not np.isfinite(std):
                    raise ValueError(f"Non-finite sensor stats on line {line} ch {ch}: mean={mean}, std={std}")
                stats[line][ch] = (mean, max(std, 1e-6))
        return stats


def compute_image_stats(images_array, df_meta=None, per_line=False):
    """
    Computes mean and std of uint8 images scaled to [0, 1] using float64 accumulation.
    If per_line is True, computes separate stats for Line A and Line B.
    """
    if not per_line or df_meta is None:
        imgs_f64 = images_array.astype(np.float64) / 255.0
        mean = float(np.mean(imgs_f64))
        std = float(np.std(imgs_f64))
        return (mean, max(std, 1e-6))
    else:
        stats = {}
        lines = df_meta['lini'].to_numpy()
        for line in ['A', 'B']:
            mask_line = (lines == line)
            imgs_f64 = images_array[mask_line].astype(np.float64) / 255.0
            mean = float(np.mean(imgs_f64)) if mask_line.sum() > 0 else 0.5
            std = float(np.std(imgs_f64)) if mask_line.sum() > 0 else 0.25
            stats[line] = (mean, max(std, 1e-6))
        return stats


class MultimodalDefectDataset(Dataset):
    def __init__(self,
                 images,
                 sensors,
                 meta_df,
                 sensor_stats=None,
                 image_stats=None,
                 is_train=False,
                 transform_img=None,
                 sensor_dropout_prob=0.15,
                 per_line_norm=False,
                 augment=False):
        """
        images: uint8 array (N, 64, 64)
        sensors: float16/float32 array (N, 3, 128)
        meta_df: DataFrame with metadata (id_unit, lini, shift, id_mesin, kelas (optional))
        """
        self.images = images
        self.meta_df = meta_df.reset_index(drop=True)
        self.is_train = is_train
        self.transform_img = transform_img
        self.sensor_dropout_prob = sensor_dropout_prob
        self.per_line_norm = per_line_norm
        self.augment = augment and is_train

        self.lines = self.meta_df['lini'].to_numpy() if 'lini' in self.meta_df.columns else np.full(len(images), 'A')
        self.id_units = self.meta_df['id_unit'].to_numpy() if 'id_unit' in self.meta_df.columns else np.arange(len(images))

        # Precompute stats if not provided
        if sensor_stats is None:
            self.sensor_stats = compute_sensor_stats(sensors, self.meta_df, per_line=per_line_norm)
        else:
            self.sensor_stats = sensor_stats

        if image_stats is None:
            self.image_stats = compute_image_stats(images, self.meta_df, per_line=per_line_norm)
        else:
            self.image_stats = image_stats

        # Vectorized pre-normalization of sensors (avoids slow per-sample pandas/numpy overhead)
        sensors_f32 = sensors.astype(np.float32, copy=True)
        is_nan = np.isnan(sensors_f32)
        self.sensor_masks = (~is_nan).astype(np.float32)

        if not per_line_norm:
            for ch in range(3):
                mean, std = self.sensor_stats[ch]
                sensors_f32[:, ch, :] = (sensors_f32[:, ch, :] - mean) / std
        else:
            for line in ['A', 'B']:
                line_mask = (self.lines == line)
                if line_mask.any():
                    for ch in range(3):
                        mean, std = self.sensor_stats[line][ch]
                        sensors_f32[line_mask, ch, :] = (sensors_f32[line_mask, ch, :] - mean) / std

        sensors_f32[is_nan] = 0.0
        self.norm_sensors = sensors_f32

        # Precompute per-sample image mean/std for fast O(1) lookup
        if not per_line_norm:
            im_m, im_s = self.image_stats
            self.img_means = np.full(len(images), im_m, dtype=np.float32)
            self.img_stds = np.full(len(images), im_s, dtype=np.float32)
        else:
            self.img_means = np.empty(len(images), dtype=np.float32)
            self.img_stds = np.empty(len(images), dtype=np.float32)
            for line in ['A', 'B']:
                line_mask = (self.lines == line)
                if line_mask.any():
                    im_m, im_s = self.image_stats[line]
                    self.img_means[line_mask] = im_m
                    self.img_stds[line_mask] = im_s

        # Targets if available
        if 'kelas' in self.meta_df.columns:
            self.labels = np.array([CLASS_TO_IDX[k] for k in self.meta_df['kelas']], dtype=np.int64)
        else:
            self.labels = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Process Image: (64, 64) -> (1, 64, 64) standardized float32
        img_np = self.images[idx].astype(np.float32) / 255.0
        if self.per_line_norm:
            img_np = (img_np - self.img_means[idx]) / self.img_stds[idx]

        if self.augment:
            if np.random.rand() < 0.5:
                img_np = np.flip(img_np, axis=1).copy()
            if np.random.rand() < 0.5:
                img_np = np.flip(img_np, axis=0).copy()

        img = torch.from_numpy(img_np).unsqueeze(0)  # (1, 64, 64)

        if self.transform_img is not None:
            img = self.transform_img(img)

        # 2. Process Sensor: (3, 128)
        if self.is_train and self.sensor_dropout_prob > 0.0 and np.random.rand() < self.sensor_dropout_prob:
            sensor_tensor = torch.zeros((3, 128), dtype=torch.float32)
            mask_tensor = torch.zeros((3, 128), dtype=torch.float32)
        else:
            sensor_tensor = torch.from_numpy(self.norm_sensors[idx])
            mask_tensor = torch.from_numpy(self.sensor_masks[idx])

        item = {
            'image': img,
            'sensor': sensor_tensor,
            'sensor_mask': mask_tensor,
            'id_unit': self.id_units[idx],
            'lini': self.lines[idx]
        }

        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item
