"""
data_loader.py

Creates tf.data.Dataset pipelines for multi-band (7-day) TIF images
based on a CSV of file paths and correct outputs.
"""

import random
import argparse
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import tensorflow as tf
import rasterio

from sklearn.preprocessing import MinMaxScaler, RobustScaler

# ------------------------------------------------------------------------
# 1. Utility Functions
# ------------------------------------------------------------------------

def read_tif(path: str) -> np.ndarray:
    """
    Reads a single TIF file with 7 bands (days), returns np.ndarray (7, H, W).
    Expects 'path' to be a Python string, so handle any tf.string -> Python str conversion.
    """
    with rasterio.open(f"./data/{path}") as src:
        arr = src.read()  # shape: (7, H, W)
        arr = arr.astype(np.float32)
    return arr

def replace_invalid_and_crop(arr: np.ndarray, target_size: int = 128) -> np.ndarray:
    """
    1) Replace inf/nan with 0
    2) Crop to (7, target_size, target_size) if arr shape is larger
    """
    arr[~np.isfinite(arr)] = 0.0

    _, H, W = arr.shape
    if H > target_size:
        arr = arr[:, :target_size, :]
    if W > target_size:
        arr = arr[:, :, :target_size]

    return arr

# ------------------------------------------------------------------------
# 2. Data Pipeline Class
# ------------------------------------------------------------------------

class TFDatasetBuilder:
    """
    Builds tf.data.Dataset for multi-day TIF data from a CSV.
    Includes:
      - Splitting 70/15/15
      - (Optional) percentile clipping
      - (Optional) robust scaling
      - Basic data augmentation with flips/rot90
    """

    def __init__(
        self,
        csv_path: str = "./data/training_data_S2/image_metadata.csv",
        seed: int = 42,
        scaling: str = 'none', #options: 'none', 'minmax', 'standard', 'robust'
        do_augmentation: bool = False,
        do_clipping: bool = True,
        lower_clip: float = 0,
        upper_clip: float = 40
    ):
        """
        Args:
            csv_path: Path to CSV with 'image_file_path', 'market_days', 'market' etc.
            seed: Random seed for reproducibility.
            clip_percentiles: Whether to do 1st/99th percentile clipping.
            do_robust_scaling: Whether to do robust scaling (fitted on train subset).
            do_augmentation: Whether to apply random flips/rotations.
        """
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.seed = seed
        self.scaling = scaling
        self.do_augmentation = do_augmentation
        self.do_clipping  = do_clipping
        self.lower_clip = lower_clip
        self.upper_clip = upper_clip

        # Robust scaler
        self.scaler = None

        # Final file/label lists
        self.train_paths, self.train_labels = [], []
        self.val_paths, self.val_labels = [], []
        self.test_paths, self.test_labels = [], []

    def build_datasets(
        self,
        train_split=0.70,
        val_split=0.15,
        sample_size=100,
        batch_size=16,
        shuffle_buffer=1000
    ):
        """
        Main method:
          1) Shuffle & split CSV -> train/val/test
          2) (Optional) compute percentile bounds from sample of training data
          3) (Optional) fit robust scaler on training data
          4) Build tf.data.Dataset for train/val/test
        """

        # 1) Shuffle & split
        indices = list(self.df.index)
        random.Random(self.seed).shuffle(indices)

        n = len(indices)
        train_end = int(train_split * n)
        val_end   = int((train_split + val_split) * n)

        train_idx = indices[:train_end]
        val_idx   = indices[train_end:val_end]
        test_idx  = indices[val_end:]

        # Extract file paths & labels
        self.train_paths, self.train_labels = self._extract_paths_and_labels(train_idx)
        self.val_paths,   self.val_labels   = self._extract_paths_and_labels(val_idx)
        self.test_paths,  self.test_labels  = self._extract_paths_and_labels(test_idx)

        # 3) Fit robust scaler on the training data if needed
        if self.scaling == 'robust':
            self._fit_robust_scaler(sample_size=sample_size)
        elif self.scaling == 'standard':
            self._fit_standard_scaler(sample_size=sample_size)
        elif self.scaling == 'minmax':
            self._fit_minmax_scaler(sample_size=sample_size)

        # 4) Build the datasets
        train_ds = self._build_tf_dataset(
            self.train_paths, self.train_labels, batch_size, shuffle=True, buffer_size=shuffle_buffer
        )
        val_ds   = self._build_tf_dataset(
            self.val_paths,   self.val_labels,   batch_size, shuffle=False
        )
        test_ds  = self._build_tf_dataset(
            self.test_paths,  self.test_labels,  batch_size, shuffle=False
        )

        return train_ds, val_ds, test_ds

    # --------------------------------------------------------------------
    # Private / Helper Methods
    # --------------------------------------------------------------------

    def _extract_paths_and_labels(self, idx_list):
        paths = []
        labels = []
        for i in idx_list:
            row = self.df.iloc[i]
            # TIF path
            tif_path = row["image_file_path"]

            # Construct label vector (7 days + 1 "market")
            market_flag = row["market"]

            paths.append(tif_path)
            labels.append(market_flag)
        return np.array(paths), np.array(labels)

    def _fit_robust_scaler(self, sample_size=100):
        """
        Fit a robust scaler on a sample of the training data.
        """
        if len(self.train_paths) == 0:
            return

        sample_paths = random.sample(list(self.train_paths), min(sample_size, len(self.train_paths)))
        features = []
        for path in sample_paths:
            arr = read_tif(path)
            arr = replace_invalid_and_crop(arr, 128)

            # Optional: clip
            if self.do_clipping:
                arr = np.clip(arr, self.lower_clip, self.upper_clip)

            # Flatten each day => shape (7, 128*128)
            flat = arr.reshape(arr.shape[0], -1)
            features.append(flat)

        if not features:
            self.scaler = None
            return

        features = np.concatenate(features, axis=0)  # (7*n_samples, 128*128)
        self.scaler = RobustScaler()
        self.scaler.fit(features)

    def _fit_standard_scaler(self, sample_size=100):
        """
        Fit a standard scaler on a sample of the training data.
        """
        if len(self.train_paths) == 0:
            return

        sample_paths = random.sample(list(self.train_paths), min(sample_size, len(self.train_paths)))
        features = []
        for path in sample_paths:
            arr = read_tif(path)
            arr = replace_invalid_and_crop(arr, 128)

            # Optional: clip
            if self.do_clipping:
                arr = np.clip(arr, self.lower_clip, self.upper_clip)

            # Flatten each day => shape (7, 128*128)
            flat = arr.reshape(arr.shape[0], -1)
            features.append(flat)

        if not features:
            self.scaler = None
            return

        features = np.concatenate(features, axis=0)
        self.scaler = StandardScaler()
        self.scaler.fit(features)

    def _fit_minmax_scaler(self, sample_size=100):
        """
        Fit a min-max scaler on a sample of the training data.
        """
        if len(self.train_paths) == 0:
            return

        sample_paths = random.sample(list(self.train_paths), min(sample_size, len(self.train_paths)))
        features = []
        for path in sample_paths:
            arr = read_tif(path)
            arr = replace_invalid_and_crop(arr, 128)

            # Optional: clip
            if self.do_clipping:
                arr = np.clip(arr, self.lower_clip, self.upper_clip)

            # Flatten each day => shape (7, 128*128)
            flat = arr.reshape(arr.shape[0], -1)
            features.append(flat)

        if not features:
            self.scaler = None
            return

        features = np.concatenate(features, axis=0)  # (7*n_samples, 128*128)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(features)

    def _build_tf_dataset(self, paths, labels, batch_size, shuffle=False, buffer_size=1000):
        """
        Creates a tf.data.Dataset from file paths & label vectors,
        applying our custom mapping logic.
        """
        # Convert to tf tensors
        path_ds = tf.data.Dataset.from_tensor_slices(paths)
        label_ds = tf.data.Dataset.from_tensor_slices(labels)
        ds = tf.data.Dataset.zip((path_ds, label_ds))

        if shuffle:
            ds = ds.shuffle(buffer_size=buffer_size, seed=self.seed)

        # Map function
        ds = ds.map(self._load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds

    def _load_and_preprocess(self, path, label):
        """
        Converts (path, label) to (processed_image, label).
        """
        # 1) Load TIF as np array
        arr = tf.py_function(self._read_and_transform, [path], tf.float32)
        # We'll set a static shape for the returned array => (7, 128, 128, 1)
        arr.set_shape((7, 128, 128, 1))

        # 2) (Optional) data augmentation
        if self.do_augmentation:
            arr = self._augment_spatial(arr)

        return arr, label

    def _read_and_transform(self, path):
        """
        Python function (wrapped by tf.py_function) to read TIF and apply:
          - replace inf/nan -> 0
          - crop
          - clip [lower_clip, upper_clip]
          - robust scaling
          - expand dims => (7,128,128,1)
        """
        path_str = path.numpy().decode('utf-8')
        arr = read_tif(path_str)
        arr = replace_invalid_and_crop(arr, 128)

        # 1) Clip
        if self.do_clipping:
            arr = np.clip(arr, self.lower_clip, self.upper_clip)

        # 2) scale
        # Flatten each day => shape (7, 128*128)
        if self.scaler is not None:
            flat = arr.reshape(arr.shape[0], -1)
            flat = self.scaler.transform(flat)
            arr = flat.reshape(arr.shape)

        # 3) Expand dims => (7, 128, 128, 1)
        arr = arr[..., np.newaxis]

        return arr.astype(np.float32)

    def _augment_spatial(self, arr):
        """
        Example augmentation: random flip and random rotation by multiples of 90.
        (arr shape: (7, 128, 128, 1))
        We'll do a random choice whether to flip horizontally/vertically,
        and rotate by 0, 90, 180, or 270 degrees.

        Note: We must do the same transform across the 7 "days" dimension.
        """
        # Convert to tf for augmentation
        # shape => (7, 128, 128, 1)
        # We want to do the same flip/rotate for all 7 channels
        # so we combine them into a single spatial dimension [7, H, W].
        # But a simpler approach is to just treat the second dimension
        # as "batch of slices" with manual ops.

        # random flip
        flip_choice = tf.random.uniform(())
        if flip_choice < 0.5:
            arr = tf.reverse(arr, axis=[2])  # horizontal flip over W dimension

        # random rotation
        rot_choice = tf.random.uniform(())
        if rot_choice < 0.25:
            # rotate 90
            arr = tf.image.rot90(arr, k=1)  # rotates height/width
        elif rot_choice < 0.5:
            # rotate 180
            arr = tf.image.rot90(arr, k=2)
        elif rot_choice < 0.75:
            # rotate 270
            arr = tf.image.rot90(arr, k=3)
        # else no rotation

        return arr

# ------------------------------------------------------------------------
# 3. Example main() demonstrating usage
# ------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="./data/training_data_S2/image_metadata.csv",
                        help="Path to CSV with TIF paths and labels")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--scaling", type=str, default='minmax',
                        help="Type of scaling to apply: 'none', 'minmax', 'standard', 'robust'")
    parser.add_argument("--sample_size", type=int, default=100,
                        help="Number of images to sample for percentile/robust-scaler estimation")
    parser.add_argument("--do_augmentation", action='store_true',
                        help="If set, apply random flips/rot90 during training")
    parser.add_argument("--do_clipping", action='store_true',
                        help="If set, clip values at min/max values (default: 0/40)")
    parser.add_argument("--lower_clip", type=int, default=0,
                        help="Lower value to clip at, default 0")
    parser.add_argument("--upper_clip", type=int, default=40,
                        help="Upper value to clip at, default 0")
    args = parser.parse_args()

    builder = TFDatasetBuilder(
        csv_path=args.csv_path,
        scaling=args.scaling,
        do_augmentation = args.do_augmentation,
        do_clipping = args.do_clipping,
        lower_clip = args.lower_clip,
        upper_clip = args.upper_clip
    )

    train_ds, val_ds, test_ds = builder.build_datasets(
        train_split=0.70, val_split=0.15,
        sample_size=args.sample_size,
        batch_size=args.batch_size
    )

    # Example usage in a Keras model
    # Suppose you already built a model `model = build_siamese_fusion_model(...)`
    # model.fit(train_ds, epochs=20, validation_data=val_ds)

    # Just show a quick sanity check
    for images, labels in train_ds.take(1):
        print("Image batch shape:", images.shape)  # (batch_size, 7, 128, 128, 1)
        print("Label batch shape:", labels.shape)  # (batch_size, 8)
        break

if __name__ == "__main__":
    main()
