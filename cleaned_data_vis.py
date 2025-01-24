#!/usr/bin/env python3
"""
cleaned_data_vis.py

Script that:
  1) Imports TFDatasetBuilder from data_pipeline.py
  2) Builds a train dataset
  3) Retrieves one batch
  4) Visualizes one day/channel of one image
"""

import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

# Import the TFDatasetBuilder class from data_pipeline.py
# Ensure data_pipeline.py is in the same directory or in your Python path
from data_loader import TFDatasetBuilder

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

    # Initialize the builder
    builder = TFDatasetBuilder(
        csv_path=args.csv_path,
        scaling=args.scaling,
        do_augmentation = args.do_augmentation,
        do_clipping = args.do_clipping,
        lower_clip = args.lower_clip,
        upper_clip = args.upper_clip
    )

    # Build datasets
    train_ds, _, _ = builder.build_datasets(
        train_split=0.70,
        val_split=0.15,
        sample_size=args.sample_size,
        batch_size=args.batch_size,

    )

    # Retrieve one batch from the train dataset
    # For demonstration, we'll just take one iteration
    for images, labels in train_ds.take(10):
        # images shape: (batch_size, 7, 128, 128, 1)
        # labels shape: (batch_size, 8)

        # Convert to numpy
        images_np = images.numpy()
        labels_np = labels.numpy()
        valmax = np.max(images_np)

        # Let's pick a random index in the batch
        idx_in_batch = random.randint(0, images_np.shape[0] - 1)

        # Extract the label vector for this sample
        label_vec = labels_np[idx_in_batch]  # shape (8,)

        # Print some info about this image
        print(f"Showing image index {idx_in_batch} in batch.")
        print(f"Label vector for this sample: {label_vec}")

        # Display all 7 day slices with matplotlib
        fig, axes = plt.subplots(1, 7, figsize=(20, 5))
        for day_index in range(7):
            # Extract that single day slice => shape (128, 128)
            day_slice = images_np[idx_in_batch, day_index, :, :, 0]

            # Print some info about this day slice
            print(f"Day {day_index} slice shape: {day_slice.shape}")
            print(f"Value range in day {day_index} slice: {np.min(day_slice):.4f}, {np.max(day_slice):.4f}")

            # Display the day slice
            ax = axes[day_index]
            ax.imshow(day_slice, cmap='gray', vmin=0, vmax=valmax)
            ax.set_title(f"Day {day_index}")
            ax.axis('off')

        plt.suptitle(f"Label vector: {label_vec}, Max val: {valmax}")
        plt.show()

        # We only visualize one sample, so break after the first
        #break

if __name__ == "__main__":
    main()
