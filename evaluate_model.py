#!/usr/bin/env python3
"""
evaluate_model.py

Loads a saved Siamese-Fusion model and evaluates its performance on the test dataset.
Computes multi-label classification metrics and visualizes results.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

from sklearn.metrics import classification_report, precision_recall_fscore_support, precision_recall_curve

# Import the TFDatasetBuilder from data_pipeline.py
from data_loader import TFDatasetBuilder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="./data/training_data_S2/image_metadata.csv",
                        help="Path to CSV file with TIF paths and labels.")
    parser.add_argument("--model_path", type=str, default="models/siamese_fusion.h5",
                        help="Path to the saved model file.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation.")
    args = parser.parse_args()

    # 1) Build the test dataset (train/val also available if needed)
    builder = TFDatasetBuilder(
        csv_path=args.csv_path,
        seed=42,
        clip_percentiles=True,    # Same config used in training
        do_robust_scaling=True,
        do_augmentation=False     # Usually disable augmentation for evaluation
    )

    # We'll just build all three; we only *need* test_ds for final evaluation
    train_ds, val_ds, test_ds = builder.build_datasets(
        train_split=0.70,
        val_split=0.15,
        sample_size=100,         # number of images for p1/p99 & robust scaling
        batch_size=args.batch_size,
        shuffle_buffer=1000
    )

    # 2) Load the saved model
    model = keras.models.load_model(args.model_path)
    model = keras.models.load_model(args.model_path)
    model.summary()

    # 3) Gather labels and predictions on the test set
    y_true = []
    y_pred = []

    # We can iterate once over test_ds to accumulate predictions
    for x_batch, y_batch in test_ds:
        preds = model.predict_on_batch(x_batch)  # shape (batch, 8)
        y_true.append(y_batch.numpy())
        y_pred.append(preds)

    y_true = np.concatenate(y_true, axis=0)  # shape (num_samples, 8)
    y_pred = np.concatenate(y_pred, axis=0)  # shape (num_samples, 8)

    # 4) Threshold predictions at 0.5 for multi-label classification
    y_pred_bin = (y_pred >= 0.5).astype(int)

    # 5) Compute metrics
    # Label names: 7 days + 1 "any_market"
    label_names = [f"Day{i+1}" for i in range(7)] + ["AnyMarket"]

    # Classification report (precision, recall, F1 per label)
    print("=== Classification Report ===")
    print(classification_report(
        y_true, y_pred_bin,
        target_names=label_names,
        zero_division=0
    ))

    # If you want array-based precision, recall, f1 scores:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred_bin, average=None, zero_division=0
    )
    # "average=None" => returns separate scores for each of the 8 labels

    # Print macro-average or micro-average if desired:
    # e.g. macro_f1 = np.mean(f1)
    macro_f1 = np.mean(f1)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred_bin, average='micro', zero_division=0
    )
    print("=== Summary Metrics ===")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Micro-F1: {micro_f1:.4f}\n")

    # 6) Visualize F1 scores per label in a bar chart
    plt.figure(figsize=(8, 4))
    x_positions = np.arange(len(label_names))
    plt.bar(x_positions, f1, color='skyblue')
    plt.xticks(x_positions, label_names, rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.title("F1 Score per Label")
    for i, score in enumerate(f1):
        plt.text(i, score + 0.02, f"{score:.2f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

    # y_true_any: shape (n_samples,), 0 or 1 for "AnyMarket"
    # y_pred_any: shape (n_samples,), continuous probability in [0,1]

    precisions, recalls, thresholds = precision_recall_curve(y_true_any, y_pred_any)

    # Compute F1 for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

    # Find the threshold that yields the maximum F1
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"Best threshold for AnyMarket by F1: {best_threshold:.3f}")
    print(f"F1 at that threshold: {best_f1:.3f}")

if __name__ == "__main__":
    main()
