#!/usr/bin/env python3
"""
train_model.py

Trains the Siamese+1D-fusion model on multi-day TIF data using the TFDatasetBuilder from data_pipeline.py
"""

import argparse
from keras import callbacks
from data_loader import TFDatasetBuilder
from models.model_fusion_anymarket import build_siamese_stacked1d_model

# ------------------------------------------------------------------------
# 2. Main Training Routine
# ------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="./data/training_data_S2/image_metadata.csv",
                        help="Path to CSV with TIF paths and labels")
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
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--save_model", type=str, default="models/siamese_fusion_anymarket.keras",
                        help="Path to save the trained model (e.g., siamese_fusion.h5).")
    args = parser.parse_args()

    # 1) Build datasets using TFDatasetBuilder
    builder = TFDatasetBuilder(
        csv_path=args.csv_path,
        scaling=args.scaling,
        do_augmentation = args.do_augmentation,
        do_clipping = args.do_clipping,
        lower_clip = args.lower_clip,
        upper_clip = args.upper_clip
    )

    train_ds, val_ds, test_ds = builder.build_datasets(
        train_split=0.70,
        val_split=0.15,
        sample_size=100,            # number of images for p1/p99 & robust scaling
        batch_size=args.batch_size,
        shuffle_buffer=1000
    )

    # 2) Build the model
    model = build_siamese_stacked1d_model(lr=args.lr)
    model.summary()

    # 3) Train
    training_callbacks = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=3, factor=0.5, verbose=1
        )
    ]

    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=training_callbacks
    )

    # 4) Evaluate on test set
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"[INFO] Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # 5) Optionally save the model
    if args.save_model:
        model.save(args.save_model)
        print(f"[INFO] Model saved to {args.save_model}")

if __name__ == "__main__":
    main()
