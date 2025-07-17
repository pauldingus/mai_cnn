import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from keras import layers, models, callbacks
from keras.optimizers import Adam
from tensorflow import config
from data_loader import TFDatasetBuilder
import datetime
import pickle
import traceback

def build_deeper_day_encoder(
    input_shape=(128, 128, 1),
    conv_filters=[16, 32, 64],
    dense_dim=128
):
    """
    Deeper day-encoder: 3 convolution blocks + BN + MaxPool, then Flatten + Dense.
    This encoder processes a single day's image of shape (128,128,1).
    """

    inputs = layers.Input(shape=input_shape)  # shape=(128,128,1)
    x = inputs

    for f in conv_filters:
        x = layers.Conv2D(f, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(dense_dim, activation='relu')(x)

    encoder = models.Model(inputs, x, name="deeper_day_encoder")
    return encoder


def build_siamese_stacked1d_model(
    days=7,
    image_height=128,
    image_width=128,
    conv1d_filters=64,
    conv1d_layers=2,
    final_dense_dim=1,
    lr=1e-4
):
    """
    Siamese approach (shared day-encoder) + stacked 1D conv for temporal fusion.
    Outputs a single sigmoid for AnyMarket.
    
    Input format: (batch_size, 128, 128, 7) - channels-last format from data_loader
    This gets reshaped to (batch_size, 7, 128, 128, 1) for TimeDistributed processing

    days: number of time steps (7)
    image_height, image_width: typically 128
    """
    # Build the day-encoder for a single day's shape (128,128,1)
    single_day_shape = (image_height, image_width, 1)
    day_encoder = build_deeper_day_encoder(input_shape=single_day_shape)

    # The model's input has shape (H, W, days) from data_loader
    # e.g. (128, 128, 7)
    model_input = layers.Input(shape=(image_height, image_width, days))

    # Reshape from (batch, H, W, days) to (batch, days, H, W, 1)
    # Split the days dimension and add a singleton channel dimension
    reshaped = layers.Reshape((image_height, image_width, days, 1))(model_input)
    reshaped = layers.Permute((3, 1, 2, 4))(reshaped)  # (batch, days, H, W, 1)

    # Encode each day => (batch, days, embedding_dim)
    day_embeddings = layers.TimeDistributed(day_encoder)(reshaped)
    # Now day_embeddings shape => (batch, 7, dense_dim)

    # Stacked 1D conv on top of day_embeddings
    x = day_embeddings
    for i in range(conv1d_layers):
        x = layers.Conv1D(filters=conv1d_filters, kernel_size=3,
                          padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)

    # Global pooling
    x = layers.GlobalMaxPooling1D()(x)

    # Final dense => single output with sigmoid
    output = layers.Dense(final_dense_dim, activation='sigmoid')(x)

    model = models.Model(inputs=model_input, outputs=output,
                         name="siamese_stacked1d_anymarket")

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    try:
        SCRATCH = os.environ.get("SCRATCH", "/scratch/users/pdingus")
        DATA_DIR = os.path.join(SCRATCH, "mai_cnn", "data", "training_data_S2")
        CHECKPOINT_DIR = os.path.join(SCRATCH, "mai_cnn", "checkpoints", "siamese_fusion_anymarket")
        MODEL_DIR = os.path.join(SCRATCH, "mai_cnn", "models", "siamese_fusion")

        print("Using data dir:", DATA_DIR)
        print("Checkpoints will be saved to:", CHECKPOINT_DIR)
        print("Model will be saved to:", MODEL_DIR)
        print("Available GPUs:", config.list_physical_devices("GPU"))

        # Set up save directory
        model_name = "siamese_fusion_anymarket"
        base_dir = os.environ.get("SCRATCH", "/scratch/users/pdingus")
        model_path = os.path.join(base_dir, "mai_cnn/models/siamese_fusion_anymarket")

        # If the model folder path doesn't exist, create it
        os.makedirs(model_path, exist_ok=True)

        # Make a subfolder of the current date, if it doesn't exist
        current_date_string = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        save_path = os.path.join(model_path, current_date_string)
        os.makedirs(save_path, exist_ok=True)

        # Training arguments
        training_args = {
            'scaling': 'standard',
            'per_image_scaling': True,
            'do_augmentation': True,
            'do_clipping': True,
            'lower_clip': 0,
            'upper_clip': 40,
        }

        # Build datasets using TFDatasetBuilder
        builder = TFDatasetBuilder(
            csv_path="./data/training_data_S2/image_metadata.csv",
            **training_args
        )

        train_ds, val_ds, test_ds = builder.build_datasets(
            train_split=0.70,
            val_split=0.15,
            sample_size=1000,  # number of images for scaling
            batch_size=16,
            shuffle_buffer=256,
        )

        # Cache and prefetch datasets
        train_ds = train_ds.cache().prefetch(buffer_size=2)
        val_ds = val_ds.cache().prefetch(buffer_size=2)
        test_ds = test_ds.cache().prefetch(buffer_size=2)

        # Build the model
        model = build_siamese_stacked1d_model(lr=1e-4)
        model.summary()

        # Set up callbacks
        checkpoint_dir = os.path.join(save_path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_callback = callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,  # Save full model, not just weights
            verbose=1,
        )

        callback_list = [
            checkpoint_callback,
            callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, verbose=1),
        ]

        # Train the model
        print("Starting training...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=30,
            callbacks=callback_list
        )

        # Evaluate the model on the test set
        test_loss, test_accuracy = model.evaluate(test_ds)
        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")

        # Save the model and artifacts
        save_model = True
        save_final_model = False
        if save_model:
            # Save the model
            if save_final_model:
                model.save(f"{save_path}/{model_name}.keras")
                print(f"[INFO] Model saved to {save_path}")

            # Save the history
            with open(f"{save_path}/history.pkl", "wb") as f:
                pickle.dump(history.history, f)

            # Save the scaler
            scaler = builder.scaler
            with open(f"{save_path}/scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)
            
            # Save the training arguments for model_application.py
            with open(f"{save_path}/training_args.pkl", "wb") as f:
                pickle.dump(training_args, f)
            print(f"[INFO] Training arguments saved to {save_path}/training_args.pkl")

        # Test the model with dummy data to verify it works
        print("Testing model architecture with dummy data...")
        import numpy as np
        dummy_input = np.random.random((2, 128, 128, 7))  # batch_size=2
        dummy_output = model.predict(dummy_input)
        print(f"Model test successful. Input shape: {dummy_input.shape}, Output shape: {dummy_output.shape}")

    # To support graceful shutdown, catch errors and capture traceback
    except Exception as e:
        print(f"[FATAL ERROR] Training/Evaluation failed: {e}")
        import traceback
        traceback.print_exc()