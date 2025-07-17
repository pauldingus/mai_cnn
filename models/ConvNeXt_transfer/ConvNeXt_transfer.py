import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from tensorflow.keras.layers import (
    Conv2D,
    GlobalAveragePooling2D,
    Dense,
    BatchNormalization,
    Activation,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras import callbacks
from tensorflow import config
from data_loader import TFDatasetBuilder
import keras
import datetime
import pickle

if __name__ == "__main__":
    try:

        SCRATCH = os.environ.get("SCRATCH", "/scratch/users/pdingus")
        DATA_DIR = os.path.join(SCRATCH, "mai_cnn", "data", "training_data_S2")
        CHECKPOINT_DIR = os.path.join(SCRATCH, "mai_cnn", "checkpoints", "ConvNeXt_transfer")
        MODEL_DIR = os.path.join(SCRATCH, "mai_cnn", "models", "ConvNeXt_transfer")

        print("Using data dir:", DATA_DIR)
        print("Checkpoints will be saved to:", CHECKPOINT_DIR)
        print("Model will be saved to:", MODEL_DIR)
        print("Available GPUs:", config.list_physical_devices("GPU"))

        # Set up save directory
        model_name = "ConvNeXt_transfer"
        base_dir = os.environ.get("SCRATCH", "/scratch/users/pdingus")
        model_path = os.path.join(base_dir, "mai_cnn/models/ConvNeXt_transfer")

        # If the model folder path doesn't exist, create it
        os.makedirs(model_path, exist_ok=True)

        # Make a subfolder of the current date, if it doesn't exist
        current_date_string = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        save_path = os.path.join(model_path, current_date_string)
        os.makedirs(save_path, exist_ok=True)

        # Load the pre-trained ConvNeXtTiny model
        base_model = keras.applications.ConvNeXtTiny(
            include_top=False,
            include_preprocessing=True,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
        )
        base_model.summary()

        # take a pre-trained binary output model, set the first channel to 7-channel
        # 2nd approach - add one layer to the network

        # Ensure that the pre-trained model is not trainable initially
        base_model.trainable = False

        # Modify the first convolutional layer to accept 7-channel input
        input_tensor = Input(shape=(128, 128, 7))

        # Use the base model's layers
        # Add a Conv2D layer to reduce the number of channels from 7 to 3
        x = Conv2D(3, (1, 1), padding="same")(input_tensor)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # Feed into base model
        x = base_model(x, training=False)

        # Classification head
        x = GlobalAveragePooling2D()(x)
        output = Dense(1, activation="sigmoid", dtype="float32")(x)

        # Final model
        model = Model(inputs=input_tensor, outputs=output)
        model.summary()

        # Compile the model for training the untrained layers
        model.compile(
            optimizer='adam',
            loss=BinaryCrossentropy(from_logits=False),
            metrics=[BinaryAccuracy()],
        )

        checkpoint_dir = os.path.join(save_path, "checkpoints")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        checkpoint_callback = callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,  # Save full model, not just weights
            verbose=1,
        )

        # Define callbacks
        callback_list = [
            checkpoint_callback,
            callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, verbose=1),
        ]


        # Import the TFDatasetBuilder class
        training_args = {
            'scaling': 'standard',
            'per_image_scaling': True,
            'do_augmentation': True,
            'do_clipping': True,
            'lower_clip': 0,
            'upper_clip': 40,
        }
        
        builder = TFDatasetBuilder(
            csv_path="./data/training_data_S2/image_metadata.csv",
            **training_args
        )

        train_ds, val_ds, test_ds = builder.build_datasets(
            train_split=0.70,
            val_split=0.15,
            sample_size=1000,  # number of images for p1/p99 & robust scaling
            batch_size=16,
            shuffle_buffer=256,
        )

        # # Small test dataset for quick testing
        # train_ds, val_ds, test_ds = builder.build_datasets( 
        #     train_split=0.70,
        #     val_split=0.15,
        #     sample_size=16,   # super small for test
        #     batch_size=8,
        #     shuffle_buffer=32,
        # )

        train_ds = train_ds.cache().prefetch(buffer_size=2)
        val_ds = val_ds.cache().prefetch(buffer_size=2)
        test_ds = test_ds.cache().prefetch(buffer_size=2)

        # Train the model with frozen base model
        # history = model.fit(train_ds, validation_data=val_ds, epochs=1, callbacks=callback_list) # test with 1 epoch
        history = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=callback_list)

        # Unfreeze the base model for fine-tuning
        base_model.trainable = True

        # Recompile the model with a lower learning rate for fine-tuning
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss=BinaryCrossentropy(from_logits=False),
            metrics=[BinaryAccuracy()],
        )

        # Fine-tune the entire model
        # history_fine = model.fit(train_ds, validation_data=val_ds, epochs=1, callbacks=callback_list) # test with 1 epoch
        history_fine = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callback_list)

        # Evaluate the model on the test set
        test_loss, test_accuracy = model.evaluate(test_ds)
        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")

        # Save the model
        save_model = True
        save_final_model = False
        if save_model:

            # Save the model
            if save_final_model == True:
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
    
    # To support graceful shutdown, catch errors and capture traceback
    except Exception as e:
        print(f"[FATAL ERROR] Training/Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
