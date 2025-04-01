from tensorflow.keras.applications.convnext import ConvNeXtTiny
from tensorflow.keras.layers import Conv2D, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras import callbacks
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
import keras_hub
import keras_hub
import numpy as np

base_model = keras.applications.ConvNeXtTiny(
        include_top=False,
        include_preprocessing=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax'
    )
base_model.summary()


# take a pre-trained binary output model, set the first channel to 7-channel
# 2nd approach - add one layer to the network

# Load the pre-trained ConvNeXtTiny model
# base_model = keras.applications.ConvNeXtTiny(
#         include_top=False, 
#         weights='imagenet', 
#         input_tensor=Input(shape=(128, 128, 3)),
#         classes=1,
#         name = 'ConvNeXtTiny_transfer'
#     )

# # Ensure that the pre-trained model is not trainable
# base_model.trainable = False

# # Modify the first convolutional layer to accept 7-channel input
# input_tensor = Input(shape=(128, 128, 7))

# # Use the base model's layers
# x = base_model(input_tensor, training=False)

# # Add a new fully connected layer for binary classification
# x = GlobalAveragePooling2D()(x)
# output = Dense(1, activation='softmax')(x)

# # Create the new model
# model = Model(inputs=input_tensor, outputs=output)

# # Print the modified model architecture
# model.summary()

# # # Compile the model
# # model.compile(
# #     optimizer='adam',
# #     loss=BinaryCrossentropy(from_logits=True),
# #     metrics=[BinaryAccuracy()]
# # )

# # Assuming you have training and validation datasets
# # train_ds and val_ds should be tf.data.Dataset objects
# # For example:
# # train_ds = ...
# # val_ds = ...

# # # Define callbacks
# # callbacks = [
# #     callbacks.EarlyStopping(
# #         monitor="val_loss", patience=5, restore_best_weights=True
# #     ),
# #     callbacks.ReduceLROnPlateau(
# #         monitor="val_loss", patience=3, factor=0.5, verbose=1
# #     ),
# # ]

# # # Train the model
# # history = model.fit(
# #     train_ds,
# #     validation_data=val_ds,
# #     epochs=50,
# #     callbacks=callbacks
# # )