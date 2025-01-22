from keras import layers, models, optimizers

def build_day_encoder(input_shape=(128, 128, 1)):
    """
    Builds a sub-network (Siamese branch) that encodes a single day's 128x128x1 image 
    into a compact feature vector.
    """
    inputs = layers.Input(shape=input_shape)

    # Example architecture (tweak as needed):
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)

    # Flatten and project to a feature vector (e.g., size 128)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    encoder_model = models.Model(inputs, x, name="DayEncoder")
    return encoder_model

def build_siamese_fusion_model(
    days=7, 
    day_input_shape=(128, 128, 1),
    conv1d_filters=64,
    conv1d_kernel_size=7,
    dense_units=8
):
    """
    Builds the full model that:
      1) Uses a shared day-encoder (Siamese approach) via TimeDistributed
      2) Fuses the 7 day-embeddings with a Conv1D layer using kernel_size=7 
      3) Outputs an 8-dimensional multi-label prediction (7 days + 1 "any market")
    """

    # Step 1: Create the day encoder
    day_encoder = build_day_encoder(day_input_shape)

    # Input shape: (batch_size, days, 128, 128, 1)
    model_input = layers.Input(shape=(days,) + day_input_shape)

    # Step 2: Encode each day independently using TimeDistributed
    # Output shape after this: (batch_size, days, embedding_dim)
    day_embeddings = layers.TimeDistributed(day_encoder)(model_input)

    # Step 3: 1D Convolution for temporal fusion
    # kernel_size=7 => covers all 7 time steps in one pass
    x = layers.Conv1D(filters=conv1d_filters, 
                      kernel_size=conv1d_kernel_size,
                      activation='relu')(day_embeddings)
    # After this Conv1D with kernel_size=7 and 'valid' padding, shape is (batch_size, days - 7 + 1, filters)
    # Which is (batch_size, 1, 64) if days=7.
    
    x = layers.Flatten()(x)  # shape => (batch_size, 64)

    # Step 4: Final Dense layer with 8 outputs (7 days + 1 any-market), all sigmoids for multi-label
    outputs = layers.Dense(dense_units, activation='sigmoid')(x)

    # Build the model
    model = models.Model(model_input, outputs, name="SiameseFusionModel")

    # Compile the model - using Binary Crossentropy
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        metrics=['accuracy']  # or any custom multi-label metrics
    )

    return model
