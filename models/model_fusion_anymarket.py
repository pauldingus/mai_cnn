from keras import layers, models
from keras.optimizers import Adam  # type: ignore

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
    image_channels=1,
    conv1d_filters=64,
    conv1d_layers=2,
    final_dense_dim=1,
    lr=1e-4
):
    """
    Siamese approach (shared day-encoder) + stacked 1D conv for temporal fusion.
    Outputs a single sigmoid for AnyMarket.

    days: number of time steps (7)
    image_height, image_width: typically 128
    image_channels: 1 (grayscale) or more if needed
    """
    # Build the day-encoder for a single day's shape (128,128,1)
    single_day_shape = (image_height, image_width, image_channels)
    day_encoder = build_deeper_day_encoder(input_shape=single_day_shape)

    # The model's top-level input has shape (days, H, W, C)
    # e.g. (7, 128, 128, 1)
    model_input = layers.Input(shape=(days, image_height, image_width, image_channels))

    # Encode each day => (batch, days, embedding_dim)
    day_embeddings = layers.TimeDistributed(day_encoder)(model_input)
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

# ------------------------------------------------------------------------
#Example main() demonstrating usage
# ------------------------------------------------------------------------

def main():
    model = build_siamese_stacked1d_model()
    model.summary()

if __name__ == "__main__":
    main()