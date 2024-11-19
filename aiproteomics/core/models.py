import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# TODO: Return "creation meta data too?"
def generate_msms_transformer(
    num_layers=6,
    num_heads=8,
    d_ff=2048,
    dropout_rate=0.1,
    seq_map=None,
    params=None):

    # This is a purely dummy model just so we can return a keras model object
    # and test saving the AIProteomicsModel

    # Input layers
    peptide = keras.Input(
        name="peptide", dtype="float32", sparse=False, batch_input_shape=(None, 30)
    )
    charge = keras.Input(
        name="charge", dtype="float32", sparse=False, batch_input_shape=(None, 6)
    )

    add_meta = layers.Concatenate()([peptide, charge])

    activation = layers.LeakyReLU(name="activation", alpha=0.30000001192092896, trainable=True)(
        add_meta
    )

    output_layer = layers.Flatten(name="out", data_format="channels_last", trainable=True)(
        activation
    )

    # Compile model
    # if this doesn't work, explicitly import masked_spectral_distance from losses
    model = keras.Model(
        inputs=[peptide, charge], outputs=output_layer
    )
    model.compile(loss="meansquarederror", optimizer="adam", metrics=["accuracy"])

    return model



