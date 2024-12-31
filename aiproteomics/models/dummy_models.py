from tensorflow import keras

def generate_dummy_msms_model(
    num_layers=6,
    num_heads=8,
    d_ff=2048,
    dropout_rate=0.1,
    params=None):

    """
    This generates a purely dummy MSMS model for testing purposes.
    It generates a `keras` model that takes a peptide sequence and charge as input,
    and outputs the predicted intensities (of length the number of possible fragments)
    and pY (a single value).
    """

    peptide = keras.Input(
        name="peptide", dtype="float32", sparse=False, batch_input_shape=(None, params.seq_len)
    )
    charge = keras.Input(
        name="charge", dtype="float32", sparse=False, batch_input_shape=(None, 1)
    )

    add_meta = keras.layers.Concatenate()([peptide, charge])

    dense = keras.layers.Dense(len(params.fragments))(add_meta)

    activation = keras.layers.LeakyReLU(name="activation", alpha=0.30000001192092896, trainable=True)(
        dense
    )

    output_intensities = keras.layers.Flatten(name="intensities", data_format="channels_last", trainable=True)(
        activation
    )

    output_pY = keras.layers.Dense(1, name="pY", trainable=True)(
        dense
    )

    model = keras.Model(
        inputs=[peptide, charge], outputs=[output_intensities, output_pY]
    )
    model.compile(loss="meansquarederror", optimizer="adam", metrics=["accuracy"])

    # Return also the metadata of the creation of this nn model
    model_creation_metadata = {
            "name": "generate_dummy_msms_model",
            "args": {
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "d_ff": d_ff,
                    "dropout_rate": dropout_rate
                }
            }

    return model, model_creation_metadata


def generate_dummy_iRT_model(
    num_layers=6,
    num_heads=8,
    d_ff=2048,
    dropout_rate=0.1,
    params=None):

    """
    This generates a purely dummy retention time model for testing purposes.
    It generates a `keras` model that takes a peptide sequence as input,
    and outputs the predicted retention time (i.e. length 1)
    """

    peptide = keras.Input(
        name="peptide", dtype="float32", sparse=False, batch_input_shape=(None, params.seq_len)
    )

    dense = keras.layers.Dense(1)(peptide)

    activation = keras.layers.LeakyReLU(name="activation", alpha=0.30000001192092896, trainable=True)(
        dense
    )

    output_iRT = keras.layers.Flatten(name="intensities", data_format="channels_last", trainable=True)(
        activation
    )

    model = keras.Model(
        inputs=[peptide], outputs=[output_iRT]
    )
    model.compile(loss="meansquarederror", optimizer="adam", metrics=["accuracy"])

    # Return also the metadata of the creation of this nn model
    model_creation_metadata = {
            "name": "generate_dummy_iRT_model",
            "args": {
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "d_ff": d_ff,
                    "dropout_rate": dropout_rate
                }
            }

    return model, model_creation_metadata


def generate_dummy_ccs_model(
    num_layers=6,
    num_heads=8,
    d_ff=2048,
    dropout_rate=0.1,
    params=None):

    """
    This generates a purely dummy ion mobility model for testing purposes.
    It generates a `keras` model that takes a peptide sequence as input,
    and outputs the predicted ion mobility (i.e. length 1)
    """

    peptide = keras.Input(
        name="peptide", dtype="float32", sparse=False, batch_input_shape=(None, params.seq_len)
    )

    dense = keras.layers.Dense(1)(peptide)

    activation = keras.layers.LeakyReLU(name="activation", alpha=0.30000001192092896, trainable=True)(
        dense
    )

    output_iRT = keras.layers.Flatten(name="intensities", data_format="channels_last", trainable=True)(
        activation
    )

    model = keras.Model(
        inputs=[peptide], outputs=[output_iRT]
    )
    model.compile(loss="meansquarederror", optimizer="adam", metrics=["accuracy"])

    # Return also the metadata of the creation of this nn model
    model_creation_metadata = {
            "name": "generate_dummy_ccs_model",
            "args": {
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "d_ff": d_ff,
                    "dropout_rate": dropout_rate
                }
            }

    return model, model_creation_metadata
