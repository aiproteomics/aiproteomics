from tensorflow import keras

# This is a purely dummy model just so we can return a keras model object
# and test saving the AIProteomicsModel
def generate_msms_transformer(
    num_layers=6,
    num_heads=8,
    d_ff=2048,
    dropout_rate=0.1,
    params=None):

    peptide = keras.Input(
        name="peptide", dtype="float32", sparse=False, batch_input_shape=(None, params.seq_len)
    )
    charge = keras.Input(
        name="charge", dtype="float32", sparse=False, batch_input_shape=(None, params.max_charge)
    )

    add_meta = keras.layers.Concatenate()([peptide, charge])

    activation = keras.layers.LeakyReLU(name="activation", alpha=0.30000001192092896, trainable=True)(
        add_meta
    )

    output_layer = keras.layers.Flatten(name="out", data_format="channels_last", trainable=True)(
        activation
    )

    model = keras.Model(
        inputs=[peptide, charge], outputs=output_layer
    )
    model.compile(loss="meansquarederror", optimizer="adam", metrics=["accuracy"])

    # Return also the metadata of the creation of this nn model
    model_creation_metadata = {
            "name": "generate_msms_transformer",
            "args": {
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "d_ff": d_ff,
                    "dropout_rate": dropout_rate
                }
            }

    return model, model_creation_metadata



