from tensorflow import keras
from tensorflow.keras import layers
from aiproteomics.models.custom_losses import masked_spectral_distance

class Attention(Layer):
    def __init__(  # pylint: disable=too-many-arguments
        self,
        context=False,
        W_regularizer=None,
        b_regularizer=None,
        u_regularizer=None,
        W_constraint=None,
        b_constraint=None,
        u_constraint=None,
        bias=True,
        **kwargs
    ):

        self.supports_masking = True
        self.init = initializers.get("glorot_uniform")
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.bias = bias
        self.context = context
        super().__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(
            shape=(input_shape[-1],),
            initializer=self.init,
            name="{}_W".format(self.name),
            regularizer=self.W_regularizer,
            constraint=self.W_constraint,
        )
        if self.bias:
            self.b = self.add_weight(
                shape=(input_shape[1],),
                initializer="zero",
                name="{}_b".format(self.name),
                regularizer=self.b_regularizer,
                constraint=self.b_constraint,
            )
        else:
            self.b = None
        if self.context:
            self.u = self.add_weight(
                shape=(input_shape[-1],),
                initializer=self.init,
                name="{}_u".format(self.name),
                regularizer=self.u_regularizer,
                constraint=self.u_constraint,
            )

        self.built = True

    def compute_mask(
        self, input, input_mask=None
    ):  # pylint: disable=arguments-renamed, redefined-builtin, unused-argument
        return None

    def call(self, x, mask=None):  # pylint: disable=arguments-differ
        a = K.squeeze(K.dot(x, K.expand_dims(self.W)), axis=-1)
        if self.bias:
            a += self.b
        a = K.tanh(a)
        if self.context:
            a = K.squeeze(K.dot(x, K.expand_dims(self.u)), axis=-1)
        a = K.exp(a)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_config(self):
        config = {
            "bias": self.bias,
            "context": self.context,
            "W_regularizer": regularizers.serialize(self.W_regularizer),
            "b_regularizer": regularizers.serialize(self.b_regularizer),
            "u_regularizer": regularizers.serialize(self.u_regularizer),
            "W_constraint": constraints.serialize(self.W_constraint),
            "b_constraint": constraints.serialize(self.b_constraint),
            "u_constraint": constraints.serialize(self.u_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def generate_prosit1_model():  # pylint: disable=too-many-locals
    """
    Creates the Prosit model:
        Gessulat, S., Schmidt, T., Zolg, D.P. et al.
        Prosit: proteome-wide prediction of peptide tandem mass spectra by deep learning.
        Nat Methods 16, 509–518 (2019). https://doi.org/10.1038/s41592-019-0426-7

    Args:
        output_format (list or str, optional): format or list of formats to save the model as.
            Set to None to not save the model.
            Defaults to 'onnx'.
    """

    # Input layers
    peptides_in = keras.Input(
        name="peptides_in", dtype="int32", sparse=False, batch_input_shape=(None, 30)
    )
    collision_energy_in = keras.Input(
        name="collision_energy_in", dtype="float32", sparse=False, batch_input_shape=(None, 1)
    )
    precursor_charge_in = keras.Input(
        name="precursor_charge_in", dtype="float32", sparse=False, batch_input_shape=(None, 6)
    )

    # iRT related branch
    embedding = layers.Embedding(
        name="embedding",
        input_dim=22,
        output_dim=32,
        trainable=True,
        dtype="float32",
        embeddings_initializer=keras.initializers.RandomUniform(
            minval=-0.05, maxval=0.05, seed=None
        ),
        embeddings_constraint=None,
        embeddings_regularizer=None,
        activity_regularizer=None,
        batch_input_shape=[None, None],
        mask_zero=False,
    )(peptides_in)

    # Warning: This was a 'CuDNNGRU' layer in the original 2019 model, it is now GRU.
    # Config options have not been changed.
    encoder1_gru = layers.GRU(
        256,
        name="encoder1_gru",
        kernel_initializer=keras.initializers.VarianceScaling(
            distribution="uniform", mode="fan_avg", scale=1.0, seed=None
        ),
        bias_initializer="zeros",
        bias_regularizer=None,
        go_backwards=False,
        kernel_constraint=None,
        kernel_regularizer=None,
        recurrent_constraint=None,
        recurrent_initializer=keras.initializers.Orthogonal(gain=1.0, seed=None),
        recurrent_regularizer=None,
        activity_regularizer=None,
        bias_constraint=None,
        return_sequences=True,
        return_state=False,
        stateful=False,
        trainable=True,
    )

    encoder1 = layers.Bidirectional(
        encoder1_gru, name="encoder1", merge_mode="concat", trainable=True
    )(embedding)

    dropout_1 = layers.Dropout(
        name="dropout_1", rate=0.3, noise_shape=None, seed=None, trainable=True
    )(encoder1)

    # Warning: This was a 'CuDNNGRU' layer in the original 2019 model, it is now GRU.
    encoder2 = layers.GRU(
        512,
        name="encoder2",
        kernel_initializer=keras.initializers.VarianceScaling(
            distribution="uniform", mode="fan_avg", scale=1.0, seed=None
        ),
        bias_initializer="zeros",
        bias_regularizer=None,
        go_backwards=False,
        kernel_constraint=None,
        kernel_regularizer=None,
        recurrent_constraint=None,
        recurrent_initializer=keras.initializers.Orthogonal(gain=1.0, seed=None),
        recurrent_regularizer=None,
        activity_regularizer=None,
        bias_constraint=None,
        return_sequences=True,
        return_state=False,
        stateful=False,
        trainable=True,
    )(dropout_1)

    dropout_2 = layers.Dropout(
        name="dropout_2", rate=0.3, noise_shape=None, seed=None, trainable=True
    )(encoder2)

    # WARNING: This was a custom implementation in the original model
    encoder_att = Attention(name="encoder_att")(dropout_2)

    # Collision energy and precursor charge branch
    meta_in = layers.Concatenate(name="meta_in", trainable=True, axis=-1)(
        [collision_energy_in, precursor_charge_in]
    )
    meta_dense = layers.Dense(
        512,
        name="meta_dense",
        activation="linear",
        activity_regularizer=None,
        bias_constraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        kernel_constraint=None,
        kernel_initializer=keras.initializers.VarianceScaling(
            distribution="uniform", mode="fan_avg", scale=1.0, seed=None
        ),
        kernel_regularizer=None,
        trainable=True,
        use_bias=True,
    )(meta_in)

    meta_dense_do = layers.Dropout(
        name="meta_dense_do", rate=0.3, noise_shape=None, seed=None, trainable=True
    )(meta_dense)

    # Joining branches
    add_meta = layers.Multiply(name="add_meta", trainable=True)([encoder_att, meta_dense_do])

    repeat = layers.RepeatVector(name="repeat", n=29, trainable=True)(add_meta)

    # Warning: This was a 'CuDNNGRU' layer in the original 2019 model, it is now GRU.
    decoder = layers.GRU(
        512,
        name="decoder",
        kernel_initializer=keras.initializers.VarianceScaling(
            distribution="uniform", mode="fan_avg", scale=1.0, seed=None
        ),
        bias_initializer="zeros",
        bias_regularizer=None,
        go_backwards=False,
        kernel_constraint=None,
        kernel_regularizer=None,
        recurrent_constraint=None,
        recurrent_initializer=keras.initializers.Orthogonal(gain=1.0, seed=None),
        recurrent_regularizer=None,
        activity_regularizer=None,
        bias_constraint=None,
        return_sequences=True,
        return_state=False,
        stateful=False,
        trainable=True,
    )(repeat)

    dropout_3 = layers.Dropout(
        name="dropout_3", rate=0.3, noise_shape=None, seed=None, trainable=True
    )(decoder)

    permute_1 = layers.Permute(name="permute_1", dims=(2, 1), trainable=True)(dropout_3)

    dense_1 = layers.Dense(
        29,
        name="dense_1",
        activation="softmax",
        activity_regularizer=None,
        bias_constraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        kernel_constraint=None,
        kernel_initializer=keras.initializers.VarianceScaling(
            distribution="uniform", mode="fan_avg", scale=1.0, seed=None
        ),
        kernel_regularizer=None,
        trainable=True,
        use_bias=True,
    )(permute_1)

    permute_2 = layers.Permute(name="permute_2", dims=(2, 1), trainable=True)(dense_1)

    multiply_1 = layers.Multiply(name="multiply_1", trainable=True)([permute_2, dropout_3])

    dense_2 = layers.Dense(
        6,
        name="dense_2",
        activation="linear",
        activity_regularizer=None,
        bias_constraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        kernel_constraint=None,
        kernel_initializer=keras.initializers.VarianceScaling(
            distribution="uniform", mode="fan_avg", scale=1.0, seed=None
        ),
        kernel_regularizer=None,
        trainable=True,
        use_bias=True,
    )
    timedense = layers.TimeDistributed(dense_2, name="timedense", trainable=True)(multiply_1)

    activation = layers.LeakyReLU(name="activation", alpha=0.30000001192092896, trainable=True)(
        timedense
    )

    output_layer = layers.Flatten(name="out", data_format="channels_last", trainable=True)(
        activation
    )

    model = keras.Model(
        inputs=[peptides_in, precursor_charge_in, collision_energy_in], outputs=output_layer
    )
    model.compile(loss="masked_spectral_distance", optimizer="adam", metrics=["accuracy"])

    # Return also the metadata of the creation of this nn model
    model_creation_metadata = {
            "name": "generate_prosit1_model",
            "args": {}
            }

    return model, model_creation_metadata
