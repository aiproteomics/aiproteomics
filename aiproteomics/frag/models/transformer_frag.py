import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers


# Uses the last part of the Prosit model on the end of the first part of the RT transformer:
# Gessulat, S., Schmidt, T., Zolg, D.P. et al.
# Prosit: proteome-wide prediction of peptide tandem mass spectra by deep learning.
# Nat Methods 16, 509–518 (2019). https://doi.org/10.1038/s41592-019-0426-7.


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


# Position
def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(
            d_model)[np.newaxis, :], d_model
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)  # pylint: disable= no-value-for-parameter, unexpected-keyword-arg


# Masking
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


# Scaled dot product attention
def scaled_dot_product_attention(q, k, v, mask):
    """
    Calculate the attention weights.

    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable 
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights
    """

    # (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += mask * -1e9

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1
    )  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


# Multi-head attention
class multi_head_attention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        # (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(k, batch_size)
        # (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model)
        )  # (batch_size, seq_len_q, d_model)

        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attention_weights


# Point wise feed forward network
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential(
        [
            # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(dff, activation="relu"),
            tf.keras.layers.Dense(d_model),  # (batch_size, seq_len, d_model)
        ]
    )


# Encoder and decoder
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()

        self.mha = multi_head_attention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        # (batch_size, input_seq_len, d_model)
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(
            out1 + ffn_output
        )  # (batch_size, input_seq_len, d_model)

        return out2


# Encoder
class EncoderBlock(tf.keras.layers.Layer):
    def __init__(  # pylint: disable=too-many-arguments
        self,
        num_layers,
        # d_embedding,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        maximum_position_encoding,
        rate=0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        """ TP: for saving """
        # self.d_embedding = d_embedding,
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = (input_vocab_size,)
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate
        """ TP END """

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(
            maximum_position_encoding, d_model)

        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training):

        seq_len = tf.shape(x)[1]

        enc_padding_mask = create_padding_mask(x)

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        # x *= tf.math.sqrt(tf.cast(self.d_embedding, tf.float32))
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, enc_padding_mask)

        return x  # (batch_size, input_seq_len, d_model)

    def get_config(self):

        config = super().get_config().copy()
        config.update(
            {
                "num_layers": self.num_layers,
                "d_model": self.d_model,
                "num_head": self.num_heads,
                "dff": self.dff,
                "input_vocab_size": self.input_vocab_size,
                "maximum_position_encoding": self.maximum_position_encoding,
                "rate": self.rate,
            }
        )
        return config


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def build_model_transformer_encoder_prosit_decoder(  # pylint: disable=too-many-arguments, too-many-locals
    num_layers,
    d_model,
    num_heads,
    d_ff,
    dropout_rate,
    vocab_size,
    max_len,
):

    # Transformer branch (peptide input)
    peptides_in = tf.keras.layers.Input(shape=(max_len,), name='peptides_in')
    collision_energy_in = keras.Input(
        name='collision_energy_in', dtype='float32', sparse=False, batch_input_shape=(None, 1))
    precursor_charge_in = keras.Input(
        name='precursor_charge_in', dtype='float32', sparse=False, batch_input_shape=(None, 6))

    encoder = EncoderBlock(num_layers, d_model, num_heads,
                           d_ff, vocab_size, max_len, dropout_rate)

    enc_output = encoder(peptides_in)  # (batch_size, inp_seq_len, d_model)

    flat_enc_output = layers.Flatten(name='flat_enc_output', trainable=True)(enc_output)
    dense_reduce = layers.Dense(512, name='dense_reduce', trainable=True, activation='linear')(flat_enc_output)

    net = tf.keras.layers.Dropout(dropout_rate)(dense_reduce)
    

    # Collision energy and precursor charge branch
    meta_in = layers.Concatenate(name='meta_in', trainable=True, axis=-1)([collision_energy_in, precursor_charge_in])
    meta_dense = layers.Dense(512,
                    name='meta_dense',
                    activation='linear',
                    activity_regularizer=None,
                    bias_constraint=None,
                    bias_initializer='zeros',
                    bias_regularizer=None,
                    kernel_constraint=None,
                    kernel_initializer=keras.initializers.VarianceScaling(distribution='uniform', mode='fan_avg', scale=1.0, seed=None),
                    kernel_regularizer=None,
                    trainable=True,
                    use_bias=True)(meta_in)

    meta_dense_do = layers.Dropout(name='meta_dense_do', rate=0.3, noise_shape=None, seed=None, trainable=True)(meta_dense)


    # Joining branches
    add_meta = layers.Multiply(name='add_meta', trainable=True)([net, meta_dense_do])

    repeat = layers.RepeatVector(name='repeat', n=29, trainable=True)(add_meta)


    # Warning: This was a 'CuDNNGRU' layer in the original 2019 model, it is now GRU.
    decoder = layers.GRU(
        512,
        name='decoder',
        kernel_initializer=keras.initializers.VarianceScaling(distribution='uniform', mode='fan_avg', scale=1.0, seed=None),
        bias_initializer='zeros',
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
        trainable=True)(repeat)


    dropout_3 = layers.Dropout(name='dropout_3', rate=0.3, noise_shape=None, seed=None, trainable=True)(decoder)

    permute_1 = layers.Permute(name='permute_1', dims=(2,1), trainable=True)(dropout_3)

    dense_1 = layers.Dense(29,
                    name='dense_1',
                    activation='softmax',
                    activity_regularizer=None,
                    bias_constraint=None,
                    bias_initializer='zeros',
                    bias_regularizer=None,
                    kernel_constraint=None,
                    kernel_initializer=keras.initializers.VarianceScaling(distribution='uniform', mode='fan_avg', scale=1.0, seed=None),
                    kernel_regularizer=None,
                    trainable=True,
                    use_bias=True)(permute_1)

    permute_2 = layers.Permute(name='permute_2', dims=(2,1), trainable=True)(dense_1)

    multiply_1 = layers.Multiply(name='multiply_1', trainable=True)([permute_2, dropout_3])

    dense_2 = layers.Dense(6,
                    name='dense_2',
                    activation='linear',
                    activity_regularizer=None,
                    bias_constraint=None,
                    bias_initializer='zeros',
                    bias_regularizer=None,
                    kernel_constraint=None,
                    kernel_initializer=keras.initializers.VarianceScaling(distribution='uniform', mode='fan_avg', scale=1.0, seed=None),
                    kernel_regularizer=None,
                    trainable=True,
                    use_bias=True)
    timedense = layers.TimeDistributed(dense_2, name='timedense', trainable=True)(multiply_1)

    activation = layers.LeakyReLU(name='activation', alpha=0.30000001192092896, trainable=True)(timedense)

    output_layer = layers.Flatten(name='out', data_format='channels_last', trainable=True)(activation)

    # Compile model
    model = keras.Model(inputs=[peptides_in, precursor_charge_in, collision_energy_in], outputs=output_layer)
    model.compile(loss='masked_spectral_distance', optimizer='adam', metrics=['accuracy'])

    return model



def build_frag_transformer_model_slice0( # pylint: disable=too-many-arguments, too-many-locals
        num_layers, 
        d_model, 
        num_heads, 
        d_ff, 
        dropout_rate, 
        vocab_size, 
        max_len,
    ):

    # Transformer branch (peptide input)
    peptides_in = tf.keras.layers.Input(shape=(max_len,), name='peptides_in')
    collision_energy_in = keras.Input(name='collision_energy_in', dtype='float32', sparse=False, batch_input_shape=(None, 1))
    precursor_charge_in = keras.Input(name='precursor_charge_in', dtype='float32', sparse=False, batch_input_shape=(None, 6))

    encoder = EncoderBlock(num_layers, d_model, num_heads, d_ff, vocab_size, max_len, dropout_rate)

    enc_output = encoder(peptides_in)  # (batch_size, inp_seq_len, d_model)

    net = enc_output[:, 0, :]
    net = tf.keras.layers.Dropout(dropout_rate)(net)


    # Collision energy and precursor charge branch
    meta_in = layers.Concatenate(name='meta_in', trainable=True, axis=-1)([collision_energy_in, precursor_charge_in])
    meta_dense = layers.Dense(512,
                    name='meta_dense',
                    activation='linear',
                    activity_regularizer=None,
                    bias_constraint=None,
                    bias_initializer='zeros',
                    bias_regularizer=None,
                    kernel_constraint=None,
                    kernel_initializer=keras.initializers.VarianceScaling(distribution='uniform', mode='fan_avg', scale=1.0, seed=None),
                    kernel_regularizer=None,
                    trainable=True,
                    use_bias=True)(meta_in)

    meta_dense_do = layers.Dropout(name='meta_dense_do', rate=0.3, noise_shape=None, seed=None, trainable=True)(meta_dense)


    # Joining branches
    add_meta = layers.Multiply(name='add_meta', trainable=True)([net, meta_dense_do])

    repeat = layers.RepeatVector(name='repeat', n=29, trainable=True)(add_meta)


    # Warning: This was a 'CuDNNGRU' layer in the original 2019 model, it is now GRU.
    decoder = layers.GRU(
        512,
        name='decoder',
        kernel_initializer=keras.initializers.VarianceScaling(distribution='uniform', mode='fan_avg', scale=1.0, seed=None),
        bias_initializer='zeros',
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
        trainable=True)(repeat)


    dropout_3 = layers.Dropout(name='dropout_3', rate=0.3, noise_shape=None, seed=None, trainable=True)(decoder)

    permute_1 = layers.Permute(name='permute_1', dims=(2,1), trainable=True)(dropout_3)

    dense_1 = layers.Dense(29,
                    name='dense_1',
                    activation='softmax',
                    activity_regularizer=None,
                    bias_constraint=None,
                    bias_initializer='zeros',
                    bias_regularizer=None,
                    kernel_constraint=None,
                    kernel_initializer=keras.initializers.VarianceScaling(distribution='uniform', mode='fan_avg', scale=1.0, seed=None),
                    kernel_regularizer=None,
                    trainable=True,
                    use_bias=True)(permute_1)

    permute_2 = layers.Permute(name='permute_2', dims=(2,1), trainable=True)(dense_1)

    multiply_1 = layers.Multiply(name='multiply_1', trainable=True)([permute_2, dropout_3])

    dense_2 = layers.Dense(6,
                    name='dense_2',
                    activation='linear',
                    activity_regularizer=None,
                    bias_constraint=None,
                    bias_initializer='zeros',
                    bias_regularizer=None,
                    kernel_constraint=None,
                    kernel_initializer=keras.initializers.VarianceScaling(distribution='uniform', mode='fan_avg', scale=1.0, seed=None),
                    kernel_regularizer=None,
                    trainable=True,
                    use_bias=True)
    timedense = layers.TimeDistributed(dense_2, name='timedense', trainable=True)(multiply_1)

    activation = layers.LeakyReLU(name='activation', alpha=0.30000001192092896, trainable=True)(timedense)

    output_layer = layers.Flatten(name='out', data_format='channels_last', trainable=True)(activation)

    # Compile model
    model = keras.Model(inputs=[peptides_in, precursor_charge_in, collision_energy_in], outputs=output_layer)
    model.compile(loss='masked_spectral_distance', optimizer='adam', metrics=['accuracy'])

    return model


def build_model_transformer_encoder_simple_prediction_head( # pylint: disable=too-many-arguments, too-many-locals
        num_layers, 
        d_model, 
        num_heads, 
        d_ff, 
        dropout_rate, 
        vocab_size, 
        max_len,
    ):

    # Transformer branch (peptide input)
    peptides_in = tf.keras.layers.Input(shape=(max_len,), name='peptides_in')
    collision_energy_in = keras.Input(name='collision_energy_in', dtype='float32', sparse=False, batch_input_shape=(None, 1))
    precursor_charge_in = keras.Input(name='precursor_charge_in', dtype='float32', sparse=False, batch_input_shape=(None, 6))

    encoder = EncoderBlock(num_layers, d_model, num_heads, d_ff, vocab_size, max_len, dropout_rate)

    enc_output = encoder(peptides_in)  # (batch_size, inp_seq_len, d_model)

    flat_enc_output = layers.Flatten(
        name='flat_enc_output', trainable=True)(enc_output)
    dense_reduce = layers.Dense(
        512, name='dense_reduce', trainable=True, activation='linear')(flat_enc_output)

    net = tf.keras.layers.Dropout(dropout_rate)(dense_reduce)

    # Collision energy and precursor charge branch
    meta_in = layers.Concatenate(
        name='meta_in', trainable=True, axis=-1)([collision_energy_in, precursor_charge_in])
    meta_dense = layers.Dense(512,
                              name='meta_dense',
                              activation='linear',
                              activity_regularizer=None,
                              bias_constraint=None,
                              bias_initializer='zeros',
                              bias_regularizer=None,
                              kernel_constraint=None,
                              kernel_initializer=keras.initializers.VarianceScaling(
                                  distribution='uniform', mode='fan_avg', scale=1.0, seed=None),
                              kernel_regularizer=None,
                              trainable=True,
                              use_bias=True)(meta_in)

    meta_dense_do = layers.Dropout(
        name='meta_dense_do', rate=0.3, noise_shape=None, seed=None, trainable=True)(meta_dense)

    # Joining branches
    add_meta = layers.Multiply(
        name='add_meta', trainable=True)([net, meta_dense_do])

    repeat = layers.RepeatVector(name='repeat', n=29, trainable=True)(add_meta)

    # Warning: This was a 'CuDNNGRU' layer in the original 2019 model, it is now GRU.
    decoder = layers.GRU(
        512,
        name='decoder',
        kernel_initializer=keras.initializers.VarianceScaling(
            distribution='uniform', mode='fan_avg', scale=1.0, seed=None),
        bias_initializer='zeros',
        bias_regularizer=None,
        go_backwards=False,
        kernel_constraint=None,
        kernel_regularizer=None,
        recurrent_constraint=None,
        recurrent_initializer=keras.initializers.Orthogonal(
            gain=1.0, seed=None),
        recurrent_regularizer=None,
        activity_regularizer=None,
        bias_constraint=None,
        return_sequences=True,
        return_state=False,
        stateful=False,
        trainable=True)(repeat)

    dropout_3 = layers.Dropout(
        name='dropout_3', rate=0.3, noise_shape=None, seed=None, trainable=True)(decoder)

    permute_1 = layers.Permute(
        name='permute_1', dims=(2, 1), trainable=True)(dropout_3)

    dense_1 = layers.Dense(29,
                           name='dense_1',
                           activation='softmax',
                           activity_regularizer=None,
                           bias_constraint=None,
                           bias_initializer='zeros',
                           bias_regularizer=None,
                           kernel_constraint=None,
                           kernel_initializer=keras.initializers.VarianceScaling(
                               distribution='uniform', mode='fan_avg', scale=1.0, seed=None),
                           kernel_regularizer=None,
                           trainable=True,
                           use_bias=True)(permute_1)

    permute_2 = layers.Permute(
        name='permute_2', dims=(2, 1), trainable=True)(dense_1)

    multiply_1 = layers.Multiply(
        name='multiply_1', trainable=True)([permute_2, dropout_3])

    dense_2 = layers.Dense(6,
                           name='dense_2',
                           activation='linear',
                           activity_regularizer=None,
                           bias_constraint=None,
                           bias_initializer='zeros',
                           bias_regularizer=None,
                           kernel_constraint=None,
                           kernel_initializer=keras.initializers.VarianceScaling(
                               distribution='uniform', mode='fan_avg', scale=1.0, seed=None),
                           kernel_regularizer=None,
                           trainable=True,
                           use_bias=True)
    timedense = layers.TimeDistributed(
        dense_2, name='timedense', trainable=True)(multiply_1)

    activation = layers.LeakyReLU(
        name='activation', alpha=0.30000001192092896, trainable=True)(timedense)

    output_layer = layers.Flatten(
        name='out', data_format='channels_last', trainable=True)(activation)

    # Compile model
    model = keras.Model(inputs=[
                        peptides_in, precursor_charge_in, collision_energy_in], outputs=output_layer)
    model.compile(loss='masked_spectral_distance',
                  optimizer='adam', metrics=['accuracy'])

    return model


