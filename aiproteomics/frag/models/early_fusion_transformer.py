import h5py
import numpy as np
import tensorflow as tf
import aiproteomics
from tensorflow import keras
from keras import layers

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


# Position
def positional_encoding(position, d_model):
    """
        Applies the sine-cosine positional encoding the
        embedded sequence layer in a transformer encoder.
    """

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
    """
        Implementation of a standard multi-head attention layer.
    """

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
    """
        A point-wise feed forward network consisting of two dense layers,
        of input dimension dff and output dimension d_model.
    """

    return tf.keras.Sequential(
        [
            # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(dff, activation="relu"),
            tf.keras.layers.Dense(d_model),  # (batch_size, seq_len, d_model)
        ]
    )


# Encoder and decoder
class EncoderLayer(tf.keras.layers.Layer):
    """
        Implementation of a standard transformer encoder layer,
        consisting of a multihead attention layer and a feed
        forward network.
    """

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


class EncoderBlockEnergyChargeEmbeddingConcat(tf.keras.layers.Layer):
    """
    A BERT-style transformer encoder block, which additionally takes two integer
    typed values - the precursor charge and the collision energy - as well as
    the usual sequence. Each of these three inputs is given a trainable embedding
    and concatenated with the (embedded) input sequence, producing a single fused
    input to the encoder block.

    The dimenesions of the embedding vectors are specified by d_model_seq,
    d_model_charge and d_model_collision_energy).
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        num_layers,
        d_model_seq,
        d_model_charge,
        d_model_collision_energy,
        num_heads,
        dff,
        input_vocab_size,
        max_charge,
        max_collision_energy,
        maximum_position_encoding,
        rate=0.1,
    ):
        super().__init__()

        self.d_model_seq = d_model_seq
        self.d_model_charge = d_model_charge
        self.d_model_collision_energy = d_model_collision_energy
        self.num_layers = num_layers


        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = (input_vocab_size,)
        self.max_charge = max_charge
        self.max_collision_energy = max_collision_energy
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate

        self.embedding_seq = tf.keras.layers.Embedding(input_vocab_size, d_model_seq)
        self.embedding_charge = tf.keras.layers.Embedding(max_charge + 1, d_model_charge)
        self.embedding_energy = tf.keras.layers.Embedding(max_collision_energy + 1, d_model_collision_energy)

        self.pos_encoding = positional_encoding(
            maximum_position_encoding, d_model_seq + d_model_charge + d_model_collision_energy)

        self.enc_layers = [
            EncoderLayer(d_model_seq + d_model_charge + d_model_collision_energy, num_heads, dff, rate) for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):

        if len(inputs) != 3:
            raise ValueError('EncoderBlockEnergyChargeEmbedding requires 3 input tensors: col_energy, charge and seq')

        col_energy, charge, seq = inputs

        seq_len = tf.shape(seq)[1]

        enc_padding_mask = create_padding_mask(seq)

        # Construct embedding vectors: a concatenation of the amino acid + energy + charge.
        # Energy and charge are the same for every amino acid.
        x = tf.keras.layers.Concatenate()(
            [
                self.embedding_seq(seq),
                self.embedding_energy(tf.ones(self.maximum_position_encoding, dtype=tf.float32) * col_energy),
                self.embedding_charge(tf.ones(self.maximum_position_encoding, dtype=tf.float32) * charge)
            ])

        # Add positional encoding
        x *= tf.math.sqrt(tf.cast(self.d_model_seq + self.d_model_charge + self.d_model_collision_energy, tf.float32))
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
                "d_model_seq": self.d_model_seq,
                "d_model_charge": self.d_model_charge,
                "d_model_collision_energy": self.d_model_collision_energy,
                "num_head": self.num_heads,
                "dff": self.dff,
                "input_vocab_size": self.input_vocab_size,
                "max_charge": self.max_charge,
                "max_collision_energy": self.max_collision_energy,
                "maximum_position_encoding": self.maximum_position_encoding,
                "rate": self.rate,
            }
        )
        return config


def build_model_early_fusion_transformer(  # pylint: disable=too-many-arguments, too-many-locals
    num_layers=6,
    d_model_seq=512,
    d_model_charge=128,
    d_model_collision_energy=128,
    num_heads=8,
    d_ff=2048,
    dropout_rate=0.1,
    vocab_size=22,
    max_charge=6,
    max_collision_energy=35,
    max_len=30,
):
    """
    Builds a fragmentation model with three input layers: a peptide sequence, a precursor charge,
    and a collision energy (same as the prosit input layers). The main bulk of the model is a
    BERT-like transformer encoder. The charge and collision energy layers are fused with the
    input sequence by concatenating embeddings (of dimensions d_model_seq, d_model_charge
    and d_model_collision_energy).

    Note that the collision energy, while generally a float, must here be discretized to whole
    numbers for the purposes of embedding.
    """

    # Transformer branch (peptide input)
    peptides_in = tf.keras.layers.Input(shape=(max_len,), name='peptides_in')
    collision_energy_in = keras.Input(
        name='collision_energy_in', dtype='float32', sparse=False, batch_input_shape=(None, 1))
    precursor_charge_in = keras.Input(
        name='precursor_charge_in', dtype='float32', sparse=False, batch_input_shape=(None, 6))

    # Convert one-hot encoding to integer
    precursor_charge_integer = layers.Dot(name='one_hot_to_int', axes=1)([precursor_charge_in, tf.constant([[1, 2, 3, 4, 5, 6]], dtype='float32')])


    encoder = EncoderBlockEnergyChargeEmbeddingConcat(num_layers, d_model_seq, d_model_charge, d_model_collision_energy, num_heads,
                           d_ff, vocab_size, max_charge, max_collision_energy, max_len, dropout_rate)

    enc_output = encoder([collision_energy_in, precursor_charge_integer, peptides_in])

    flat_enc_output = layers.Flatten(
        name='flat_enc_output', trainable=True)(enc_output)
    dense_reduce = layers.Dense(
        174, name='dense_reduce', trainable=True, activation='linear')(flat_enc_output)

    net = tf.keras.layers.Dropout(dropout_rate)(dense_reduce)

    activation = layers.Activation(tf.keras.activations.gelu, name='gelu', trainable=True)(net)

    output_layer = layers.Flatten(
        name='out', data_format='channels_last', trainable=True)(activation)

    # Compile model
    model = keras.Model(inputs=[
                        peptides_in, precursor_charge_in, collision_energy_in], outputs=output_layer)
    model.compile(loss='masked_spectral_distance',
                  optimizer='adam', metrics=['accuracy'])

    return model





def early_fusion_transformer_128(load_weights=True):

    # Set the dimensions of the embeddings for the sequence, charge and collision
    # energy layers.
    d_model_seq = 512
    d_model_charge = 128
    d_model_collision_energy = 128

    # Build the early fusion transformer and load weights
    model = build_model_transformer_embedding_energy_charge_concat(
            num_layers = 6,                 # number of layers
            d_model_seq = d_model_seq,
            d_model_charge = d_model_charge,
            d_model_collision_energy = d_model_collision_energy,
            num_heads = 8,                  # Number of attention heads
            d_ff = 2048,                    # Hidden layer size in feed forward network inside transformer
            dropout_rate = 0.1,             #
            vocab_size = 22,                # number of aminoacids
            max_charge = 6,
            max_collision_energy = 35,
            max_len = 30                    # maximal peptide length
    )

    if load_weights:
        model.load_weights(best_weights)
