import glob
import sys

import h5py
import numpy as np
import tensorflow as tf
from keras import layers
from tensorflow import keras


from aiproteomics.datasets.DataSetPrositFrag import DataSetPrositFrag
from aiproteomics.frag.models import transformer_frag

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# Encoder
class EncoderBlockEnergyChargeEmbeddingConcat(tf.keras.layers.Layer):
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
        self.embedding_energy = tf.keras.layers.Embedding(max_collision_energy + 1,
                                                          d_model_collision_energy)

        self.pos_encoding = transformer_frag.positional_encoding(
            maximum_position_encoding, d_model_seq + d_model_charge + d_model_collision_energy)

        self.enc_layers = [
            transformer_frag.EncoderLayer(d_model_seq + d_model_charge +
                                          d_model_collision_energy, num_heads,
                                          dff,
                                          rate) for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):

        if len(inputs) != 3:
            raise ValueError(
                'EncoderBlockEnergyChargeEmbedding requires 3 input tensors: col_energy, charge and seq')

        col_energy, charge, seq = inputs

        seq_len = tf.shape(seq)[1]

        enc_padding_mask = transformer_frag.create_padding_mask(seq)

        # Construct embedding vectors: a concatenation of the amino acid + energy + charge.
        # Energy and charge are the same for every amino acid.
        x = tf.keras.layers.Concatenate()(
            [
                self.embedding_seq(seq),
                self.embedding_energy(
                    tf.ones(self.maximum_position_encoding, dtype=tf.float32) * col_energy),
                self.embedding_charge(
                    tf.ones(self.maximum_position_encoding, dtype=tf.float32) * charge)
            ])

        # Add positional encoding
        x *= tf.math.sqrt(
            tf.cast(self.d_model_seq + self.d_model_charge + self.d_model_collision_energy,
                    tf.float32))
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


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def __call__(self, step):
        arg1 = tf.math.rsqrt(tf.cast(step, dtype=tf.float32))
        arg2 = tf.cast(step, dtype=tf.float32) * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def build_model_transformer_embedding_energy_charge_concat(
        # pylint: disable=too-many-arguments, too-many-locals
        num_layers,
        d_model_seq,
        d_model_charge,
        d_model_collision_energy,
        num_heads,
        d_ff,
        dropout_rate,
        vocab_size,
        max_charge,
        max_collision_energy,
        max_len,
):
    # Transformer branch (peptide input)
    peptides_in = tf.keras.layers.Input(shape=(max_len,), name='peptides_in')
    collision_energy_in = keras.Input(
        name='collision_energy_in', dtype='float32', sparse=False, batch_input_shape=(None, 1))
    precursor_charge_in = keras.Input(
        name='precursor_charge_in', dtype='float32', sparse=False, batch_input_shape=(None, 6))

    # Convert one-hot encoding to integer
    precursor_charge_integer = layers.Dot(name='one_hot_to_int', axes=1)(
        [precursor_charge_in, tf.constant([[1, 2, 3, 4, 5, 6]], dtype='float32')])

    encoder = EncoderBlockEnergyChargeEmbeddingConcat(num_layers, d_model_seq, d_model_charge,
                                                      d_model_collision_energy, num_heads,
                                                      d_ff, vocab_size, max_charge,
                                                      max_collision_energy, max_len, dropout_rate)

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


def masked_spectral_distance(true, pred):
    # Note, fragment ions that cannot exists (i.e. y20 for a 7mer) must have the value  -1.
    import tensorflow
    import keras.backend as k

    epsilon = k.epsilon()
    pred_masked = ((true + 1) * pred) / (true + 1 + epsilon)
    true_masked = ((true + 1) * true) / (true + 1 + epsilon)
    pred_norm = k.l2_normalize(true_masked, axis=-1)
    true_norm = k.l2_normalize(pred_masked, axis=-1)
    product = k.sum(pred_norm * true_norm, axis=1)
    arccos = tensorflow.acos(product)
    return 2 * arccos / np.pi


def get_callbacks(model_dir_path):
    import keras

    loss_format = "{val_loss:.5f}"
    epoch_format = "{epoch:02d}"
    weights_file = "{}/weight_{}_{}.hdf5".format(
        model_dir_path, epoch_format, loss_format
    )
    save = keras.callbacks.ModelCheckpoint(weights_file, save_best_only=True,
                                           save_weights_only=True)
    decay = keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.2)
    return [save, decay]


def get_best_weights_file():
    # Get best weights file

    weight_files = sorted(glob.glob("full_transfo_embeddings_concat_128/data/weight_*.hdf5"))
    best_weights = weight_files[-1]
    return best_weights


def load_xy(path):
    N = 6787933

    with h5py.File(path, "r") as f:
        x = [f['sequence_integer'][:N], f['precursor_charge_onehot'][:N],
             f['collision_energy_aligned_normed'][:N]]
        y = f['intensities_raw'][:N]

    return x, y


def main():
    d_model_seq = 512
    d_model_charge = 128
    d_model_collision_energy = 128

    model = build_model_transformer_embedding_energy_charge_concat(
        num_layers=6,  # number of layers
        d_model_seq=d_model_seq,
        d_model_charge=d_model_charge,
        d_model_collision_energy=d_model_collision_energy,
        num_heads=8,  # Number of attention heads
        d_ff=2048,  # Hidden layer size in feed forward network inside transformer
        dropout_rate=0.1,  #
        vocab_size=22,  # number of aminoacids
        max_charge=6,
        max_collision_energy=35,
        max_len=30  # maximal peptide length
    )

    if sys.argv[1] == "train":
        # Get best weights file
        best_weights = get_best_weights_file()
        print("Using weights file:", best_weights)
        model.load_weights(best_weights)
        val_split = 0.8

        print("Training...")
        x, y = load_xy('traintest_hcd.hdf5')

        callbacks = get_callbacks('full_transfo_embeddings_concat_128/data')
        loss = masked_spectral_distance
        learning_rate = CustomSchedule(d_model_seq + d_model_charge + d_model_collision_energy)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                             epsilon=1e-9)
        model.compile(optimizer=optimizer, loss=loss)
        history = model.fit(
            x=x,
            y=y,
            epochs=150,
            batch_size=1024,
            validation_split=1 - val_split,
            callbacks=callbacks
        )

        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        print('train_loss:\n', train_loss)
        print('val_loss:\n', val_loss)
    elif sys.argv[1] == "plot":
        from aiproteomics.comparison.ComparisonPrositFrag import ComparisonPrositFrag
        print("Plotting spectral angle comparison...")

        # Get best weights file
        best_weights = get_best_weights_file()
        print("Using weights file:", best_weights)

        model.load_weights(best_weights)

        holdout_dataset = DataSetPrositFrag('holdout_hcd.hdf5')

        print("Plotting...")

        sa_plot = ComparisonPrositFrag.plot_spectral_angle_distributions(holdout_dataset, model)

        print("Saving...")
        sa_plot.get_figure().savefig("spectral_angle_comparison.png")
    elif sys.argv[1] == "predict":
        print("Predicting...")
        print("Loading training set...")
        x, y = load_xy('traintest_hcd.hdf5')

        # Get best weights file
        best_weights = get_best_weights_file()
        print("Using weights file:", best_weights)
        model.load_weights(best_weights)

        predictions = model.predict(x)
        with h5py.File("predictions.hdf5", "w") as f:
            dset = f.create_dataset("predictions", predictions.shape, predictions.dtype)
            dset[:, :] = predictions

        #np.save("predictions.npy", predictions)


if __name__ == "__main__":
    main()
