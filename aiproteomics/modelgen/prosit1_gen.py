import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tf2onnx

from .prosit1.layers import Attention

def masked_spectral_distance(true, pred):
    """
    Function obtained from https://github.com/kusterlab/prosit/tree/master/prosit/losses.py
    """
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


def build_prosit1_model(save_format='onnx'):
    """
    Creates the Prosit model:
        Gessulat, S., Schmidt, T., Zolg, D.P. et al.
        Prosit: proteome-wide prediction of peptide tandem mass spectra by deep learning. 
        Nat Methods 16, 509–518 (2019). https://doi.org/10.1038/s41592-019-0426-7
    
    args:
        save_format (str): 'onnx' or 'keras' or 'both' or None. Defaults to 'onnx'. Use None if you do not wish to save the model
    """

    valid_save_formats = ['onnx', 'keras', 'both', None]
    if save_format not in valid_save_formats:
        raise ValueError(
            f'Invalid save_format given ({save_format}).\n'
            f'Select valid save_format from {valid_save_formats}.'
        )

    # Input layers
    peptides_in = keras.Input(name='peptides_in', dtype='int32', sparse=False, batch_input_shape=(None, 30))
    collision_energy_in = keras.Input(name='collision_energy_in', dtype='float32', sparse=False, batch_input_shape=(None, 1))
    precursor_charge_in = keras.Input(name='precursor_charge_in', dtype='float32', sparse=False, batch_input_shape=(None, 6))

    # iRT related branch
    embedding = layers.Embedding(
                    name='embedding',
                    input_dim=22,
                    output_dim=32,
                    trainable=True,
                    dtype='float32',
                    embeddings_initializer=keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None),
                    embeddings_constraint=None,
                    embeddings_regularizer=None,
                    activity_regularizer=None,
                    batch_input_shape=[None, None],
                    mask_zero=False)(peptides_in)

    # Warning: This was a 'CuDNNGRU' layer in the original 2019 model, it is now GRU.
    # Config options have not been changed.
    encoder1_gru = layers.GRU(
        256,
        name='encoder1_gru',
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
        trainable=True)

    encoder1 = layers.Bidirectional(encoder1_gru, name='encoder1', merge_mode='concat', trainable=True)(embedding)

    dropout_1 = layers.Dropout(name='dropout_1', rate=0.3, noise_shape=None, seed=None, trainable=True)(encoder1)

    # Warning: This was a 'CuDNNGRU' layer in the original 2019 model, it is now GRU.
    encoder2 = layers.GRU(
        512,
        name='encoder2',
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
        trainable=True)(dropout_1)

    dropout_2 = layers.Dropout(name='dropout_2', rate=0.3, noise_shape=None, seed=None, trainable=True)(encoder2)

    # WARNING: This was a custom implementation in the original model
    encoder_att = Attention(name='encoder_att')(dropout_2)

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
    add_meta = layers.Multiply(name='add_meta', trainable=True)([encoder_att, meta_dense_do])

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

    # Save model
    output_location = './aiproteomics/modelgen/saved_models/'
    if save_format == None:
        # do not save
        pass
    else:
        if save_format == 'onnx' or save_format == 'both':
            # save as onnx
            output_path = output_location + model.name + ".onnx"
            # using default opset and spec settings for now, might need to be hardcoded if it doesn't work for all cases in the future            
            # for some idea on how to set this, see example on https://github.com/onnx/tensorflow-onnx/blob/main/tutorials/keras-resnet50.ipynb
            tf2onnx.convert.from_keras(model, output_path=output_path)
        if save_format == 'keras' or save_format == 'both':
            # save as keras
            model.save(output_location + model.name)

    return model


if __name__ == "__main__":
    build_prosit1_model(save_format='onnx')
