import tensorflow as tf

from aiproteomics.core.modeltypes import ModelParamsMSMS, ModelParamsRT, ModelParamsCCS
from aiproteomics.models.dummy_models import generate_dummy_msms_model, generate_dummy_iRT_model, generate_dummy_ccs_model
from aiproteomics.models.prosit1 import generate_prosit1_model


def test_build_dummy_msms_model():
    """
    Tests generating a (dummy) MSMS model using a `ModelParamsMSMS` class
    """

    params = ModelParamsMSMS(seq_len=50, ions=['y','b'], max_charge=2, neutral_losses=['', 'H3PO4'])
    nn_model, creation_meta = generate_dummy_msms_model(
        params=params,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        dropout_rate=0.1
    )
    assert isinstance(nn_model, tf.keras.Model)
    assert isinstance(creation_meta, dict)


def test_build_dummy_iRT_model():
    """
    Tests generating a (dummy) MSMS model using a `ModelParamsRT` class
    """

    params = ModelParamsRT(seq_len=50, iRT_rescaling_mean=101.11514, iRT_rescaling_var=46.5882)
    nn_model, creation_meta = generate_dummy_iRT_model(
        params=params,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        dropout_rate=0.1
    )
    assert isinstance(nn_model, tf.keras.Model)
    assert isinstance(creation_meta, dict)


def test_build_dummy_ccs_model():
    """
    Tests generating a (dummy) CCS model using a `ModelParamsMSMS` class
    """

    params = ModelParamsCCS(seq_len=50)
    nn_model, creation_meta = generate_dummy_ccs_model(
        params=params,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        dropout_rate=0.1
    )
    assert isinstance(nn_model, tf.keras.Model)
    assert isinstance(creation_meta, dict)


def test_build_prosit1_model():
    """
    Tests generating an implementation of the original 2019 Prosit model.
    """

    nn_model, creation_meta = generate_prosit1_model()
    assert isinstance(nn_model, tf.keras.Model)
    assert isinstance(creation_meta, dict)
