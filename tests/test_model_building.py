import tensorflow as tf
from aiproteomics.rt.models import build_rt_transformer_model
from aiproteomics.frag.models.early_fusion_transformer import build_model_early_fusion_transformer

def test_build_rt_transformer_model():
    model_irt = build_rt_transformer_model(
        num_layers=6,  # number of layers
        d_model=512,
        num_heads=8,  # Number of attention heads
        d_ff=2048,  # Hidden layer size in feed forward network inside transformer
        dropout_rate=0.1,  #
        vocab_size=22,  # number of aminoacids
        max_len=30,
    )
    assert model_irt is not None
    assert isinstance(model_irt, tf.keras.Model)


def test_build_frag_transformer_model():
    model_frag = build_model_early_fusion_transformer()
    assert model_frag is not None
    assert isinstance(model_frag, tf.keras.Model)
