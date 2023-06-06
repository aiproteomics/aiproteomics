import os.path
import tensorflow as tf
from aiproteomics.rt.models import build_rt_transformer_model
from aiproteomics.frag.models import build_model_transformer_encoder_prosit_decoder
from aiproteomics.e2e.speclibgen import csv_to_msp

def test_spectral_lib_generation(tmp_path):
    # Build an iRT model
    model_irt = build_rt_transformer_model(
        num_layers = 6,                 # number of layers
        d_model = 512,
        num_heads = 8,                  # Number of attention heads
        d_ff = 2048,                    # Hidden layer size in feed forward network inside transformer
        dropout_rate = 0.1,             #
        vocab_size = 22,                # number of aminoacids
        max_len = 30 
        )

    # Build a fragmentation model
    model_frag = build_model_transformer_encoder_prosit_decoder(
        num_layers = 6,                 # number of layers
        d_model = 512,
        num_heads = 8,                  # Number of attention heads
        d_ff = 2048,                    # Hidden layer size in feed forward network inside transformer
        dropout_rate = 0.1,             #
        vocab_size = 22,                # number of aminoacids
        max_len = 30                    # maximal peptide length
        )

    # Use the models to build a spectral library for a test csv of peptides
    msp_loc = tmp_path / 'test.msp'
    csv_to_msp('tests/assets/example.csv',
           msp_loc,
           model_frag=model_frag,
           model_irt=model_irt,
           batch_size_frag=1024,
           batch_size_iRT=1024,
           iRT_rescaling_mean = 56.35363441,
           iRT_rescaling_var = 1883.0160689
          )
    
    assert os.path.isfile(msp_loc)
