import numpy as np
from aiproteomics.core.spectrum import ModelParams, output_layer_to_spectrum


def test_model_params():

    model_params = ModelParams(seq_len=50, ions=['y','b'], num_charges=2, neutral_losses=['', 'H3PO4'])

    # Check that the number of fragments generated is correct
    frag_list = model_params.generate_fragment_list()
    assert len(frag_list) == 392


def test_output_layer_to_spectrum():

    model_params = ModelParams(seq_len=50, ions=['y','b'], num_charges=2, neutral_losses=['', 'H3PO4'])

    frag_list = model_params.generate_fragment_list()

    # Create random output layer
    output_layer = np.random.random(392)

    # Convert to spectrum
    spectrum = output_layer_to_spectrum(output_layer, model_params, "*SSS1TT221", 1, pY=0.97, iRT=1, ccs=0, thresh=0.9)

    # Convert to pandas dataframe
    spectrum_df = spectrum.to_dataframe()

    # Output to csv and parquet
    spectrum_df.to_csv('test_spectrum.tsv', sep='\t')
    spectrum_df.to_parquet('test_spectrum.parquet')
