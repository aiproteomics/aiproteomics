from dataclasses import dataclass
import itertools as it
from typing import Optional

import numpy as np
import pandas as pd
from pyteomics import mass

#from definitions import MASS_pY, ANNOTATION_pY, get_ion_mz, get_precursor_mz, aa_mass, mass_neutral_loss
#from utils import generate_unmodified_peptide_sequence, unimod_to_single_char_sequence

from spectrum import ModelParams, output_layer_to_spectrum

if __name__ == "__main__":
    model_params = ModelParams(seq_len=50, ions=['y','b'], num_charges=2, neutral_losses=['', 'H3PO4'])
    print(model_params)

    frag_list = model_params.generate_fragment_list()

    print("Length:", len(frag_list))
    print([frag.annotation for frag in frag_list])

    # Create random output layer
    output_layer = np.random.random(392)

    # Convert to spectrum
    spectrum = output_layer_to_spectrum(output_layer, model_params, "*SSS1TT221", 1, pY=0.97, iRT=1, ccs=0, thresh=0.9)


    spectrum_df = spectrum.to_dataframe()
    print(spectrum_df)

    spectrum_df.to_csv('test_spectrum.tsv', sep='\t')
    spectrum_df.to_parquet('test_spectrum.parquet')
