from . import sanitize
import pandas as pd
import numpy as np
from aiproteomics.e2e import convert_to_msp, tensorize
from aiproteomics.e2e.constants import MAX_FRAG_CHARGE, MAX_NLOSSES


def _predict(
    data,
    model_frag,
    model_irt,
    batch_size_frag=None,
    batch_size_iRT=None,
    iRT_rescaling_mean=None,
    iRT_rescaling_var=None,
):
    # Get fragmentation model predictions
    x = [
        data["sequence_integer"],
        data["precursor_charge_onehot"],
        data["collision_energy_aligned_normed"],
    ]
    prediction = model_frag.predict(x, verbose=False, batch_size=batch_size_frag)
    data["intensities_pred"] = prediction
    data = sanitize.sanitize_prediction_output(data)

    # Get iRT model predictions
    x = data["sequence_integer"]
    prediction = model_irt.predict(x, verbose=False, batch_size=batch_size_iRT)
    data["iRT"] = prediction * np.sqrt(iRT_rescaling_var) + iRT_rescaling_mean

    return data


def _read_peptides_csv(fname, chunksize):
    for df in pd.read_csv(fname, chunksize=chunksize):
        df.reset_index(drop=True, inplace=True)
        assert "modified_sequence" in df.columns
        assert "collision_energy" in df.columns
        assert "precursor_charge" in df.columns
        data = {
            "collision_energy_aligned_normed": tensorize.get_numbers(df.collision_energy) / 100.0,
            "sequence_integer": tensorize.get_sequence_integer(df.modified_sequence),
            "precursor_charge_onehot": tensorize.get_precursor_charge_onehot(df.precursor_charge),
        }
    
        # Calculate length of each (integer) peptide sequence
        lengths = (data["sequence_integer"] > 0).sum(1)

        # Output shape: (
        #                 N_sequences,
        #                 MAX_SEQUENCE - 1 (the max number of ions),
        #                 NUM ION TYPES (default 2: 'y', 'b'),
        #                 MAX number of losses (default 3: '', 'H20', 'NH3'),
        #                 MAX FRAGMENT CHARGE (default 6)
        #               )
        masses_pred = tensorize.get_mz_applied(df)

        masses_pred = sanitize.cap(masses_pred, MAX_NLOSSES, MAX_FRAG_CHARGE)
        masses_pred = sanitize.mask_outofrange(masses_pred, lengths)
        masses_pred = sanitize.mask_outofcharge(masses_pred, df.precursor_charge)

        # Output shape: N_sequences x 174
        masses_pred = sanitize.reshape_flat(masses_pred)
        data["masses_pred"] = masses_pred

        yield data



def csv_to_msp(
    in_csv_fname,
    out_msp_fname,
    model_frag,
    model_irt,
    batch_size_frag,
    batch_size_iRT,
    iRT_rescaling_mean,
    iRT_rescaling_var,
    chunksize=10000,
):
    """
    Outputs spectral library in msp format, for the peptide sequence list in the provided csv file

    Input:
        in_csv_fname: string
                      Path to a csv formatted file containing a list of (modified) peptide
                      sequences, collision energies and precursor charges.
                      For example:
                          modified_sequence,collision_energy,precursor_charge
                          MMPAAALIM(ox)R,35,3
                          MLAPPPIM(ox)K,30,2
                          MRALLLIPPPPM(ox)R,30,6
        out_msp_fname: string
                       Name of the (msp format) spectral library file that output should be
                       written to.                   
        model_frag: tensor flow model
                    The fragmentation model (prosit-like) to use for spectra prediction
        model_irt: tensor flow model
                   The normalized retention time model to use for prediction

        chunksize: integer (optional)
                   For very large input files, chunksize can be set to only read in
                   and process a fixed number of rows (chunksize) at a time. This helps
                   avoid out-of-memory situations.
 
    """

    # Read in peptide rows from the input csv file, 'chunksize' rows at a time
    with open(out_msp_fname, 'w', encoding='utf-8') as speclibout:

        for peptidedata in _read_peptides_csv(in_csv_fname, chunksize=chunksize):
            # Run fragmentation and iRT prediction models for this chunk of peptides
            predictiondata = _predict(
                peptidedata,
                model_frag,
                model_irt,
                batch_size_frag=batch_size_frag,
                batch_size_iRT=batch_size_iRT,
                iRT_rescaling_mean=iRT_rescaling_mean,
                iRT_rescaling_var=iRT_rescaling_var,
            )

            # Convert the predictions to the chosen speclib format text
            speclibtxt = convert_to_msp.convert_to_msp(predictiondata)
            speclibout.write(speclibtxt)
    
            print(len(speclibtxt))
