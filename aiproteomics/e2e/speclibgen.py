from . import sanitize
import pandas as pd
import numpy as np
from aiproteomics.e2e import speclib_writers, tensorize
from aiproteomics.e2e.constants import MAX_FRAG_CHARGE, MAX_NLOSSES


def _predict(
    peptide_data,
    model_frag=None,
    model_irt=None,
    model_ccs=None,
    batch_size_frag=None,
    batch_size_iRT=None,
    batch_size_ccs=None,
    iRT_rescaling_mean=None,
    iRT_rescaling_var=None,
):
    """
    Use the provided models to predict the fragmentation spectra, normalized
    retention times and ion mobility for the sequence information provided in
    `peptide_data`.
    """

    # Get fragmentation model predictions
    if model_frag:
        x = [
            peptide_data["sequence_integer"],
            peptide_data["precursor_charge_onehot"],
            peptide_data["collision_energy_aligned_normed"],
        ]
        prediction = model_frag.predict(x, verbose=False, batch_size=batch_size_frag)
        peptide_data["intensities_pred"] = prediction
        peptide_data = sanitize.sanitize_prediction_output(peptide_data)

    # Get iRT model predictions
    if model_irt:
        x = peptide_data["sequence_integer"]
        prediction = model_irt.predict(x, verbose=False, batch_size=batch_size_iRT)
        peptide_data["iRT"] = prediction * np.sqrt(iRT_rescaling_var) + iRT_rescaling_mean

    # Get ion mobility model predictions
    if model_ccs:
        x = peptide_data["sequence_integer"]
        prediction = model_ccs.predict(x, verbose=False, batch_size=batch_size_ccs)
        peptide_data["ccs"] = prediction

    return peptide_data


def _read_peptides_csv(fname, chunksize, unknown_value_str='NA'):
    for df in pd.read_csv(fname, chunksize=chunksize):
        df.reset_index(drop=True, inplace=True)
        assert "modified_sequence" in df.columns
        assert "collision_energy" in df.columns
        assert "precursor_charge" in df.columns
        data = {
            "collision_energy_aligned_normed": tensorize.get_numbers(df.collision_energy) / 100.0,
            "sequence_integer": tensorize.get_sequence_integer(df.modified_sequence),
            "precursor_charge_onehot": tensorize.get_precursor_charge_onehot(df.precursor_charge)
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

        # Add Protein ID and gene name if available
        if "protein_id" in df.columns:
            data["protein_id"] = df["protein_id"]

        if "gene_name" in df.columns:
            data["gene_name"] = df["gene_name"]

        yield data



def csv_to_speclib(
    in_csv_fname,
    out_msp_fname,
    model_frag,
    model_irt=None,
    model_ccs=None,
    batch_size_frag=1024,
    batch_size_iRT=1024,
    batch_size_ccs=1024,
    iRT_rescaling_mean=0.0,
    iRT_rescaling_var=0.0,
    chunksize=10000,
    fmt='tsv'
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
        model_ccs: tensor flow model
                   The ion mobility model to use for prediction

        chunksize: integer (optional)
                   For very large input files, chunksize can be set to only read in
                   and process a fixed number of rows (chunksize) at a time. This helps
                   avoid out-of-memory situations.
 
    """

    # Read in peptide rows from the input csv file, 'chunksize' rows at a time
    with open(out_msp_fname, 'w', encoding='utf-8') as speclibout:

        if fmt == 'tsv':
            speclibout.write(speclib_writers.get_tsv_format_header())

        for peptidedata in _read_peptides_csv(in_csv_fname, chunksize=chunksize):
            # Run fragmentation and iRT prediction models for this chunk of peptides
            predictiondata = _predict(
                peptidedata,
                model_frag=model_frag,
                model_irt=model_irt,
                model_ccs=model_ccs,
                batch_size_frag=batch_size_frag,
                batch_size_iRT=batch_size_iRT,
                batch_size_ccs=batch_size_ccs,
                iRT_rescaling_mean=iRT_rescaling_mean,
                iRT_rescaling_var=iRT_rescaling_var,
            )

            # Convert the predictions to the chosen speclib format text
            speclibtxt = speclib_writers.convert_to_speclib(predictiondata, fmt=fmt)
            speclibout.write(speclibtxt)
