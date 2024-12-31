import pandas as pd
import numpy as np

from aiproteomics.core.definitions import ANNOTATION_pY, ALLOWED_IONS
from aiproteomics.core.fragment import Fragment
from aiproteomics.core.spectrum import output_layer_to_spectrum
from aiproteomics.core.modeltypes import AIProteomicsModel


def parse_ion_annotation(ion):
    """
        For a given ion annotation string, `ion` (e.g. "y3(2+)-H2O") this function
        will parse the constituent information into:
        `ion_type` (e.g. 'y')
        `ion_break` (e.g. 3, the point in the sequence where breakage occured)
        `ion_charge` (e.g. 12)
        `neutral_loss` (e.g. "H2O". If no loss, this is an empty string)

        The above extracted info is returned as a `Fragment` object.
    """

    if 'nan' in ion:
        return None

    if ion == ANNOTATION_pY:
        return Fragment(ANNOTATION_pY, 1, 0, '')

    # Get single letter ion identifier e.g. 'y', 'b', 'a'
    ion_type = ion[0]
    if ion_type not in ALLOWED_IONS:
        raise ValueError(f'Ion type {ion_type} not in expected ion types: {ALLOWED_IONS}')

    # Attempt to split into ion and neutral loss
    ion_split = ion[1:].split('-')
    ion_part = ion_split[0]
    neutral_loss = ""

    # If neutral loss
    if len(ion_split) == 2:
        ion_part = ion_split[0]
        neutral_loss = ion_split[1]

    # Determine ion charge
    ion_part_split = ion_part.split('(')
    ion_charge = 1
    if len(ion_part_split) == 2:
        ion_charge = int(ion_part_split[1].split('+')[0])

    # Get ion breakage position
    ion_break_str = ion_part_split[0]
    if ion_break_str[-1] == '*':
        # Check if asterisk after breakage, corresponding to phospho loss
        ion_break_str = ion_break_str[:-1]
        neutral_loss = "H3PO4"

    try:
        ion_break = int(ion_break_str)
    except ValueError as ve:
        raise ValueError(f'Exception when converting ion breakage str {ion_break_str}: {ve}') from ve

    return Fragment(ion_type, ion_charge, ion_break, neutral_loss)




def build_spectral_library(inputs: pd.DataFrame,
                           msms: AIProteomicsModel = None,
                           rt: AIProteomicsModel = None,
                           ccs: AIProteomicsModel = None,
                           pY_threshold = 0.8):
    """
    A utility function that generates a spectral library for a batch of inputs, given one or more prediction models.

    Args:
        `inputs`: A `pandas.DataFrame` with two columns: "peptide" (a string representation of a peptide, including UniMod modifications)
                  and "charge" (an integer value giving the charge on this precursor sequence).
        `msms`: An `AIProteomicsModel` for predicting the msms spectra (including pY) for a given peptide sequence and charge.
        `rt` (Optional): An `AIProteomicsModel` for predicting the normalized retention time for a given peptide sequence.
        `ccs`(Optional): An `AIProteomicsModel` for predicting the ion mobility for a given peptide sequence.
        `pY_threshold`: An intensity value below which the predicted pY value is ignored (not significant)

    Returns:
        A `pandas.DataFrame` containing the predicted spectra of all sequences provided in the `inputs` `DataFrame`.
    """

    if msms is None:
        raise ValueError("At least an msms model must be provided or no spectral library can be generated")

    # Check columns of inputs dataframe
    required_columns = [
        "peptide",
        "charge"
    ]
    for col in required_columns:
        if col not in inputs:
            raise ValueError(f"Inputs dataframe must have the column {col}")

    # Map inputs to nn model inputs
    input_seq = np.stack(inputs["peptide"].map(msms.seq_map.map_to_int))
    input_charge = inputs["charge"].values

    # Run model inference
    if msms:
        msms_intensities, msms_pY = msms.nn_model.predict([input_seq, input_charge])
    if rt:
        rt_out = rt.nn_model.predict([input_seq])
    if ccs:
        ccs_out = ccs.nn_model.predict([input_seq])


    # Generate a spectrum (as a dataframe) for each sequence based on the model predictions
    dfs = []
    for index, row in inputs.iterrows():

        if msms:
            intensities = msms_intensities[index]
            pY = msms_pY[index]
        else:
            intensities = None
            pY = None

        if rt:
            iRT = rt_out[index]
        else:
            iRT = None

        if ccs:
            ccs = ccs_out[index]
        else:
            ccs = None

        unmodified_peptide_sequence = msms.seq_map.generate_unmodified_peptide_sequence(row["peptide"])

        df = output_layer_to_spectrum(
                intensities,
                msms.model_params,
                row["peptide"],
                row["charge"],
                pY=pY,
                iRT=iRT,
                ccs=ccs,
                thresh=pY_threshold,
                unmodified_seq=unmodified_peptide_sequence)
        dfs.append(df.to_dataframe())

    # Concatenate all spectra into one dataframe and return it
    df = pd.concat(dfs).reset_index()
    return df
