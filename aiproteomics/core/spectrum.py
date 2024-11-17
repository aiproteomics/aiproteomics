from dataclasses import dataclass, asdict
import itertools as it
from typing import Optional

import numpy as np
import pandas as pd

from aiproteomics.core.fragment import Fragment
from aiproteomics.core.definitions import ANNOTATION_pY
from aiproteomics.core.mz import MASS_pY, get_ion_mz, get_precursor_mz, aa_mass

@dataclass
class SpectrumEntry:
    """
        One fragment in a `Spectrum` object. Contains the `Fragment` description (that
        includes annotation, break point, ion type etc.), the intensity of the peak
        and the corresponding m/z value.
    """

    frag: Fragment
    intensity: float
    mz: float


@dataclass
class Spectrum:
    """
        An object to describe the MS spectrum for a given peptide. This means the list of
        product fragments (e.g. "y3", "b2" etc), their m/z and their intensity.
    """

    precursor_mz: float
    precursor_sequence: str
    precursor_charge: int
    pY: float
    iRT: float
    ccs: float
    products: list
    precursor_unmodified: Optional[str] = None
    protein_id: Optional[str] = None
    gene_name: Optional[str] = None

    def to_dataframe(self):
        """
            Returns a representation of this spectrum as a pandas `DataFrame` object.
            The dataframe can then be converted to either a `tsv` (using `.to_csv(sep='\t')`
            or to `parquet` (using `.to_parquet()`).
        """

        num_products = len(self.products)
        if self.pY:
            num_products += 1

        # Set the data types of the output columns
        # First set the values that are the same for all product fragments
        out_dict = {
                "PrecursorMz": np.full(num_products, self.precursor_mz, dtype="float32"),
                "ProductMz": np.full(num_products, 0.0, dtype="float32"),
                "Annotation": np.empty(num_products, dtype="object"),
                "ProteinId": np.full(num_products, self.protein_id, dtype="object"),
                "GeneName": np.full(num_products, self.gene_name, dtype="object"),
                "PeptideSequence": np.full(num_products, self.precursor_unmodified, dtype="object"),
                "ModifiedPeptideSequence": np.full(num_products, self.precursor_sequence, dtype="object"),
                "PrecursorCharge": np.full(num_products, self.precursor_charge, dtype="int32"),
                "LibraryIntensity": np.full(num_products, 0.0, dtype="float32"),
                "NormalizedRetentionTime": np.full(num_products, self.iRT, dtype="float32"),
                "PrecursorIonMobility": np.full(num_products, self.ccs, dtype="float32"),
                "FragmentType": np.empty(num_products, dtype="object"),
                "FragmentCharge": np.full(num_products, 0, dtype="int32"),
                "FragmentSeriesNumber": np.full(num_products, 0, dtype="int32"),
                "FragmentLossType": np.empty(num_products, dtype="object")
        }

        # Now add the fragment specific data
        for i, product in enumerate(self.products):
            out_dict["ProductMz"][i] = product.mz
            out_dict["Annotation"][i] = product.frag.annotation
            out_dict["LibraryIntensity"][i] = product.intensity
            out_dict["FragmentType"][i] = product.frag.fragment_type
            out_dict["FragmentCharge"][i] = product.frag.fragment_charge
            out_dict["FragmentSeriesNumber"][i] = product.frag.fragment_series_number
            out_dict["FragmentLossType"][i] = product.frag.fragment_loss_type

        # If the pY diagnostic peak is provided, add that as the final entry
        if self.pY:
            out_dict["ProductMz"][-1] = MASS_pY
            out_dict["Annotation"][-1] = ANNOTATION_pY
            out_dict["LibraryIntensity"][-1] = self.pY
            out_dict["FragmentType"][-1] = ANNOTATION_pY
            out_dict["FragmentCharge"][-1] = 0
            out_dict["FragmentSeriesNumber"][-1] = 0
            out_dict["FragmentLossType"][-1] = ''

        return pd.DataFrame(out_dict)



def output_layer_to_spectrum(output_layer, model_params, sequence, precursor_charge, pY=None, iRT=None, ccs=None, thresh=0.1):
    """
        Calculates the spectrum (product fragments, with annotations and intensities) corresponding to the
        given model parameters and output layer(s).

        Inputs:
            `model_params`: parameters of the AI msms prediction model
            `output_layer` of the AI msms model (assumed to be something like a numpy Array)
            `precursor_charge`, the charge of the peptide prior to fragmentation
            `pY`, if available (otherwise `None`). The predicted diagnostic peak (for phospho)
            `iRT`, if available (otherwise `None`). The predicted normalized retention time of the precursor peptide.
            `ccs`, if available (otherwise `None`). The predicted ion mobility of the precursor peptide.
        Outputs:
            A `Spectrum` object containing the list of predicted product fragments and their intensities

    """

    # Work out length of input sequence (in amino acids) so that the
    # relevant portion of the output layer can be considered
    input_seq_len = len(sequence)
    if sequence[0] == '*':
        input_seq_len -= 1

    # Get mask for possible fragments
    mask = model_params.generate_mask(input_seq_len=input_seq_len, precursor_charge=precursor_charge)

    # Identify which intensity peaks are above the defined threshold
    # and get the list of fragments that correspond to that. Also
    # mask the fragments that are not valid for the given length
    # and precursor charge.
    peaks = (mask) & (output_layer > thresh)
    frag_list = zip(model_params.fragments[peaks], output_layer[peaks])

    # Calc precursor m/z
    precursor_mz = get_precursor_mz(sequence, precursor_charge, aa_mass=aa_mass)

    # Build the list of spectrum entries for each product fragment
    products = [SpectrumEntry(
                                frag,
                                intensity,
                                get_ion_mz(sequence, frag, aa_mass)
                                ) for frag, intensity in frag_list]

    # Include pY diagnostic peak only if above threshold
    # (and provided)
    if pY and pY < thresh:
        pY = None

    return Spectrum(
                precursor_mz=precursor_mz,
                precursor_sequence=sequence,
                precursor_charge=precursor_charge,
                pY=pY,
                iRT=iRT,
                ccs=ccs,
                products=products)
