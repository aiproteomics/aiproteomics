from dataclasses import dataclass
import itertools as it
from typing import Optional

import numpy as np
import pandas as pd

from fragment import Fragment
from definitions import ANNOTATION_pY
from mz import MASS_pY, get_ion_mz, get_precursor_mz, aa_mass, mass_neutral_loss
from utils import generate_unmodified_peptide_sequence, unimod_to_single_char_sequence


@dataclass
class ModelParams:
    seq_len: int
    ions: list
    num_charges: int
    neutral_losses: list

    def __post_init__(self):
        """
            Fragment list is generated and the result stored with the object
            since it won't change and is used a lot.
        """
        self.fragments = self.generate_fragment_list()


    def generate_fragment_list(self):
        """
            Generate permutations of fragments possible for the output layer
            of the model described by this `model_params` object.
        """
        frag_iter = it.product(
                            self.neutral_losses,
                            range(1, self.seq_len),
                            self.ions,
                            range(1, self.num_charges + 1))

        frag_list = np.array([Fragment(ion, charge, cleavage, loss) for loss, cleavage, ion, charge in frag_iter])

        return frag_list


    def generate_mask(self, input_seq_len=None, precursor_charge=None):
        """
            Generate a mask for an input sequence of length `input_seq_len` and
            charge `precursor_charge`. The mask is an array of booleans in which
            True means the corresponding fragment is possible and False otherwise.

            This is useful because a breakage point cannot be greater than the length
            of the sequence - 1, and typically fragments are not charged greater than
            their precursor was.
        """
        mask = np.array([
            (frag.fragment_series_number < input_seq_len and frag.fragment_charge <= precursor_charge)
            for frag in self.fragments])

        return mask


@dataclass
class SpectrumEntry:
    frag: Fragment
    intensity: float
    mz: float


@dataclass
class Spectrum:
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


