from dataclasses import dataclass
import itertools as it

import numpy as np
from pyteomics import mass


@dataclass
class Fragment:
    fragment_type: str
    fragment_charge: int
    fragment_series_number: int
    fragment_loss_type: str

    def __post_init__(self):
        self.annotation = self.make_annotation_str()

    def make_annotation_str(self) -> str:
        s = self.fragment_type + str(self.fragment_series_number)
        if self.fragment_charge > 1:
            s += f'({str(self.fragment_charge)}+)'
        if self.fragment_loss_type:
            s += f'-{self.fragment_loss_type}'
        return s


@dataclass
class ModelParams:
    seq_len: int
    ions: list
    num_charges: int
    neutral_losses: list

    def generate_fragment_list(self, input_seq_len=None):
        """
            Generate permutations of fragments possible for the output layer
            of the model described by this `model_params` object.

            By default, this function assumes a sequence of the maximum size
            allowed in this model (`seq_len`). However, if `input_seq_len` is
            passed, the returned list is truncated to only those fragments which
            are possible for a sequence of that length.
        """

        frag_iter = it.product(
                            range(1, self.seq_len),
                            self.ions,
                            range(self.num_charges),
                            self.neutral_losses)

        frag_list = np.array([Fragment(ion, charge, cleavage, loss) for cleavage, ion, charge, loss in frag_iter])

        if input_seq_len:
            return frag_list[:input_seq_len * len(self.ions) * self.num_charges * len(self.neutral_losses)]

        return frag_list

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
    iRT: float
    ccs: float
    products: list

    def to_tsv(self, unknown_value_str='NA', intensity_scale=10000):

        s = ''

        # Add rows for each fragment
        for product in self.products:

            # Collect the values for each column describing a particular peak
            row = [
                str(self.precursor_mz),
                str(product.mz),
                product.frag.annotation,
                unknown_value_str,
                unknown_value_str,
                self.precursor_sequence,
                self.precursor_sequence,
                str(self.precursor_charge),
                str(product.intensity * intensity_scale),
                str(self.iRT),
                str(self.ccs),
                product.frag.fragment_type,
                str(product.frag.fragment_charge),
                str(product.frag.fragment_series_number),
                product.frag.fragment_loss_type
            ]
            # Add the row to the output string
            s += '\t'.join(row) + '\n'

        return s

    def get_tsv_format_header(self):
        tsv_columns = [
            "PrecursorMz",
            "ProductMz",
            "Annotation",
            "ProteinId",
            "GeneName",
            "PeptideSequence",
            "ModifiedPeptideSequence",
            "PrecursorCharge",
            "LibraryIntensity",
            "NormalizedRetentionTime",
            "PrecursorIonMobility",
            "FragmentType",
            "FragmentCharge",
            "FragmentSeriesNumber",
            "FragmentLossType"
        ]
        return '\t'.join(tsv_columns) + '\n'


def generate_aa_mass():
    db = mass.Unimod()
    aa_comp = mass.std_aa_comp.copy()

    # Get relevant modifcations from unimod database
    oxidation = db.by_title("Oxidation")["composition"]
    phospho = db.by_title("Phospho")["composition"]
    carbamidomethyl = db.by_title("Carbamidomethyl")["composition"]
    acetyl = db.by_title("Acetyl")["composition"]

    # Generate modified amino acid compositions
    aa_comp["1"] = aa_comp["M"] + oxidation
    aa_comp["2"] = aa_comp["S"] + phospho
    aa_comp["3"] = aa_comp["T"] + phospho
    aa_comp["4"] = aa_comp["Y"] + phospho
    aa_comp["*"] = acetyl
    aa_comp["C"] = aa_comp["C"] + carbamidomethyl

    # Get masses
    aa_mass = {k: mass.calculate_mass(v) for k, v in aa_comp.items()}

    return aa_mass


# Neutral losses
mass_neutral_loss = {
    "H2O": mass.calculate_mass(formula='H2O'),
    "NH3": mass.calculate_mass(formula='NH3'),
    "H3PO4": mass.calculate_mass(formula='H3PO4')
}

aa_mass = generate_aa_mass()


def get_ion_mz(seq, frag: Fragment, aa_mass):
    """
    Calculate the mass of the given fragment for the given sequence.
    """

    ion_type = frag.fragment_type
    ion_break = frag.fragment_series_number
    ion_charge = frag.fragment_charge

    if ion_break > len(seq):
        print(seq, ion_break, ion_type)

    if ion_type[0] in 'abc':
        frag_seq = seq[:ion_break]
    else:
        # If the first entry is acetylation, skip it as not real amino acid (check this!)
        if seq[0] == '*':
            frag_seq = seq[ion_break + 1:]
        else:
            frag_seq = seq[ion_break:]

    return mass.fast_mass(frag_seq, ion_type=ion_type, charge=ion_charge, aa_mass=aa_mass)



def output_layer_to_spectrum(output_layer, model_params, sequence, precursor_charge, iRT, ccs, thresh=0.1):

    # Work out length of input sequence (in amino acids) so that the
    # relevant portion of the output layer can be considered
    input_seq_len = len(sequence)
    if sequence[0] == '*':
        input_seq_len -= 1

    # Identify which intensity peaks are above the defined threshold
    # and get the list of fragments that correspond to that.
    frag_key = model_params.generate_fragment_list(input_seq_len=input_seq_len)
    output_layer_truncated = output_layer[:len(frag_key)]
    peaks = output_layer_truncated > thresh
    frag_list = zip(frag_key[peaks], output_layer_truncated[peaks])


    # Calc precursor m/z
    precursor_mz = mass.fast_mass(sequence=sequence, charge=precursor_charge, aa_mass=aa_mass, ion_type='M')

    # Build the list of spectrum entries for each product fragment
    products = [SpectrumEntry(
                                frag,
                                intensity,
                                get_ion_mz(sequence, frag, aa_mass)
                                ) for frag, intensity in frag_list]

    return Spectrum(
                precursor_mz=precursor_mz,
                precursor_sequence=sequence,
                precursor_charge=precursor_charge,
                iRT=iRT,
                ccs=ccs,
                products=products)


model_params = ModelParams(seq_len=50, ions=['y','b'], num_charges=2, neutral_losses=['', 'H3PO4'])
print(model_params)

frag_list = model_params.generate_fragment_list()

print("Length:", len(frag_list))
print([frag.annotation for frag in frag_list])

# Create random output layer
output_layer = np.random.random(392)

# Convert to spectrum
spectrum = output_layer_to_spectrum(output_layer, model_params, "*SSS1TT221", 1, 0, 0, thresh=0.9)

print(spectrum.get_tsv_format_header())
print(spectrum.to_tsv())

