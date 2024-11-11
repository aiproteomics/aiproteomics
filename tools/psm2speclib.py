# LOOKS WRONG in pyteomics - Check if y and b ions have correct OH, H additions

# Check about pY charge and fragment values (need integers)

import sys
import argparse
import re

import dask.dataframe as dd
from dask.diagnostics import ProgressBar

from pyteomics import mass

# TODO: Move this to settings
ALLOWED_IONS = ['y', 'b', 'a']
MASS_pY = 216.043 # phosphorylation diagnostic peak


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

aa_mod_map = {
        'M(UniMod:35)': '1',
        'S(UniMod:21)': '2',
        'T(UniMod:21)': '3',
        'Y(UniMod:21)': '4',
        '(UniMod:1)':   '*',
        'C(UniMod:4)':  'C'
}

aa_mass = generate_aa_mass()


def generate_unmodified_peptide_sequence(modified_seq):
    """ For a given peptide sequence, `modified_seq`, containing modification
        notation in the form '(UniMod:X)', this function will return the
        sequence absent any modification text (by simply removing anything
        in brackets `()`).
    """
    return re.sub(r"[\(].*?[\)]", "", modified_seq)


def unimod_to_single_char_sequence(seq, ignore_unsupported=False):
    seq = seq.strip()
    seq = seq.strip('_')

    for k, v in aa_mod_map.items():
        if '(' not in seq:
            break
        seq = seq.replace(k, v)

    # If there are still modifications present, then they
    # are not supported.
    if '(' in seq:
        if ignore_unsupported:
            return None
        raise ValueError(f'Sequence {seq} contains unsupported amino acid modifications. List of supported mods: {aa_mod_map.keys()}')

    return seq


def get_ion_mz(seq, ion_type, ion_break, ion_charge, aa_mass):

    if ion_type[0] in 'abc':
        # If the first entry is acetylation, skip it as not real amino acid (check this!)
        if seq[0] == '*':
            frag_seq = seq[:ion_break+1]
        else:
            frag_seq = seq[:ion_break]
    else:
        frag_seq = seq[-ion_break:]

    return mass.fast_mass(frag_seq, ion_type=ion_type, charge=ion_charge, aa_mass=aa_mass)


def parse_ion(ion):
    if 'nan' in ion:
        return None

    if ion == 'pY':
        return ('pY', 0, 1, None)

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

    return (ion_type, ion_break, ion_charge, neutral_loss)


def parse_ions(str_matches, str_intensities, seq):

    # Parse each ion in the semi-colon separated list of matches.
    # Extract the type (y, b, a or pY), the breakage point,
    # and the neutral loss (H2O, NH3, H3PO4 or none). Use these
    # to calculate the m/z of the corresponding fragment.

    annotations = []
    intensities = []
    ion_mzs = []
    ion_types = []
    ion_breaks = []
    ion_charges = []
    losses = []

    for annotation, intensity in zip(str_matches.strip().split(';'), str_intensities.strip().split(';')):

        parsed_ion = parse_ion(annotation)

        if parsed_ion is None:
            continue

        ion_type, ion_break, ion_charge, neutral_loss = parsed_ion

        if ion_type == 'pY':
            ion_mz = MASS_pY
        else:
            ion_mz = get_ion_mz(seq, ion_type, ion_break, ion_charge, aa_mass)

        if neutral_loss:
            ion_mz -= mass_neutral_loss[neutral_loss]

        annotations.append(annotation)
        intensities.append(intensity)
        ion_mzs.append(ion_mz)
        ion_types.append(ion_type)
        ion_breaks.append(ion_break)
        ion_charges.append(ion_charge)
        losses.append(neutral_loss)


    return {"Annotation": annotations,
            "LibraryIntensity": intensities,
            "ProductMz": ion_mzs,
            "FragmentType": ion_types,
            "FragmentSeriesNumber": ion_breaks,
            "FragmentCharge": ion_charges,
            "FragmentLossType": losses}


# Specify types for each of the output columns from the initial mapping.
# "object" is used for those columns that will contain a list (since the
# value is different for each fragment).
out_dtypes = {
    "PrecursorMz": "string",
    "ProductMz": "object",
    "Annotation": "object",
    "PeptideSequence": "string",
    "ModifiedPeptideSequence": "string",
    "PrecursorCharge": "int32",
    "LibraryIntensity": "object",
    "NormalizedRetentionTime": "float32",
    "PrecursorIonMobility": "float32",
    "FragmentType": "object",
    "FragmentCharge": "object",
    "FragmentSeriesNumber": "object",
    "FragmentLossType": "object"
}


def map_psm_row(row, ignore_unsupported=True):
    modified_peptide_sequence = row['Modified.sequence']
    precursor_charge = row['Charge']
    matches = row['Matches']
    intensities = row['Intensities']
    normalized_retention_time = row['Retention.time']
    precursor_ion_mobility = row['CCS']

    # Map peptide sequence with unimod modifications so that there is
    # only one character per amino acid (as defined in aa_mod_map)
    seq = unimod_to_single_char_sequence(modified_peptide_sequence, ignore_unsupported=ignore_unsupported)

    if seq is None:
        return None

    precursor_mz = mass.fast_mass(sequence=seq, charge=precursor_charge, aa_mass=aa_mass, ion_type='M')
    unmodified_peptide_sequence = generate_unmodified_peptide_sequence(modified_peptide_sequence)


    # First set the properties that come from the precursor
    result = dict.fromkeys(out_dtypes.keys())
    result["PrecursorMz"] = precursor_mz
    result["PeptideSequence"] = unmodified_peptide_sequence
    result["ModifiedPeptideSequence"] = modified_peptide_sequence
    result["PrecursorCharge"] = precursor_charge
    result["NormalizedRetentionTime"] = normalized_retention_time
    result["PrecursorIonMobility"] = precursor_ion_mobility

    # Next get the properties of each fragment
    ions_dict = parse_ions(matches, intensities, seq)
    result.update(ions_dict)

    if len(ions_dict["Annotation"]) == 0:
        return None

    return result


if __name__ == "__main__":

    # Parse commandline arguments. Input and output filenames must be provided.
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', type=str, help='The input PSM file you wish to convert.', required=True)
    parser.add_argument('-o', '--outfile', type=str, help='The output tsv format speclib.', required=True)
    parser.add_argument('-I', '--ignore-unsupported', action="store_true", default=False, help='Ignore unsupported modified sequences.')
    parser.add_argument('-n', '--num-partitions', type=int, default=1, help='Number of partitions to use with Dask.')
    args = parser.parse_args(sys.argv[1:len(sys.argv)])

    # Read input PSM tsv file and make lists of fragment info for each row
    psm_df = dd.read_csv(args.infile, sep='\t')
    psm_df = psm_df.repartition(npartitions=args.num_partitions)
    out_series = psm_df.map_partitions(lambda part : part.apply(
        lambda row: map_psm_row(row, ignore_unsupported=args.ignore_unsupported), axis=1, result_type='expand'), meta=out_dtypes)

    # Explode the lists so that we get 1 row per fragment
    explode_cols = [
        "ProductMz",
        "Annotation",
        "LibraryIntensity",
        "FragmentType",
        "FragmentCharge",
        "FragmentSeriesNumber",
        "FragmentLossType"
    ]
    out_series = out_series.explode(column=explode_cols)

    # Drop empty rows (corresponds to input sequences that had no matches - e.g. set to nan)
    out_series = out_series.dropna()

    # Write the resulting speclib to file.
    with ProgressBar():
        out_series.to_csv(args.outfile, sep='\t', na_rep='NaN', index=False, header=True, single_file=True)
