import sys
import argparse

import dask.dataframe as dd
from dask.diagnostics import ProgressBar

from aiproteomics.core.definitions import MASS_pY, ANNOTATION_pY, get_ion_mz, get_precursor_mz, aa_mass, mass_neutral_loss
from aiproteomics.core.utils import generate_unmodified_peptide_sequence, unimod_to_single_char_sequence

# The ions supported by this conversion tool
ALLOWED_IONS = ['y', 'b', 'a']

# Specify types for each of the output columns of the speclib file.
out_dtypes = {
    "PrecursorMz": "float32",
    "ProductMz": "float32",
    "Annotation": "string",
    "PeptideSequence": "string",
    "ModifiedPeptideSequence": "string",
    "PrecursorCharge": "int32",
    "LibraryIntensity": "float32",
    "NormalizedRetentionTime": "float32",
    "PrecursorIonMobility": "float32",
    "FragmentType": "string",
    "FragmentCharge": "int32",
    "FragmentSeriesNumber": "int32",
    "FragmentLossType": "string"
}

# Columns corresponding to info about each fragment
explode_cols = [
    "ProductMz",
    "Annotation",
    "LibraryIntensity",
    "FragmentType",
    "FragmentCharge",
    "FragmentSeriesNumber",
    "FragmentLossType"
]


def parse_ion(ion):
    """
        For a given ion annotation, `ion` (e.g. "y3(2+)-H2O") this function
        will parse the constituent information, returning a tuple of:
        `ion_type` (e.g. 'y')
        `ion_break` (e.g. 3, the point in the sequence where breakage occured)
        `ion_charge` (e.g. 12)
        `neutral_loss` (e.g. "H2O". If no loss, this is an empty string)
    """

    if 'nan' in ion:
        return None

    if ion == ANNOTATION_pY:
        return (ANNOTATION_pY, 0, 1, None)

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
    """
        Parse each ion in the semi-colon separated list of matches.
        Extract the type (y, b, a or pY), the breakage point,
        and the neutral loss (H2O, NH3, H3PO4 or none). Use these
        to calculate the m/z of the corresponding fragment.

        Returns a dict containing the above information. Values are
        lists of length N where N is the number of product fragments.
    """

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

        if ion_type == ANNOTATION_pY:
            ion_mz = MASS_pY
        else:
            ion_mz = get_ion_mz(seq, ion_type, ion_break, ion_charge, aa_mass)

        if neutral_loss:
            ion_mz -= mass_neutral_loss[neutral_loss]

        annotations.append(annotation)
        intensities.append(float(intensity))
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


def map_psm_row(row, ignore_unsupported=True):
    """
    Parses all ions in the `Matches` column of the input row, and calculates the m/z
    for each corresponding product. Returns a dict containing all the columns specified
    in `out_dtypes`. For columns that have more than one value (e.g. the `LibraryIntensity`
    column) a list will be given.
    """

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


    precursor_mz = get_precursor_mz(seq, precursor_charge, aa_mass=aa_mass)
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
    parser.add_argument('-i', '--inpath', type=str, help='The input PSM file you wish to convert.', required=True)
    parser.add_argument('-o', '--outpath', type=str, help='The path to write the output (a file for tsv, a directory for parquet).', required=True)
    parser.add_argument('-I', '--ignore-unsupported', action="store_true", default=False, help='Ignore unsupported modified sequences.')
    parser.add_argument('-f', '--informat', type=str, choices=['tsv', 'parquet'],
                        help='The input format. If tsv, inpath is expected to be a tsv file. '
                             'If parquet, inpath is expected to be a directory of parquet files.',
                        required=True)
    parser.add_argument('-g', '--outformat', type=str, choices=['tsv', 'parquet'], help='The output format.', required=True)
    parser.add_argument('-n', '--num-partitions', type=int, default=1, help='Number of partitions to use with Dask.')
    args = parser.parse_args(sys.argv[1:len(sys.argv)])

    # Read input PSM file according to specified format
    if args.informat == 'tsv':
        psm_df = dd.read_csv(args.inpath, sep='\t')
    else:
        psm_df = dd.read_parquet(args.inpath)
    psm_df = psm_df.repartition(npartitions=args.num_partitions)

    # Make lists of fragment info for each row, then explode the lists
    # so we get 1 row per fragment
    speclib_df = psm_df.map_partitions(
            lambda part : part.apply(
            lambda row: map_psm_row(row, ignore_unsupported=args.ignore_unsupported), axis=1, result_type='expand'
            ).explode(column=explode_cols), meta=out_dtypes)

    # Drop empty rows (corresponds to input sequences that had no matches - e.g. set to nan)
    speclib_df = speclib_df.dropna()

    # Write the resulting speclib to file.
    with ProgressBar():
        if args.outformat == 'tsv':
            speclib_df.to_csv(args.outpath, sep='\t', na_rep='NaN', index=False, header=True, single_file=True)
        elif args.outformat == 'parquet':
            speclib_df.to_parquet(args.outpath)
