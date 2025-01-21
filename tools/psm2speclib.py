import sys
import argparse

import dask.dataframe as dd
from dask.diagnostics import ProgressBar

from aiproteomics.core.definitions import ANNOTATION_pY
from aiproteomics.core.mz import MASS_pY, get_ion_mz, get_precursor_mz, aa_mass, mass_neutral_loss
from aiproteomics.core.sequence import SequenceMapper, PHOSPHO_MAPPING
from aiproteomics.core.utils import parse_ion_annotation

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

        frag = parse_ion_annotation(annotation)

        if frag is None:
            continue

        if frag.fragment_type == ANNOTATION_pY:
            ion_mz = MASS_pY
        else:
            ion_mz = get_ion_mz(seq, frag, aa_mass)

        annotations.append(annotation)
        intensities.append(float(intensity))
        ion_mzs.append(ion_mz)
        ion_types.append(frag.fragment_type)
        ion_breaks.append(frag.fragment_series_number)
        ion_charges.append(frag.fragment_charge)
        losses.append(frag.fragment_loss_type)


    return {"Annotation": annotations,
            "LibraryIntensity": intensities,
            "ProductMz": ion_mzs,
            "FragmentType": ion_types,
            "FragmentSeriesNumber": ion_breaks,
            "FragmentCharge": ion_charges,
            "FragmentLossType": losses}


def map_psm_row(row, sequence_mapper=None, ignore_unsupported=True):
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
    seq = sequence_mapper.unimod_to_single_char_sequence(modified_peptide_sequence, ignore_unsupported=ignore_unsupported)

    if seq is None:
        return None


    precursor_mz = get_precursor_mz(seq, precursor_charge, aa_mass=aa_mass)
    unmodified_peptide_sequence = sequence_mapper.generate_unmodified_peptide_sequence(modified_peptide_sequence)


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

    # Build a sequence mapper with the desired mapping characteristics
    seqmap = SequenceMapper(min_seq_len=7, max_seq_len=50, mapping=PHOSPHO_MAPPING)

    # Make lists of fragment info for each row, then explode the lists
    # so we get 1 row per fragment
    speclib_df = psm_df.map_partitions(
            lambda part : part.apply(
            lambda row: map_psm_row(row, sequence_mapper=seqmap, ignore_unsupported=args.ignore_unsupported), axis=1, result_type='expand'
            ).explode(column=explode_cols), meta=out_dtypes)

    # Drop empty rows (corresponds to input sequences that had no matches - e.g. set to nan)
    speclib_df = speclib_df.dropna()

    # Write the resulting speclib to file.
    with ProgressBar():
        if args.outformat == 'tsv':
            speclib_df.to_csv(args.outpath, sep='\t', na_rep='NaN', index=False, header=True, single_file=True)
        elif args.outformat == 'parquet':
            speclib_df.to_parquet(args.outpath)
