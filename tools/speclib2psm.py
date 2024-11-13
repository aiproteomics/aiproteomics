import sys
import argparse
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

# Columns from the input file that are not needed for creating
# the output file and can be dropped for performance reasons
drop_columns = [
    "FragmentType",
    "FragmentCharge",
    "FragmentSeriesNumber",
    "FragmentLossType"]

# Columns to group by, for determining which rows belong to
# a unique entry in the output PSM file
group_columns = ['ModifiedPeptideSequence',
                 'PeptideSequence',
                 'PrecursorMz',
                 'PrecursorCharge',
                 'NormalizedRetentionTime',
                 'PrecursorIonMobility']

# Specify the mapping of column headings in the input file
# to the desired column headings in the output file.
column_map = {
    "ModifiedPeptideSequence": "Modified.sequence",
    "PrecursorCharge": "Charge",
    "NormalizedRetentionTime": "Retention.time",
    "PrecursorIonMobility": "CCS"}

# Desired order of columns in the output tsv file
output_headings_order = ["Modified.sequence", "Charge", "Matches", "Intensities", "Retention.time", "CCS"]

# Specify types for each of the output columns
out_dtypes = {"Modified.sequence": "string",
             "Charge": "int64",
             "Retention.time": "float64",
             "CCS": "float64",
             "Intensities": "string",
             "Matches": "string"}

def map_row(row):
    """
    Maps the row headings from the input file to the desired headings
    for the output file. Also adds two columns, "Intensities" and
    "Matches", containing a semi-colon separated list of ion annotations
    and intensities.
    """

    new_row = { new_key: row[old_key] for old_key, new_key in column_map.items() }
    new_row["Intensities"] = ";".join(str(i) for i in row["LibraryIntensity"])
    new_row["Matches"] = ";".join(row["Annotation"])

    return new_row


if __name__ == "__main__":

    # Parse commandline arguments. Input and output filenames must be provided.
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inpath', type=str, help='The input speclib file you wish to convert.', required=True)
    parser.add_argument('-o', '--outpath', type=str, help='The output (PSM) filename.', required=True)
    parser.add_argument('-f', '--informat', type=str, choices=['tsv', 'parquet'],
                        help='The input format. If tsv, inpath is expected to be a tsv file. '
                        'If parquet, inpath is expected to be a directory of parquet files.',
                        required=True)
    parser.add_argument('-g', '--outformat', type=str, choices=['tsv', 'parquet'], help='The output format.', required=True)
    parser.add_argument('-n', '--num-partitions', type=int, default=1, help='Number of partitions to use with Dask.')
    args = parser.parse_args(sys.argv[1:len(sys.argv)])

    # Read input speclib file according to specified format
    if args.informat == 'tsv':
        speclib_df = dd.read_csv(args.inpath, sep='\t')
    else:
        speclib_df = dd.read_parquet(args.inpath)

    # Drop unneeded columns (this can take about 30% off the compute, depending on sizes)
    speclib_df = speclib_df.drop(drop_columns, axis=1)

    # Aggregate annotations and intensities to a semi-colon separated list
    speclib_df = speclib_df.groupby(group_columns).agg({'Annotation': list, 'LibraryIntensity': list}).reset_index()
    speclib_df = speclib_df.map_partitions(lambda part : part.apply(map_row, axis=1, result_type='expand'), meta=out_dtypes)

    # Compute and write to tsv
    speclib_df = speclib_df[output_headings_order]
    with ProgressBar():
        if args.outformat == 'tsv':
            speclib_df.to_csv(args.outpath, sep='\t', single_file=True, header=True, index=False)
        elif args.outformat == 'parquet':
            speclib_df.to_parquet(args.outpath)
