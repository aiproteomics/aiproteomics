# `aiproteomics` Tools

This directory contains various tools for pre- and post- processing, and conversion.

## `psm2speclib`
This tool converts a PSM style tsv or parquet formatted file(s) to a corresponding spectral library (also tsv or parquet formatted)
containing a row per product.

### Running
It can be run with, e.g.
```
python psm2speclib.py -i in_psm.tsv -o out_speclib.parquet --informat tsv --outformat parquet
```
to convert a psm in tsv format to a spectral lib in parquet format. The input and output formats
do not have to be the same.

To see help on all options:
```
python psm2speclib.py --help
```

### Ignoring unsupported modifications
`psm2speclib` is designed mainly for phosphoproteomics data so many UniMod/Amino acid combinations are
not supported. By default, the tool with throw an error if it finds a sequence with an unsupported modification.
To ignore any such sequences, use the `--ignore-unsupported` (or simply `-I`) option:

```
python psm2speclib.py -i in_psm.parquet/ -o out_speclib.tsv --ignore_unsupported --informat parquet --outformat tsv
```

## `speclib2psm`
This tool does the reverse conversion as `psm2speclib`. It takes an input tsv formatted spectral library
and converts to a PSM style tsv file.

### Running
It can be run with e.g.
```
python speclib2psm.py -i speclib.parquet/ -o test_psm.parquet/ --informat parquet --outformat parquet
```

The format options are the same as for `psm2speclib`.

To see help on all options:
```
python speclib2psm.py --help
```
