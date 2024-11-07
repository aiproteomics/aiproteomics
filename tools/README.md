# `aiproteomics` Tools

This directory contains various tools for pre- and post- processing, and conversion.

## `psm2speclib`
This tool converts a PSM style tsv formatted file to a corresponding spectral library (also tsv formatted)
containing a row per product.

### Requirements
Currently it requires a minimum of `dask 2024.10.0` and `pyteomics 4.7.5`. The dask version
may clash with the current tensorflow version requirements (which are very rigid).

### Running
It can be run with, e.g.
```
python psm2speclib.py -i in_psm.tsv -o out_speclib.tsv
```

To see help on all options:
```
python psm2speclib.py --help
```

### Ignoring unsupported modifications
`psm2speclib` is designed mainly for phosphoproteomics data so many UniMod/Amino acid combinations are
not supported. By default, the tool with throw an error if it finds a sequence with an unsupported modification.
To ignore any such sequences, use the `--ignore-unsupported` (or simply `-I`) option:

```
python psm2speclib.py -i in_psm.tsv -o out_speclib.tsv --ignore_unsupported
```
