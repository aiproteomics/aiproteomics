[![RSD](https://img.shields.io/badge/rsd-aiproteomics-00a3e3.svg)](https://www.research-software.nl/software/aiproteomics)
[![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu) [![build](https://github.com/aiproteomics/aiproteomics/actions/workflows/build.yml/badge.svg)](https://github.com/aiproteomics/aiproteomics/actions/workflows/build.yml) [![cffconvert](https://github.com/aiproteomics/aiproteomics/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/aiproteomics/aiproteomics/actions/workflows/cffconvert.yml) [![sonarcloud](https://github.com/aiproteomics/aiproteomics/actions/workflows/sonarcloud.yml/badge.svg)](https://github.com/aiproteomics/aiproteomics/actions/workflows/sonarcloud.yml) [![markdown-link-check](https://github.com/aiproteomics/aiproteomics/actions/workflows/markdown-link-check.yml/badge.svg)](https://github.com/aiproteomics/aiproteomics/actions/workflows/markdown-link-check.yml)  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7890716.svg)](https://doi.org/10.5281/zenodo.7890716)

## `aiproteomics` python package
This package contains various tools, datasets and ML model implementations from the field of (phospho-)proteomics. It is intended to facilitate the testing and comparison of different neural network architectures and existing models, using the same datasets. Both retention time and fragmentation (MSMS) models are included.

Implementations of existing models from the literature are intended to be modifiable/extendable. For example, so that tests may be carried out with different peptide input lengths etc.

## Installation instructions

The current package can be installed using poetry after cloning the repository.  
Installation instructions for poetry itself can be found [here](https://python-poetry.org/docs/).  
Once poetry is installed, run:

```
git clone git@github.com:aiproteomics/aiproteomics.git
cd aiproteomics/
poetry install
```

## Try demo notebook
After installation, you can try out the demo notebook by running the following:
```
poetry run jupyter lab demo/uniprot_e2e.ipynb
```
This will open the notebook using jupyter lab.

## Redesign in progress
This package is in the process of being redesigned to make it more general and portable. The redesign is focussing on the creation of:
1. Generators of models (in the open and portable ONNX format)
2. Converters from .msp format to input for each model type
3. Converters from each model type to .msp

Below is a diagram showing how the proposed tools will be combined to produce a pipeline for training proteomics models and using them to generate synthetic spectral libraries:

![Proposed aiproteomics pipeline](proposed_aiproteomics_pipeline.png)

## Contributing

If you want to contribute to the development of aiproteomics,
have a look at the [contribution guidelines](CONTRIBUTING.md).
