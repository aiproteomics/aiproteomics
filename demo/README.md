# Demo notebooks

Some demo notebooks are provided here to provide examples of use of the `aiproteomics` library. Hopefully these
may also act as templates for common uses, allowing users to quickly start using the library by adapting to their
specific use cases.

## Setup
Before running these notebooks, some setup is needed to download certain data files that are not in this repository
due to size and licensing reasons.

1. Download the Prosit training and validation sets
From [here](https://figshare.com/articles/dataset/ProteomeTools_-_Prosit_fragmentation_-_Data/6860261) you can download the
`traintest_hcd.hdf5` and `holdout_hcd.hdf5` files. Please place them in the `demo/` directory so that the demo notebooks
may find them.

2. Download the Prosit model weights
Similarly, download the weights files for the [fragmentation](https://figshare.com/articles/dataset/Prosit_-_Model_-_Fragmentation/6965753)
and [retention time](https://figshare.com/articles/dataset/Prosit_-_Model_-_iRT/6965801) models, and place them in the `demo/` directory.
The weights files are the `.hdf5` files whose names begin with `weight_`.

## Spectral Library Generation
To see an example of loading a model and producing a spectral library with it, see the <uniprot_e2e.ipynb> notebook. This can be run using
```bash
poetry run jupyter lab uniprot_e2e.ipynb
```

## Spectral Angle Comparison
To compare the performance of different models with respect to the holdout set, see the <demo_spectral_angle_comparison.ipynb> notebook.
This can be run with
```bash
poetry run jupyter lab demo_spectral_angle_comparison.ipynb
```
