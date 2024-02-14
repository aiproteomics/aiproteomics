"""prosit_fragmentation dataset."""

from . import prosit_fragmentation_dataset_builder
import tensorflow_datasets as tfds


class PrositFragmentationTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for prosit_fragmentation dataset."""

    # TODO(prosit_fragmentation):
    DATASET_CLASS = prosit_fragmentation_dataset_builder.Builder
    SPLITS = {
        "train": 4,  # Number of fake train example
        "validate": 4,  # Number of fake test example
    }
    DL_EXTRACT_RESULT = {"train": "sample.hdf5", "validate": "val_sample.hdf5"}


if __name__ == "__main__":
    tfds.testing.test_main()
