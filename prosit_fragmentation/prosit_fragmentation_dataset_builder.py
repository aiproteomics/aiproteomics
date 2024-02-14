import logging

import h5py
import tensorflow_datasets as tfds
import tensorflow as tf

# TODO(AIProteomicsProsit1Frag): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Prosit 1.0 training and holdout dataset, fetched from 
https://figshare.com/articles/dataset/ProteomeTools_-_Prosit_fragmentation_-_Data/6860261
"""

# TODO(AIProteomicsProsit1Frag): BibTeX citation
_CITATION = """
"""

BATCH_SIZE = int(1e5)

logger = logging.getLogger(__name__)


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Prosit fragmentation dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    MAX_SEQUENCE_LENGTH = 30

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "sequence": tfds.features.Tensor(
                        shape=(self.MAX_SEQUENCE_LENGTH,), dtype=tf.int64
                    ),
                    "precursor_charge_onehot": tfds.features.Tensor(
                        shape=(6,), dtype=tf.int64
                    ),
                    "collision_energy_aligned_normed": tfds.features.Tensor(
                        shape=(1,), dtype=tf.float64
                    ),
                    "intensities_raw": tfds.features.Tensor(
                        shape=(174,), dtype=tf.float64
                    ),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=("image", "label"),  # Set to `None` to disable
            homepage="https://dataset-homepage/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # Downloads the data and defines the splits
        path = dl_manager.download_and_extract(
            {
                "train": "https://figshare.com/ndownloader/files/12785291",
                "validate": "https://figshare.com/ndownloader/files/12506534",
            }
        )

        # Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples(path["train"]),
            "validate": self._generate_examples(path["validate"]),
        }

    def _generate_examples(self, path):
        """Yields examples."""

        logger.info("Loading data into memory")
        with h5py.File(path, "r") as f:
            sequence_integer = f["sequence_integer"][()]
            precursor_charge_onehot = f["precursor_charge_onehot"][()]
            collision_energy_aligned_normed = f["collision_energy_aligned_normed"][()]
            intensities_raw = self._batch_load_hdf5(f["intensities_raw"])

            logger.info("Loaded examples in memory")
            seqid = 0
            for sequence, precursor_charge, collision_energy, intensities_raw in zip(
                sequence_integer,
                precursor_charge_onehot,
                collision_energy_aligned_normed,
                intensities_raw,
            ):
                seqid += 1
                yield seqid, {
                    "sequence": sequence,
                    "precursor_charge_onehot": precursor_charge,
                    "collision_energy_aligned_normed": collision_energy,
                    "intensities_raw": intensities_raw,
                }

    @staticmethod
    def _batch_load_hdf5(dataset: h5py.Dataset, batch_size=BATCH_SIZE):
        idx = 0
        while idx < dataset.shape[0]:
            end = min(idx + batch_size, dataset.shape[0])
            batch = dataset[idx:end]

            yield from batch
