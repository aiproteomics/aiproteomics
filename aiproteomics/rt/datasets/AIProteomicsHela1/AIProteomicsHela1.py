"""AIProteomicsHela1 dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import csv
import numpy as np
import re

# TODO(AIProteomicsHela1): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
DeepDIA training, testing and validation sets. Peptide sequences are preprocessed to Prosit alphabet and iRT normalized.
"""

# TODO(AIProteomicsHela1): BibTeX citation
_CITATION = """
"""


class Aiproteomicshela1(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for AIProteomicsHela1 dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
          '1.0.0': 'Initial release.',
    }


    Prosit_ALPHABET = {
        "A": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
        "H": 7,
        "I": 8,
        "K": 9,
        "L": 10,
        "M": 11,
        "N": 12,
        "P": 13,
        "Q": 14,
        "R": 15,
        "S": 16,
        "T": 17,
        "V": 18,
        "W": 19,
        "Y": 20,
        "o": 21,
    }

    MAX_SEQUENCE_LENGTH = 50


    def sequence_to_integer(self, sequence):
        array = np.zeros(self.MAX_SEQUENCE_LENGTH, dtype=tf.int32)
        for j, s in enumerate(re.sub('M\(ox\)', 'o', sequence)):
            array[j] = self.Prosit_ALPHABET[s]
        return array


    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'sequence': tfds.features.Tensor(shape=(self.MAX_SEQUENCE_LENGTH,), dtype=tf.int32),
                'irt': tfds.features.Tensor(shape=(1,), dtype=tf.float32),
                }),
            supervised_keys=('sequence', 'irt'),
            homepage=None,
            citation=_CITATION,
            )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        # Download the data and define the splits
        path = dl_manager.download_and_extract({
            'train'   : 'https://surfdrive.surf.nl/files/index.php/s/f8oS3spyFINzJ8Y/download',
            'test'    : 'https://surfdrive.surf.nl/files/index.php/s/t2a97IWyVlyMA5E/download',
            'validate': 'https://surfdrive.surf.nl/files/index.php/s/zyEqgQxB6HuvJxd/download'
            })

        return {
            'train'   : self._generate_examples(path['train']),
            'test'    : self._generate_examples(path['test']),
            'validate': self._generate_examples(path['validate'])
        }

    def _generate_examples(self, path):
        """Yields examples."""
        with open(path, 'r') as csvfile:
            for row in csv.DictReader(csvfile):
                yield row['sequence'], {
                    'sequence': self.sequence_to_integer(row['sequence']),
                    'irt': np.array([float(row['irt'])/100.0], dtype=tf.float32)
                }
