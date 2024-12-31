from dataclasses import dataclass, asdict
import re

import numpy as np

@dataclass(frozen=True)
class SequenceMapping:
    """
        description: A string explaining where this mapping came from or what kind of
        problem it is seeking to solve. This is intended to help in future if someone
        is looking at an older model and wondering why the mapping is a certain way.

        aa_int_map: The direct mapping from a single letter (referring to an amino acid
        or modified amino acid) to an integer in the model input layer.

        aa_mod_map: Supported amino acid modifications and corresponding mapping to the
        model alphabet


    """

    description: str
    aa_int_map: dict
    aa_mod_map: dict


PHOSPHO_MAPPING = SequenceMapping(

    description = "Mapping for the phospho model",

    aa_int_map = {
            ' ': 0,
            'A': 1,
            'C': 2,
            'D': 3,
            'E': 4,
            'F': 5,
            'G': 6,
            'H': 7,
            'I': 8,
            'K': 9,
            'L': 10,
            'M': 11,
            'N': 12,
            'P': 13,
            'Q': 14,
            'R': 15,
            'S': 16,
            'T': 17,
            'V': 18,
            'W': 19,
            'Y': 20,
            '1': 21,
            '2': 22,
            '3': 23,
            '4': 24,
            '*': 25
        },

    aa_mod_map = {
            'M(UniMod:35)': '1',
            'S(UniMod:21)': '2',
            'T(UniMod:21)': '3',
            'Y(UniMod:21)': '4',
            '(UniMod:1)':   '*',
            'C(UniMod:4)':  'C'
    }
)


PROSIT_MAPPING = SequenceMapping(

    description = "Mapping (adapted for Unimod strings) used in the 2019 prosit model",

    # Mapping from amino acid to integers in the model input layer
    # Note that this includes amino acids with modifications.
    aa_int_map = {
            ' ': 0,
            'A': 1,
            'C': 2,
            'D': 3,
            'E': 4,
            'F': 5,
            'G': 6,
            'H': 7,
            'I': 8,
            'K': 9,
            'L': 10,
            'M': 11,
            'N': 12,
            'P': 13,
            'Q': 14,
            'R': 15,
            'S': 16,
            'T': 17,
            'V': 18,
            'W': 19,
            'Y': 20,
            '1': 21
        },

    # Supported amino acid modifications and corresponding
    # mapping to the model alphabet
    aa_mod_map = {
            'M(UniMod:35)': '1',
            'C(UniMod:4)':  'C'
    }
)


@dataclass
class SequenceMapper:

    min_seq_len: int
    max_seq_len: int
    mapping: SequenceMapping


    # Mapping from amino acid to integers in the model input layer
    def map_to_int(self, seq):
        single_char_seq = self.unimod_to_single_char_sequence(seq)

        l = len(single_char_seq)
        if l < self.min_seq_len or l > self.max_seq_len:
            raise ValueError(f"Sequence {seq} does not fit in range defined by min_seq_len={self.min_seq_len} and max_seq_len={self.max_seq_len}")

        single_char_seq = single_char_seq.ljust(self.max_seq_len, ' ')

        return np.array([self.mapping.aa_int_map[letter] for letter in single_char_seq], dtype=np.int32)
#        return [self.mapping.aa_int_map[letter] for letter in single_char_seq]


    def generate_unmodified_peptide_sequence(self, modified_seq):
        """ For a given peptide sequence, `modified_seq`, containing modification
            notation in the form '(UniMod:X)', this function will return the
            sequence absent any modification text (by simply removing anything
            in brackets `()`).
        """
        return re.sub(r"[\(].*?[\)]", "", modified_seq)


    def unimod_to_single_char_sequence(self, seq, ignore_unsupported=False):
        """
            Takes a peptide sequence `seq` as input, encoded as a string with UniMod modifiers.
            For example, "_(UniMod:1)AAAAKPNNLS(UniMod:21)LVVHGPGDLR_".
            Converts this to a sequence of 1 character per amino acid, according to the
            mapping given in `definitions.aa_mod_map`.
        """

        seq = seq.strip().strip('_')

        for k, v in self.mapping.aa_mod_map.items():
            if '(' not in seq:
                break
            seq = seq.replace(k, v)

        # If there are still modifications present, then they
        # are not supported.
        if '(' in seq:
            if ignore_unsupported:
                return None
            raise ValueError(f'Sequence {seq} contains unsupported amino acid modifications. List of supported mods: {self.mapping.aa_mod_map.keys()}')

        return seq


    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return SequenceMapper(
                min_seq_len=int(d["min_seq_len"]),
                max_seq_len=int(d["max_seq_len"]),
                mapping = SequenceMapping(
                        description=d["mapping"]["description"],
                        aa_int_map=d["mapping"]["aa_int_map"],
                        aa_mod_map=d["mapping"]["aa_mod_map"]
                    )
                )
