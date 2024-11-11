import re
from definitions import aa_mod_map


def generate_unmodified_peptide_sequence(modified_seq):
    """ For a given peptide sequence, `modified_seq`, containing modification
        notation in the form '(UniMod:X)', this function will return the
        sequence absent any modification text (by simply removing anything
        in brackets `()`).
    """
    return re.sub(r"[\(].*?[\)]", "", modified_seq)


def unimod_to_single_char_sequence(seq, ignore_unsupported=False):
    seq = seq.strip()
    seq = seq.strip('_')

    for k, v in aa_mod_map.items():
        if '(' not in seq:
            break
        seq = seq.replace(k, v)

    # If there are still modifications present, then they
    # are not supported.
    if '(' in seq:
        if ignore_unsupported:
            return None
        raise ValueError(f'Sequence {seq} contains unsupported amino acid modifications. List of supported mods: {aa_mod_map.keys()}')

    return seq
