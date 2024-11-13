import re
from aiproteomics.core.definitions import aa_mod_map, ANNOTATION_pY, ALLOWED_IONS
from aiproteomics.core.fragment import Fragment


def generate_unmodified_peptide_sequence(modified_seq):
    """ For a given peptide sequence, `modified_seq`, containing modification
        notation in the form '(UniMod:X)', this function will return the
        sequence absent any modification text (by simply removing anything
        in brackets `()`).
    """
    return re.sub(r"[\(].*?[\)]", "", modified_seq)


def unimod_to_single_char_sequence(seq, ignore_unsupported=False):
    """
        Takes a peptide sequence `seq` as input, encoded as a string with UniMod modifiers.
        For example, "_(UniMod:1)AAAAKPNNLS(UniMod:21)LVVHGPGDLR_".
        Converts this to a sequence of 1 character per amino acid, according to the
        mapping given in `definitions.aa_mod_map`.
    """

    seq = seq.strip().strip('_')

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


def parse_ion_annotation(ion):
    """
        For a given ion annotation string, `ion` (e.g. "y3(2+)-H2O") this function
        will parse the constituent information into:
        `ion_type` (e.g. 'y')
        `ion_break` (e.g. 3, the point in the sequence where breakage occured)
        `ion_charge` (e.g. 12)
        `neutral_loss` (e.g. "H2O". If no loss, this is an empty string)

        The above extracted info is returned as a `Fragment` object.
    """

    if 'nan' in ion:
        return None

    if ion == ANNOTATION_pY:
        return Fragment(ANNOTATION_pY, 1, 0, '')

    # Get single letter ion identifier e.g. 'y', 'b', 'a'
    ion_type = ion[0]
    if ion_type not in ALLOWED_IONS:
        raise ValueError(f'Ion type {ion_type} not in expected ion types: {ALLOWED_IONS}')

    # Attempt to split into ion and neutral loss
    ion_split = ion[1:].split('-')
    ion_part = ion_split[0]
    neutral_loss = ""

    # If neutral loss
    if len(ion_split) == 2:
        ion_part = ion_split[0]
        neutral_loss = ion_split[1]

    # Determine ion charge
    ion_part_split = ion_part.split('(')
    ion_charge = 1
    if len(ion_part_split) == 2:
        ion_charge = int(ion_part_split[1].split('+')[0])

    # Get ion breakage position
    ion_break_str = ion_part_split[0]
    if ion_break_str[-1] == '*':
        # Check if asterisk after breakage, corresponding to phospho loss
        ion_break_str = ion_break_str[:-1]
        neutral_loss = "H3PO4"

    try:
        ion_break = int(ion_break_str)
    except ValueError as ve:
        raise ValueError(f'Exception when converting ion breakage str {ion_break_str}: {ve}') from ve

    return Fragment(ion_type, ion_charge, ion_break, neutral_loss)


