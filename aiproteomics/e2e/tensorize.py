import collections
import numpy as np

# from .constants import CHARGES, MAX_SEQUENCE, ALPHABET, MAX_ION, NLOSSES, CHARGES, ION_TYPES, ION_OFFSET
from . import constants
from . import utils
from . import match
from . import annotate


def stack(queue):
    listed = collections.defaultdict(list)
    for t in queue.values():
        if t is not None:
            for k, d in t.items():
                listed[k].append(d)
    stacked = {}
    for k, d in listed.items():
        if isinstance(d[0], list):
            stacked[k] = [item for sublist in d for item in sublist]
        else:
            stacked[k] = np.vstack(d)
    return stacked


def get_numbers(vals, dtype=float):
    """
    Takes input list and converts values to specified numpy dtype.
    Outputs numpy array in "column format", i.e.
        If input looks like:
            [35, 30, 30],
        Output looks like:
            array([[35.],
            [30.],
            [30.]])
    """
    a = np.array(vals).astype(dtype)
    return a.reshape([len(vals), 1])


def get_precursor_charge_onehot(charges):
    """
    Input:
        charges: int
    Output:
        onehot encoded array of length max(CHARGES)
    Example:
        If charges=3, and the max charge number is 6, then
        the output will be [0, 0, 1, 0, 0, 0]
    """
    array = np.zeros([len(charges), max(constants.CHARGES)], dtype=int)
    for i, precursor_charge in enumerate(charges):
        array[i, precursor_charge - 1] = 1
    return array


def get_sequence_integer(sequences):
    """
    Takes modified sequence (string) as input. For example, "MMPAAALIM(ox)R"
    Maps it to an array of integers, according to the prosit alphabet.
    """
    array = np.zeros([len(sequences), constants.MAX_SEQUENCE], dtype=int)
    for i, sequence in enumerate(sequences):
        for j, s in enumerate(utils.peptide_parser(sequence)):
            array[i, j] = constants.ALPHABET[s]
    return array


def parse_ion(string):
    ion_type = constants.ION_TYPES.index(string[0])
    if ("-") in string:
        ion_n, suffix = string[1:].split("-")
    else:
        ion_n = string[1:]
        suffix = ""
    return ion_type, int(ion_n) - 1, constants.NLOSSES.index(suffix)


def get_mz_applied(df, ion_types="yb"):
    ito = {it: constants.ION_OFFSET[it] for it in ion_types}

    def calc_row(row):
        array = np.zeros(
            [
                constants.MAX_ION,
                len(constants.ION_TYPES),
                len(constants.NLOSSES),
                len(constants.CHARGES),
            ]
        )
        fw, bw = match.get_forward_backward(row.modified_sequence)
        for z in range(row.precursor_charge):
            zpp = z + 1
            annotation = annotate.get_annotation(fw, bw, zpp, ito)
            for ion, mz in annotation.items():
                it, _in, nloss = parse_ion(ion)
                array[_in, it, nloss, z] = mz
        return [array]

    mzs_series = df.apply(calc_row, 1)
    out = np.squeeze(np.stack(mzs_series))
    if len(out.shape) == 4:
        out = out.reshape([1] + list(out.shape))
    return out
