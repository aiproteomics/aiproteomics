# Code from (or adapted from) https://github.com/kusterlab/prosit/ Apache License 2.0
# See README.md

import numpy
import functools
from .constants import *

def reshape_dims(array):
    n, dims = array.shape
    assert dims == 174
    nlosses = 1

    return array.reshape(
        [array.shape[0], MAX_SEQUENCE - 1, len(ION_TYPES), nlosses, MAX_FRAG_CHARGE]
    )


def reshape_flat(array):
    s = array.shape
    flat_dim = [s[0], functools.reduce(lambda x, y: x * y, s[1:], 1)]
    return array.reshape(flat_dim)


def normalize_base_peak(array):
    # flat
    maxima = array.max(axis=1)
    array = array / maxima[:, numpy.newaxis]
    return array


def mask_outofrange(array, lengths, mask=-1.0):
    # dim
    for i in range(array.shape[0]):
        array[i, lengths[i] - 1 :, :, :, :] = mask
    return array


def cap(array, nlosses=1, z=3):
    return array[:, :, :, :nlosses, :z]


def mask_outofcharge(array, charges, mask=-1.0):
    # dim
    for i in range(array.shape[0]):
        if charges[i] < 3:
            array[i, :, :, :, charges[i] :] = mask
    return array


def prediction(data, batch_size=600):
    assert "sequence_integer" in data
    assert "intensities_pred" in data
    assert "precursor_charge_onehot" in data

    sequence_lengths = numpy.count_nonzero(data["sequence_integer"], axis=1)
    intensities = data["intensities_pred"]
    charges = list(data["precursor_charge_onehot"].argmax(axis=1) + 1)

    intensities[intensities < 0] = 0
    intensities = normalize_base_peak(intensities)
    intensities = reshape_dims(intensities)
    intensities = mask_outofrange(intensities, sequence_lengths)
    intensities = mask_outofcharge(intensities, charges)
    intensities = reshape_flat(intensities)

    data["intensities_pred"] = intensities

    if "intensities_raw" in data:
        data["spectral_angle"] = get_spectral_angle(
            data["intensities_raw"], data["intensities_pred"], batch_size=batch_size
        )
    return data
