# Code from (or adapted from) https://github.com/kusterlab/prosit/ Apache License 2.0
# See README.md

import numpy
import functools
from aiproteomics.e2e.constants import MAX_SEQUENCE, ION_TYPES, MAX_NLOSSES, MAX_FRAG_CHARGE


def reshape_dims(array):
    """
    "Deflatten" the intensities array (from the size 174 prosit output layer)
    to the form N_sequences x MAX_SEQUENCE - 1 x len(ION_TYPES) x MAX_NLOSSES x MAX_FRAG_CHARGE
    For default output layer size 174, this corresponds to the shape:
    N_sequences x 29 x 2 x 1 x 3
    """
    _, dims = array.shape
    assert dims == 174

    return array.reshape(
        [array.shape[0], MAX_SEQUENCE - 1, len(ION_TYPES), MAX_NLOSSES, MAX_FRAG_CHARGE]
    )


def reshape_flat(array):
    """
    Reshape the given array to shape (N_sequences x flat_array_length)
    where N_sequences is the length of the first axis,
    and flat_array_length is the combined flattened length of all remaining axes.
    For standard prosit defaults, this results in a shape of N_sequences x 174.
    """

    s = array.shape
    flat_dim = [s[0], functools.reduce(lambda x, y: x * y, s[1:], 1)]
    return array.reshape(flat_dim)


def normalize_base_peak(array):
    """
    Normalize by the highest intensity peak within each sequence
    """
    maxima = array.max(axis=1)
    array = array / maxima[:, numpy.newaxis]
    return array


def mask_outofrange(array, lengths, mask=-1.0):
    """
    Sets all values to mask for entries beyond the peptide sequence length.
    The max length is 29 (for a peptide length of 30 amino acids) but many
    sequences will be shorter, and so the unused remainder of the array is masked.
    """
    for i in range(array.shape[0]):
        array[i, lengths[i] - 1 :, :, :, :] = mask
    return array


def cap(array, nlosses=1, z=3):
    """
    Caps the neutral losses and frag charges to nlosses and z, respectively.

    Caps the shape to (:, :, :, nlosses, z)
    For example, if nlosses=1 and z=3, then
    an array of shape (3, 29, 2, 3, 6) -> (3, 29, 2, 1, 3)
    """

    return array[:, :, :, :nlosses, :z]


def mask_outofcharge(array, charges, mask=-1.0):
    """
    For each quence, mask any entries corresponding to charges
    greater than the threshold provided in 'charges'.
    """
    for i in range(array.shape[0]):
        if charges[i] < 3:
            array[i, :, :, :, charges[i] :] = mask
    return array


def sanitize_prediction_output(data):
    """
    Default prosit output layer is 174, coming from a
    flattening of array with dimensions: 29 x 2 x 1 x 3

    This comes from:
    * Max number of ions = MAX_SEQUENCE_LENGTH - 1 = 30 - 1 = 29
    * Number of ion types = 2 ('y' and 'b')
    * nlosses (nothing, H2O or NH3) capped at 1
    * charges (max of 6) capped at 3

    """

    assert "sequence_integer" in data
    assert "intensities_pred" in data
    assert "precursor_charge_onehot" in data

    sequence_lengths = numpy.count_nonzero(data["sequence_integer"], axis=1)
    intensities = data["intensities_pred"]
    charges = list(data["precursor_charge_onehot"].argmax(axis=1) + 1)

    # Set all negative intensities to zero
    intensities[intensities < 0] = 0

    intensities = normalize_base_peak(intensities)
    intensities = reshape_dims(intensities)
    intensities = mask_outofrange(intensities, sequence_lengths)
    intensities = mask_outofcharge(intensities, charges)

    # Shape = N_sequences x 174
    intensities = reshape_flat(intensities)

    data["intensities_pred"] = intensities

    return data
