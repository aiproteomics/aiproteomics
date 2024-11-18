from enum import Enum
from dataclasses import dataclass, asdict
import itertools as it
from typing import Optional

import numpy as np

from aiproteomics.core.fragment import Fragment

class ModelType(Enum):
    MSMS = "msms"
    RT = "rt"
    CCS = "ccs"


class ModelParams:

    def get_model_type(self):
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError

    def from_dict(self, d):
        raise NotImplementedError



@dataclass
class ModelParamsMSMS(ModelParams):

    """
        Describes a particular MSMS fragmentation AI model with fixed input sequence
        length `seq_len`, handling ions of types `ions`, precursor charges of up to
        `max_charge`, and neutral losses in the list `neutral_losses`.
    """

    seq_len: int
    ions: list
    max_charge: int
    neutral_losses: list

    def __post_init__(self):
        """
            Fragment list is generated and the result stored with the object
            since it won't change and is used a lot.
        """
        self.fragments = self.generate_fragment_list()


    def generate_fragment_list(self):
        """
            Generate permutations of fragments possible for the output layer
            of the model described by this `model_params` object.
        """
        frag_iter = it.product(
                            self.neutral_losses,
                            range(1, self.seq_len),
                            self.ions,
                            range(1, self.max_charge + 1))

        frag_list = np.array([Fragment(ion, charge, cleavage, loss) for loss, cleavage, ion, charge in frag_iter])

        return frag_list


    def get_model_type(self):
        return ModelType.MSMS


    def generate_mask(self, input_seq_len=None, precursor_charge=None):
        """
            Generate a mask for an input sequence of length `input_seq_len` and
            charge `precursor_charge`. The mask is an array of booleans in which
            True means the corresponding fragment is possible and False otherwise.

            This is useful because a breakage point cannot be greater than the length
            of the sequence - 1, and typically fragments are not charged greater than
            their precursor was.
        """
        mask = np.array([
            (frag.fragment_series_number < input_seq_len and frag.fragment_charge <= precursor_charge)
            for frag in self.fragments])

        return mask


    def to_dict(self):
        """
            Return model parameters as a `dict`
        """
        return asdict(self)



@dataclass
class ModelParamsRT:

    """
        Describes a particular retention time AI model with fixed input sequence
        length `seq_len`, and output rescaling parameters `iRT_rescaling_mean`
        and `iRT_rescaling_var`
    """

    seq_len: int
    iRT_rescaling_mean: Optional[float] = None
    iRT_rescaling_var: Optional[float] = None

    def get_model_type(self):
        return ModelType.RT


    def scale_output_iRT(model_output):
        """
            For a given retention time value predicted by this model (`model_output`),
            this function rescales it using the iRT variance and mean parameters
            to get the true iRT prediction
        """
        return model_output * np.sqrt(iRT_rescaling_var) + iRT_rescaling_mean


    def to_dict(self):
        """
            Return model parameters as a `dict`
        """
        return asdict(self)


@dataclass
class ModelParamsCCS:

    """
        Describes a particular ion mobility AI model with fixed input sequence
        length `seq_len`
    """

    seq_len: int


    def get_model_type(self):
        return ModelType.CCS


    def to_dict(self):
        """
            Return model parameters as a `dict`
        """
        return asdict(self)
