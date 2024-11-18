import os
import json
import datetime

from enum import Enum
from dataclasses import dataclass, asdict
import itertools as it
from typing import Optional
from pathlib import Path

import numpy as np

import tensorflow as tf
import tf2onnx

import aiproteomics
from aiproteomics.core.fragment import Fragment
from aiproteomics.core.sequence import SequenceMapper

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
    iRT_rescaling_mean: float
    iRT_rescaling_var: float

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




@dataclass
class AIProteomicsModel:

    seq_map: SequenceMapper
    model_params: ModelParams
    nn_model: tf.keras.Model


    def process_inputs(self, sequence: np.array, charge: np.array):
        pass


    def process_outputs():
        # return df of all generated spectra
        pass

    def to_dir(self, dirpath, overwrite=False, config_fname="config.json", nn_model_fname="nn.onnx"):

        dirpath = Path(dirpath)
        config_fname = Path(config_fname)
        confpath = dirpath / config_fname
        nnpath = dirpath / nn_model_fname

        # Create timestamp for time this was written to disk
        tz = datetime.timezone.utc
        fmt = "%Y-%m-%dT%H%M%S"
        timestamp = datetime.datetime.now(tz=tz).strftime(fmt)

        params_dict = {
                "aiproteomics_version": aiproteomics.__VERSION__,
                "creation_time": timestamp,
                "model_type": self.model_params.get_model_type().value,
                "model_params": self.model_params.to_dict(),
                "seq_map": self.seq_map.to_dict(),
                "nn_model": nn_model_fname
                }

        # Create the output dir (if not existing)
        if os.path.exists(dirpath):
            if not overwrite:
                raise ValueError(f"Directory {dirpath} already exists. If you want to overwrite it, use overwrite=True")
        else:
            os.makedirs(dirpath)

        # Write params of this model to the config file
        with open(confpath, "w") as conffile:
            json.dump(params_dict, conffile, indent=4)

        # Write NN model to the model directory too
        tf2onnx.convert.from_keras(self.nn_model, output_path=nnpath)


    def from_dir(self):
        # Previously generate model as ONNX? Then can easily load from or save to? Then dump in dir along with rest?
        pass







import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Return "creation meta data too?"
def generate_msms_transformer(
    num_layers=6,
    num_heads=8,
    d_ff=2048,
    dropout_rate=0.1,
    seq_map=None,
    params=None):


    # Input layers
    peptide = keras.Input(
        name="peptide", dtype="float32", sparse=False, batch_input_shape=(None, 30)
    )
    charge = keras.Input(
        name="charge", dtype="float32", sparse=False, batch_input_shape=(None, 6)
    )

    add_meta = layers.Concatenate()([peptide, charge])

    activation = layers.LeakyReLU(name="activation", alpha=0.30000001192092896, trainable=True)(
        add_meta
    )

    output_layer = layers.Flatten(name="out", data_format="channels_last", trainable=True)(
        activation
    )

    # Compile model
    # if this doesn't work, explicitly import masked_spectral_distance from losses
    model = keras.Model(
        inputs=[peptide, charge], outputs=output_layer
    )
    model.compile(loss="meansquarederror", optimizer="adam", metrics=["accuracy"])

    return model



