import os
import json
import datetime

from enum import Enum
from dataclasses import dataclass, asdict
import itertools as it
from pathlib import Path

import numpy as np
import pandas as pd

import tensorflow as tf

import aiproteomics
from aiproteomics.core.fragment import Fragment
from aiproteomics.core.sequence import SequenceMapper
from aiproteomics.core.spectrum import output_layer_to_spectrum

class ModelType(Enum):
    MSMS = "msms"
    RT = "rt"
    CCS = "ccs"


class ModelParams:

    def get_model_type(self):
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError

    @staticmethod
    def from_dict(ptype: ModelType, d: dict):
        """
            Given a model type `ptype` and a `dict` of the
            serialized parameters for that model, makes an
            instance of the corresponding ModelParams subclass
            and returns it.
        """

        if ptype == ModelType.MSMS:
            return ModelParamsMSMS(
                    seq_len=int(d["seq_len"]),
                    ions=d["ions"],
                    max_charge=int(d["max_charge"]),
                    neutral_losses=d["neutral_losses"]
            )

        if ptype == ModelType.RT:
            return ModelParamsRT(
                    seq_len=int(d["seq_len"]),
                    iRT_rescaling_mean=float(d["iRT_rescaling_mean"]),
                    iRT_rescaling_var=float(d["iRT_rescaling_var"])
            )

        if ptype == ModelType.CCS:
            return ModelParamsCCS(
                    seq_len=int(d["seq_len"])
            )

        raise ValueError(f"No matching ModelParams builder for type {ptype}")


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


    def scale_output_iRT(self, model_output):
        """
            For a given retention time value predicted by this model (`model_output`),
            this function rescales it using the iRT variance and mean parameters
            to get the true iRT prediction
        """
        return model_output * np.sqrt(self.iRT_rescaling_var) + self.iRT_rescaling_mean


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

    """
        Holds the three important components of an AI model used for generating
        MSMS spectra, retention times or ion mobility: the input sequence mapping (`seq_map`),
        the info needed for output layer processing (`model_params`), and the neural network
        model itself (`nn_model`).

        Can be dumped to a directory using `to_dir()` and later reloaded using the `.from_dir()`
        static method.
    """

    seq_map: SequenceMapper
    model_params: ModelParams
    nn_model: tf.keras.Model
    nn_model_creation_metadata: dict

    def process_inputs(self, sequence: np.array, charge: np.array):
        pass


    def process_outputs(self):
        # return df of all generated spectra, or iRT values, or ccs values
        pass


    def to_dir(self, dirpath, overwrite=False, config_fname="config.json", nn_model_fname="model.keras"):
        """
        Dump this `AIProteomicsModel` to a directory, containing a config json file and the AI model
        itself. The model can then be reloaded again using the `AIProteomicsModel.from_dir()` method.
        """

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
                "nn_model": nn_model_fname,
                "nn_model_creation_metadata": self.nn_model_creation_metadata
                }

        # Create the output dir (if not existing)
        if os.path.exists(dirpath):
            if not overwrite:
                raise ValueError(f"Directory {dirpath} already exists. If you want to overwrite it, use overwrite=True")
        else:
            os.makedirs(dirpath)

        # Write params of this model to the config file
        with open(confpath, "w", encoding="utf-8") as conffile:
            json.dump(params_dict, conffile, indent=4)

        # Write NN model to the model directory too
        self.nn_model.save(nnpath)


    @staticmethod
    def from_dir(dirpath, config_fname="config.json"):
        """
        Reload an `AIProteomicsModel` using the information in the given directory, `dirpath`.

        Returns a new `AIProteomicsModel`.
        """

        dirpath = Path(dirpath)
        config_fname = Path(config_fname)
        confpath = dirpath / config_fname

        with open(confpath, "r", encoding="utf-8") as conffile:
            params_dict = json.load(conffile)


        # Check that expected essential keys are present
        assert "model_type" in params_dict
        assert "model_params" in params_dict
        assert "seq_map" in params_dict
        assert "nn_model" in params_dict

        # Get (enum form) of the model type (e.g. MSMS)
        model_type = ModelType(params_dict["model_type"])

        # Build ModelParams and SequenceMapper objects using the information in the config file
        model_params = ModelParams.from_dict(model_type, params_dict["model_params"])
        seq_map = SequenceMapper.from_dict(params_dict["seq_map"])

        # Load the AI model from file
        nnpath = dirpath / params_dict["nn_model"]
        nn_model = tf.keras.models.load_model(nnpath)

        return AIProteomicsModel(seq_map=seq_map,
                                 model_params=model_params,
                                 nn_model=nn_model,
                                 nn_model_creation_metadata=params_dict["nn_model_creation_metadata"]
                                )


def build_spectral_library(inputs: pd.DataFrame, msms: AIProteomicsModel = None, rt: AIProteomicsModel = None, ccs: AIProteomicsModel = None, pY_threshold = 0.8):

    if msms is None:
        raise ValueError("At least an msms model must be provided or no spectral library can be generated")

    # Check columns of inputs dataframe
    required_columns = [
        "peptide",
        "charge"
    ]
    for col in required_columns:
        if col not in inputs:
            raise ValueError(f"Inputs dataframe must have the column {col}")

    # Map inputs to nn model inputs
    input_seq = np.stack(inputs["peptide"].map(msms.seq_map.map_to_int))
    input_charge = inputs["charge"].values

    # Run model inference
    if msms:
        msms_intensities, msms_pY = msms.nn_model.predict([input_seq, input_charge])
    if rt:
        rt_out = rt.nn_model.predict([input_seq])
    if ccs:
        ccs_out = ccs.nn_model.predict([input_seq])


    # Generate a spectrum (as a dataframe) for each sequence based on the model predictions
    dfs = []
    for index, row in inputs.iterrows():

        if msms:
            intensities = msms_intensities[index]
            pY = msms_pY[index]
        else:
            intensities = None
            pY = None

        if rt:
            iRT = rt_out[index]
        else:
            iRT = None

        if ccs:
            ccs = ccs_out[index]
        else:
            ccs = None

        unmodified_peptide_sequence = msms.seq_map.generate_unmodified_peptide_sequence(row["peptide"])

        df = output_layer_to_spectrum(
                intensities,
                msms.model_params,
                row["peptide"],
                row["charge"],
                pY=pY,
                iRT=iRT,
                ccs=ccs,
                thresh=pY_threshold,
                unmodified_seq=unmodified_peptide_sequence)
        dfs.append(df.to_dataframe())

    # Concatenate all spectra into one dataframe and return it
    df = pd.concat(dfs).reset_index()
    return df
