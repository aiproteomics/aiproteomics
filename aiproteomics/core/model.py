import os
import json
import datetime
from pathlib import Path

from dataclasses import dataclass

import numpy as np

import tensorflow as tf
import tf2onnx

import aiproteomics
from aiproteomics.core.sequence import SequenceMapper, PHOSPHO_MAPPING, PROSIT_MAPPING
from aiproteomics.core.modeltypes import ModelParams, ModelParamsMSMS, ModelParamsRT, generate_msms_transformer


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










#    model_params: ModelParams
#    input_mapper: SequenceMapper
#    output_mapper: FixedOutputMapper
#


# sequence mapping
# returns aa_mod and aa_mass for itself

#    frag_list = model_params.generate_fragment_list()
#
#    # Create random output layer
#    output_layer = np.random.random(392)
#
#    # Convert to spectrum
#    spectrum = output_layer_to_spectrum(output_layer, model_params, "*SSS1TT221", 1, pY=0.97, iRT=1, ccs=0, thresh=0.9)
#
#    # Convert to pandas dataframe
#    spectrum_df = spectrum.to_dataframe()


if __name__ == "__main__":


    # INPUT:
    # Choose what sequence mapping you want to use
    seqmap = SequenceMapper(min_seq_len=7, max_seq_len=50, mapping=PHOSPHO_MAPPING)

    # OUTPUT:
    # Choose your model type (msms) and the parameters for its output
    params = ModelParamsMSMS(seq_len=50, ions=['y','b'], max_charge=2, neutral_losses=['', 'H3PO4'])

    # Make a compatible NN model
    nn_model = generate_msms_transformer(
        seq_map=seqmap,
        params=params,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        dropout_rate=0.1
    )

    # Build the model
    msmsmodel = AIProteomicsModel(seq_map=seqmap, model_params=params, nn_model=nn_model)

    # Save the model
    msmsmodel.to_dir("testmodelfrag/", overwrite=True)

    # -----------

    # Try making a prosit-style retention time model
    seqmap = SequenceMapper(min_seq_len=7, max_seq_len=30, mapping=PROSIT_MAPPING)
    params = ModelParamsRT(seq_len=30, iRT_rescaling_mean=101.11514, iRT_rescaling_var=46.5882)
    rtmodel = AIProteomicsModel(seq_map=seqmap, model_params=params, nn_model=nn_model)
    rtmodel.to_dir("testmodelrt/", overwrite=True)





#    fragmodel = AIProteomicsModel(model_params=params, input_mapper=SequenceMapper, output_mapper=FixedOutputMapper)
#
#    ###
#
#    fragmodel = AIProteomicsModel.from_dir('fragmodel1/')
#
#    input_seq_df = pd.DataFrame.from_csv('peptides_to_predict.tsv', sep='\t')
#    speclib_df = fragmodel.predict_to_speclib(input_seq_df)
#
#
#    speclib_df.to_csv('speclib.parquet', sep='\t')
