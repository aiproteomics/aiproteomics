# coding=utf-8
# Copyright 2024 Thang V Pham
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import argparse
import re

import pandas as pd
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import aiproteomics

def main():

    '''
    Example:
    python ./create_spectral_library.py -from_csv
    python create_spectral_library.py -from_csv 240912-diann191-library-from-fasta/report-lib.prosit.csv-noU.csv
    -msms prosit
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-from_csv', '--from_csv', type = str, help = 'Input data.')
    parser.add_argument('-msms', '--msms', default = 'tf', type = str, help = 'MS/MS model.')
    args = parser.parse_args(sys.argv[1:len(sys.argv)])

    print('Input =', args.from_csv)

    '''
    MS/MS model
    '''
    print('Loading MS/MS model')
    if args.msms == 'prosit':
        from aiproteomics.modelgen.prosit1_gen import build_prosit1_model

        model_msms = build_prosit1_model()

        # Load trained weights from a model
        model_msms.load_weights('/d0/tp/data_transformer/prosit/weight_32_0.10211.hdf5')
        
        # Make a plot of the model
        tf.keras.utils.plot_model(model_msms, to_file = args.from_csv + "-prosit-msms.png", show_shapes=True)
    else:
        from aiproteomics.frag.models import build_frag_transformer_model_slice0
        model_frag = build_frag_transformer_model_slice0(
            num_layers = 6,                 # number of layers
            d_model = 512,
            num_heads = 8,                  # Number of attention heads
            d_ff = 2048,                    # Hidden layer size in feed forward network inside transformer
            dropout_rate = 0.1,             #
            vocab_size = 22,                # number of aminoacids
            max_len = 30                    # maximal peptide length
        )

        # Load trained weights from (old) fragmentation transformer model
        model_frag.load_weights('/d0/tp/data_transformer/weight_67_0.22422.hdf5')

        # Make a plot of the model
        tf.keras.utils.plot_model(model_frag, to_file = args.from_csv + "-tf-msms.png", show_shapes=True)

    '''
    rt model
    '''
    print('Loading iRT model')
    from aiproteomics.rt.models import build_rt_transformer_model

    model_irt = build_rt_transformer_model(
        num_layers = 6,                 # number of layers
        d_model = 512,
        num_heads = 8,                  # Number of attention heads
        d_ff = 2048,                    # Hidden layer size in feed forward network inside transformer
        dropout_rate = 0.1,             #
        vocab_size = 22,                # number of aminoacids
        max_len = 30 
    )
    # Make a plot of the model
    tf.keras.utils.plot_model(model_irt, to_file = args.from_csv + '-rt.png', show_shapes=True)

    # Don't have the weights right now but can uncomment when retrained
    #model.load_weights('./trained_prosit_irt/weight_24_0.03205.hdf5')

    '''
    ccs model
    '''
    # We have not trained one yet
    model_ccs = None

    '''
    removing line with U
    cat report-lib.prosit.csv | awk -F ',' '{if(/U/) {} else {print}}' > report-lib.prosit.filtered.csv
    head report-lib.prosit.filtered.csv
    '''

    from aiproteomics.e2e.speclibgen import csv_to_speclib

    csv_to_speclib(
        args.from_csv,
        args.from_csv + '-out.tsv',
        #model_frag=model_msms,
        model_frag=model_frag,
        model_irt=model_irt,
        model_ccs=model_ccs,
        batch_size_frag=1024,
        iRT_rescaling_mean = 56.35363441,
        iRT_rescaling_var = 1883.0160689,
        chunksize=10000,
        fmt='tsv'
    )

if __name__ == '__main__':
    main()
    