from aiproteomics.core.sequence import SequenceMapper, PHOSPHO_MAPPING, PROSIT_MAPPING
from aiproteomics.core.modeltypes import ModelParamsMSMS, ModelParamsRT, ModelParamsCCS, AIProteomicsModel
from aiproteomics.core.models import generate_msms_transformer


if __name__ == "__main__":


    # INPUT:
    # Choose what sequence mapping you want to use
    seqmap = SequenceMapper(min_seq_len=7, max_seq_len=50, mapping=PHOSPHO_MAPPING)

    # OUTPUT:
    # Choose your model type (msms) and the parameters for its output
    params = ModelParamsMSMS(seq_len=50, ions=['y','b'], max_charge=2, neutral_losses=['', 'H3PO4'])

    # Make a compatible NN model
    nn_model, creation_meta = generate_msms_transformer(
        seq_map=seqmap,
        params=params,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        dropout_rate=0.1
    )

    # Build the model
    msmsmodel = AIProteomicsModel(seq_map=seqmap, model_params=params, nn_model=nn_model, nn_model_creation_metadata=creation_meta)

    # Save the model
    msmsmodel.to_dir("testmodelfrag/", overwrite=True)

    # -----------

    # Try making a prosit-style retention time model
    seqmap = SequenceMapper(min_seq_len=7, max_seq_len=30, mapping=PROSIT_MAPPING)
    params = ModelParamsRT(seq_len=30, iRT_rescaling_mean=101.11514, iRT_rescaling_var=46.5882)
    rtmodel = AIProteomicsModel(seq_map=seqmap, model_params=params, nn_model=nn_model, nn_model_creation_metadata=creation_meta)
    rtmodel.to_dir("testmodelrt/", overwrite=True)


    # -----------

    # Try making a phospho-style CCS time model
    seqmap = SequenceMapper(min_seq_len=7, max_seq_len=50, mapping=PHOSPHO_MAPPING)
    params = ModelParamsCCS(seq_len=30)
    ccsmodel = AIProteomicsModel(seq_map=seqmap, model_params=params, nn_model=nn_model, nn_model_creation_metadata=creation_meta)
    ccsmodel.to_dir("testmodelccs/", overwrite=True)




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
