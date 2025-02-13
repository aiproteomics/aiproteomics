import pandas as pd

from aiproteomics.core.sequence import SequenceMapper, PHOSPHO_MAPPING, PROSIT_MAPPING
from aiproteomics.core.modeltypes import ModelParamsMSMS, ModelParamsRT, ModelParamsCCS, AIProteomicsModel
from aiproteomics.core.utils import build_spectral_library
from aiproteomics.models.dummy_models import generate_dummy_msms_model, generate_dummy_iRT_model, generate_dummy_ccs_model


def test_phospho_msms_save_and_reload(tmp_path):
    """
    Builds an `AIProteomicsModel` with phospho mapping and dummy MSMS model.
    Tests building the model, saving it to directory, then reloading from directory.
    """

    seqmap = SequenceMapper(min_seq_len=7, max_seq_len=50, mapping=PHOSPHO_MAPPING)
    params = ModelParamsMSMS(seq_len=50, ions=['y','b'], max_charge=2, neutral_losses=['', 'H3PO4'])

    nn_model, creation_meta = generate_dummy_msms_model(
        params=params,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        dropout_rate=0.1
    )

    msmsmodel = AIProteomicsModel(seq_map=seqmap, model_params=params, nn_model=nn_model, nn_model_creation_metadata=creation_meta)

    # Save the model
    msmsmodel.to_dir(tmp_path / "testmodelfrag/", overwrite=True)

    # Load the model back in as a new AIProteomicsModel instance
    reloaded_msms = AIProteomicsModel.from_dir(tmp_path / "testmodelfrag/")


def test_phospho_irt_save_and_reload(tmp_path):
    """
    Builds an `AIProteomicsModel` with phospho mapping and dummy iRT model.
    Tests building the model, saving it to directory, then reloading from directory.
    """

    seqmap = SequenceMapper(min_seq_len=7, max_seq_len=50, mapping=PHOSPHO_MAPPING)
    params = ModelParamsRT(seq_len=50, iRT_rescaling_mean=101.11514, iRT_rescaling_var=46.5882)
    nn_model, creation_meta = generate_dummy_iRT_model(
        params=params,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        dropout_rate=0.1
    )
    rtmodel = AIProteomicsModel(seq_map=seqmap, model_params=params, nn_model=nn_model, nn_model_creation_metadata=creation_meta)
    rtmodel.to_dir(tmp_path / "testmodelrt/", overwrite=True)
    reloaded_rt = AIProteomicsModel.from_dir(tmp_path / "testmodelrt/")


def test_phospho_ccs_save_and_reload(tmp_path):
    """
    Builds an `AIProteomicsModel` with phospho mapping and dummy CCS model.
    Tests building the model, saving it to directory, then reloading from directory.
    """

    seqmap = SequenceMapper(min_seq_len=7, max_seq_len=50, mapping=PHOSPHO_MAPPING)
    params = ModelParamsCCS(seq_len=50)
    nn_model, creation_meta = generate_dummy_ccs_model(
        params=params,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        dropout_rate=0.1
    )
    ccsmodel = AIProteomicsModel(seq_map=seqmap, model_params=params, nn_model=nn_model, nn_model_creation_metadata=creation_meta)
    ccsmodel.to_dir(tmp_path / "testmodelccs/", overwrite=True)
    reloaded_ccs = AIProteomicsModel.from_dir(tmp_path / "testmodelccs/")


def test_build_spec_lib(tmp_path, request):
    """
    Build an MSMS, iRT and CCS model, then use them to generate a spectral library for
    a list of peptides read in from a `tsv` file.
    """

    # Use the same sequence mapper for all three models
    seqmap = SequenceMapper(min_seq_len=7, max_seq_len=50, mapping=PHOSPHO_MAPPING)

    # MSMS model
    params = ModelParamsMSMS(seq_len=50, ions=['y','b'], max_charge=2, neutral_losses=['', 'H3PO4'])
    nn_model, creation_meta = generate_dummy_msms_model(
        params=params,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        dropout_rate=0.1
    )
    msmsmodel = AIProteomicsModel(seq_map=seqmap, model_params=params, nn_model=nn_model, nn_model_creation_metadata=creation_meta)

    # iRT model
    params = ModelParamsRT(seq_len=50, iRT_rescaling_mean=101.11514, iRT_rescaling_var=46.5882)
    nn_model, creation_meta = generate_dummy_iRT_model(
        params=params,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        dropout_rate=0.1
    )
    rtmodel = AIProteomicsModel(seq_map=seqmap, model_params=params, nn_model=nn_model, nn_model_creation_metadata=creation_meta)


    # CCS model
    params = ModelParamsCCS(seq_len=50)
    nn_model, creation_meta = generate_dummy_ccs_model(
        params=params,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        dropout_rate=0.1
    )
    ccsmodel = AIProteomicsModel(seq_map=seqmap, model_params=params, nn_model=nn_model, nn_model_creation_metadata=creation_meta)

    # Use the three models to generate a spectral library
    input_peptides = pd.read_csv(request.path.parent / 'assets/peptides_to_predict.tsv', sep='\t')
    speclib_df = build_spectral_library(input_peptides, msms=msmsmodel, rt=rtmodel, ccs=ccsmodel)

    # Output speclib to tsv
    speclib_df.to_csv(tmp_path / 'speclib.tsv', sep='\t')
