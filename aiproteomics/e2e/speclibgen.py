from . import sanitize
import pandas as pd
import numpy as np
from aiproteomics.e2e import convert_to_msp
from . import tensorize

def _predict(data, model_frag, model_irt, batch_size_frag=None, batch_size_iRT=None, iRT_rescaling_mean=None, iRT_rescaling_var=None):
    # Get fragmentation model predictions
    x = [data['sequence_integer'], data['precursor_charge_onehot'], data['collision_energy_aligned_normed']]
    prediction = model_frag.predict(x, verbose=False, batch_size=batch_size_frag)
    data["intensities_pred"] = prediction
    data = sanitize.prediction(data)

    # Get iRT model predictions
    x = data['sequence_integer']
    prediction = model_irt.predict(x, verbose=False, batch_size=batch_size_iRT)
    data["iRT"] = prediction * np.sqrt(iRT_rescaling_var) + iRT_rescaling_mean

    return data


def _read_peptides_csv(fname):

    df = pd.read_csv(fname)    
    df.reset_index(drop=True, inplace=True)
    assert "modified_sequence" in df.columns
    assert "collision_energy" in df.columns
    assert "precursor_charge" in df.columns
    data = {
        "collision_energy_aligned_normed": tensorize.get_numbers(df.collision_energy) / 100.0,
        "sequence_integer": tensorize.get_sequence_integer(df.modified_sequence),
        "precursor_charge_onehot": tensorize.get_precursor_charge_onehot(df.precursor_charge),
        "masses_pred": tensorize.get_mz_applied(df),
    }
    nlosses = 1
    z = 3
    lengths = (data["sequence_integer"] > 0).sum(1)

    masses_pred = tensorize.get_mz_applied(df)
    masses_pred = sanitize.cap(masses_pred, nlosses, z)
    masses_pred = sanitize.mask_outofrange(masses_pred, lengths)
    masses_pred = sanitize.mask_outofcharge(masses_pred, df.precursor_charge)
    masses_pred = sanitize.reshape_flat(masses_pred)
    data["masses_pred"] = masses_pred

    return data


def csv_to_msp(in_csv_fname, out_msp_fname, model_frag, model_irt, batch_size_frag, batch_size_iRT, iRT_rescaling_mean, iRT_rescaling_var):
    data = _read_peptides_csv(in_csv_fname)
    data = _predict(data,
               model_frag,
               model_irt,
               batch_size_frag=batch_size_frag,
               batch_size_iRT=batch_size_iRT,
               iRT_rescaling_mean = iRT_rescaling_mean,
               iRT_rescaling_var = iRT_rescaling_var
              )

    c = convert_to_msp.Converter(data, out_msp_fname)
    c.convert()

