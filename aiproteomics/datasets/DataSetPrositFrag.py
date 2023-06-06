import tensorflow as tf
import h5py

class DataSetPrositFrag:
    EXPECTED_PROSIT_FRAG_DATA_KEYS = [
        'sequence_integer',
        'precursor_charge_onehot',
        'collision_energy',
        'collision_energy_aligned_normed',
        'intensities_raw'
    ]
    
    def __init__(self, dataset_fname, N=-1):
        """
        Reads in fragmentaion dataset from the specified hdf5 file.
        Format is assumed to be the style of the Prosit (2019) model.
        Reads N data points (by default N=-1, for all data)
        """

        with h5py.File(dataset_fname, "r") as f:

            for expected_key in self.EXPECTED_PROSIT_FRAG_DATA_KEYS:
                if expected_key not in f.keys():
                    raise ValueError(f"Expected key {expected_key} in provided hdf5 dataset {dataset_fname}. Are you sure this is a Prosit 2019 Fragmentation Dataset?")
            
            self.sequence = f['sequence_integer'][:N]
            self.precursor_charge = f['precursor_charge_onehot'][:N]
            self.collision_energy = f['collision_energy'][:N]
            self.collision_energy_aligned_normed = f['collision_energy_aligned_normed'][:N]
            self.intensities_raw = f['intensities_raw'][:N]
    
        # Cast intensities to float32 to match prediction type
        self.intensities_raw = tf.cast(self.intensities_raw, tf.float32)
    
        self.model_X = [self.sequence, self.precursor_charge, self.collision_energy_aligned_normed]
        self.model_Y = self.intensities_raw

    def get_predictions(self, model):
        """
        Apply the given tensorflow model to the input layer data in this dataset, returning the predictions
        as numpy array. Also returns the true (Y) outputs corresponding to each prediction.
        """
        return model.predict(self.model_X), self.model_Y
