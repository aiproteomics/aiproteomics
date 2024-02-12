import tensorflow as tf
import keras.backend as k
import numpy as np
import pandas as pd
import seaborn as sns

from aiproteomics.datasets.DataSetPrositFrag import DataSetPrositFrag


class ComparisonPrositFrag:

    @staticmethod
    def compare_spectral_angle_distributions(dataset: DataSetPrositFrag, frag_model):
        """
        Returns a pandas dataframe containing the spectral distance between true and predicted
        values in the provided dataset, using predictions from the provided tensorflow model.
        """
        pred_all, true_all = dataset.get_predictions(frag_model)

        # TODO: The following is a very slow way to get the spectral angle distance for every spectrum.
        # Refactor to use numpy exclusively.
        spectral_distance = []
        for true, pred in np.stack((true_all, pred_all), axis=1):
            true = true.clip(min=0)
            pred = pred.clip(min=0)
            spectral_distance.append(
                ComparisonPrositFrag.normalized_spectral_contrast_distance(true, pred)
            )
        spectral_distance = np.array(spectral_distance)

        result_dict = {
            "normalized_spectral_contrast_distance": spectral_distance,
            "collision_energy": dataset.collision_energy.flatten(),
        }
        return pd.DataFrame.from_dict(result_dict)

    @staticmethod
    def plot_spectral_angle_distributions(dataset: DataSetPrositFrag, frag_model, save_fname=None):
        # Calculate the spectral angle distances
        df = ComparisonPrositFrag.compare_spectral_angle_distributions(dataset, frag_model)

        # Plot the spectral angle distances, one for each collision energy
        sns.set_theme()
        sa_plot = sns.violinplot(
            data=df, x="collision_energy", y="normalized_spectral_contrast_distance"
        )

        # If save filename is provided, save the figure
        if save_fname is not None:
            sa_plot.get_figure().savefig(save_fname)

        return sa_plot

    @staticmethod
    def normalized_spectral_contrast_distance(true, pred):
        """
        Calculate the (normalized) spectral contrast distance for two spectra. 1 represents total overlap.
        """
        pred_norm = k.l2_normalize(pred)
        true_norm = k.l2_normalize(true)
        product = k.sum(pred_norm * true_norm)
        arccos = tf.acos(product)
        return 1 - 2 * arccos / np.pi
