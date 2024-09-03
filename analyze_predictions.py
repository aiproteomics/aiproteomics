# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import h5py
from dask import array as da
import matplotlib.pyplot as plt
import seaborn as sns
from dask import dataframe as dd
from dask.diagnostics import ProgressBar
import pandas as pd
from pathlib import Path
import os
import sys
# %matplotlib inline

# %%
PLOTS_DIR = Path("plots")


# %%
# Turns out it was better to store the results in a hdf5 file
# predictions = np.load("predictions.npy")

# with h5py.File("predictions.hdf5", "w") as f:
#     dset = f.create_dataset("predictions", predictions.shape, predictions.dtype)
#     dset[:, :] = predictions
# %%
# The following only makes sense as shell script
try:
    print(sys.argv)
    chunksize = int(sys.argv[1])
    sample_size = -1
    print(f"Found chunksize of {chunksize}")
except Exception:
    print("No chunksize found")
    chunksize = 1000
    sample_size = 1000

# %%
def load_dask_array_from_hdf5(filename, key, chunksize=1000, sample_size=-1):
    f = h5py.File(filename)
    a = da.from_array(f[key], chunks=chunksize)
    a = a[:sample_size]
    return a


# %%
predictions = load_dask_array_from_hdf5("predictions.hdf5", "predictions", chunksize, sample_size)
labels = load_dask_array_from_hdf5("traintest_hcd.hdf5", "intensities_raw", chunksize, sample_size)
collision_energy = load_dask_array_from_hdf5("traintest_hcd.hdf5", "collision_energy", chunksize, sample_size)
precursor_charge = load_dask_array_from_hdf5("traintest_hcd.hdf5", "precursor_charge_onehot", chunksize, sample_size)
sequences = load_dask_array_from_hdf5("traintest_hcd.hdf5", "sequence_integer", chunksize, sample_size)

# %%
full_sequences = sequences[:,-1] > 0

full_sequences.shape

# %%
sequence_lengths= da.argmin(sequences, axis=1)

# %%
sequence_lengths[full_sequences] = 29

# %%
f = h5py.File("traintest_hcd.hdf5")

f.keys()

# %%
["collision_energy_aligned_normed", "precursor_charge_onehot", "sequence_integer"]

# %%
labels.shape

# %%
predictions.shape

# %%
labels[0].compute()

# %%
from sklearn.preprocessing import normalize


# TODO: Check if results are the same as with ComparisonPrositFrag
def normalized_spectral_contrast_distance(true, pred):
    """
    Calculate the (normalized) spectral contrast distance for two spectra. 1 represents total overlap.
    """
    pred_norm = normalize(pred)
    true_norm = normalize(true)
    
    product =  pred_norm * true_norm
    product = product.sum(axis=1)
    
    arccos = np.arccos(product)
    return 1 - 2 * arccos / np.pi



# %%
normalized_spectral_contrast_distance(labels[:2], predictions[:2])

# %%
limit=None

comparisons = da.map_blocks(normalized_spectral_contrast_distance, labels[:limit], predictions[:limit], drop_axis=1, dtype=float)

comparisons

# %%
computation, edges = da.histogram(comparisons, bins=50, range=(0, 1))

with ProgressBar():
    hist = computation.compute()


# %%


ax = sns.barplot(x=edges[:-1], y=hist)

_=ax.set_xticks(ticks = range(len(edges[:-1])), labels=[e if i%4 == 0 else "" for i, e in enumerate(edges[:-1])])
_ = ax.set_xlabel("Spectral distance (higher is better)")
_ = ax.set_ylabel("Number of predictions")

plt.savefig(PLOTS_DIR/"accuracy_hist.png")

# %%
sequence_column_names = [f"seq_{i}" for i in range(sequences.shape[1])]

sequence_df = dd.from_dask_array(sequences, columns=sequence_column_names)

concatenated = dd.concat([dd.from_dask_array(sequence_lengths), dd.from_dask_array(collision_energy), dd.from_dask_array(comparisons), sequence_df], 1)
concatenated.columns = ["sequence_length", "collision_energy", "distance"] + sequence_column_names

concatenated

# %%
concatenated.head()

# %%
# Let's take a sample to quickly try out analyses
sample_size= 100000
sample = concatenated.head(sample_size)

# %%
collision_energy[:3].compute()

# %%
corr_columns = ["sequence_length", "collision_energy", "distance"]

with ProgressBar():
    sample = concatenated[corr_columns].head(100000)
    correlations = sample.corr()
    
correlations


# %%
with ProgressBar():
    full_set_correlations = concatenated[corr_columns].corr().compute()

sns.heatmap(full_set_correlations)

plt.savefig(PLOTS_DIR/"correlations")

# %% [markdown]
# It seems that sequence length and distance have some correlation. I can imagine that longer sequences are more easy to predict because there is more info there? Or maybe there are unique sequences that are memorized. I wonder what the distribution is of sequence length.

# %%
sample.sequence_length.hist()

# %%
# On the full set

freq, bins = da.histogram(concatenated["sequence_length"], bins=range(30))

with ProgressBar():
    freq = freq.compute()
    
bins = bins.compute()

# %%
fig, ax = plt.subplots()

ax.bar(x=range(freq.shape[0]), height= freq)
ax.set_xlabel("Sequence length")
ax.set_ylabel("Frequency")

ax.set_title("Number of sequences per sequence length")

plt.savefig(PLOTS_DIR/"sequence_length_histogram.png")

# %% [markdown]
# I wonder if there are any duplicates and whether it is possible that the model overfits on them.

# %%
counts = sequence_df.groupby(sequence_df.columns.tolist()).size()


counts[counts["size"] > 1]

with ProgressBar():
    counts =  counts.compute()



# %% [markdown]
# Looks like there is one sequence that is in the dataset 2843 times! I wonder how that affects the training set. I also wonder if this sequence has consistent target values.

# %%
sorted_duplicates = counts.sort_values(ascending=False)

pd.DataFrame(sorted_duplicates, columns=["number_of_duplicates"])


# %%
with ProgressBar():
    all_duplicates = concatenated[(concatenated[sequence_column_names] == sorted_duplicates.index[0]).all(axis=1)].compute()

all_duplicates

all_duplicates.to_csv(PLOTS_DIR/"duplicates.csv")

# %% [markdown]
# ## Distinguishing samples that perform well or badly
#
#

# %%
concatenated.head()

# %%
with ProgressBar():
    best_performing = concatenated.sort_values("distance", ascending=False).head(1000)

best_performing

# %%
best_performing.iloc[0]

# %%
