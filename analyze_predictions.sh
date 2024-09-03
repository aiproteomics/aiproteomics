#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=18
#SBATCH --mem=40G

# module load 2022
# module load Python/3.10.4-GCCcore-11.3.0
# module load h5py/3.7.0-foss-2022a
# module load matplotlib/3.5.2-foss-2022a

module load 2023
module load Python-bundle-PyPI/2023.06-GCCcore-12.3.0
module load scikit-learn/1.3.1-gfbf-2023a
module load h5py/3.9.0-foss-2023a
module load matplotlib/3.7.2-gfbf-2023a
module load dask/2023.9.2-foss-2023a

echo "Analyzing predictions"
python analyze_predictions.py 1000000
