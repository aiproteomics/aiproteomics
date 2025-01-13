#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH -p gpu
#SBATCH --gpus=4
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=18
#SBATCH --mem=40G


module load 2022
module load Python/3.10.4-GCCcore-11.3.0
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
module load h5py/3.7.0-foss-2022a

echo "training new transformer model"

data_dir=/gpfs/work2/0/einf3550/randomized-c1000-mf4-l1

python train.py "$data_dir" --distributed --epochs 100 --batch-size 1024 --file-limit 800