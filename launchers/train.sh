#!/bin/bash
#SBATCH --job-name="test_mnist"
#SBATCH --workdir=.
#SBATCH --qos=training
#SBATCH --ntasks=8
#SBATCH --gres gpu:2
#SBATCH --time=02:00:00

module purge; module load K80/default impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML

python train.py
