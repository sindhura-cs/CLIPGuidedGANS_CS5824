#!/bin/bash

### train_stylegan.sh
###########################################################################
## environment & variable setup
####### job customization
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -t 72:00:00
#SBATCH -p t4_normal_q
#SBATCH -A cs5824
#SBATCH --gres=gpu:1
####### end of job customization
# end of environment & variable setup
###########################################################################
module load Anaconda3/2020.11 cuda10.1/toolkit/10.1.243 pytorch-py37-cuda10.1-gcc/1.6.0 Ninja/1.9.0-GCCcore-8.3.0
module list ## make sure cuda is loaded if you are using the GPU
nvidia-smi  ## make sure you see GPUs
python train.py --iter 13500 --wandb --augment_p 0.5 --ckpt ./checkpoint/005000.pt fashion_IQ_256.lmdb
