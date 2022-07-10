#!/bin/bash

#SBATCH --job-name=RCGAN

#SBATCH --time=01:10:00

#SBATCH --mail-user=daniele.regecambrin@studenti.polito.it
#SBATCH --mail-type=ALL

#SBATCH --output=%j_log.log

#SBATCH --mem=4GB

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:1
#SBATCH --partition=cuda

module load nvidia/cudasdk/11.6

source ~/.bashrc
conda activate mlinapp

python train.py --training_approach="specific" --epochs=30 --batch_size=128 --num_workers=4 --indices_path="data/chosen_indices_eyeglasses_smaller.npy" --experiment_name="dg3 freeze 4 rec 10" --target_attr="Eyeglasses" --dg_ratio=3 --upload_weights --max_time "00:01:00:00" --freeze_layers 4 --mode wgan --lambda_1 10 --use_alternate_dataset