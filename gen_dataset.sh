#!/bin/bash
#SBATCH --job-name=cartpole_gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=gen_dataset_%j.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate icl_ebonye_pendulum

cd /home/etran52/data/transformer_control

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python generate_dataset_cartpole_ebonye.py --config conf/_XandYtest_cartpole_ebonye.yaml
