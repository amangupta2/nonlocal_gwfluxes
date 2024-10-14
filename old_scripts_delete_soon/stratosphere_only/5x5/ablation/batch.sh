#!/bin/bash
#SBATCH --job-name=5x5_2cnn
#SBATCH --partition=serc
#SBATCH -c 10
#SBATCH -G 1
#SBATCH --gpus-per-node=1
#SBATCH --time=168:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=gpu_slurm-%j.out
#SBATCH -C GPU_MEM:80GB
#SBATCH --mem-per-cpu=10GB # 5GB for regional, but 10 GB for global 5x5
###SBATCH --job-name=jupyter_notebook
###SBATCH --mail-user=ag4680@stanford.edu
###SBATCH --mail-type=BEGIN,END,FAIL
###SBATCH --partition=serc
###SBATCH -p gpu (use serc, not gpu, which has 80 GB A100s too)

# from Mark:
# use sh_node_feat -p serc (or gpu) to see the node structure of the partition and what GPUs are available
# -c indicates cpu_per_task
# -G is the number of GPUs you want to request
# -p is the partition
# requesting SBATCH -G 4 || AND || --gpus-per-node=4 allocated 4 GPUs within a single node
# if you want the distributed over two nodes, do: -G 4 || --gpus-per-node=2

# for more information on GPUs on sherlock: https://www.sherlock.stanford.edu/docs/user-guide/gpu/#gpu-types

source /home/groups/aditis2/ag4680/miniconda3/etc/profile.d/conda.sh
conda activate siv2

python 5x5global_training_ablation_2cnnlayers.py
#python 5x5global_training_ablation_2cnnlayers_uvthetaw.py
# 1andes, 2scand, 3himalaya, 4newfound, 5south_ocn, 6se_asia, 7natlantic, 8npacific
#python regional1x1.py 8npacific
#python troposphere_regional1x1.py 1andes
