#!/bin/bash
#SBATCH --job-name=inferIFS
#SBATCH --partition=serc
#SBATCH -c 10
#SBATCH -G 1
#SBATCH --gpus-per-node=1
#SBATCH --time=168:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=gpu_slurm-%j.out
#SBATCH -C GPU_MEM:80GB
#SBATCH --mem-per-cpu=5GB

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

# 'global'/'stratosphere_only' and 'feature_set'
# TRAINING
#python training_attention_unet.py global uvthetaw





# INFERENCE
# Most optimal epochs to use for respective configs
# stratosphere_only | uvtheta  | epoch=100 | month
# stratosphere_only | uvthetaw | epoch=100 | month
# Usage: python inference.py <vertical> <features> <epoch> <month>
python inference2.py stratosphere_only uvthetaw 100 $month 1
python inference2.py stratosphere_only uvtheta 100 $month 1



