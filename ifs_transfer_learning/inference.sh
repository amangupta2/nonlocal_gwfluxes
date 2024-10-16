#!/bin/bash
#SBATCH --job-name=TLINFER
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

# use training_ifs_transfer_learning.py for transfer learning training

# 'attention/ann' 'global'(horizontal) 'global'/'stratosphere_only'(vertical) and 'feature_set', 'CHECKPOINT_EPOCH'
# TRAINING

# use attn_inference.py for TL inference using attention UNet models
# Inference from transfer learning save checkpoints
# attention unet is only being used for global horizontal, not regional 
# for IFS and dropout=0: <vertical> <features> <ckpt#> <test_on>
#python attn_inference.py stratosphere_only uvtheta 100 IFS
#python attn_inference.py stratosphere_only uvthetaw 100 IFS

#python ANN_inference.py stratosphere_only uvtheta 100 IFS 1
#python attn_inference.py stratosphere_only uvthetaw 100 IFS


# for ERA5 and dropout=0: <vertical> <features> <ckpt#> <test_on> <validation month>
for month in 1 2 3 4 5 6 7 8 9 10 11 12;
do
        #python attn_inference.py stratosphere_only uvtheta 100 ERA5 $month
        #python attn_inference.py stratosphere_only uvthetaw 100 ERA5 $month
	python ANN_inference.py stratosphere_only uvtheta 100 ERA5 $month 1
done
#python attn_inference.py attention global stratosphere_only uvthetaw 100



#for IFS and dropout!=0 (need ensemble numbers are well): # for IFS and dropout=0: <vertical> <features> <ckpt#> <test_on> <ensemble_no.>


#for ERA5 and dropout!=0 (need ensemble numbers are well) # for IFS and dropout=0: <vertical> <features> <ckpt#> <test_on> <validation month> <ensemble_no.>



