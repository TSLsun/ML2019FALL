#!/bin/bash
#SBATCH --job-name="ML"
#SBATCH --partition=v100-16g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00
#SBATCH --workdir=.
#SBATCH --output=cout.txt
#SBATCH --error=cerr.txt
###SBATCH --test-only

echo
echo "============================ Messages from Goddess ============================"
echo " * Job starting from: "`date`
echo " * Job ID           : "$SLURM_JOBID
echo " * Job name         : "$SLURM_JOB_NAME
echo " * Job partition    : "$SLURM_JOB_PARTITION
echo " * Nodes            : "$SLURM_JOB_NUM_NODES
echo " * Cores            : "$SLURM_NTASKS
echo " * Working directory: "${SLURM_SUBMIT_DIR/$HOME/"~"}
echo "==============================================================================="
echo

module load opt gcc cuda/9.0 python/3.6.8-gnu-gpu 

bash ./cluster.sh ../data/trainX.npy ./pred.csv
#/share/home/tslsun025/.conda/envs/ml2019fall_hw3/bin/python3.6 test.py  ../data/test_img  ../submit_output/pred.csv

#CUDA_VISIBLE_DEVICES=1 python3 unet2D-norm.py

echo
echo "============================ Messages from Goddess ============================"
echo " * Job ended at     : "`date`
echo "==============================================================================="
echo
