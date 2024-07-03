#!/bin/sh
#SBATCH -p crtai3
#SBATCH -N 1
#SBATCH --gpus-per-node=1

#conda activate /home/mshaka/anaconda3/envs/llm

#python prompt_preparation.py
python main.py
