#!/bin/bash

#SBATCH -J gcp-cpu-test
#SBATCH -o slurm/gcp-cpu-test.%j.out
#SBATCH -e slurm/gcp-cpu-test.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --partition=compute
#SBATCH -t 120:00:00

source /contrib/Tim.Smith/miniconda3/etc/profile.d/conda.sh
conda activate xesn
mprof run --python python test_scaling.py
