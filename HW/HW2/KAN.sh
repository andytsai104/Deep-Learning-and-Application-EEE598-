#!/bin/bash

#SBATCH -N 1		# number of nodes
#SBATCH --mem=20G	# amount of memory for the job
#SBATCH -G a100:1	# number of gpus
#SBATCH -c 1		# number of cores
#SBATCH -t 0-10:00:00	# time in d-hh:mm:ss
#SBATCH -p general	# partition
#SBATCH -q public	# QOS
#SBATCH -o KAN.%j.out	# file to save job's STDOUT (%j = JobId)
#SBATCH -e KAN.%j.err	# file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL		# sent an email when a job starts, stops, or fails
#SBATCH --mail-user="ctsai67@asu.edu"
#SBATCH --export=NONE		# Purge the job-submitting shell environment

# Load required software
module load mamba/latest

# Activate our environment
source activate Tsai

# Change the directory  of our script
cd ~/EEE598/A2/

# Run the software/python script
# python P3-1.py
python KAN.py