#!/bin/bash

#SBATCH -N 1		# number of nodes
#SBATCH --mem=256G	# amount of memory for the job
#SBATCH -G a100:1	# number of gpus
#SBATCH -c 4		# number of cores
#SBATCH -t 0-20:00:00	# time in d-hh:mm:ss
#SBATCH -p general	# partition
#SBATCH -q public	# QOS
#SBATCH --output="/home/ctsai67/EEE598/Assignment 3/P3/3-1/out/3-1.%j.out"	# file to save job's STDOUT (%j = JobId)
#SBATCH --error "/home/ctsai67/EEE598/Assignment 3/P3/3-1/err/3-1.%j.err"	# file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL		# sent an email when a job starts, stops, or fails
#SBATCH --mail-user="ctsai67@asu.edu"
#SBATCH --export=NONE		# Purge the job-submitting shell environment

# Load required software
module load mamba/latest

# Activate our environment
source activate Tsai

# Change the directory  of our script
cd ~/EEE598/"Assignment 3"/P3/3-1

# Run the software/python script
python P3-1.py