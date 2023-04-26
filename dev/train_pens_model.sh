#!/bin/bash -l

# Set SCC project

#$ -P sparkgrp

# Request my job to run on Buy-in Compute group hardware my_project has access to
#$ -l buyin

# Specify hard time limit for the job. 
#   The job will be aborted if it runs longer than this time.
#   The default time is 12 hours

#$ -l h_rt=48:00:00

# Send an email when the job finishes or if it is aborted (by default no email is sent).

#$ -m ea

# Request a whole 36 processor node with at least 1 TB of RAM

#$ -pe omp 16

# Combine output and error files into a single file
#$ -j y

module load python3/3.8.10
source /projectnb/sparkgrp/ds-naacp-media-bias/venvs/gensim-venv/bin/activate
python topicModel.py