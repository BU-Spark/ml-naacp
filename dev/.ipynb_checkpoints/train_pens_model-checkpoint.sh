#!/bin/bash -l

# Set SCC project

#$ -P sparkgrp

# Send an email when the job finishes or if it is aborted (by default no email is sent).

#$ -m ea

# Request a whole 28 processor node with at least 512 GB of RAM

#$ -pe omp 28
#$ -l mem_per_core=18G

# Combine output and error files into a single file
#$ -j y

module load python3/3.8.10
source /projectnb/sparkgrp/ds-naacp-media-bias/venvs/gensim-venv/bin/activate
python topicModel.py