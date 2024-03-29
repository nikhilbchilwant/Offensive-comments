#!/bin/bash
#----------------------------------
# Specifying grid engine options
#----------------------------------
#$ -S /bin/bash  
# the working directory where the commands below will
# be executed: (make sure to specify)
#$ -wd /nethome/nchilwant/projects/offensive-comments
#
# logging files will go here: (make sure to specify)
#$ -e /data/users/nchilwant/log/ -o /data/users/nchilwant/log/
#  
# Specify the node on which to run
#$ -l hostname=cl12lx
#----------------------------------
#  Running some bash commands 
#----------------------------------
export PATH="/nethome/nchilwant/miniconda3/bin:$PATH"
source activate pytorch_cuda101
nvidia-smi
#----------------------------------
# Running your code (here we run some python script as an example)
#----------------------------------
pwd
echo "Starting the python script"
python train.py -c germ-eval-config.json
echo "Finished execution."