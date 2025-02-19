#!/bin/bash

#SBATCH --partition=roma
#SBATCH --account=neutrino:ml-dev
#
#SBATCH --job-name=gmpana
#SBATCH --output=logs/output-%j.txt
#SBATCH --error=logs/output-%j.txt
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10g
#
#SBATCH --time=2:00:00

SINGULARITY_IMAGE_PATH=/sdf/group/neutrino/images/larcv2_ub20.04-cuda11.3-cudnn8-pytorch1.10.0-larndsim-2022-11-03.sif

# I've been using a UUID from the OS to make unique filenames
# these are very long and can be quite ugly
# you can also just use the slurm batch index
GAMPANAROOT=${HOME}/studies/gampana/gampana
GAMPIXROOT=${HOME}/studies/GAMpix/GAMPixPy/gampixpy

TRAINDIR=/sdf/data/neutrino/dougl215/gampixpy/point_source_batch1_250214/gampixD
TRAINOUTPUT=$1/gampixD
LEARNINGRATE=$2

mkdir $1
mkdir $TRAINOUTPUT

COMMAND="python3 ${GAMPANAROOT}/train_position_estimator.py -o ${TRAINOUTPUT} -r ${GAMPIXROOT}/readout_config/GAMPixD.yaml -n 20 --train ${TRAINDIR} -lr ${LEARNINGRATE}"

echo $COMMAND
singularity exec -B /sdf,/lscratch ${SINGULARITY_IMAGE_PATH} ${COMMAND}
