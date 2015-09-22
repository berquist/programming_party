#!/usr/bin/env bash

# Source, don't run:
# source prepare_frank_env_gnu.sh

# This will load the proper environment modules for using the
# GNU compilers on Frank.

module use /home/dlambrecht/erb74/modules/frank
module use /home/dlambrecht/software/modules

module purge

module load python/anaconda3
module load emacs
module load git
module load cmake
module load gcc/5.2.0-rhel
module load mkl/2015.1.133/icc-st
module load armadillo/5.600.2-g5.2.0-mkl2015-wrapper
