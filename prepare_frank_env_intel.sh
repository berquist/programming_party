#!/usr/bin/env bash

# Source, don't run:
# source prepare_frank_env_intel.sh

# This will load the proper environment modules for using the
# Intel compilers on Frank.

module use /home/dlambrecht/erb74/modules/frank
module use /home/dlambrecht/software/modules

module purge

module load python/anaconda3
module load emacs
module load git
module load cmake
module load intel/2015.1.133
module load mkl/2015.1.133/icc-st
module load armadillo/5.600.2-i2015.1.133-wrapper
