#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH -A ICT23_MHPC
#SBATCH --time 0:40:00
#SBATCH -p boost_usr_prod
#SBATCH -e ../error-%j.err
#SBATCH -o ../outputs/output-%j-RF16.out


#module load profile/deeplrn
#module load cuda/11.8
#module load gcc/11.3.0
#module load nccl
#module load llvm
#module load gsl
#module load openmpi
#export PATH=/leonardo/home/userexternal/sdigioia/R_package/bin/R:${PATH}
#export LD_LIBRARY_PATH=/leonardo/home/userexternal/sdigioia/R_package/lib/R/lib:${LD_LIBRARY_PATH}
#export R_HOME=/leonardo/home/userexternal/sdigioia/R_package/
#export PKG_CONFIG_PATH=/leonardo/home/userexternal/sdigioia/R_package/lib/R/pkgconfig/:${PKG_CONFIG_PATH}

#conda activate /leonardo_work/ICT23_MHPC/sdigioia/env/RLenv

python pinesClass.py --pca 20 --RF 20 
