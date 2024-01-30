module load profile/deeplrn
module load cuda/11.8
module load gcc
module load nccl
module load llvm
module load gsl
module load openmpi
export PATH=/leonardo/home/userexternal/sdigioia/R_package/bin/R:${PATH}
export LD_LIBRARY_PATH=/leonardo/home/userexternal/sdigioia/R_package/lib/R/lib:${LD_LIBRARY_PATH}
export R_HOME=/leonardo/home/userexternal/sdigioia/R_package/
export PKG_CONFIG_PATH=/leonardo/home/userexternal/sdigioia/R_package/lib/R/pkgconfig/:${PKG_CONFIG_PATH}

