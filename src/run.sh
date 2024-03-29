#! /bin/bash

#SBATCH --job-name="Python MM"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH  -A ICT23_MHPC
#SBATCH --time 0:10:00
#SBATCH --gres=gpu:1
#SBATCH --mem=49000MB
#SBATCH  -p boost_usr_prod
#SBATCH -e %j.err
#SBATCH -o %j.out

python pinesClass.py --pca 20 --SVC > SVC_PCA_20
