#! /bin/bash

#SBATCH --job-name="Python MM"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH  -A ICT23_MHPC
#SBATCH --time 2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=49000MB
#SBATCH  -p boost_usr_prod
#SBATCH -e %j.err
#SBATCH -o %j.out

python pinesClass.py --pca 20 --lda 10 --RF 20 --LogR 200