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

python pinesClass.py --lda 12 --RF 20 -f > RF_20_LDA_12
