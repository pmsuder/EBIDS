#!/bin/bash
#SBATCH -e output_slurm/slurm_%A_%a.err
#SBATCH -o output_slurm/slurm_%A_%a.out
#SBATCH --array=181-200
#SBATCH --mail-user=pms69@duke.edu
#SBATCH --mail-type=END
#SBATCH -c 10
#SBATCH --mem-per-cpu=2GB
#SBATCH --account=laberlabs

conda init
conda activate mafab
python src/simulations_main.py --setting "conservative_B"
