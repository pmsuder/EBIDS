#!/bin/bash

# List of directories to check/create
dirs=("plots" "tables" "output_slurm" "results_slurm" "results_slurm/10_arms_fixed_variances" "results_slurm/10_arms_fixed_variances_sensitivity" "results_slurm/10_arms_random_variances" "results_slurm/10_arms_random_variances_sensitivity" "results_slurm/20_arms_fixed_variances" "results_slurm/20_arms_fixed_variances_sensitivity" "results_slurm/20_arms_random_variances" "results_slurm/20_arms_random_variances_sensitivity" "results_slurm/anti_conservative_B" "results_slurm/conservative_B" "results_slurm/splines" "results_slurm/splines_sensitivity")

for dir in "${dirs[@]}"; do
  if [ ! -d "$dir" ]; then
    echo "Creating directory: $dir"
    mkdir -p "$dir"
  else
    echo "Directory already exists: $dir"
  fi
done
