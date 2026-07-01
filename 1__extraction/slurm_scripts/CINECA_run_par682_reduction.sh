#!/bin/bash -l

#SBATCH --job-name=job-par682           # Job name
#SBATCH --time=06:00:00                 # Walltime (hh:mm:ss)
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=8            # One MPI task per node
#SBATCH --cpus-per-task=1             # Number of physical CPU cores per task (adjust to 32 for MARCONI100)
#SBATCH --partition=boost_usr_prod   # Partition to submit to
#SBATCH --qos=normal               # Quality of Service
#SBATCH --mem=256G                  # Memory per node (e.g., 128G)
#SBATCH --output=par682-reduction.out             # Standard output file
#SBATCH --error=par682-reduction.err              # Standard error file
#SBATCH --account=<insert-here>    # Project account number

# Set environment variables for OpenMP
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # Set number of OpenMP threads

# Run the application using srun
mpirun --bind-to=none python $HOME/code/PASSAGE_scripts/1__extraction/A__reduction.py $HOME/code/PASSAGE_scripts/1__extraction/config_files/CINECA_config_par682.toml
