#!/bin/bash -l

#SBATCH --job-name=job-par676           # Job name
#SBATCH --time=03:00:00                 # Walltime (hh:mm:ss)
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=1            # One MPI task per node
#SBATCH --cpus-per-task=8             # Number of physical CPU cores per task (adjust to 32 for MARCONI100)
#SBATCH --partition=dcgp_usr_prod   # Partition to submit to
#SBATCH --qos=normal               # Quality of Service
#SBATCH --mem=32G                  # Memory per node (e.g., 128G)
#SBATCH --output=job-par676.out             # Standard output file
#SBATCH --error=job-par676.err              # Standard error file
#SBATCH --account=INA24_C7B12    # Project account number

# Load required modules
module load anaconda3@2023.09-0
#conda env update -f ./setup_env.yml -v
conda activate niriss-reduction

# Set environment variables for OpenMP
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # Set number of OpenMP threads

# Run the application using srun
#srun ./myprogram < myinput > myoutput

#salloc -N 1 -n 8
