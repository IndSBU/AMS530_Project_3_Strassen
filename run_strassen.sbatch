#!/bin/bash
#SBATCH --job-name=strassen
#SBATCH --output=seawulf_strassen_%j.out
#SBATCH --error=seawulf_strassen_%j.err
#SBATCH --partition=short-96core
#SBATCH --nodes=1
#SBATCH --ntasks=49                 # 49 ranks = 49 leaves
#SBATCH --ntasks-per-node=49        # ensure all ranks land on the one node
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=0

# --- env info (helpful for debugging) ---
echo "Start: $(date)"
echo "Node list: $SLURM_NODELIST"
echo "CWD: $(pwd)"
echo "SLURM_NTASKS=${SLURM_NTASKS}"

# --- toolchain ---
module purge
# (No need to 'module load slurm' inside a batch job)
module load gcc/13.2.0
module load openmpi/gcc13.2/5.0.2

# --- MPI/UCX prefs (good on Milan/Seawulf) ---
export OMPI_MCA_pml=ucx
export OMPI_MCA_osc=ucx
export OMPI_MCA_btl=^openib
# Keep ranks on distinct cores
export SLURM_CPU_BIND=cores
# Make sure any OpenMP in dependencies doesn't oversubscribe
export OMP_NUM_THREADS=1

# --- build (adjust filename if needed) ---
mpicxx -O3 -march=native -std=c++17 strassen.cpp -o seawulf_strassen

# --- run ---
# Prefer Slurm launcher with PMIx so OpenMPI sees all ranks
srun --mpi=pmix -n ${SLURM_NTASKS} ./seawulf_strassen

# If your site prefers mpirun, comment the srun line above and use:
# mpirun -np ${SLURM_NTASKS} --map-by core ./seawulf_strassen

echo "End: $(date)"

