#!/bin/bash
#SBATCH --job-name=strassen
#SBATCH --output=seawulf_strassen_%j.out
#SBATCH --error=seawulf_strassen_%j.err
#SBATCH --partition=short-96core      
#SBATCH --nodes=1                     
#SBATCH --ntasks=49                    
#SBATCH --cpus-per-task=1             
#SBATCH --time=00:30:00              
#SBATCH --mem=0                        

echo "Start: $(date)"
echo "Node list: $SLURM_NODELIST"
echo "CWD: $(pwd)"
echo "SLURM_NTASKS=${SLURM_NTASKS}"

module purge
module load slurm
module load gcc/13.2.0
module load openmpi/gcc13.2/5.0.2

export OMPI_MCA_pml=ucx
export OMPI_MCA_osc=ucx
export OMPI_MCA_btl=^openib

export SLURM_CPU_BIND=cores

export OMP_NUM_THREADS=1

mpicxx -O3 -march=native -std=c++17 strassen.cpp -o seawulf_strassen

echo "mpicxx: $(mpicxx --version | head -n1)"
echo "mpirun:  $(mpirun --version | head -n1)"
echo "ompi_info PMIx summary:"
ompi_info --parsable --all | grep -E 'mpi:|pmix:|prte:' | head -n 20

PLACEMENT=/gpfs/scratch/inijjar/placement.txt
mpirun -np  ${SLURM_NTASKS} -map-by core --bind-to core ./seawulf_strassen "$PLACEMENT"

echo "End: $(date)"

 
