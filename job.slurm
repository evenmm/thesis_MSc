#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-imf
#SBATCH --time=70:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=12000
#SBATCH --job-name="parallel_test"
#SBATCH --output=output-parallel-test.out
#SBATCH --mail-user=evenmm@stud.ntnu.no
#SBATCH --mail-type=NONE

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module purge
module load GCC/7.3.0-2.30
module load CUDA/9.2.88
module load OpenMPI/3.1.1
module load Python/3.6.6

#virtualenv pythonhome
source pythonhome/bin/activate
#pip install scipy numpy matplotlib
export OMP_NUM_THREADS=4

/usr/bin/time -v python parallel-robustness-evaluation.py

uname -a

deactivate
