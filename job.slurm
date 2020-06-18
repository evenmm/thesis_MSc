#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-imf
#SBATCH --time=90:00:00
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=5
#SBATCH --mem=128000
#SBATCH --job-name="b.5-200-parallel"
#SBATCH --output=out-b.5-T-200-lambda-index-%a.out
#SBATCH --mail-user=evenmm@stud.ntnu.no
#SBATCH --mail-type=NONE
#SBATCH --array=0-20 

# 21 different lambdas. Array id is lambda index
# Other version:
# 420 = 21 lambdas times 21 seeds
# _%A_%a gives Master job id and array task id

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
#pip install scipy numpy matplotlib sklearn
#export OMP_NUM_THREADS=5 # This is set by cpus-per-task

/usr/bin/time -v python cluster-parallel-robustness-evaluation.py $SLURM_ARRAY_TASK_ID

uname -a

deactivate
