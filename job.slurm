#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=ie-imf     ###share-ie-imf
#SBATCH --time=99:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1    ### Number of tasks (MPI processes)
#SBATCH --cpus-per-task=2      ### Number of threads per task (OMP threads)
#SBATCH --mem=12000 # Do not set to 128.000 even if that's the limit because then it is too big
#SBATCH --job-name="b.5-200-parallel"
#SBATCH --output=out-b.5-T-200-lambda-index-%a.out
#SBATCH --mail-user=evenmm@stud.ntnu.no
#SBATCH --mail-type=NONE
#SBATCH --array=0-20 

# 21 different lambdas. Array id is lambda index
# Hver job i arrayet er en egen jobb.
# Hver jobb får 1 node, kjører 1 OMP task (pythonskript)
# OMP_NUM_THREADS = # cpu-per-task = 4 eller 5

# Other version:
# 420 = 21 lambdas times 21 seeds
# %A gives Master job id, %a gives array task id

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "The job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
#echo "Total of $SLURM_NTASKS cores"

module purge
module load GCC/7.3.0-2.30
module load CUDA/9.2.88
module load OpenMPI/3.1.1
module load Python/3.6.6

#virtualenv pythonhome
source pythonhome/bin/activate
#pip install scipy numpy matplotlib sklearn
#export OMP_NUM_THREADS=5 # This is set by cpus-per-task

echo "Running and timing the code parallelly, but no printing until it's done..."
/usr/bin/time -v python cluster-parallel-robustness-evaluation.py $SLURM_ARRAY_TASK_ID

uname -a

deactivate
