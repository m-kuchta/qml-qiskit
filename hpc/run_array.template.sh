#!/bin/bash
#SBATCH --job-name=<YOUR_JOB_NAME>
#SBATCH --partition=<PARTITION_NAME>
#SBATCH --account=<YOUR_ACCOUNT_GRANT>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00            
#SBATCH --array=1-10             
#SBATCH --output=<PATH_TO_LOGS>/logs/out/job_%A_%a.out 
#SBATCH --error=<PATH_TO_LOGS>/logs/error/job_%A_%a.err

cd "${SLURM_SUBMIT_DIR}" || exit 1

JOBS_FILE="hpc/jobs_array.txt"
if [ ! -f "${JOBS_FILE}" ]; then
    echo "ERROR: ${JOBS_FILE} not found in $(pwd)" >&2
    exit 1
fi

source /path/to/.venv/bin/activate

COMMAND=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${JOBS_FILE}")

echo "$COMMAND"
eval "$COMMAND"

echo "Task $SLURM_ARRAY_TASK_ID finished"