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
#SBATCH --output=logs/job_%A_%a.out 
#SBATCH --error=logs/job_%A_%a.err

source /path/to/.venv/bin/activate

cd $SLURM_SUBMIT_DIR

COMMAND=$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs_array.txt)

echo "$COMMAND"
eval $COMMAND

echo "Task $SLURM_ARRAY_TASK_ID finished"