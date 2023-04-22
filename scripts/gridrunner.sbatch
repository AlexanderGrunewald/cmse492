#!/bin/bash
#
#SBATCH --qos=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-999
#SBATCH --job-name tardis_TNS_train 
#SBATCH --output=/mnt/home/grunew14/Documents/project/logs/tardis_tns_train_%A_%a.out 
#SBATCH --error=/mnt/home/grunew14/Documents/project/logs/tardis_tns_train_%A_%a.err 
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=[grunew14@msu.edu](mailto:grunew14@msu.edu)
date;hostname;pwd

NUMSIMS=10
START=$((SLURM_ARRAY_TASK_ID*NUMSIMS))
STOP=$((START+NUMSIMS))
echo "START=$START"
echo "STOP=$STOP"

for (( N = $START; N <= $STOP; N++ ))
do
    echo "RUNNING INDEX $N"
        python script.py "$N"
        rm core*
done
date;hostname;pwd