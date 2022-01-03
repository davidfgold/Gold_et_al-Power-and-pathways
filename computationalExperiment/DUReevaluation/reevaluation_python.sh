DATA_DIR="/scratch/dfg42/WaterPaths_Sedento_reevaluation/DU_reeval/"
N_REALIZATIONS=1000
SOLS_FILE_NAME="compSol_DVs.csv"
RDM_PER_JOB=2
N_NODES=2
RDM=1
#for RDM in $(seq 0 $RDM_PER_JOB 4)
#do
SLURM="#!/bin/bash\n\
#SBATCH -n $N_NODES -N $N_NODES\n\
#SBATCH --time=4:00:00\n\
#SBATCH --job-name=python_reeval_${RDM}_to_$(($RDM+$RDM_PER_JOB-1))\n\
#SBATCH --output=output/python_reeval_${RDM}_to_$(($RDM+$RDM_PER_JOB-1)).out\n\
#SBATCH --error=output/python_reeval_${RDM}_to_$(($RDM+$RDM_PER_JOB-1)).err\n\
#SBATCH --exclusive\n\
export OMP_NUM_THREADS=16\n\
module load python/3.6.9\n
time mpirun -np $N_NODES python3 reeval.py \$OMP_NUM_THREADS $N_REALIZATIONS $DATA_DIR $RDM $SOLS_FILE_NAME $N_NODES $RDM_PER_JOB"
echo -e $SLURM | sbatch
sleep 0.5
#donemod
