DATA_DIR=/scratch/dfg42/WaterPaths_Sedento_reevaluation/DU_reeval/

N_REALIZATIONS=1000
START_RDM=1
END_RDM=2000


for RDM in $(seq $START_RDM 1 $END_RDM)
do
	SLURM="#!/bin/bash\n\
#SBATCH -n 1 -N 1\n\
#SBATCH --job-name=cube_dev_tables_${RDM}\n\
#SBATCH --output=output/cube_tables_${RDM}.out\n\
#SBATCH --error=output/cube_tables_${RDM}.err\n\
#SBATCH --exclusive
#SBATCH --time=1:00:00\n\
export OMP_NUM_THREADS=16\n\
cd \$SLURM_SUBMIT_DIR\n\
time ./triangleSimulation -T 16 -t 2344 -r ${N_REALIZATIONS} -d ${DATA_DIR} -C 1 -O DU_tables/rof_tables_rdm_${RDM} -s TestFiles/FB_comp.csv -U TestFiles/rdm_utilities_test_problem_reeval.csv -W TestFiles/rdm_water_sources_test_problem_reeval.csv -P TestFiles/rdm_dmp_test_problem_reeval.csv -m 0 -R ${RDM} -p false"
#echo "-T 16 -t 2344 -r ${N_REALIZATIONS} -d ${DATA_DIR} -C 1 -O DU_tables/rof_tables_rdm_${RDM}/ -s TestFiles/FB_comp.csv -U TestFiles/rdm_utilities_test_problem_reeval.csv -W TestFiles/rdm_water_sources_test_problem_reeval.csv -P TestFiles/rdm_dmp_test_problem_reeval.csv -m 0 -R ${RDM} -p false"
	echo -e $SLURM | sbatch
	sleep 0.5


	#MOD=$(( $RDM % 25))
	#if [[ $MOD -eq 0 ]]
	#then
	#	echo "Pausing"
	#	sleep 2700
	#else
		#sleep 0.5
	#fi
done

