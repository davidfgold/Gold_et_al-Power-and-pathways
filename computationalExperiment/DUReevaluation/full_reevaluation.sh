#!/bin/bash
DATA_DIR=/scratch/dfg42/WaterPaths_Sedento_reevaluation/DU_reeval/
N_REALIZATIONS=1000

for REL in {1..1}
do
SLURM="#!/bin/bash\n\
#SBATCH -n 1 -N 1\n\
#SBATCH --job-name=FB_reeval_R${REL}\n\
#SBATCH --output=output/FB_reeval_R${REL}.out\n\
#SBATCH --error=output/FB_reeval_R${REL}.err\n\
#SBATCH --exclusive\n\
#SBATCH --time=4:00:00\n\
export OMP_NUM_THREADS=16\n\
cd \$SLURM_SUBMIT_DIR\n\
time ./triangleSimulation -T 16 -t 2344 -r ${N_REALIZATIONS} -d ${DATA_DIR} -C -1 -O DU_tables/rof_tables_rdm_${REL}/ -s TestFiles/FB_comp.csv -U TestFiles/rdm_utilities_test_problem_reeval.csv -W TestFiles/rdm_water_sources_test_problem_reeval.csv -P TestFiles/rdm_dmp_test_problem_reeval.csv -f 0 -l 1 -R ${REL} -p false"

echo triangleSimulation -T 16 -t 2344 -r ${N_REALIZATIONS} -d ${DATA_DIR} -C -1 -O DU_tables/rof_tables_rdm_${REL}/ -s TestFiles/FB_comp.csv -U TestFiles/rdm_utilities_test_problem_reeval.csv -W TestFiles/rdm_water_sources_test_problem_reeval.csv -P TestFiles/rdm_dmp_test_problem_reeval.csv -f 0 -l 1 -R ${REL} -p false	
#echo -e $SLURM | sbatch
#echo $REL
MOD=$(( $REL % 20))
if [[ $MOD -eq 0 ]]
then
	echo "Pausing"
	sleep 510
else
	sleep 0.5
fi
done

