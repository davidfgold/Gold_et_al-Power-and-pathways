DATA_DIR=/scratch/dfg42/WaterPaths_Sedento_reevaluation/RegionalDefection/
N_REALIZATIONS=1000
OMP_NUM_THREADS=16

SLURM="#!/bin/bash\n\
#SBATCH -n 4 -N 4\n\
#SBATCH --job-name=dry_test\n\
#SBATCH --output=FBCompOutput/dry_test.out\n\
#SBATCH --error=FBCompOutput/dry_test.err\n\
#SBATCH --time=12:00:00\n\
#SBATCH --mail-user=dgoldri25@gmail.com\n\
#SBATCH --mail-type=all\n\
#SBATCH --exclusive
export OMP_NUM_THREADS=16\n\
cd \$SLURM_SUBMIT_DIR\n\
mpirun -np 4 ./triangleSimulation -T \${OMP_NUM_THREADS} -t 2344 -r ${N_REALIZATIONS} -d ${DATA_DIR} -C -1 -O rof_tables/ -e 0 -U TestFiles/rdm_utilities_test_problem_opt.csv -W TestFiles/rdm_water_sources_test_problem_opt.csv -P TestFiles/rdm_dmp_test_problem_opt.csv -b true -o 100 -n 1000"
	
echo -e $SLURM | sbatch
sleep 0.5


