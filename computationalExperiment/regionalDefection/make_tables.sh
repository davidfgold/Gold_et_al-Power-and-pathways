DATA_DIR=/work/04528/tg838274/stampede2/Sedento_Valley_Coalition/Optimizaton_files/WaterPaths/
N_REALIZATIONS=1000


SLURM="#!/bin/bash\n\
#SBATCH -n 1 -N 1\n\
#SBATCH --job-name=skx_dev_tables\n\
#SBATCH --output=output/tables.out\n\
#SBATCH --error=output/tables.err\n\
#SBATCH -p skx-dev\n\
#SBATCH --time=2:00:00\n\
#SBATCH --mail-user=dgoldri25@gmail.com\n\
#SBATCH --mail-type=all\n\
export OMP_NUM_THREADS=96\n\
cd \$SLURM_SUBMIT_DIR\n\
time ./triangleSimulation -T \${OMP_NUM_THREADS} -t 2344 -r ${N_REALIZATIONS} -d ${DATA_DIR} -C 1 -s compSol_DVs.csv -U TestFiles/rdm_utilities_test_problem_opt.csv -W TestFiles/rdm_water_sources_test_problem_opt.csv -P TestFiles/rdm_dmp_test_problem_opt.csv"
	
echo -e $SLURM | sbatch
sleep 0.5


