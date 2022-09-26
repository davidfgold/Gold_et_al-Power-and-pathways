# Gold_et_al-Power-and-pathways
Code and data for Gold et al. (2022)
https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021EF002472

To recreate all figures, run the Python scripts within the code directory.
Data from Trindade et al., (2020) can be found in the data directory, all results can be found in the results directory.

To replicate the compuational experiment, follow the steps outlined below.
Note: this experiment was run using high performance computing and cannot easily be replicated on a personal computer. To accurately replicate results, the parallel master worker version of the Borg MOEA should be used, access can be requested here: http://borgmoea.org/

## Regional Defection Analysis
1. clone this repository and extrac the following compressed data
    a. regaionalDefection/src.zip
    b. regionalDefection/TestFiles/demands.zip
    c. all files within regionalDefection/TestFiles/inflows/
    d. all files within regionalDefection/TestFiles/evaporation/
2. compile waterpaths using with a gcc compiler by typing "make gcc" into the command line
3. construct ROF tables using the make_tables.sh bash script (this will need to be edited in accordance with the cluster being used, the file path must be edited as well as the number of cores, partitions etc., note that the timing may need to be changed depending on the number of available processors)
4. add the borg moea to the regionalDefection directory in a subdirectory titled "Borg"
5. compile borg using OpenMPI (as long as the cluster has openMPI this only requires "make")
6. copy the libborgms.a file to a directory titled "lib" within the regionalDefection directory
7. Run regional defection for Watertown:
    a. in regionalDefection/src/Utils/Constants.h change the number of decision variables to 11
    b. make sure line 4 of regionalDefection/src/main.cpp reads: #include "Problem/WatertownOptimization.h"
    c. make sure line 24 reads: WatertownOptimization \*problem_ptr;
    d. make sure line 240 reads: WatertownOptimization problem(n_weeks, import_export_rof_table);
    e. recompile WaterPaths by first entering "make clean" then "make borg" into the command line
    f. create an empty folder titled "output"
    h. run the regional defection optimization using the bash script (watertown.sh). Note this will have to be modified for the specific cluster in use
    i. save all output to a local machine
9. Run regional defection for Dryville:
    a. in regionalDefection/src/Utils/Constants.h change the number of decision variables to 9
    b. change line 4 of regionalDefection/src/main.cpp to read: #include "Problem/DryvilleOptimization.h"
    c. change line 24 to read: DryvilleOptimization \*problem_ptr;
    d. change line 240 to readd: DryvilleOptimization problem(n_weeks, import_export_rof_table);
    e. recompile WaterPaths by first entering "make clean" then "make borg" into the command line
    f. run the regional defection optimization using the bash script (dryville.sh). Note this will have to be modified for the specific cluster in use
    g. save all output to a local machine
11. Run regional defection for Fallsland:
    a. in regionalDefection/src/Utils/Constants.h change the number of decision variables to 9
    b. change line 4 of regionalDefection/src/main.cpp to read: #include "Problem/Fallsland.h"
    c. change line 24 to read: FallslandOptimization \*problem_ptr;
    d. change line 240 to readd: FallslandOptimization problem(n_weeks, import_export_rof_table);
    e. recompile WaterPaths by first entering "make clean" then "make borg" into the command line
    f. run the regional defection optimization using the bash script (dryville.sh). Note this will have to be modified for the specific cluster in use
    g. save all output to a local machine

## DU reevaluation
1. Extract the compressed source files from compuationalExperiment/DUReevaluation/src.zip
2. copy the TestFiles folder from the regionalDefection directory
3. upload the Pareto Approximate fronts from each regional defection optimization
4. compile WaterPaths with "make gcc"
5. use make_RDM_tables.sh to create tables for each DU SOW (note this method of parallelization may not suit all clusters and final tables will take a significant amount of memory, this may require using a python script similar to reeval.py)
6. peform DU reevaluation using the python_reevaluation.sh bash file (will need to be edited for the cluster being used)
