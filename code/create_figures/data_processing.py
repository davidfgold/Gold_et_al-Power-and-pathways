import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from copy import deepcopy
from matplotlib import cm, colors
from sklearn.ensemble import GradientBoostingClassifier



def normalize_objectives(bounds, columns, objective_vector, fmin, num_sols):
    '''
    Normalizes objectives for plotting on a parallel axis plot based on a given
    set of bounds.
    
    Params:
        bounds: a vector with upper and lower bounds [upper, lower]
        columns: columns of the objectives to be normalized
        objective_vector: vector to be normalized
        fmin: boolean for if objective is minized or maximized (true=minimized)
    Returns: 
        normalized_vector: a numpy array with normalized objective values
    '''
    print('num sols = ' + str(num_sols))
    
    normalized_vector = deepcopy(objective_vector)
    if num_sols < 2:       
        for i in columns:
            if not fmin:
                normalized_vector[i] = 1-(objective_vector[i] - \
                         bounds[1])/(bounds[0]-bounds[1])
            else:
                normalized_vector[i] = (objective_vector[i] - \
                         bounds[1])/(bounds[0]-bounds[1])
    else:
        for j in range(0,num_sols):
            for i in columns:
                if not fmin:
                    normalized_vector[j, i] = 1-(objective_vector[i] - \
                             bounds[1])/(bounds[0]-bounds[1])
                else:
                    normalized_vector[j, i] = (objective_vector[j, i] - \
                             bounds[1])/(bounds[0]-bounds[1])
    
    return normalized_vector


def load_normalize_brush(comp_name, utility_name, brush_criteria):
    """
    loads reoptimized reference set, normalizes the objectives for plotting
    on a p-axis and brushes solutions to a given set of criteria
    
    Parameters:
        comp_name: name of the compromise solution (ie "LS" for least squares)
        utility_letter: a string of the first letter of the utility name
        brush_criteria: a vector containing [rel criteria, rf criteria, 
        inf criteria and wcc criteria]
        
    Returns:
        normalized: a vector containing the reference set normalized to fit
        into a p-axis
        normalized_brush: a vector containing only solutions that meet criteria
        also normalized for a p-axis
        reoptimization: a vector with containing original objective values (not
        normalized)
    """
    REL_bounds = [0.03, 0]
    RF_bounds = [.3, 0]
    INF_bounds = [120, 0]
    PFC_bounds = [0.65, 0]
    WCC_bounds = [.2, 0]
    
    # load reference sets from optimization
    reoptimization = np.loadtxt('../../results/reoptimized_solution_sets/' \
        + comp_name + '/individual_ref_sets/' + comp_name + '_' + \
        utility_name + '_obj_filtered.csv', delimiter=',')
    
    reoptimization[:,0] = reoptimization[:,0]
    
    # find solutions that meet brushing criteria
    brush = reoptimization[(reoptimization[:,0] <= brush_criteria[0]),:]
    brush =  brush[(brush[:,1] <= brush_criteria[1]),:]
    brush =  brush[(brush[:,2] <= brush_criteria[2]),:]
    brush = brush[(brush[:,4] <= brush_criteria[3]),:]
    
    # normalize
    normalized = deepcopy(reoptimization)
    normalized_brushed = deepcopy(brush)
    
    #REL_bounds_min = [0.05, 0] 
    # rel   (minimized in this set, so I'm using bounds 1-bounds)      
    normalized = normalize_objectives(REL_bounds, [0], normalized, True, 
                                        len(normalized))
    normalized_brushed = normalize_objectives(REL_bounds, [0],
                                                normalized_brushed, True, 
                                                len(normalized_brushed))
    
    # rf
    normalized = normalize_objectives(RF_bounds, [1], normalized, True, 
                                        len(normalized))
    normalized_brushed = normalize_objectives(RF_bounds, [1],
                                                normalized_brushed, True, 
                                                len(normalized_brushed))
    
    # INF
    normalized = normalize_objectives(INF_bounds, [2], normalized, True,
                                        len(normalized))
    normalized_brushed = normalize_objectives(INF_bounds, [2],
                                                normalized_brushed, True,
                                                len(normalized_brushed))
    
    # PFC
    normalized = normalize_objectives(PFC_bounds, [3], normalized, True,
                                        len(normalized))
    normalized_brushed = normalize_objectives(PFC_bounds, [3],
                                                normalized_brushed, True, 
                                                len(normalized_brushed))
    
    # WCC
    normalized = normalize_objectives(WCC_bounds, [4], normalized, True,
                                        len(normalized))
    normalized_brushed = normalize_objectives(WCC_bounds, [4],
                                                normalized_brushed, True,
                                                len(normalized_brushed))
    
    return normalized, normalized_brushed, reoptimization


def check_rdm_meet_criteria(objectives, crit_objs, crit_vals):
    # check max and min criteria for each objective
    meet_low = objectives[:, crit_objs] >= crit_vals[0]
    meet_high = objectives[:, crit_objs] <= crit_vals[1]

    # check if max and min criteria are met at the same time
    robustness_utility_solution = np.hstack((meet_low, meet_high)).all(axis=1)

    return robustness_utility_solution


def normalize(min, max, values):
    norm = (values-min)/(max-min)
    return norm

def create_obj_df_raw(objs_by_utility):
    '''  
    
    By: DG
    Date: 6/20
    
    Returns a df of raw objective values
    
    '''
    # make arrays with inidividual utilities' objs    
    W_df = pd.DataFrame(objs_by_utility[0], columns = 
                        ['W_Rel', 'W_RF', 'W_INPV', 'W_FC', 'W_PFC'])
    D_df = pd.DataFrame(objs_by_utility[1], columns = 
                        ['D_Rel', 'D_RF', 'D_INPV', 'D_FC', 'D_PFC'])
    F_df = pd.DataFrame(objs_by_utility[2], columns = 
                        ['F_Rel', 'F_RF', 'F_INPV', 'F_FC', 'F_PFC'])
     
    objs_df = pd.concat([W_df, D_df, F_df], axis=1)
    
    return objs_df
  

def calculate_robustness(all_rdm_folder, start_rdm, rdms, num_sols, sol):
    '''
    Returns: a single array with objectives, utility rob and individual rob
    
    Calculates the robustness for each utilty for the satisficing criteria used
    in Trindade et al., (2020)
    Rel >= 98%
    RF <= 10%
    WCC <= 10%
    
    Parameters: 
        all_rdm_folder: a folder with an objective file for each RDM of a given
        solution
        
        rdms: the total number of RDM SOWs
        
        num_sols: total number of solutions in the rdm files
        
        sol: index of solution of interest
    
    Returns: 
        RDM_objectives: a np array with objective values from all SOWs
        
        rob_out: a 1x9 numpy array containing the robustness
        values for each utility and each criteria individually
        
    preconditions: all_rdm_folder contains a set of Objective files for each
                   SOW. Each file should be named 
                   "Objectives_RDM#_sols0_to_$.csv", where # is the SOW and $ 
                   is the total number of solutions
    '''
        
    # read files and load into an array
    for RDM in range(start_rdm, rdms+1):
        if RDM==start_rdm:
            RDM_objectives = np.loadtxt(all_rdm_folder + '/Objectives_RDM' + 
                                        str(RDM) + '_sols0_to_' + 
                                        str(num_sols) + '.csv', 
                                        delimiter=',', skiprows=sol,
                                                   max_rows=1)
        else:
            try:
                cur_RDM = np.loadtxt(all_rdm_folder + 
                                                       '/Objectives_RDM' + 
                                                       str(RDM) + '_sols0_to_' + 
                                                       str(num_sols) + '.csv',
                                                       delimiter=',', skiprows=sol, max_rows=1)
                RDM_objectives = np.vstack((RDM_objectives, cur_RDM))
            except:
                print('RDM ' + str(RDM) + ' does not have all sols')
                     
            
            
    
    # determine if sol meets individual criteria for each SOW
    # using bernardo's function above (produces a boolean array for each obj)
    
    # Calculate Watertown Robustness
    w_rel_rob = check_rdm_meet_criteria(RDM_objectives, [0], [0.98, 1])
    w_rf_rob = check_rdm_meet_criteria(RDM_objectives, [1], [0, 0.1])
    w_wcc_rob = check_rdm_meet_criteria(RDM_objectives, [4], [0, 0.1])
    w_all_rob = np.vstack((w_rel_rob, w_rf_rob, w_wcc_rob)).all(axis=0)
    
    # add to a single array
    w_full_rob = np.zeros(4)
    w_full_rob[0] = w_rel_rob.mean()
    w_full_rob[1] = w_rf_rob.mean()
    w_full_rob[2] = w_wcc_rob.mean()
    w_full_rob[3] = w_all_rob.mean()
    
    
    # Calculate Dryville Robustness
    d_rel_rob = check_rdm_meet_criteria(RDM_objectives, [5], [0.98, 1])
    d_rf_rob = check_rdm_meet_criteria(RDM_objectives, [6], [0, 0.1])
    d_wcc_rob = check_rdm_meet_criteria(RDM_objectives, [9], [0, 0.1])
    d_all_rob = np.vstack((d_rel_rob, d_rf_rob, d_wcc_rob)).all(axis=0)
    
    # add to a single array
    d_full_rob = np.zeros(4)
    d_full_rob[0] = d_rel_rob.mean()
    d_full_rob[1] = d_rf_rob.mean()
    d_full_rob[2] = d_wcc_rob.mean()
    d_full_rob[3] = d_all_rob.mean()
    
    # Calculate Fallsland Robustness
    f_rel_rob = check_rdm_meet_criteria(RDM_objectives, [10], [0.98, 1])
    f_rf_rob = check_rdm_meet_criteria(RDM_objectives, [11], [0, 0.1])
    f_wcc_rob = check_rdm_meet_criteria(RDM_objectives, [14], [0, 0.1])
    f_all_rob = np.vstack((f_rel_rob, f_rf_rob, f_wcc_rob)).all(axis=0)
    
    f_full_rob = np.zeros(4)
    f_full_rob[0] = f_rel_rob.mean()
    f_full_rob[1] = f_rf_rob.mean()
    f_full_rob[2] = f_wcc_rob.mean()
    f_full_rob[3] = f_all_rob.mean()
    
   # add all three to a single array
    rob_out = np.hstack((w_full_rob, d_full_rob, f_full_rob))

    return RDM_objectives, rob_out


def create_sd_input(RDM_objectives, criteria):
    '''
    Create a boolean array of whether each SOW meets satisfying criteria 
    
    Rel >= 98%
    RF <= 10%
    WCC <= 10%
    
    Parameters: 
        RDM_objectives: an array with objective values across SOWs for a given
        solution (created by calculate_robustness function). Each row is a SOW
        and each column is an objective
                
        criteria = satisficing criteria [lower, upper]. Defaults to
        [[0.98, 1], [0,]]
    
    returns:
        satisficing, a boolean array containing meets/fails for each 
        robustness criteria. Columns = [WRel, WRF, WWCC, DRel, DRF, DWCC, FRel,
        FRF, FWCC]
    '''
    w_rel = check_rdm_meet_criteria(RDM_objectives, [0], [0.98, 1])
    w_rf = check_rdm_meet_criteria(RDM_objectives, [1], [0, 0.1])
    w_wcc = check_rdm_meet_criteria(RDM_objectives, [4], [0, 0.1])
    w_all = np.zeros(1000)
    for i in range(0,len(w_rel)):
        w_all[i] = (w_rel[i] and w_rf[i] and w_wcc[i])*1
    
    d_rel = check_rdm_meet_criteria(RDM_objectives, [5], [0.98, 1])
    d_rf = check_rdm_meet_criteria(RDM_objectives, [6], [0, 0.1])
    d_wcc = check_rdm_meet_criteria(RDM_objectives, [9], [0, 0.1])
    d_all = np.zeros(1000)
    for i in range(0,len(d_rel)):
        d_all[i] = (d_rel[i] and d_rf[i] and d_wcc[i])*1
    
    f_rel = check_rdm_meet_criteria(RDM_objectives, [10], [0.98, 1])
    f_rf = check_rdm_meet_criteria(RDM_objectives, [11], [0, 0.1])
    f_wcc = check_rdm_meet_criteria(RDM_objectives, [14], [0, 0.1])
    f_all = np.zeros(1000)
    for i in range(0,len(f_rel)):
        f_all[i] = (f_rel[i] and f_rf[i] and f_wcc[i])*1
    
    indiv_satisficing = [w_rel*1, w_rf*1, w_wcc*1, w_all, d_rel*1, d_rf*1, \
                         d_wcc*1, d_all, f_rel*1, f_rf*1, f_wcc*1, f_all]
    
    return indiv_satisficing



def boosted_tree_sd(satisficing, rdm_factors, n_trees, tree_depth):
    '''
    Performs boosted trees scenario discovery for a given satisficing criteria
    
    Parameters:
        satisficing: a boolean array containing whether each SOW met
        criteria for a given solution
        
        rdm_factors: an array with rdm factors comprising each SOW
        
        n_trees: number of trees for boosting
        
        tree_depth: max depth of trees
        
    returns:
        gbc: fit classifier on all data
        
        gbc_2factors: classifier fit to only the top two factors for plotting
    
    '''
    
    gbc = GradientBoostingClassifier(n_estimators=n_trees,
                                            learning_rate=0.1,
                                            max_depth=tree_depth)
    
    gbc.fit(rdm_factors, satisficing[0]*1)
    print('Boosted Trees score: {}'.format(gbc.score(rdm_factors, satisficing[0]*1)))
    
    feature_importances = deepcopy(gbc.feature_importances_)
    most_influential_factors = np.argsort(feature_importances)[::-1]
    
    
    gbc_2factors = GradientBoostingClassifier(n_estimators=n_trees,
                                            learning_rate=0.1,
                                            max_depth=tree_depth)
    most_important_rdm_factors = rdm_factors[:, most_influential_factors[:2]]
    gbc_2factors.fit(most_important_rdm_factors, satisficing[0]*1)
    
    return gbc, gbc_2factors, most_important_rdm_factors



def filter_criteria(refSetFile, criteria_cols, criteria):
    """
        A function to filter a solution set by a given criteria
        
        inputs
            refSetFile: the location of the reference set of reoptomized 
            solutions. First three lines describe problem
            
            criteria_cols: the column indicies of the filter criteria
            
            criteria: thresholds to be filtered (assumes minimization)
            
        returns
            filtered_ref_set: an array containing only solutions that meet filter criteria
    """
    reoptimizedRefSet = np.loadtxt(open(refSetFile, 'rt').readlines()[:-1],
                                   skiprows=3)
    # first criteria from reoptomized
    filtered_ref_set = reoptimizedRefSet[(reoptimizedRefSet[:, criteria_cols[0]] <=criteria[0]),:]
    
    # remaining criteria recursively
    for i in range(1, len(criteria_cols)):
        filtered_ref_set = filtered_ref_set[(filtered_ref_set[:, criteria_cols[i]] <=criteria[i]),:]

    return filtered_ref_set


def calc_obj_regret(original_obj, reoptimized_obj, utility, optimized_utility):
    '''
    Calculates the regret for each objective
    
    Params 
        original_obj: trindade et al optimized solution (np array)
        reoptomized_obj: objectives from reoptimization (np array)
        utility: the utility regret is being calculated for (str)
        optimized_utility: binary, True if utility is the one reoptimized for        
    returns:
        regret: a vector of regret, each element is an objective (np array)
        regret is normalized by the original objective value
    '''
    criteria = [.02, .1,  50, .5, .1]
    # extract objectives for utility of interest
    reoptimized_individual_obj = np.zeros([len(reoptimized_obj), 5])
    if utility=='Watertown':
        for i in range(0,5):
            reoptimized_individual_obj[:,i] = reoptimized_obj[:, i]
    
    elif utility == 'Dryville':
        for i in range(0,5):
            reoptimized_individual_obj[:,i] = reoptimized_obj[:,i+5]
    elif utility == 'Fallsland':
        for i in range(0,5):
            reoptimized_individual_obj[:,i] = reoptimized_obj[:,i+10]
    else:
        print('Error, utility not listed')
        return 
    # calculate regret
    regret = np.zeros(5)
    
    # if this is the optimized utility, calculate benefit
    if optimized_utility:
        # reliability (maximized)
        regret[0] = -(max(reoptimized_individual_obj[:,0]) - 
              original_obj[0]) / criteria[0]
        # all other obj are maximized (RF, NPC, PFC, WCC)
        for i in range(1,5):
            regret[i] = -(original_obj[i] -
                  min(reoptimized_individual_obj[:,i])) / criteria[i]
        
    # if not optimized utility, calculate loss    
    else:
        # reliability (maximized)
        regret[0] = (original_obj[0] - 
              min(reoptimized_individual_obj[:,0])) / criteria[0]
        # all other obj are maximized (RF, NPC, PFC, WCC)
        for i in range(1,5):
            regret[i] = (max(reoptimized_individual_obj[:,i]) - 
                  original_obj[i]) / criteria[i]
        
    return regret*100


def calc_rob_regret(original_rob, reoptimized_rob, utility, defecting_utility):
    '''
    Returns robustness space regret for a utility
    
    Params 
        original_rob: robustness of original compromise (np array)
        reoptimized_rob: robustness of defection sols (np array)
        utility: the utility regret is being calculated for (str)
        defecting_utility: binary, True if utility is the one reoptimized for        
    Returns:
        regret: a vector of regret, each element is an objective (np array)
    '''
            
    # extract objectives for utility of interest
    reoptimized_individual_rob = np.zeros([len(reoptimized_rob), 5])
    if utility=='Watertown':
        utility_original_all_rob = original_rob[4]
        utility_original = original_rob[1:5]
        for i in range(1,5):
            reoptimized_individual_rob[:,i] = reoptimized_rob[:, i]
        defection_choices = reoptimized_individual_rob[reoptimized_individual_rob[:,4] >= utility_original_all_rob]
        #defection_choices = defection_choices[:,1:5]
    
    elif utility == 'Dryville':
        utility_original_all_rob = original_rob[8]
        utility_original = original_rob[5:9]
        for i in range(1,5):
            reoptimized_individual_rob[:,i] = reoptimized_rob[:,i+4]
        defection_choices = reoptimized_individual_rob[reoptimized_individual_rob[:,4] >= utility_original_all_rob]
        #defection_choices = defection_choices[:,5:9]

    elif utility == 'Fallsland':
        utility_original_all_rob = original_rob[12]
        utility_original = original_rob[9:]
        for i in range(1,5):
            reoptimized_individual_rob[:,i] = reoptimized_rob[:,i+8]
        defection_choices = reoptimized_individual_rob[reoptimized_individual_rob[:,4] >= utility_original_all_rob]
        #defection_choices = defection_choices[:,9:]

    else:
        print('Error, utility not listed')
        return 
    # calculate regret
    regret = np.zeros(4)
    # if this is the optimized utility, calculate benefit
    if defecting_utility:
        # select solutions that outperform original     
        if len(defection_choices) > 0:
            for i in range(0,4):
                print('Defecting utility:')
                print(max(defection_choices[:,i+1]))
                print(utility_original[i])
                regret[i] = max(defection_choices[:,i+1]) - utility_original[i]
                print(regret[i])
        else:
            print(utility + ' does not gain in robustness!')
            for i in range(0,4):
                print('Defecting utility:')
                print(max(reoptimized_individual_rob[:,i+1]))
                print(utility_original[i])
                regret[i] = max(reoptimized_individual_rob[:,i+1]) - utility_original[i]
                print(regret[i])

    # if not optimized utility, calculate loss    
    else:
        for i in range(0,4):
            print('Cooperative utility:')
            print(min(reoptimized_individual_rob[:,i+1]))
            print(utility_original[i])    
            regret[i] = min(reoptimized_individual_rob[:,i+1]) - utility_original[i] 
            print(regret[i])
        
    return regret
