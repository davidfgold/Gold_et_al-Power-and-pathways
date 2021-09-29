# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 10:00:00 2021

@author: dgold
"""

from sklearn.ensemble import GradientBoostingClassifier
from copy import deepcopy
from data_processing import *
import numpy as np

def boosted_tree_sd(satisficing, rdm_factors, n_trees, tree_depth, crit_idx):
    '''
    Performs boosted trees scenario discovery for a given satisficing criteria
    
    inputs:
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
    
    gbc.fit(rdm_factors, satisficing[crit_idx])
    #print('Boosted Trees score: {}'.format(gbc.score(rdm_factors[:], satisficing[crit_idx]*1)))
    
    feature_importances = deepcopy(gbc.feature_importances_)
    most_influential_factors = np.argsort(feature_importances)[::-1]
    
    
    gbc_2factors = GradientBoostingClassifier(n_estimators=n_trees,
                                            learning_rate=0.1,
                                            max_depth=tree_depth)
    
    
    most_important_rdm_factors = rdm_factors[:, most_influential_factors[:2]]
    gbc_2factors.fit(most_important_rdm_factors, satisficing[crit_idx])
    
    return gbc, gbc_2factors, most_important_rdm_factors, feature_importances

def perform_and_plot_SD(defection_sol, defecting_util, utility, comp, \
                        num_defections, ax):
    '''
    Performs gbc classification and plots a factor map
    
    inputs:
        defection_sol: a boolean, true if analyzing a defection scenario
        defecting_util: string, name of the utility defecting (all lowercase)
        utility: string, name of the utilty (all lowercase)
        comp: string, either LS_comp_98 or PW_comp_113
        num_defections: number of dection solutions if applicable
        ax: the axis object to be plotted on
    '''
    
    # if this is a defection scenario, choose either the max or min robustness
    if defection_sol:
        # load robustness of defection solutions
        defection_robustness = np.loadtxt('../../results/' + \
                   'reoptimized_solution_sets/' + comp + '/' + \
                   'du_reevaluation/'  + defecting_util + '/' + \
                   defecting_util + '_all_robustness.csv', delimiter=',')
        
        if defecting_util == utility:
            # select the maximum robustness that results from defection
            if utility == 'watertown':
                selected_solution = np.argmax(defection_robustness[:,4])
            elif utility == 'dryville':
                selected_solution = np.argmax(defection_robustness[:,8])
            else:
                selected_solution = np.argmax(defection_robustness[:,12])
            
        else:
            # select the minimum robustness that results from defection
            if utility == 'watertown':
                selected_solution = np.argmin(defection_robustness[:,4])
            elif utility == 'dryville':
                selected_solution = np.argmin(defection_robustness[:,8])
            else:
                selected_solution = np.argmin(defection_robustness[:,12])
    
        RDM_objectives, indiv_robustness = calculate_robustness('../' + \
            '../results/du_reevaluation_data/'+ comp + '/' + defecting_util  + \
            '/Objs_by_SOW', 1, 1000, num_defections , selected_solution)
        
        
    # otherwise load the objectives from the original compromise
    else:
        if comp == 'LS_comp_98':
            selected_solution = 98
        else:
            selected_solution = 113
        RDM_objectives, indiv_robustness = calculate_robustness('../' + \
    '../data/allSols_DU_reeval/Obj_by_SOW', 1, 1000, 229, selected_solution)


    # load rdm files
    utility_rdm = np.loadtxt('../../data/RDM_SOWs/rdm_utilities_test_problem_reeval_2021.csv', delimiter=',')
    dmp_rdm = np.loadtxt('../../data/RDM_SOWs/rdm_dmp_test_problem_reeval_2021_noise.csv', delimiter=',')
    water_sources_rdm = np.loadtxt('../../data/RDM_SOWs/rdm_water_sources_test_problem_reeval_2021.csv', delimiter=',')
    inflows_rdm =np.loadtxt('../../data/RDM_SOWs/rdm_inflows_test_problem_reeval_2021.csv', delimiter=',')
    
    rdm_factors = np.hstack((utility_rdm, dmp_rdm, water_sources_rdm, inflows_rdm))
    #rdm_factors = rdm_factors[1:1001]
    
    # if defection, remove broken rdm 924
    #if defection_sol and defecting_util == utility:
    #    rdm_factors = np.delete(rdm_factors,923,0)
    
    
    SD_input = create_sd_input(RDM_objectives,[])
    
    if utility == 'watertown':
                rob_idx = 3
    elif utility == 'dryville':
        rob_idx = 7
    else:
        rob_idx = 11
    
    rdm_names = ['Demand growth multiplier', 'Bond interest rate multiplier',\
                 'Bond term multiplier', 'Discount rate multiplier', \
                 'Restriction effectiveness \nmultiplier Watertown', \
                 'Restriction effectiveness \nmultiplier Dryville',
                 'Restriction effectiveness \nmultiplier Fallsland', \
                'NRR permitting time multiplier', 'NRR construction time multiplier', \
                'SC permitting time multiplier', 'SC construction time multiplier', \
                'CRR Low expansion \npermitting time multiplier', \
                'CRR Low expansion \nconstruction time multiplier',\
                'CRR High expansion \npermitting time multiplier',\
                'CRR High expansion \nconstruction time multiplier', \
                'Watertown Resuse I \npermitting time multiplier', \
                'Watertown Reuse I \nconstruction time multiplier', \
                'Watertown Resuse II \npermitting time multiplier', \
                'Watertown Reuse II \nconstruction time multiplier', \
                'Dryville Reuse \npermitting time multiplier', \
                'Dryville Reuse \nconstruction time multiplier', \
                'Fallsland Reuse \npermitting time multiplier', \
                'Watertown Reuse \nconstruction time multiplier', \
                'Inflow Amplitude', 'Inflow frequency', 'Inflow phase']

    if indiv_robustness[rob_idx] < 0.98:
        gbc, gbc_2factors, most_important_rdm_factors, \
        feature_importances = boosted_tree_sd(SD_input, rdm_factors, 500, 4, \
                                              rob_idx)
        
        # predict across 2D features space
        gbc_2factors.fit(most_important_rdm_factors, SD_input[rob_idx])
           
        # plot prediction contours
        x_data = most_important_rdm_factors[:,0]
        y_data = most_important_rdm_factors[:,1]
        
        x_min, x_max = (x_data.min(), x_data.max())
        y_min, y_max = (y_data.min(), y_data.max())
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max * 1.001, (x_max - x_min) / 100),
                             np.arange(y_min, y_max * 1.001, (y_max - y_min) / 100))
                             
        dummy_points = list(zip(xx.ravel(), yy.ravel()))
        
        z = gbc_2factors.predict_proba(dummy_points)[:, 1]
        z[z < 0] = 0.
        z = z.reshape(xx.shape)
              
        
        ax.contourf(xx, yy, z, [0, 0.5, 0.98, 1.], cmap='RdBu',
                    alpha=.3, vmin=0., vmax=1.)
        ax.scatter(most_important_rdm_factors[:,0], most_important_rdm_factors[:,1],\
                   c=SD_input[rob_idx], cmap='Reds_r', edgecolor='grey', 
                   alpha=.6, s= 25)
        xlabel = rdm_names[np.argsort(feature_importances)[::-1][0]] + \
        ' (' + str(int(feature_importances[np.argsort(feature_importances)[::-1][0]]*100)) + '%)'
        ylabel = rdm_names[np.argsort(feature_importances)[::-1][1]]  + \
        ' (' + str(int(feature_importances[np.argsort(feature_importances)[::-1][1]]*100)) + '%)'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim([min(most_important_rdm_factors[:,0]), max(most_important_rdm_factors[:,0])])
        ax.set_ylim([min(most_important_rdm_factors[:,1]), max(most_important_rdm_factors[:,1])])

    
    # if robustness is very high (>98%), just plot a contour map of uniform color
    else:
        # plot prediction contours
        x_data = rdm_factors[:,0]
        y_data = rdm_factors[:,4]
        
        x_min, x_max = (x_data.min(), x_data.max())
        y_min, y_max = (y_data.min(), y_data.max())
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max * 1.001, (x_max - x_min) / 100),
                             np.arange(y_min, y_max * 1.001, (y_max - y_min) / 100))
                             
        dummy_points = list(zip(xx.ravel(), yy.ravel()))
        
        z = np.ones(len(xx)**2)
        z = z.reshape(xx.shape)
        ax.contourf(xx, yy, z, [0, 0.5, 0.98, 1.], cmap='RdBu',
                    alpha=.3, vmin=0., vmax=1.)
    
        ax.scatter(rdm_factors[:,0], rdm_factors[:,4], c=SD_input[rob_idx], \
                   cmap='Reds_r', edgecolor='grey', alpha=.6, s=25)
        ax.set_xlabel(rdm_names[0])
        ax.set_ylabel(rdm_names[4])
        ax.set_xlim([min(rdm_factors[:,0]), max(rdm_factors[:,0])])        
        ax.set_ylim([min(rdm_factors[:,4]), max(rdm_factors[:,4])]) 
