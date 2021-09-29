"""
Created on Wed Jan  6 17:27:56 2021

This script will create figure 4 of Gold et al., 2021
The figure shows parallel axes plots which compare the results from defection
to the original compromise objectives
"""

from data_processing import *
from matplotlib import cm
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
import seaborn as sns
sns.set(style='white')

def load_and_format_optimization_output(comp_name):
    # load compromise values from original solution
    original_objectives = np.loadtxt('../../data/selectedSols_objs/' \
        + comp_name + '.csv', delimiter=',')
    
    
    # normalize objectives so all axes plot values range from 0-1
    REL_bounds = [1, .97]
    RF_bounds = [.3, 0]
    INF_bounds = [120, 0]
    PFC_bounds = [0.65, 0]
    WCC_bounds = [.2, 0]
    
    normalized_objectives = deepcopy(original_objectives)
      
    # normalize reliability (use function from data_analysis) (columns 0, 5, 10)                   
    normalized_objectives = normalize_objectives(REL_bounds, [0, 5, 10], 
                                            normalized_objectives, False,1)
    # normalize rf (columns 1, 6 and 11)
    normalized_objectives = normalize_objectives(RF_bounds, [1, 6, 11], 
                                            normalized_objectives, True, 1)
    # normalize INF_NPC (columns 2, 7, 12)
    normalized_objectives = normalize_objectives(INF_bounds, [2, 7, 12], 
                                            normalized_objectives, True, 1)
    # normalize PFC (columns 3, 8, 13)
    normalized_objectives = normalize_objectives(PFC_bounds, [3, 8, 13], 
                                            normalized_objectives, True, 1)
    # normalize WCC (columns 4, 9, 14)
    normalized_objectives = normalize_objectives(WCC_bounds, [4, 9, 14], 
                                                 normalized_objectives, True, 1)
                                                 
    
    Watertown_obj = normalized_objectives[0:5]
    Dryville_obj = normalized_objectives[5:10]
    Fallsland_obj = normalized_objectives[10:15]
    
    
    # load, brush and normalize reoptimization sets
    W_plot, W_plot_brushed, W_reoptimized = load_normalize_brush(comp_name, 'watertown',
                                                                         [.03, .2,
                                                                          200, .1])
    D_plot, D_plot_brushed, D_reoptimized = load_normalize_brush(comp_name, 'dryville',
                                                                         [.03, .2, 
                                                                          10, .1])
    F_plot, F_plot_brushed, F_reoptimized = load_normalize_brush(comp_name, 'fallsland',
                                                                         [.03, .2, 
                                                                          10, .1])
                
    return Watertown_obj, Dryville_obj, Fallsland_obj, W_plot_brushed, D_plot_brushed, F_plot_brushed


def plot_utility_paxis(original_obj, reoptimized_brushed_obj, ax, name, color):
    """
    Plots a parallel axis subplot comparing original solution and reoptimization
    
    Parameters:
        original_obj: the objective values for the original solution
        reoptimized_brushed_obj: objectives from reoptimizatoin, brushed
        ax: an axes object
        name: string with the name of the utility to be plotted
        color: hex id of color for each compromise
    """
    
    objectives_names_mini = ['REL', 'RF', 'INF', 'PFC', 'WFPC']
    y_parallel_axes = [0, 1, 1, 0, 0 , 1, 1,0, 0,1]
    x_parallel_axes = [0,0, 1, 1, 2, 2, 3, 3, 4, 4]
    ax.plot(x_parallel_axes, y_parallel_axes, linewidth=1.5, c='slategrey')
    
    # plot the reevaluated solutions
    for i in range(len(reoptimized_brushed_obj[:,0])):
        ys = (reoptimized_brushed_obj[i,:])
        xs = range(len(ys))
        ax.plot(xs, ys, c='#ffd166', alpha=.8, linewidth=2)

    # plot the original solution
    ys = (original_obj)
    xs = range(len(ys))
    ax.plot(xs, ys, c=color, alpha=1, linewidth=3)
            
    # plot the reevaluated solutions
    #ys = (reoptimized_brushed_obj[3,:])
    #xs = range(len(ys))
    #ax.plot(xs, ys, c='orange', alpha=1, linewidth=3)        
            
    
    #ax.set_ylabel("Objective Value \n $\longleftarrow$Preference ", size= 14)
    ax.set_ylim([0,1])
    ax.set_yticklabels('')
    ax.set_xticks([0,1,2,3,4])
    ax.set_xticklabels([objectives_names_mini[i] for i in range(0, len(objectives_names_mini))], fontsize=12)
    ax.set_xlim([0,4])
    ax.set_title(name+ ' Reoptimization', fontsize=14)


# LS comp 98
LS_W_original_obj, LS_D_original_obj, LS_F_original_obj, LS_W_plot_brushed, LS_D_plot_brushed, LS_F_plot_brushed = load_and_format_optimization_output('LS_comp_98')

# plot figure 4
objectives_names_mini = ['REL', 'RF', 'INF', 'PFC', 'WCC']
fig = plt.figure(figsize=(12,6)) # create the figure

ax1 = fig.add_subplot(3, 2, 1)   
ax2 = fig.add_subplot(3, 2, 3)   
ax3 = fig.add_subplot(3, 2, 5) 
plot_utility_paxis(LS_W_original_obj, LS_W_plot_brushed, ax1, "Watertown", '#073b4c')
plot_utility_paxis(LS_D_original_obj, LS_D_plot_brushed, ax2, "Dryville",'#073b4c')
plot_utility_paxis(LS_F_original_obj, LS_F_plot_brushed, ax3, "Fallsland", '#073b4c')
  

# PW comp 113
PW_W_original_obj, PW_D_original_obj, PW_F_original_obj, PW_W_plot_brushed, PW_D_plot_brushed, PW_F_plot_brushed = load_and_format_optimization_output('PW_comp_113')

                   
ax1 = fig.add_subplot(3, 2, 2)   
ax2 = fig.add_subplot(3, 2, 4)   
ax3 = fig.add_subplot(3, 2, 6) 
plot_utility_paxis(PW_W_original_obj, PW_W_plot_brushed, ax1, "Watertown", '#ef476f')
plot_utility_paxis(PW_D_original_obj, PW_D_plot_brushed, ax2, "Dryville",'#ef476f')
plot_utility_paxis(PW_F_original_obj, PW_F_plot_brushed, ax3, "Fallsland", '#ef476f')
                   
plt.tight_layout()
plt.savefig('Figure-5.png', bbox_inches='tight')

