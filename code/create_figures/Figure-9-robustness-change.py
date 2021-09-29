# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:06:12 2021

@author: dgold
"""
import numpy as np
from matplotlib import pyplot as plt
from data_processing import calc_rob_regret


def plot_points_and_arrows(utility_idx, obj, obj_y_vals, original_LS, \
                           original_PW, LS_regret, PW_regret, ax):
    '''
    Plots original robustness value, defection robustness value and regret
    
    Parameters:
        -utility_idx: reprents the first column robustness values are stored
        in the robustness files for each utility (Watertown = 0, Dryville=4
        and Fallsland=8)
        -obj: the objective being plotted (REL=0, RF=1, WFPC=2, ALL=3)
        -obj_y_values: a list of y coordinates for each objective
        -original_LS: the orginal robustness of LS comp
        -original_PW: the original robustness of PW comp
        -LS_regret: the calculated regret for LS comp
        -PW_regret: the calculated regret for PW comp
        -ax: axes object to be plotted on
    '''
    
    # LS points and arrow
    # plot points
    ax.scatter(original_LS[utility_idx+obj+1], obj_y_vals[obj][0], \
               edgecolor='#073b4c', color='white', s =50, zorder=2)
    ax.scatter((original_LS[utility_idx+obj+1]+LS_regret[obj]), \
               obj_y_vals[obj][0], edgecolor='#073b4c', color='#073b4c', s =50, \
               zorder=2)
    
    # if regret is large enough draw arrow  
    line_x = [original_LS[utility_idx+obj+1], \
                  original_LS[utility_idx+obj+1]+LS_regret[obj]]
    line_y = [obj_y_vals[obj][0], obj_y_vals[obj][0]]
        
    if LS_regret[obj] < 0:
        ax.plot(line_x, line_y, color='#b56576', linestyle = '--', linewidth=3, zorder=0)
    else:
        ax.plot(line_x, line_y, color='#577590', linewidth=3, zorder=0)
    
    '''
    if abs(LS_regret[obj])>.075:
        if LS_regret[obj] < 0:
            ax.arrow(original_LS[utility_idx+obj+1], obj_y_vals[obj][0], LS_regret[obj]+.05, 0, \
              head_width = 0.20, head_length=0.03, linewidth=3, color='grey')
        else:
            ax.arrow(original_LS[utility_idx+obj+1], obj_y_vals[obj][0], LS_regret[obj]-.05, 0, \
              head_width = 0.20, head_length=0.03, linewidth=3, color='black')
    
    else:
        line_x = [original_LS[utility_idx+obj+1], \
                  original_LS[utility_idx+obj+1]+LS_regret[obj]]
        line_y = [obj_y_vals[obj][0], obj_y_vals[obj][0]]
        
        
        ax.plot(line_x, line_y, color='#ef476f', linewidth=2, zorder=0)
    '''
    # PW points and arrow
    # plot points
    ax.scatter(original_PW[utility_idx+obj+1], obj_y_vals[obj][1], \
               edgecolor='#ef476f', color='white', s =50, zorder=2)
    ax.scatter((original_PW[utility_idx+obj+1]+PW_regret[obj]), \
               obj_y_vals[obj][1], edgecolor='#ef476f', color='#ef476f', s =50, zorder=2)
    
    # draw arrow 
    # if regret is large enough draw arrow  
    line_x = [original_PW[utility_idx+obj+1], \
                  original_PW[utility_idx+obj+1]+PW_regret[obj]]
    line_y = [obj_y_vals[obj][1], obj_y_vals[obj][1]]
    if PW_regret[obj] < 0:
        ax.plot(line_x, line_y, color='#b56576',linestyle='--', linewidth=3, zorder=0)
    else:
        ax.plot(line_x, line_y, color='#577590', linewidth=3, zorder=0)
                    
    
    '''
    if abs(PW_regret[obj])>.075:
        if PW_regret[obj] < 0:
            
           ax.arrow(original_PW[utility_idx+obj+1], obj_y_vals[obj][1], \
             PW_regret[obj]+.04, 0, head_width = 0.20, linewidth=3, head_length=0.03, \
             color='grey')
        else:
            ax.arrow(original_PW[utility_idx+obj+1], obj_y_vals[obj][1], \
             PW_regret[obj]-.05, 0, head_width = 0.20, linewidth=3, head_length=0.03, \
            color='black')
    
    else:
        line_x = [original_PW[utility_idx+obj+1], \
                  original_PW[utility_idx+obj+1]+PW_regret[obj]]
        line_y = [obj_y_vals[obj][1], obj_y_vals[obj][1]]
        
        
        ax.plot(line_x, line_y, color='#ef476f', linewidth=2, zorder=0)
'''    



def robustness_regret_subplot(utility_idx, obj_y_vals, original_LS, \
                              original_PW, LS_regret, PW_regret, y_labels,
                              plot_idx, ax):
    '''
    Adds regret elements to subplot for both LS and PW comps for a utility
    
    
    Parameters:
        -utility_idx: reprents the first column robustness values are stored
        in the robustness files for each utility (Watertown = 0, Dryville=4
        and Fallsland=8)
        -obj_y_values: a list of y coordinates for each objective
        -original_LS: the orginal robustness of LS comp
        -original_PW: the original robustness of PW comp
        -LS_regret: the calculated regret for LS comp
        -PW_regret: the calculated regret for PW comp
        -y_labels: a list of strings to label the y-axis ticks 
        -plot_idx: the index of the subplot on the 3x3 grid
        -ax: axes object to be plotted on
    ''' 
    
    # plot points and arrows for each objective
    plot_points_and_arrows(utility_idx, 0, obj_y_vals, original_LS,\
                           original_PW, LS_regret, PW_regret,ax)
    plot_points_and_arrows(utility_idx, 1, obj_y_vals, original_LS, \
                           original_PW, LS_regret, PW_regret,ax)
    plot_points_and_arrows(utility_idx, 2, obj_y_vals, original_LS, \
                           original_PW, LS_regret, PW_regret,ax)
    plot_points_and_arrows(utility_idx, 3, obj_y_vals, original_LS, \
                           original_PW, LS_regret, PW_regret,ax)

    
    # format the plot
    # if the plot is on the top row, add the appropriate title
    titles = ['Watertown Defection', 'Dryville Defection', \
              'Fallsland Defection']
    if plot_idx == 1:
        ax.set_title(titles[0], fontsize=16)
    elif plot_idx == 2:
        ax.set_title(titles[1], fontsize=16)
    elif plot_idx==3:
        ax.set_title(titles[2], fontsize=16)
    
    # format the y-axis
    ax.set_yticks([0.75, 2.25, 3.75, 5.25])

    # if the plot is in the left column, add labels
    if plot_idx == 1 or plot_idx == 4 or plot_idx == 7:
        ax.set_yticklabels(y_labels)
        if utility_idx == 0:
            ax.set_ylabel('Watertown Criteria', fontsize=14)
        elif utility_idx == 4:
            ax.set_ylabel('Dryville Criteria', fontsize=14)
        elif utility_idx == 8:
            ax.set_ylabel('Fallsland Criteria', fontsize=14)
    else:
        ax.set_yticklabels([])
        
    ax.set_xlim([0,1.05])
    
    if plot_idx>6:
        ax.set_xlabel('Robustness (%)', fontsize=14)
    else:
        ax.set_xticks([])




# create coordinates for each criteria
# [REL, RF, WFPC, ALL]
obj_y_vals = [[5,5.5], [3.5,4], [2, 2.5], [.5,1]]

# load data
# load compromise values from original solution
original_robustness_all = np.loadtxt('../../results/du_reevaluation_data/all_robustness.csv', delimiter=',')
original_robustness_LS_98 = original_robustness_all[98, :]
original_robustness_PW_113 = original_robustness_all[113, :]

# load the re-evaluated files for LS defection
LS_98_W_reeval = np.loadtxt('../../results/reoptimized_solution_sets/LS_comp_98/du_reevaluation/Watertown/watertown_all_robustness.csv', delimiter=',')
LS_98_D_reeval = np.loadtxt('../../results/reoptimized_solution_sets/LS_comp_98/du_reevaluation/Dryville/dryville_all_robustness.csv', delimiter=',')
LS_98_F_reeval = np.loadtxt('../../results/reoptimized_solution_sets/LS_comp_98/du_reevaluation/Fallsland/fallsland_all_robustness.csv', delimiter=',')

# load the re-evaluated files for PW defection
PW_113_W_reeval = np.loadtxt('../../results/reoptimized_solution_sets/PW_comp_113/du_reevaluation/Watertown/watertown_all_robustness.csv', delimiter=',')
PW_113_D_reeval = np.loadtxt('../../results/reoptimized_solution_sets/PW_comp_113/du_reevaluation/Dryville/dryville_all_robustness.csv', delimiter=',')
PW_113_F_reeval = np.loadtxt('../../results/reoptimized_solution_sets/PW_comp_113/du_reevaluation/Fallsland/fallsland_all_robustness.csv', delimiter=',')


# set up the figure
fig = plt.figure(figsize=(10,8))
y_labels = ['ALL', 'WFPFC', 'RF', 'REL']


################# Watertown defection ################
# Effect on watertown robustness
LS_W_W_regret = calc_rob_regret(original_robustness_LS_98, LS_98_W_reeval, 'Watertown', True)
PW_W_W_regret = calc_rob_regret(original_robustness_PW_113, PW_113_W_reeval, 'Watertown', True)

ax1 = fig.add_subplot(3,3,1)
robustness_regret_subplot(0, obj_y_vals, original_robustness_LS_98, \
                              original_robustness_PW_113, LS_W_W_regret, PW_W_W_regret, y_labels,
                              1, ax1)



# Effect on Dryville robustness
LS_W_D_regret = calc_rob_regret(original_robustness_LS_98, LS_98_W_reeval, 'Dryville', False)
PW_W_D_regret = calc_rob_regret(original_robustness_PW_113, PW_113_W_reeval, 'Dryville', False)

ax4 = fig.add_subplot(3,3,4)
robustness_regret_subplot(4, obj_y_vals, original_robustness_LS_98, \
                              original_robustness_PW_113, LS_W_D_regret, PW_W_D_regret, y_labels,
                              4, ax4)


# Effect on Fallsland robustness
LS_W_F_regret = calc_rob_regret(original_robustness_LS_98, LS_98_W_reeval, 'Fallsland', False)
PW_W_F_regret = calc_rob_regret(original_robustness_PW_113, PW_113_W_reeval, 'Fallsland', False)

ax7 = fig.add_subplot(3,3,7)
robustness_regret_subplot(8, obj_y_vals, original_robustness_LS_98, \
                              original_robustness_PW_113, LS_W_F_regret, PW_W_F_regret, y_labels,
                              7, ax7)

################ Dryville Defection ################
# effect on Watertown Robustness
LS_D_W_regret = calc_rob_regret(original_robustness_LS_98, LS_98_D_reeval, 'Watertown', False)
PW_D_W_regret = calc_rob_regret(original_robustness_PW_113, PW_113_D_reeval, 'Watertown', False)

ax2 = fig.add_subplot(3,3,2)
robustness_regret_subplot(0, obj_y_vals, original_robustness_LS_98, \
                              original_robustness_PW_113, LS_D_W_regret, PW_D_W_regret, y_labels,
                              2, ax2)


# effect on Dryville Robustness (Note: something is wrong here, regret too high!)
LS_D_D_regret = calc_rob_regret(original_robustness_LS_98, LS_98_D_reeval, 'Dryville', True)
PW_D_D_regret = calc_rob_regret(original_robustness_PW_113, PW_113_D_reeval, 'Dryville', True)

# plot
ax5 = fig.add_subplot(3,3,5)
robustness_regret_subplot(4, obj_y_vals, original_robustness_LS_98, \
                              original_robustness_PW_113, LS_D_D_regret, PW_D_D_regret, y_labels,
                              5, ax5)


# effect on Fallsland Robustness  (Note: something is wrong here, regret too high!)
LS_D_F_regret = calc_rob_regret(original_robustness_LS_98, LS_98_D_reeval, 'Fallsland', False)
PW_D_F_regret = calc_rob_regret(original_robustness_PW_113, PW_113_D_reeval, 'Fallsland', False)

# plot
ax8 = fig.add_subplot(3,3,8)
robustness_regret_subplot(8, obj_y_vals, original_robustness_LS_98, \
                              original_robustness_PW_113, LS_D_F_regret, PW_D_F_regret, y_labels,
                              8, ax8)

################ Fallsland Defection ################
# effect on Watertown Robustness
LS_F_W_regret = calc_rob_regret(original_robustness_LS_98, LS_98_F_reeval, 'Watertown', False)
PW_F_W_regret = calc_rob_regret(original_robustness_PW_113, PW_113_F_reeval, 'Watertown', False)

ax3 = fig.add_subplot(3,3,3)
robustness_regret_subplot(0, obj_y_vals, original_robustness_LS_98, \
                              original_robustness_PW_113, LS_F_W_regret, PW_F_W_regret, y_labels,
                              3, ax3)

# effect on Dryville Robustness
LS_F_D_regret = calc_rob_regret(original_robustness_LS_98, LS_98_F_reeval, 'Dryville', False)
PW_F_D_regret = calc_rob_regret(original_robustness_PW_113, PW_113_F_reeval, 'Dryville', False)

ax6 = fig.add_subplot(3,3,6)
robustness_regret_subplot(4, obj_y_vals, original_robustness_LS_98, \
                              original_robustness_PW_113, LS_F_D_regret, PW_F_D_regret, y_labels,
                              6, ax6)

# effect on Dryville Robustness
LS_F_F_regret = calc_rob_regret(original_robustness_LS_98, LS_98_F_reeval, 'Fallsland', True)
PW_F_F_regret = calc_rob_regret(original_robustness_PW_113, PW_113_F_reeval, 'Fallsland', True)


ax9 = fig.add_subplot(3,3,9)
robustness_regret_subplot(8, obj_y_vals, original_robustness_LS_98, \
                              original_robustness_PW_113, LS_F_F_regret, PW_F_F_regret, y_labels,
                              9, ax9)


plt.tight_layout()
plt.savefig('figure_9_rob.png')


