"""
Created on Fri Mar  6 15:53:05 2020

This script will create the visualization in Figure 3 of Gold et al. (2021)
"""


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates
import seaborn as sns
import matplotlib.cm as cm 
#from plotting_functions_DG import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

### Figure 3a ###
# plots a parallel axis plot of the objective space from the Trindade et al., 
# (2020) pareto approximate set

# load data
pf_objectives = pd.read_csv('../../data/Trindade_et_al_2020_data/' + \
                            'reference_set/Trindade_reference_set_objectives_final.csv')

# normalize objectives so all axes plot values range from 0-1
REL_bounds = [1, .98]
RF_bounds = [.3, 0]
INF_bounds = [120, 0]
PFC_bounds = [0.65, 0]
WCC_bounds = [.1, 0]

normalized_objectives = pf_objectives

normalized_objectives['REL'] = 1-(normalized_objectives['REL'] - \
                     REL_bounds[1])/(REL_bounds[0]-REL_bounds[1])

normalized_objectives['RF'] = (normalized_objectives['RF'] - \
                     RF_bounds[1])/(RF_bounds[0]-RF_bounds[1])

normalized_objectives['INF NPC'] = (normalized_objectives['INF NPC'] - \
                     INF_bounds[1])/(INF_bounds[0]-INF_bounds[1])

normalized_objectives['PFC'] = (normalized_objectives['PFC'] - \
                     PFC_bounds[1])/(PFC_bounds[0]-PFC_bounds[1])

normalized_objectives['WCC'] = (normalized_objectives['WCC'] - \
                     WCC_bounds[1])/(WCC_bounds[0]-WCC_bounds[1])


# use pandas to create the parallel axis plot
fig = plt.figure(figsize=(12,10))
plt.rcParams['axes.titlepad'] = 14
#spec = fig.add_gridspec(ncols=2, nrows=2, figure=fig, height_ratios = [.75,1], 
#                        width_ratios = [1, .75])
spec = fig.add_gridspec(ncols=4, nrows=9, figure=fig,
                        height_ratios = [.5,.5,.5, .1, 1, .05, 1,.05, 1], \
                        width_ratios = [1, .1, .5, .5])


ax = fig.add_subplot(spec[0:3,:])
parallel_coordinates(normalized_objectives[['REL', 'RF','INF NPC','PFC', 'WCC', 
                                            'Name']], 'Name', ax=ax, color= \
                                            ['#dee2e6', '#073b4c','#ef476f'],
                                           Linewidth=5, alpha=.75)
ax.set_ylim([0,1])
ax.set_xticklabels(['Rel', 'RF', 'INF NPC', 'PFC', 'WCC'], fontsize=14)
ax.legend(loc='upper right', fontsize =14)
ax.get_yaxis().set_ticks([])
ax.set_ylabel('Objective Value\n$\longleftarrow$ Direction of preference', 
               labelpad=10, fontsize =14)
ax.set_title('Objective Space', fontsize=16)
#plt.savefig('Figure3a.pdf')
# Final touches to figure made with Adobe Illustrator (legend, labels etc.)


### Figure 3b ###
# plots a 3D scatter plot of the robustness values for the three utilities

robustness = np.loadtxt('../../results/du_reevaluation_data/' + \
                         'utility_robustness.csv', delimiter=',')

# identify the solutions of interest
LS = robustness[98, :]
PW = robustness[113, :]
#BRo_W = robustness[21, :]
#BRo_D = robustness[142, :]
#BRo_F = robustness[85, :]

#robustness = np.delete(robustness, [21, 82, 85, 98, 142], axis=0)
robustness = np.delete(robustness, [98, 113], axis=0)

#fig = plt.figure(figsize=(8,8), dpi=300)
ax1 = fig.add_subplot(spec[4:,0:2],projection='3d')

ax1.scatter3D(robustness[:,1]*100, robustness[:,2]*100, 
             robustness[:,3]*100, s=50, color='grey', alpha = .3)
ax1.scatter3D(LS[1]*100, LS[2]*100, LS[3]*100,
             s=200, color='#073b4c', marker='d', alpha = 1)
#ax1.scatter3D(BRo_W[1]*100, BRo_W[2]*100, BRo_W[3]*100, 
#             s=200, color='#06d6a0', marker = '^', alpha = 1)             
#ax1.scatter3D(BRo_D[1]*100, BRo_D[2]*100, BRo_D[3]*100, 
#             s=200, color='#118ab2', marker ='^', alpha = 1)
#ax1.scatter3D(BRo_F[1]*100, BRo_F[2]*100, BRo_F[3]*100, 
#             s=200, color='#ffd166', marker='^', alpha = 1)             
ax1.scatter3D(PW[2]*100, PW[2]*100, PW[3]*100, 
             s=200, color='#ef476f', marker = 'd', alpha = 1)
             
ax1.scatter3D(100, 100, 100, marker='*', s=200, color='k')
ax1.set_xlim([40,100])
ax1.set_ylim([40, 100])
ax1.set_zlim([40,100])
ax1.set_xlabel('Watertown Robustness (%)\n$\longleftarrow$ Direction of preference', 
               labelpad=10, fontsize =14)
ax1.set_ylabel('Dryville Robustness (%)\n$\longleftarrow$ Direction of preference', 
               labelpad=10, fontsize =14)
ax1.set_zlabel('Fallsland Robustness (%)\nDirection of preference $\longrightarrow$', 
               labelpad=10, fontsize =14)
ax1.azim=160
ax1.elev=20
ax1.set_title('Robustness Space', fontsize=16)
#plt.tight_layout()
#plt.savefig('Figure3_march.pdf')


#### plot decision variables on radial plots


# load original solution (1-D array)
LS_DVs = np.loadtxt('../../data/selectedSols_DVs/LScompSol98_DVs.csv', 
                          delimiter=',')
PW_DVs = np.loadtxt('../../data/selectedSols_DVs/PWcompSol113_DVs.csv', 
                          delimiter=',')

# extract DVs for each utility (not including IP)
W_DVs_LS = np.take(LS_DVs,[0, 8, 5,  11, 17])
W_inf_LS = np.take(LS_DVs,[20, 21, 22, 23, 27])
# reverse direction of ROFs, since lower values mean increased use of the DV
W_DVs_LS[0] = 1- W_DVs_LS[0]
W_DVs_LS[1] = W_DVs_LS[1]/.10
W_DVs_LS[2] = W_DVs_LS[2]/.9
W_DVs_LS[3] = 1- W_DVs_LS[3]
W_DVs_LS[4] = 1- W_DVs_LS[4]
W_DVs_LS = np.append(W_DVs_LS, W_DVs_LS[0])


# extract DVs for each utility (not including IP)
W_DVs_PW = np.take(PW_DVs,[0, 8,  5, 11, 17])
W_inf_PW = np.take(PW_DVs,[20, 21, 22, 23, 27])
# reverse direction of ROFs, since lower values mean increased use of the DV
W_DVs_PW[0] = 1- W_DVs_PW[0]
W_DVs_PW[1] = W_DVs_PW[1]/.1
W_DVs_PW[2] = W_DVs_PW[2]/.90
W_DVs_PW[3] = 1- W_DVs_PW[3]
W_DVs_PW[4] = 1- W_DVs_PW[4]
W_DVs_PW = np.append(W_DVs_PW, W_DVs_PW[0])


# dryville
D_DVs_LS = np.take(LS_DVs,[1,3,6,9,12,18])
D_inf_LS = np.take(LS_DVs,[24, 28])
D_DVs_LS[0] = 1- D_DVs_LS[0]
D_DVs_LS[1] = 1- D_DVs_LS[1]
D_DVs_LS[2] = D_DVs_LS[2]/.33
D_DVs_LS[3] = D_DVs_LS[3]/.1
D_DVs_LS[4] = 1- D_DVs_LS[4]
D_DVs_LS[5] = 1- D_DVs_LS[5]
D_DVs_LS = np.append(D_DVs_LS, D_DVs_LS[0])


D_DVs_PW = np.take(PW_DVs,[1,3,6,9,12,18])
D_inf_PW = np.take(PW_DVs,[24, 28])
D_DVs_PW[0] = 1- D_DVs_PW[0]
D_DVs_PW[1] = 1- D_DVs_PW[1]
D_DVs_PW[2] = D_DVs_PW[2]/.33
D_DVs_PW[3] = D_DVs_PW[3]/.1
D_DVs_PW[4] = 1- D_DVs_PW[4]
D_DVs_PW[5] = 1- D_DVs_PW[5]
D_DVs_PW = np.append(D_DVs_PW, D_DVs_PW[0])

# fallsland
F_DVs_LS = np.take(LS_DVs,[2, 4, 7, 10, 13, 19])
F_inf_LS= np.take(LS_DVs,[26, 29])
F_DVs_LS[0] = 1- F_DVs_LS[0]
F_DVs_LS[1] = 1- F_DVs_LS[1]
F_DVs_LS[2] = F_DVs_LS[2]/.33
F_DVs_LS[3] = F_DVs_LS[3]/.1
F_DVs_LS[4] = 1- F_DVs_LS[4]
F_DVs_LS[5] = 1- F_DVs_LS[5]
F_DVs_LS = np.append(F_DVs_LS, F_DVs_LS[0])

F_DVs_PW = np.take(PW_DVs,[2, 4, 7, 10, 13, 19])
F_inf_PW= np.take(PW_DVs,[26, 29])
F_DVs_PW[0] = 1- F_DVs_PW[0]
F_DVs_PW[1] = 1- F_DVs_PW[1]
F_DVs_PW[2] = F_DVs_PW[2]/.33
F_DVs_PW[3] = F_DVs_PW[3]/.1
F_DVs_PW[4] = 1- F_DVs_PW[4]
F_DVs_PW[5] = 1- F_DVs_PW[5]
F_DVs_PW = np.append(F_DVs_PW, F_DVs_PW[0])


# watertown dvs (LS)
W_DV_names = ['RT', 'RC', 'LMA', 'IT', 'INF']
ax3 = fig.add_subplot(spec[4,2], projection='polar')
#ax3 = fig.add_subplot(111, projection='polar')
theta = np.linspace(0,2*np.pi, len(W_DVs_LS))
ax3.plot(theta, W_DVs_LS, color='#073b4c')
lines, labels = ax3.set_thetagrids(range(0, 360, int(360/len(W_DV_names))), (W_DV_names))
ax3.set_yticklabels([])
ax3.set_title('LS', fontsize=16)


# watertown dvs (PW)
ax4 = fig.add_subplot(spec[4,3], projection='polar')
#ax3 = fig.add_subplot(111, projection='polar')
theta = np.linspace(0,2*np.pi, len(W_DVs_PW))
ax4.plot(theta, W_DVs_PW, color='#ef476f')
lines, labels = ax4.set_thetagrids(range(0, 360, int(360/len(W_DV_names))), (W_DV_names))
ax4.set_yticklabels([])
ax4.set_title('PW', fontsize=16)


# dryille dvs (LS)
D_DV_names = ['RT', 'TT', 'LMA', 'RC', 'IT', 'INF']
ax5 = fig.add_subplot(spec[6,2], projection='polar')
#ax3 = fig.add_subplot(111, projection='polar')
theta = np.linspace(0,2*np.pi, len(D_DVs_LS))
ax5.plot(theta, D_DVs_LS, color='#073b4c')
lines, labels = ax5.set_thetagrids(range(0, 360, int(360/len(D_DV_names))), (D_DV_names))
ax5.set_yticklabels([])

# dryille dvs (PW)
ax6 = fig.add_subplot(spec[6,3], projection='polar')
#ax3 = fig.add_subplot(111, projection='polar')
theta = np.linspace(0,2*np.pi, len(D_DVs_PW))
ax6.plot(theta, D_DVs_PW, color='#ef476f')
lines, labels = ax6.set_thetagrids(range(0, 360, int(360/len(D_DV_names))), (D_DV_names))
ax6.set_yticklabels([])


# Fallsland dvs (LS)
F_DV_names = ['RT', 'TT', 'LMA', 'RC', 'IT', 'INF']
ax7 = fig.add_subplot(spec[8,2], projection='polar')
#ax3 = fig.add_subplot(111, projection='polar')
theta = np.linspace(0,2*np.pi, len(F_DVs_LS))
ax7.plot(theta, F_DVs_LS, color='#073b4c')
lines, labels = ax7.set_thetagrids(range(0, 360, int(360/len(F_DV_names))), (F_DV_names))
ax7.set_yticklabels([])


# Fallsland dvs (PW)
F_DV_names = ['RT', 'TT', 'LMA', 'RC', 'IT', 'INF']
ax8 = fig.add_subplot(spec[8,3], projection='polar')
#ax3 = fig.add_subplot(111, projection='polar')
theta = np.linspace(0,2*np.pi, len(F_DVs_PW))
ax8.plot(theta, F_DVs_PW, color='#ef476f')
lines, labels = ax8.set_thetagrids(range(0, 360, int(360/len(F_DV_names))), (F_DV_names))
ax8.set_yticklabels([])



plt.tight_layout()
plt.savefig('Figure-3-April.svg', bbox_inches='tight')




'''
ax3.scatter([0,1,2,3,4], W_DVs_LS, color='#073b4c', marker = 'd')
ax3.scatter([0,1,2,3,4], W_DVs_PW, color = '#ef476f', marker = 'd')
ax3.set_xticks([0,1,2,3,4])
ax3.get_yaxis().set_ticks([])
ax3.set_xticklabels(['RT', 'LMA', 'RC', 'IT', 'INF'], fontsize=14)
ax3.set_title('Decision Space \nWatertown DVs')













ax4 = fig.add_subplot(spec[4,2], porjection='polar')
theta = np.linspace(0,2*np.pi, len(F_DVs_LS))



ax4.plot([0,1,2,3,4, 5], D_DVs_LS, color='#073b4c', marker = 'd')
ax4.scatter([0,1,2,3,4, 5], D_DVs_PW, color = '#ef476f', marker = 'd')
ax4.set_xticks([0,1,2,3,4, 5])
ax4.get_yaxis().set_ticks([])
ax4.set_ylabel('DV Value\n Increased Use $\longrightarrow$', 
               labelpad=10, fontsize =14)
ax4.set_xticklabels(['RT', 'TT', 'LMA', 'RC', 'IT', 'INF'], fontsize=14)
ax4.set_title('Dryville DVs')

ax4 = fig.add_subplot(spec[5,2])
ax4.scatter([0,1,2,3,4, 5], F_DVs_LS, color='#073b4c', marker = 'd')
ax4.scatter([0,1,2,3,4, 5], F_DVs_PW, color = '#ef476f', marker = 'd')
ax4.set_xticks([0,1,2,3,4, 5])
ax4.get_yaxis().set_ticks([])
ax4.set_xticklabels(['RT', 'TT', 'LMA', 'RC', 'IT', 'INF'], fontsize=14)
ax4.set_title('Fallsland DVs')
plt.tight_layout()
plt.savefig('Figure-3-march-21.pdf')
'''




