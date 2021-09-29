# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:06:12 2021

@author: dgold
"""
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(style='whitegrid')




# load original solution (1-D array)
LS_DVs = np.loadtxt('../../data/selectedSols_DVs/LScompSol98_DVs.csv', 
                          delimiter=',')
PW_DVs = np.loadtxt('../../data/selectedSols_DVs/PWcompSol113_DVs.csv', 
                          delimiter=',')

# extract DVs for each utility (not including IP)
W_original_LS = np.take(LS_DVs,[0, 8, 5,  11, 17])
W_original_inf_LS = np.take(LS_DVs,[20, 21, 22, 23, 27])
# reverse direction of ROFs, since lower values mean increased use of the DV
W_original_LS[0] = 1- W_original_LS[0]
W_original_LS[1] = W_original_LS[1]/.10
W_original_LS[2] = W_original_LS[2]/.9
W_original_LS[3] = 1- W_original_LS[3]
W_original_LS[4] = 1- W_original_LS[4]
W_original_LS = np.append(W_original_LS, W_original_LS[0])

W_DV_names = ['RT', 'RC', 'LMA', 'IT', 'INF']




# load defection solutions
W_LS_defection = np.loadtxt('../../results/reoptimized_solution_sets/' + \
                             'LS_comp_98/decision_variables/Watertown_LS_reeval.csv', \
                             delimiter=',')
W_LS_defection = W_LS_defection[:,[0, 8, 5, 11, 17]]

#W_original_inf_LS = np.take(LS_DVs,[20, 21, 22, 23, 27])
# reverse direction of ROFs, since lower values mean increased use of the DV
W_LS_defection[:,0] = 1- W_LS_defection[:,0]
W_LS_defection[:,1] = W_LS_defection[:,1]/.10
W_LS_defection[:,2] = W_LS_defection[:,2]/.9
W_LS_defection[:,3] = 1- W_LS_defection[:,3]
W_LS_defection[:,4] = 1- W_LS_defection[:,4]
W_LS_defection = np.concatenate((W_LS_defection, W_LS_defection[:,0].reshape(28,1)),axis=1)
#W_LS_defection =np.reshape(W_LS_defection, [28,6])


fig = plt.figure(figsize=(10,6))

# watertown dvs (PW)
ax1 = fig.add_subplot(231, projection='polar')
theta = np.linspace(0,2*np.pi, len(W_original_LS))
for i in range(0, len(W_LS_defection)):
    ax1.plot(theta, W_LS_defection[i,:], color='#ffd166')
ax1.plot(theta, W_original_LS, color='#073b4c')
lines, labels = ax1.set_thetagrids(range(0, 360, int(360/len(W_DV_names))), (W_DV_names))
ax1.set_yticklabels([])
#ax1.set_title('LS', fontsize=16)


# extract DVs for each utility (not including IP)
W_original_PW = np.take(PW_DVs,[0, 8, 5,  11, 17])
W_original_inf_PW = np.take(PW_DVs,[20, 21, 22, 23, 27])
# reverse direction of ROFs, since lower values mean increased use of the DV
W_original_PW[0] = 1- W_original_PW[0]
W_original_PW[1] = W_original_PW[1]/.10
W_original_PW[2] = W_original_PW[2]/.9
W_original_PW[3] = 1- W_original_PW[3]
W_original_PW[4] = 1- W_original_PW[4]
W_original_PW = np.append(W_original_PW, W_original_PW[0])




# load defection solutions
W_PW_defection = np.loadtxt('../../results/reoptimized_solution_sets/' + \
                             'PW_comp_113/decision_variables/Watertown_PW_reeval.csv', \
                             delimiter=',')
W_PW_defection = W_PW_defection[:,[0, 8, 5, 11, 17]]

#W_original_inf_LS = np.take(LS_DVs,[20, 21, 22, 23, 27])
# reverse direction of ROFs, since lower values mean increased use of the DV
W_PW_defection[:,0] = 1- W_PW_defection[:,0]
W_PW_defection[:,1] = W_PW_defection[:,1]/.10
W_PW_defection[:,2] = W_PW_defection[:,2]/.9
W_PW_defection[:,3] = 1- W_PW_defection[:,3]
W_PW_defection[:,4] = 1- W_PW_defection[:,4]
W_PW_defection = np.concatenate((W_PW_defection, W_PW_defection[:,0].reshape(42,1)),axis=1)
#W_LS_defection =np.reshape(W_LS_defection, [28,6])


# watertown dvs (PW)
ax4 = fig.add_subplot(234, projection='polar')
theta = np.linspace(0,2*np.pi, len(W_original_LS))
for i in range(0, len(W_PW_defection)):
    ax4.plot(theta, W_PW_defection[i,:], color='#ffd166')
ax4.plot(theta, W_original_PW, color='#ef476f')
lines, labels = ax4.set_thetagrids(range(0, 360, int(360/len(W_DV_names))), (W_DV_names))
ax4.set_yticklabels([])
#ax1.set_title('LS', fontsize=16)




D_original_LS = np.take(LS_DVs,[1,3,6,9,12,18])
D_original_inf_LS = np.take(LS_DVs,[24, 28])
D_DV_names = ['RT', 'TT',  'LMA', 'RC', 'IT', 'INF']

D_original_LS[0] = 1- D_original_LS[0]
D_original_LS[1] = 1- D_original_LS[1]
D_original_LS[2] = D_original_LS[2]/.33
D_original_LS[3] = D_original_LS[3]/.1
D_original_LS[4] = 1- D_original_LS[4]
D_original_LS[5] = 1- D_original_LS[5]
D_original_LS = np.append(D_original_LS, D_original_LS[0])


# load defection solutions
D_LS_defection = np.loadtxt('../../results/reoptimized_solution_sets/' + \
                             'LS_comp_98/decision_variables/Dryville_LS_reeval.csv', \
                             delimiter=',')
D_LS_defection = D_LS_defection[:,[1,3,6,9,12,18]]

#W_original_inf_LS = np.take(LS_DVs,[20, 21, 22, 23, 27])
# reverse direction of ROFs, since lower values mean increased use of the DV
D_LS_defection[:,0] = 1- D_LS_defection[:,0]
D_LS_defection[:,1] = 1- D_LS_defection[:,1]
D_LS_defection[:,2] = D_LS_defection[:,2]/.330
D_LS_defection[:,3] = D_LS_defection[:,3]/.1
D_LS_defection[:,4] = 1- D_LS_defection[:,4]
D_LS_defection[:,5] = 1- D_LS_defection[:,5]
D_LS_defection = np.concatenate((D_LS_defection, D_LS_defection[:,0].reshape(90,1)),axis=1)



ax2 = fig.add_subplot(232, projection='polar')
theta = np.linspace(0,2*np.pi, len(D_original_LS))
for i in range(0, len(D_LS_defection)):
    ax2.plot(theta, D_LS_defection[i,:], color='#ffd166')
ax2.plot(theta, D_original_LS, color='#073b4c')

lines, labels = ax2.set_thetagrids(range(0, 360, int(360/len(D_DV_names))), (D_DV_names))
ax2.set_yticklabels([])



D_original_PW = np.take(PW_DVs,[1,3,6,9,12,18])
D_original_inf_PW = np.take(PW_DVs,[24, 28])

D_original_PW[0] = 1- D_original_PW[0]
D_original_PW[1] = 1- D_original_PW[1]
D_original_PW[2] = D_original_PW[2]/.33
D_original_PW[3] = D_original_PW[3]/.1
D_original_PW[4] = 1- D_original_PW[4]
D_original_PW[5] = 1- D_original_PW[5]
D_original_PW = np.append(D_original_PW, D_original_PW[0])


# load defection solutions
D_PW_defection = np.loadtxt('../../results/reoptimized_solution_sets/' + \
                             'PW_comp_113/decision_variables/Dryville_PW_reeval.csv', \
                             delimiter=',')
D_PW_defection = D_PW_defection[:,[1,3,6,9,12,18]]

# reverse direction of ROFs, since lower values mean increased use of the DV
D_PW_defection[:,0] = 1- D_PW_defection[:,0]
D_PW_defection[:,1] = 1- D_PW_defection[:,1]
D_PW_defection[:,2] = D_PW_defection[:,2]/.330
D_PW_defection[:,3] = D_PW_defection[:,3]/.1
D_PW_defection[:,4] = 1- D_PW_defection[:,4]
D_PW_defection[:,5] = 1- D_PW_defection[:,5]
D_PW_defection = np.concatenate((D_PW_defection, D_PW_defection[:,0].reshape(75,1)),axis=1)



ax5 = fig.add_subplot(235, projection='polar')
theta = np.linspace(0,2*np.pi, len(D_original_PW))
for i in range(0, len(D_PW_defection)):
    ax5.plot(theta, D_PW_defection[i,:], color='#ffd166')
ax5.plot(theta, D_original_PW, color='#ef476f')

lines, labels = ax5.set_thetagrids(range(0, 360, int(360/len(D_DV_names))), (D_DV_names))
ax5.set_yticklabels([])




F_original_LS = np.take(LS_DVs,[2, 4, 7, 10, 13, 19])
F_original_inf_LS = np.take(LS_DVs,[24, 28])
F_DV_names = ['RT', 'TT',  'LMA', 'RC', 'IT', 'INF']

F_original_LS[0] = 1- F_original_LS[0]
F_original_LS[1] = 1- F_original_LS[1]
F_original_LS[2] = F_original_LS[2]/.33
F_original_LS[3] = F_original_LS[3]/.1
F_original_LS[4] = 1- F_original_LS[4]
F_original_LS[5] = 1- F_original_LS[5]
F_original_LS = np.append(F_original_LS, F_original_LS[0])


# load defection solutions
F_LS_defection = np.loadtxt('../../results/reoptimized_solution_sets/' + \
                             'LS_comp_98/decision_variables/Fallsland_LS_reeval.csv', \
                             delimiter=',')
F_LS_defection = F_LS_defection[:,[2, 4, 7, 10, 13, 19]]

#W_original_inf_LS = np.take(LS_DVs,[20, 21, 22, 23, 27])
# reverse direction of ROFs, since lower values mean increased use of the DV
F_LS_defection[:,0] = 1- F_LS_defection[:,0]
F_LS_defection[:,1] = 1- F_LS_defection[:,1]
F_LS_defection[:,2] = F_LS_defection[:,2]/.330
F_LS_defection[:,3] = F_LS_defection[:,3]/.1
F_LS_defection[:,4] = 1- F_LS_defection[:,4]
F_LS_defection[:,5] = 1- F_LS_defection[:,5]
F_LS_defection = np.concatenate((F_LS_defection, F_LS_defection[:,0].reshape(31,1)),axis=1)



ax3 = fig.add_subplot(233, projection='polar')
theta = np.linspace(0,2*np.pi, len(F_original_LS))
for i in range(0, len(F_LS_defection)):
    ax3.plot(theta, F_LS_defection[i,:], color='#ffd166')
ax3.plot(theta, F_original_LS, color='#073b4c')

lines, labels = ax3.set_thetagrids(range(0, 360, int(360/len(F_DV_names))), (F_DV_names))
ax3.set_yticklabels([])



F_original_PW = np.take(PW_DVs,[2, 4, 7, 10, 13, 19])
F_original_inf_PW = np.take(PW_DVs,[24, 28])

F_original_PW[0] = 1- F_original_PW[0]
F_original_PW[1] = 1- F_original_PW[1]
F_original_PW[2] = F_original_PW[2]/.33
F_original_PW[3] = F_original_PW[3]/.1
F_original_PW[4] = 1- F_original_PW[4]
F_original_PW[5] = 1- F_original_PW[5]
F_original_PW = np.append(F_original_PW, F_original_PW[0])


# load defection solutions
F_PW_defection = np.loadtxt('../../results/reoptimized_solution_sets/' + \
                             'PW_comp_113/decision_variables/Fallsland_PW_reeval.csv', \
                             delimiter=',')
F_PW_defection = F_PW_defection[0:31,[2, 4, 7, 10, 13, 19]]

# reverse direction of ROFs, since lower values mean increased use of the DV
F_PW_defection[:,0] = 1- F_PW_defection[:,0]
F_PW_defection[:,1] = 1- F_PW_defection[:,1]
F_PW_defection[:,2] = F_PW_defection[:,2]/.330
F_PW_defection[:,3] = F_PW_defection[:,3]/.1
F_PW_defection[:,4] = 1- F_PW_defection[:,4]
F_PW_defection[:,5] = 1- F_PW_defection[:,5]
F_PW_defection = np.concatenate((F_PW_defection, F_PW_defection[:,0].reshape(31,1)),axis=1)



ax6 = fig.add_subplot(236, projection='polar')
theta = np.linspace(0,2*np.pi, len(F_original_PW))
for i in range(0, len(F_PW_defection)):
    ax6.plot(theta, F_PW_defection[i,:], color='#ffd166')
ax6.plot(theta, F_original_PW, color='#ef476f')

lines, labels = ax6.set_thetagrids(range(0, 360, int(360/len(F_DV_names))), (F_DV_names))
ax6.set_yticklabels([])


plt.tight_layout()
plt.savefig('fig-7-dvs.svg')


'''
# watertown defection



W_DVs_defection = W_defection[:,[0, 5, 8, 11, 14, 17]]
W_inf_defection = W_defection[:,[20, 21, 22, 23, 27]]

W_DVs_defection[:,0] = 1- W_DVs_defection[:,0]

W_DVs_defection[:,3] = 1- W_DVs_defection[:,3]
W_DVs_original[3] = 1- W_DVs_original[3]

W_DVs_defection[:,5] = 1- W_DVs_defection[:,5]
W_DVs_original[5] = 1 -W_DVs_original[5]







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






fig = plt.figure(figsize=(12,4))


# watertown dvs (PW)
ax1 = fig.add_subplot(231, projection='polar')
#ax3 = fig.add_subplot(111, projection='polar')
theta = np.linspace(0,2*np.pi, len(W_original_LS))
ax1.plot(theta, W_original_LS, color='#073b4c')
lines, labels = ax1.set_thetagrids(range(0, 360, int(360/len(W_DV_names))), (W_DV_names))
ax1.set_yticklabels([])
ax1.set_title('LS', fontsize=16)
'''
















'''
# load original solution (1-D array)
original_DVs = np.loadtxt('../../data/selectedSols_DVs/PWcompSol113_DVs.csv', 
                          delimiter=',')


# extract DVs for each utility
W_DVs_original = np.take(original_DVs,[0, 5, 8, 11, 14, 17])
W_inf_original = np.take(original_DVs,[20, 21, 22, 23, 27])
D_DVs_original = np.take(original_DVs,[1,3,6,9,12,15,18])
D_inf_original = np.take(original_DVs,[24, 28])
F_DVs_original = np.take(original_DVs,[2, 4, 7, 10, 13, 16, 19])
F_inf_original= np.take(original_DVs,[26, 29])


# watertown defection
# load defection solutions
W_defection = np.loadtxt('../../results/reoptimized_solution_sets/' + \
                             'PW_comp_113/decision_variables/Watertown_PW_reeval_filtered_higher_RF.csv', \
                             delimiter=',')


W_DVs_defection = W_defection[:,[0, 5, 8, 11, 14, 17]]
W_inf_defection = W_defection[:,[20, 21, 22, 23, 27]]

W_DVs_defection[:,0] = 1- W_DVs_defection[:,0]
W_DVs_original[0] = 1- W_DVs_original[0]

W_DVs_defection[:,3] = 1- W_DVs_defection[:,3]
W_DVs_original[3] = 1- W_DVs_original[3]

W_DVs_defection[:,5] = 1- W_DVs_defection[:,5]
W_DVs_original[5] = 1 -W_DVs_original[5]

W_plot = pd.DataFrame(W_DVs_defection, columns=['RT', 'LMA', 'RC', 'IT', 'IP', 'INF'])













ax1 = fig.add_subplot(1,3,1)


ax1=sns.swarmplot(data=W_plot, color='grey')
ax1.scatter(np.arange(6),  W_DVs_original, c='#ef476f', s=80, marker='v', zorder=2)


#dryville
# load defection solutions
D_defection = np.loadtxt('../../results/reoptimized_solution_sets/' + \
                             'PW_comp_113/decision_variables/Dryville_PW_filterend_reeval.csv', \
                             delimiter=',')

D_DVs_defection = D_defection[:, [1,3,6,9,12,15,18]]
D_inf_defection = D_defection[:,[24, 28]]

D_DVs_defection[:,2] = D_DVs_defection[:,2]/.36

D_DVs_defection[:,0] = 1- D_DVs_defection[:,0]
D_DVs_original[0] = 1 - D_DVs_original[0]

D_DVs_defection[:,1] = 1- D_DVs_defection[:,1]
D_DVs_original[1] = 1 - D_DVs_original[1]

D_DVs_defection[:,4] = 1- D_DVs_defection[:,4]
D_DVs_original[4] = 1 - D_DVs_original[4]

D_DVs_defection[:,6] = 1- D_DVs_defection[:,6]
D_DVs_original[6] = 1 - D_DVs_original[6]

D_plot = pd.DataFrame(D_DVs_defection, columns=['RT', 'TT', 'LMA', 'RC', 'IT', 'IP', 'INF'])

ax2 = fig.add_subplot(1,3,2)
sns.set(style='whitegrid')
ax= sns.swarmplot(data=D_plot, color='grey')
ax.scatter(np.arange(7),  D_DVs_original, c='#ef476f', s=80, marker='v', zorder=2)

# Fallsland
# load defection solutions
F_defection = np.loadtxt('../../results/reoptimized_solution_sets/' + \
                             'PW_comp_113/decision_variables/Fallsland_PW_reeval_filtered.csv', \
                             delimiter=',')


F_DVs_defection = F_defection[:,[2, 4, 7, 10, 13, 16, 19]]
F_inf_defection = F_defection[:,[26, 29]]

# Format DVs for plotting
F_DVs_defection[:,2] = F_DVs_defection[:,2]/.36

F_DVs_defection[:,0] = 1- F_DVs_defection[:,0]
F_DVs_original[0] = 1 - F_DVs_original[0]

F_DVs_defection[:,1] = 1- F_DVs_defection[:,1]
F_DVs_original[1] = 1 - F_DVs_original[1]

F_DVs_defection[:,4] = 1- F_DVs_defection[:,4]
F_DVs_original[4] = 1 - F_DVs_original[4]

F_DVs_defection[:,6] = 1- F_DVs_defection[:,6]
F_DVs_original[6] = 1 - F_DVs_original[6]


F_plot = pd.DataFrame(F_DVs_defection, columns=['RT', 'TT', 'LMA', 'RC', 'IT', 'IP', 'INF'])


ax3 = fig.add_subplot(1,3,3)
sns.set(style='whitegrid')
ax3= sns.swarmplot(data=F_plot, color='grey', zorder=1)
ax3.scatter(np.arange(7),  F_DVs_original, c='#ef476f', s=80, marker='v', zorder=2)

plt.savefig('PW defection dvs.png')

'''



