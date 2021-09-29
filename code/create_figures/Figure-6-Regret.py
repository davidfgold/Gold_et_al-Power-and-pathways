# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 15:37:52 2020

@author: dgold
"""

import sys
#sys.path.append('../')
#from data_analysis_scripts import *
from data_processing import *
import os
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
sns.set(style='white')




comp_name ='LS_comp_98'
# load compromise values from original solution
LS_original_objectives = np.loadtxt('../../data/selectedSols_objs/' \
    + comp_name + '.csv', delimiter=',')

PW_comp = ['PW', 'PW','PW','PW','PW']
LS_comp = ['LS', 'LS','LS','LS','LS']
objs = ['Rel', 'RF', 'NPV', 'PFC', 'WFPC']

LS_W_original = LS_original_objectives[0:5]
LS_D_original = LS_original_objectives[5:10]
LS_F_original = LS_original_objectives[10:15]


comp_name ='PW_comp_113'
# load compromise values from original solution
PW_original_objectives = np.loadtxt('../../data/selectedSols_objs/' \
    + comp_name + '.csv', delimiter=',')

PW_W_original = PW_original_objectives[0:5]
PW_D_original = PW_original_objectives[5:10]
PW_F_original = PW_original_objectives[10:15]



# load the LS re-evaluated files
LS_D_reeval = np.loadtxt('../../results/reoptimized_solution_sets/LS_comp_98/all_utility_reeval/LS_dryville_all_util.csv', delimiter=',')
LS_F_reeval = np.loadtxt('../../results/reoptimized_solution_sets/LS_comp_98/all_utility_reeval/LS_fallsland_all_util.csv', delimiter=',')
LS_W_reeval = np.loadtxt('../../results/reoptimized_solution_sets/LS_comp_98/all_utility_reeval/LS_watertown_all_util.csv', delimiter=',')

# load the PW re-evaluated files
PW_D_reeval = np.loadtxt('../../results/reoptimized_solution_sets/PW_comp_113/all_utility_reeval/PW_dryville_all_util.csv', delimiter=',')
PW_F_reeval = np.loadtxt('../../results/reoptimized_solution_sets/PW_comp_113/all_utility_reeval/PW_fallsland_all_util.csv', delimiter=',')
PW_W_reeval = np.loadtxt('../../results/reoptimized_solution_sets/PW_comp_113/all_utility_reeval/PW_watertown_all_util.csv', delimiter=',')


# LS W_W
LS_W_W_regret = calc_obj_regret(LS_W_original, LS_W_reeval, 'Watertown', True)
PW_W_W_regret = calc_obj_regret(PW_W_original, PW_W_reeval, 'Watertown', True)


LS_W_W = list(zip(objs, LS_W_W_regret*-1, LS_comp))
LS_W_W_df = pd.DataFrame(LS_W_W, columns = ['Objs', 'Regret', 'Comp'])

PW_W_W = list(zip(objs, PW_W_W_regret*-1, PW_comp))
PW_W_W_df = pd.DataFrame(PW_W_W, columns = ['Objs', 'Regret', 'Comp'])

W_W_all_regret = pd.concat([LS_W_W_df, PW_W_W_df])


# PW W_D
LS_W_D_regret = calc_obj_regret(LS_D_original, LS_W_reeval, 'Dryville', False)
PW_W_D_regret = calc_obj_regret(PW_D_original, PW_W_reeval, 'Dryville', False)


LS_W_D = list(zip(objs, LS_W_D_regret*-1, LS_comp))
LS_W_D_df = pd.DataFrame(LS_W_D, columns = ['Objs', 'Regret', 'Comp'])

PW_W_D = list(zip(objs, PW_W_D_regret*-1, PW_comp))
PW_W_D_df = pd.DataFrame(PW_W_D, columns = ['Objs', 'Regret', 'Comp'])

W_D_all_regret = pd.concat([LS_W_D_df, PW_W_D_df])



# LS W_F
LS_W_F_regret = calc_obj_regret(LS_F_original, LS_W_reeval, 'Fallsland', False)
PW_W_F_regret = calc_obj_regret(PW_F_original, PW_W_reeval, 'Fallsland', False)

LS_W_F = list(zip(objs, LS_W_F_regret*-1, LS_comp))
LS_W_F_df = pd.DataFrame(LS_W_F, columns = ['Objs', 'Regret', 'Comp'])

PW_W_F = list(zip(objs, PW_W_F_regret*-1, PW_comp))
PW_W_F_df = pd.DataFrame(PW_W_F, columns = ['Objs', 'Regret', 'Comp'])

W_F_all_regret = pd.concat([LS_W_F_df, PW_W_F_df])

# D_W
LS_D_W_regret = calc_obj_regret(LS_W_original, LS_D_reeval, 'Watertown', False)
PW_D_W_regret = calc_obj_regret(PW_W_original, PW_D_reeval, 'Watertown', False)

LS_D_W = list(zip(objs, LS_D_W_regret*-1, LS_comp))
LS_D_W_df = pd.DataFrame(LS_D_W, columns = ['Objs', 'Regret', 'Comp'])

PW_D_W = list(zip(objs, PW_D_W_regret*-1, PW_comp))
PW_D_W_df = pd.DataFrame(PW_D_W, columns = ['Objs', 'Regret', 'Comp'])

D_W_all_regret = pd.concat([LS_D_W_df, PW_D_W_df])

# LS D_D
LS_D_D_regret = calc_obj_regret(LS_D_original, LS_D_reeval, 'Dryville', True)
PW_D_D_regret = calc_obj_regret(PW_D_original, PW_D_reeval, 'Dryville', True)

LS_D_D = list(zip(objs, LS_D_D_regret*-1, LS_comp))
LS_D_D_df = pd.DataFrame(LS_D_D, columns = ['Objs', 'Regret', 'Comp'])

PW_D_D = list(zip(objs, PW_D_D_regret*-1, PW_comp))
PW_D_D_df = pd.DataFrame(PW_D_D, columns = ['Objs', 'Regret', 'Comp'])

D_D_all_regret = pd.concat([LS_D_D_df, PW_D_D_df])


# LS D_F
LS_D_F_regret = calc_obj_regret(LS_F_original, LS_D_reeval, 'Fallsland', False)
PW_D_F_regret = calc_obj_regret(PW_F_original, PW_D_reeval, 'Fallsland', False)

LS_D_F = list(zip(objs, LS_D_F_regret*-1, LS_comp))
LS_D_F_df = pd.DataFrame(LS_D_F, columns = ['Objs', 'Regret', 'Comp'])

PW_D_F = list(zip(objs, PW_D_F_regret*-1, PW_comp))
PW_D_F_df = pd.DataFrame(PW_D_F, columns = ['Objs', 'Regret', 'Comp'])

D_F_all_regret = pd.concat([LS_D_F_df, PW_D_F_df])


# LS F_W
LS_F_W_regret = calc_obj_regret(LS_W_original, LS_F_reeval, 'Watertown', False)
PW_F_W_regret = calc_obj_regret(PW_W_original, PW_F_reeval, 'Watertown', False)


LS_F_W = list(zip(objs, LS_F_W_regret*-1, LS_comp))
LS_F_W_df = pd.DataFrame(LS_F_W, columns = ['Objs', 'Regret', 'Comp'])

PW_F_W = list(zip(objs, PW_F_W_regret*-1, PW_comp))
PW_F_W_df = pd.DataFrame(PW_F_W, columns = ['Objs', 'Regret', 'Comp'])

F_W_all_regret = pd.concat([LS_F_W_df, PW_F_W_df])

# F_D
LS_F_D_regret = calc_obj_regret(LS_D_original, LS_F_reeval, 'Dryville', False)
PW_F_D_regret = calc_obj_regret(PW_D_original, PW_F_reeval, 'Dryville', False)

LS_F_D = list(zip(objs, LS_F_D_regret*-1, LS_comp))
LS_F_D_df = pd.DataFrame(LS_F_D, columns = ['Objs', 'Regret', 'Comp'])

PW_F_D = list(zip(objs, PW_F_D_regret*-1, PW_comp))
PW_F_D_df = pd.DataFrame(PW_F_D, columns = ['Objs', 'Regret', 'Comp'])

F_D_all_regret = pd.concat([LS_F_D_df, PW_F_D_df])

# F_F
LS_F_F_regret = calc_obj_regret(LS_F_original, LS_F_reeval, 'Fallsland', True)
PW_F_F_regret = calc_obj_regret(PW_F_original, PW_F_reeval, 'Fallsland', True)

LS_F_F = list(zip(objs, LS_F_F_regret*-1, LS_comp))
LS_F_F_df = pd.DataFrame(LS_F_F, columns = ['Objs', 'Regret', 'Comp'])

PW_F_F = list(zip(objs, PW_F_F_regret*-1, PW_comp))
PW_F_F_df = pd.DataFrame(PW_F_F, columns = ['Objs', 'Regret', 'Comp'])

F_F_all_regret = pd.concat([LS_F_F_df, PW_F_F_df])



# Create an array with the colors you want to use
colors = ['#073b4c','#ef476f'] # Set your custom color palette
customPalette = sns.set_palette(sns.color_palette(colors))


fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(331)
sns.barplot(x='Regret', y='Objs', data=W_W_all_regret, ax=ax1, hue='Comp', \
            palette = customPalette, orient='h')
ax1.set_xlim([-250,250])
ax1.get_legend().remove()


ax2 = fig.add_subplot(334)
sns.barplot(x='Regret', y='Objs', data=W_D_all_regret, ax=ax2, hue='Comp', \
            palette = customPalette, orient='h')
ax2.set_xlim([-250,250])
ax2.get_legend().remove()

ax3 = fig.add_subplot(337)
sns.barplot(x='Regret', y='Objs', data=W_F_all_regret, ax=ax3, hue='Comp', \
            palette = customPalette, orient='h')
ax3.set_xlim([-250,250])
ax3.get_legend().remove()

ax4 = fig.add_subplot(332)
sns.barplot(x='Regret', y='Objs', data=D_W_all_regret, ax=ax4, hue='Comp', \
            palette = customPalette, orient='h')
ax4.set_xlim([-450,250])
ax4.get_legend().remove()
ax4.set_ylabel('')

ax5 = fig.add_subplot(335)
sns.barplot(x='Regret', y='Objs', data=D_D_all_regret, ax=ax5, hue='Comp', \
            palette = customPalette, orient='h')
ax5.set_xlim([-250,250])
ax5.get_legend().remove()
ax5.set_ylabel('')


ax6 = fig.add_subplot(338)
sns.barplot(x='Regret', y='Objs', data=D_F_all_regret, ax=ax6, hue='Comp', \
            palette = customPalette, orient='h')
ax6.set_xlim([-450,250])
ax6.get_legend().remove()
ax6.set_ylabel('')


ax7 = fig.add_subplot(333)
sns.barplot(x='Regret', y='Objs', data=F_W_all_regret, ax=ax7, hue='Comp', \
            palette = customPalette, orient='h')
ax7.set_xlim([-450,250])
ax7.get_legend().remove()
ax7.set_ylabel('')

ax8 = fig.add_subplot(336)
sns.barplot(x='Regret', y='Objs', data=F_D_all_regret, ax=ax8, hue='Comp', \
            palette = customPalette, orient='h')
ax8.set_xlim([-250,250])
ax8.get_legend().remove()
ax8.set_ylabel('')

ax9 = fig.add_subplot(339)
sns.barplot(x='Regret', y='Objs', data=F_F_all_regret, ax=ax9, hue='Comp', \
            palette = customPalette, orient='h')
ax9.set_xlim([-250,250])
ax9.get_legend().remove()
ax9.set_ylabel('')


plt.tight_layout()
plt.savefig('Figure-6-Regret.png', bbox_inches='tight')






