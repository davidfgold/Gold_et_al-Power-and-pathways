# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:53:10 2020

@author: dgold
"""

from sklearn.ensemble import GradientBoostingClassifier
from copy import deepcopy
from data_processing import *
from matplotlib import pyplot as plt
import seaborn
sns.set(style='white')

# get original robustness values for comps
OG_RDM_objectives, indiv_robustness = calculate_robustness('../' + \
    '../data/Trindade_et_al_2020_data/data_sp2/reeval/objectives/extracted', 1, 1000, 229, 98)



# get original robustness values for comps
Defection_RDM_objectives, defection_indiv_robustness = calculate_robustness('../' + \
    '../results/du_reevaluation_data/LS_comp_98/Watertown/Objs_by_SOW', 1, 1000, 29, 0)



# load rdm files
utility_rdm = np.loadtxt('../../data/RDM_SOWs/rdm_utilities_test_problem_reeval.csv', delimiter=',')
dmp_rdm = np.loadtxt('../../data/RDM_SOWs/rdm_dmp_test_problem_reeval.csv', delimiter=',')
water_sources_rdm = np.loadtxt('../../data/RDM_SOWs/rdm_water_sources_test_problem_reeval.csv', delimiter=',')
inflows_rdm =np.loadtxt('../../data/RDM_SOWs/rdm_inflows_test_problem_reeval.csv', delimiter=',')

rdm_factors = np.hstack((utility_rdm, dmp_rdm, water_sources_rdm, inflows_rdm))

original_SD_input = create_sd_input(OG_RDM_objectives,[])
defection_SD_input = create_sd_input(Defection_RDM_objectives,[])

int_OG_SD_input = original_SD_input
int_def_SD_input = defection_SD_input

for i in range(9):
    int_OG_SD_input[i] = original_SD_input[i]*1
    int_def_SD_input[i] = defection_SD_input[i]*1


fig = plt.figure()
plt.scatter(utility_rdm[1:1001,0], dmp_rdm[1:1001,0], c=int_OG_SD_input[6], cmap='Reds_r', edgecolor='k')
plt.xlim([0.5,2])

fig = plt.figure()
plt.scatter(utility_rdm[1:1001,0], dmp_rdm[1:1001,0], c=int_def_SD_input[6], cmap='Blues_r', edgecolor='k')
plt.xlim([0.5,2])


# create watertown all three obj satisficing vector
watertown_OG_sat = np.zeros(1000)
watertown_def_sat = np.zeros(1000)

for i in range(1000):
    watertown_OG_sat[i] = original_SD_input[0][i] and original_SD_input[1][i] and original_SD_input[2][i]
    watertown_def_sat[i] = defection_SD_input[0][i] and defection_SD_input[1][i] and defection_SD_input[2][i]


n_trees=500
tree_depth = 4
region_alpha=0.6

gbc = GradientBoostingClassifier(n_estimators=n_trees,
                                            learning_rate=0.1,
                                            max_depth=tree_depth)

gbc.fit(rdm_factors[1:1001,:], watertown_OG_sat.astype(bool))
print('Boosted Trees score: {}'.format(gbc.score(rdm_factors[1:1001,:], watertown_OG_sat*1)))
    
feature_importances = deepcopy(gbc.feature_importances_)
most_influential_factors = np.argsort(feature_importances)[::-1]


gbc_2factors = GradientBoostingClassifier(n_estimators=n_trees,
                                            learning_rate=0.1,
                                            max_depth=tree_depth)


most_important_rdm_factors = rdm_factors[1:1001, most_influential_factors[:2]]
gbc_2factors.fit(most_important_rdm_factors, watertown_OG_sat*1)

x_data = rdm_factors[1:1001,0]
y_data = rdm_factors[1:1001,4]

x_min, x_max = (x_data.min(), x_data.max())
y_min, y_max = (y_data.min(), y_data.max())

xx, yy = np.meshgrid(np.arange(x_min, x_max * 1.001, (x_max - x_min) / 100),
                     np.arange(y_min, y_max * 1.001, (y_max - y_min) / 100))
                     
dummy_points = list(zip(xx.ravel(), yy.ravel()))

z = gbc_2factors.predict_proba(dummy_points)[:, 1]
z[z < 0] = 0.
z = z.reshape(xx.shape)


fig = plt.figure()
ax=fig.add_subplot(111)
ax.contourf(xx, yy, z, [0, 0.5, 0.98, 1.], cmap='RdGy',
            alpha=.3, vmin=0., vmax=1.)
plt.scatter(utility_rdm[1:1001,0], dmp_rdm[1:1001,0], c=int_OG_SD_input[6], cmap='Reds_r', edgecolor='k', alpha=.6)





gbc = GradientBoostingClassifier(n_estimators=n_trees,
                                            learning_rate=0.1,
                                            max_depth=tree_depth)

gbc.fit(rdm_factors[1:1001,:], watertown_OG_sat.astype(bool))
print('Boosted Trees score: {}'.format(gbc.score(rdm_factors[1:1001,:], watertown_def_sat*1)))
    
feature_importances = deepcopy(gbc.feature_importances_)
most_influential_factors = np.argsort(feature_importances)[::-1]


gbc_2factors = GradientBoostingClassifier(n_estimators=n_trees,
                                            learning_rate=0.1,
                                            max_depth=tree_depth)


most_important_rdm_factors = rdm_factors[1:1001, most_influential_factors[:2]]
gbc_2factors.fit(most_important_rdm_factors, watertown_def_sat*1)

x_data = rdm_factors[1:1001,0]
y_data = rdm_factors[1:1001,4]

x_min, x_max = (x_data.min(), x_data.max())
y_min, y_max = (y_data.min(), y_data.max())

xx, yy = np.meshgrid(np.arange(x_min, x_max * 1.001, (x_max - x_min) / 100),
                     np.arange(y_min, y_max * 1.001, (y_max - y_min) / 100))
                     
dummy_points = list(zip(xx.ravel(), yy.ravel()))

z = gbc_2factors.predict_proba(dummy_points)[:, 1]
z[z < 0] = 0.
z = z.reshape(xx.shape)


fig = plt.figure()
ax=fig.add_subplot(111)
ax.contourf(xx, yy, z, [0, 0.5, 0.98, 1.], cmap='RdGy',
            alpha=.3, vmin=0., vmax=1.)
plt.scatter(utility_rdm[1:1001,0], dmp_rdm[1:1001,0], c=watertown_def_sat*1, cmap='Reds_r', edgecolor='k', alpha=.6)










def boosted_tree_sd(satisficing, rdm_factors, n_trees, tree_depth):
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
    
    gbc.fit(rdm_factors[0:1008,:], satisficing[0]*1)
    print('Boosted Trees score: {}'.format(gbc.score(rdm_factors[:], satisficing[0]*1)))
    
    feature_importances = deepcopy(gbc.feature_importances_)
    most_influential_factors = np.argsort(feature_importances)[::-1]
    
    
    gbc_2factors = GradientBoostingClassifier(n_estimators=n_trees,
                                            learning_rate=0.1,
                                            max_depth=tree_depth)
    
    
    most_important_rdm_factors = rdm_factors[:, most_influential_factors[:2]]
    gbc_2factors.fit(most_important_rdm_factors, satisficing[0]*1)
    
    return gbc, gbc_2factors, most_important_rdm_factors, feature_importances



def plot_factor_map( gbc_2factors, most_important_rdm_factors):

    '''
    creates a contour plot showing the predicted failure regions of the top two
    RDM factors
    
    inputs:
        rdm_factors: an array with rdm factors comprising each SOW
        
        gbc_2factors: classifier fit to only the top two factors
        
        most_important_rdm_factors: an array with indicies of most influential
        rdm factors
    
    returns: contour plot
    '''
    x_data = most_important_rdm_factors[0]
    y_data = most_important_rdm_factors[1]
    
    x_min, x_max = (x_data.min(), x_data.max())
    y_min, y_max = (y_data.min(), y_data.max())
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max * 1.001, (x_max - x_min) / 100),
                         np.arange(y_min, y_max * 1.001, (y_max - y_min) / 100))
                         
    dummy_points = list(zip(xx.ravel(), yy.ravel()))
    
    z = gbc_2factors.predict_proba(dummy_points)[:, 1]
    z[z < 0] = 0.
    z = z.reshape(xx.shape)
    
    
    fig = plt.figure()
    ax=fig.add_subplot(111)
    ax.contourf(xx, yy, z, [0, 0.5, 0.98, 1.], cmap='RdGy',
                alpha=.1, vmin=0., vmax=1.)