# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 12:47:51 2020

@author: dgold
"""

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from matplotlib import pyplot as plt

#from plotting_functions_DG import *



def cluster_pathways(solution, utility, num_clusters, reoptimization=False, reoptimization_utility = '/'):
    """
    clusters infrastructure pathways be the week each option is constructed
    creates "representative pathways" for diagnostics and communication
    
    Parameters:
        solution: name of the solution to be plotted (should be folder name)
        utility: a string (lowercase) of the name of the utility of interest
        num_clusters: number of clusters used (should be carefully assessed)
        reoptimization: boolean for if the solution is a reoptimization
    
    returns:
        cluster_pathways: a 3-d list containing the pathways for each cluster
        cluster_medians: an array containing median construction weeks for each
        inf option in each clustercluster
    
    """
    # read pathways file
    if reoptimization:
        pathways_df = pd.read_csv('../../results/reoptimized_solution_sets/' 
                          + solution + 
                          '/all_utility_reeval/' + 
                          reoptimization_utility+'/Pathways.out', sep='\t')
    else:
        pathways_df = pd.read_csv('../../data/Trindade_et_al_2020_data/selected_sols_policies_pathways/' 
                              +solution+'/Pathways.out', sep='\t')
    
    # reformat for clustering (need an array with each row a realization and
    # each column a different infrastructure option. elements are const weeks)
    cluster_input = np.ones([1000,13])*2344

    # loop through each realization
    for real in range(0,1000):
    # extract the realization
        current_real = pathways_df[pathways_df['Realization']==real]
        # find the infrastructure option (ids 0-2 are already built, 6 is off)
        for inf in [3,4,5,7,8,9,10,11,12]:
            if not current_real.empty:
                for index, row in current_real.iterrows():
                    if row['infra.']==inf:
                        cluster_input[real, inf] = row['week']
    
    # post process to remove inf options never constructed and normalize weeks
    # to [0-1] by dividing by total weeks, 2344
    cluster_input = cluster_input[:,[4,5,7,8,9,10,11,12]]/2344
    #inf_options = inf_options[4:6]+ inf_options[7:13]
    
    # extract columns for each utility
    if utility == 'watertown':
        # watertown has NRR, CRR_low, CRR_high, WR, WRII
        cluster_input = cluster_input[:,[0, 2, 3, 4, 5]]
    elif utility == 'dryville':
        # dryville has SCR, DR
        cluster_input = cluster_input[:,[1, 6]]
    else:
        # fallsland has NRR, FR
        cluster_input = cluster_input[:,[0, 7]]

    # k-means clustering
    k_means = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    
    k_means.fit(cluster_input)
    k_means_cluster_centers = k_means.cluster_centers_	
    k_means_labels = pairwise_distances_argmin(cluster_input, k_means_cluster_centers)
    
    # assign each realization to a pathway, and calculate the median week
    # each infrstructure option is constructed in each cluster
    cluster_pathways = []
    cluster_medians = []
    for i in range(0, num_clusters):
        current_cluster =  cluster_input[k_means_labels==i,:]*2344
        cluster_pathways.append(current_cluster)
        
        current_medians = np.zeros(len(current_cluster[0,:]))
        for j in range(0, len(current_cluster[0,:])):
            current_medians[j]= np.median(current_cluster[:,j])
        
        cluster_medians.append(current_medians)
    
    
    # sort clusters by average of medians to get heavy, mod and light clusters
    cluster_means = [np.mean(cluster_medians[0]), np.mean(cluster_medians[1]), 
                     np.mean(cluster_medians[2])]
    
    sorted_indicies = np.argsort(cluster_means)
    
    # re-order based on sorting
    #cluster_pathways = np.vstack((cluster_pathways[sorted_indicies[0]], 
    #                             cluster_pathways[sorted_indicies[1]],
    #                             cluster_pathways[sorted_indicies[2]]))
    
    cluster_medians = np.vstack((cluster_medians[sorted_indicies[2]], 
                                 cluster_medians[sorted_indicies[1]],
                                 cluster_medians[sorted_indicies[0]]))
    
    
    return cluster_pathways, cluster_medians





def plot_single_pathway(cluster_medians, cluster_pathways, inf_options_idx, 
                        c, cmap, ax, inf_names, y_offset, ylabeling, xlabeling, 
                        plot_legend):
    """
    Makes a plot of an infrastructure Pathway
    
    Parameters:
        cluster_medias: an array with median weeks each option is built
        cluster_pathways: an array with every pathway in the cluster
        inf_options: an array with numbers representing each option (y-axis vals)
        should start at zero to represent the "baseline"
        c: color to plot pathway
        ax: axes object to plot pathway
        inf_names: a list of strings with the names of each pathway
    
    """
    # get array of the infrastructure options without baseline
    inf_options_idx_no_baseline=inf_options_idx[1:]
    
    sorted_inf = np.argsort(cluster_medians)
    
    # plot heatmap of construction times
    cluster_pathways = np.rint(cluster_pathways/45)
    inf_im = np.zeros((45, np.shape(cluster_pathways)[1]+1))
    
    for k in range(1,np.shape(cluster_pathways)[1]+1) :
        for i in range(0,45):
            for j in range(0, len(cluster_pathways[:,k-1])):
                if cluster_pathways[j,k-1] == i:
                    inf_im[i,k] +=1
   
    ax.imshow(inf_im.T, cmap=cmap, aspect='auto', alpha = 0.35)     
    
    
    # sort by construction order
    #cluster_medians = np.rint(cluster_medians/45)
    #sorted_inf = np.argsort(cluster_medians)
    
    # plot pathways
    # create arrays to plot the pathway lines. To ensure pathways have corners 
    # we need an array to have length 2*num_inf_options
    pathway_x = np.zeros(len(cluster_medians)*2+2)
    pathway_y = np.zeros(len(cluster_medians)*2+2)
    
    # to make corners, each inf option must be in the slot it is triggered, and
    # the one after
    cluster_medians = np.rint(cluster_medians/45)
    for i in range(0,len(cluster_medians)):
        for j in [1,2]:
            pathway_x[i*2+j] = cluster_medians[sorted_inf[i]]
            pathway_y[i*2+j+1] = inf_options_idx_no_baseline[sorted_inf[i]]
    
    # end case
    pathway_x[-1] = 45

    # plot the pathway line    
    ax.plot(pathway_x, pathway_y+y_offset, color=c, linewidth=5, 
            alpha = .9, zorder=1)
     
    # plot pathway nodes    
    #inf_scatter = np.zeros(len(cluster_medians)+1)
    #inf_scatter[1:] = inf_options_idx_no_baseline[sorted_inf]           
    #ax.scatter(cluster_medians[sorted_inf],inf_scatter[:-1], color=c, edgecolor='k', s=100, zorder=2, alpha = .5)

    # format plot (might need editing)  
    ax.set_xlim([0,44])
    inf_name_lables = ['']+inf_names
    inf_options_idx = np.hstack((inf_options_idx, inf_options_idx[-1]+1))
    ax.set_yticks(inf_options_idx-1)
    if ylabeling:
        ax.set_yticklabels(inf_name_lables)
    else:
        ax.set_yticklabels([''])
    if not xlabeling:
        ax.set_xticks([],[])
        #ax.set_xticklabels([''])
    else:
        ax.set_xticklabels(np.arange(2020,2070,10))    
    ax.set_ylim([-0.5,len(cluster_medians)+.5])


  



def create_cluster_plots(w_meds, d_meds, f_meds, w_pathways, d_pathways, 
                         f_pathways, n_clusters, cluster_colors, cmaps, fig,
                         gspec, fig_col, ylabeling, plot_legend):
    """
    creates a figure with three subplots, each representing a utility
    
    Parameters:
        w_meds: median values for each cluster for watertown
        d_meds: median values for each cluster for dryville
        f_meds: median values for each cluster for fallsland
        w_pathways: all pathways in each watertown cluster
        d_pathways: all pathways in each dryville cluster
        f_pathways: all pathways in each fallsland cluster
        n_clusters: number of clusters
        cluster_colors: an array of colors for each cluster
        cmaps: an array of colormaps for coloring the heatmap
        fig: a figure object for plotting
        fig_dims: an array with the number of rows and columns of subplots
        
        NOTE: DOES NOT SAVE THE FIGURE
    """
    
    watertown_inf = ['Baseline', 'New River\nReservoir', 
                     'College Rock\nExpansion Low',
                     'College Rock\nExpansion High', 'Water Reuse', 
                     'Water Reuse II']
    dryville_inf = ['Baseline', 'Sugar Creek\nReservoir', 'Water Reuse', '','','']
    fallsland_inf = ['Baseline', 'New River\nReservoir', 'Water Reuse', '','','']
    
    ax1 = fig.add_subplot(gspec[0, fig_col])
    ax2 = fig.add_subplot(gspec[1, fig_col])
    ax3 = fig.add_subplot(gspec[2, fig_col])
    
    y_offsets = [-.15, 0, .15]
    
    '''
    w_plot_order = np.argsort([np.mean(w_meds[0]), np.mean(w_meds[1]), 
                               np.mean(w_meds[2])])
    d_plot_order = np.argsort([np.mean(d_meds[0]), np.mean(d_meds[1]),
                               np.mean(d_meds[2])])
    f_plot_order = np.argsort([np.mean(f_meds[0]), np.mean(f_meds[1]), 
                              np.mean(f_meds[2])])
    '''
    
    
    
    for i in np.arange(n_clusters):

        plot_single_pathway(w_meds[i], w_pathways[i], np.array([0, 1,2,3,4, 5]), 
                              cluster_colors[i], 
                              cmaps[i], ax1, watertown_inf, 
                              y_offsets[i], ylabeling, False, plot_legend)
        plot_single_pathway(d_meds[i], d_pathways[i], np.array([0, 1,2]), 
                              cluster_colors[i], cmaps[i], ax2, dryville_inf, 
                              y_offsets[i], ylabeling, False, plot_legend)
        plot_single_pathway(f_meds[i], f_pathways[i], np.array([0, 1,2]), 
                              cluster_colors[i], cmaps[i], ax3, fallsland_inf, 
                             y_offsets[i], ylabeling, True, plot_legend)
    
    if plot_legend:
        ax1.legend(['Light inf.', 'Moderat inf.', 'Heavy inf.'], 
                   loc='upper left')
    
    ax1.tick_params(axis = "y", which = "both", left = False, right = False)
    ax2.tick_params(axis = "y", which = "both", left = False, right = False)
    ax3.tick_params(axis = "y", which = "both", left = False, right = False)
    
    #if fig_row == 1:
    #    ax1.set_title('Watertown Pathways')
    #    ax2.set_title('Dryville Pathways')
    #    ax3.set_title('Fallsland Pathways')
    
    plt.tight_layout()


solution = 'LS_comp_98'


heavy_inf_color = '#233E56'
mid_inf_color = '#286994'
light_inf_color = '#1789D0'


# original
watertown_cluster_pathways, watertown_cluster_meds = cluster_pathways(solution, 'watertown', 3)
dryville_cluster_pathways, dryville_cluster_meds = cluster_pathways(solution, 'dryville', 3)
fallsland_cluster_pathways, fallsland_cluster_meds = cluster_pathways(solution, 'fallsland', 3)

# sort clusters
#watertown_sorted_meds = [watertown_cluster_meds[0], watertown_cluster_meds[2],
#                         watertown_cluster_meds[1]]
#dryville_sorted_meds = [dryville_cluster_meds[2], dryville_cluster_meds[1],
#                         dryville_cluster_meds[0]]
#fallsland_sorted_meds =[fallsland_cluster_meds[2], fallsland_cluster_meds[1],
#                         fallsland_cluster_meds[0]]


fig = plt.figure(figsize=(16,9), dpi=300)
gspec = fig.add_gridspec(ncols=4, nrows=3, height_ratios =[1,.5,.5] )

create_cluster_plots(watertown_cluster_meds, dryville_cluster_meds, 
                     fallsland_cluster_meds, watertown_cluster_pathways,
                     dryville_cluster_pathways, fallsland_cluster_pathways,
                      3, [light_inf_color, mid_inf_color, heavy_inf_color], 
                     ['bone_r', 'bone_r', 'bone_r'], fig, gspec, 0, True, True)    

# watertown reoptimization
watertown_cluster_pathways, watertown_cluster_meds = cluster_pathways(solution, 'watertown', 3, True, 'watertown')
dryville_cluster_pathways, dryville_cluster_meds = cluster_pathways(solution, 'dryville', 3, True, 'watertown')
fallsland_cluster_pathways, fallsland_cluster_meds = cluster_pathways(solution, 'fallsland', 3, True, 'watertown')

create_cluster_plots(watertown_cluster_meds, dryville_cluster_meds, 
                     fallsland_cluster_meds, watertown_cluster_pathways,
                     dryville_cluster_pathways, fallsland_cluster_pathways,
                      3, [light_inf_color, mid_inf_color, heavy_inf_color], 
                     ['bone_r', 'bone_r', 'bone_r'], fig, gspec, 1, False, 
                     False)      

# Dryville reoptimization
watertown_cluster_pathways, watertown_cluster_meds = cluster_pathways(solution, 'watertown', 3, True, 'dryville')
dryville_cluster_pathways, dryville_cluster_meds = cluster_pathways(solution, 'dryville', 3, True, 'dryville')
fallsland_cluster_pathways, fallsland_cluster_meds = cluster_pathways(solution, 'fallsland', 3, True, 'dryville')

create_cluster_plots(watertown_cluster_meds, dryville_cluster_meds, 
                     fallsland_cluster_meds, watertown_cluster_pathways,
                     dryville_cluster_pathways, fallsland_cluster_pathways,
                     3, [light_inf_color, mid_inf_color, heavy_inf_color], 
                     ['bone_r', 'bone_r', 'bone_r'], fig, gspec, 2, False, 
                     False)  

# fallsland reoptimization
watertown_cluster_pathways, watertown_cluster_meds = cluster_pathways(solution, 'watertown', 3, True, 'fallsland')
dryville_cluster_pathways, dryville_cluster_meds = cluster_pathways(solution, 'dryville', 3, True, 'fallsland')
fallsland_cluster_pathways, fallsland_cluster_meds = cluster_pathways(solution, 'fallsland', 3, True, 'fallsland')

create_cluster_plots(watertown_cluster_meds, dryville_cluster_meds, 
                     fallsland_cluster_meds, watertown_cluster_pathways,
                     dryville_cluster_pathways, fallsland_cluster_pathways,
                     3, [light_inf_color, mid_inf_color, heavy_inf_color], 
                     ['bone_r', 'bone_r', 'bone_r'], fig, gspec, 3, False, 
                     False)  

plt.savefig('LS_98_pathways.svg')



'''
watertown_cluster_pathways, watertown_cluster_meds = cluster_pathways(solution, 'watertown', 3)
dryville_cluster_pathways, dryville_cluster_meds = cluster_pathways(solution, 'dryville', 3)
fallsland_cluster_pathways, fallsland_cluster_meds = cluster_pathways(solution, 'fallsland', 3)

create_cluster_plots(watertown_cluster_meds, dryville_cluster_meds, 
                     fallsland_cluster_meds, watertown_cluster_pathways,
                     dryville_cluster_pathways, fallsland_cluster_pathways,
                     3, ['tab:red', 'tab:blue', 'tab:green'], 
                     ['Reds', 'Blues', 'Greens'])    


plt.savefig('../../results/figures/FB_comp_cluster_pathways.png')
'''