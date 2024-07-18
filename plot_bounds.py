import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import entropy, iv_inequality_constraint_check, mutual_information_vec, conditional_mutual_information_vec, entropy_vec
from tqdm import tqdm
# plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Paired.colors)



np.random.seed(0)

plt.figure(figsize=(10,10))
cce_iv_list = []
Hu_list = []

CCE_count = 0
iv_ineq_count = 0
valid_count = 0
total = 0

# graph= 'invalid_iv'
# graph = 'weak_zy'
graph = 'iv'



experiments_list1 = [[2,2,2,2,0.1], [2,2,2,2,0.2],  [2,2,2,2,0.5]]
experiments_list3 = [[4,4,4,4,0.1], [4,4,4,4,0.2], [4,4,4,4,0.5]]

all_experiments = [experiments_list1, experiments_list3]

col = 0

# Get the 'tab20c' colormap
cmap = plt.cm.get_cmap('tab20c')

# Total colors available
total_colors = cmap.N  # Should be 20 for 'tab20c'

# Generate indices for 3 out of every 4 colors
color_indices = [i for i in range(total_colors) if i % 4 != 3]
# Extract those colors
colors = [cmap(i / (total_colors - 1)) for i in color_indices]


for experiments_list in all_experiments:
    natural_bounds_all = []
    tp_bounds_all = []
    entr_bounds_all = []
    entropy_all = []

    for experiment in experiments_list:
        
        ## set the number of states
        x_states = experiment[0]
        y_states = experiment[1]
        z_states = experiment[2]
        u_states = experiment[3]
        alpha_u = experiment[4]
        num_dist = 100
        m_states = 2

        folder_name = '{}_x{}_y{}_z{}_u{}_a{}'.format(graph, x_states, y_states, z_states, u_states, str(alpha_u).replace('.', ''))
        y_xzu = np.load('experiments/{}/y_xzu.npy'.format(folder_name))
        x_zu = np.load('experiments/{}/x_zu.npy'.format(folder_name))
        z = np.load('experiments/{}/z.npy'.format(folder_name))
        u = np.load('experiments/{}/u.npy'.format(folder_name))

        zu = np.einsum('nk, nl -> nkl', z, u)
        xzu = np.einsum('njkl, nkl -> njkl', x_zu, zu)
        yxzu = np.einsum('nijkl, njkl -> nijkl', y_xzu, xzu)

        yx_zu  = np.einsum('nijkl, njkl -> nijkl', y_xzu, x_zu)
        yx_z = np.einsum('nijkl, nl -> nijk', yx_zu, u)


        xu = xzu.sum(axis=2)
        zxu = xzu.transpose((0, 2, 1, 3))
        z_xu = np.einsum('nijk, njk -> nijk', zxu, np.divide(1, xu))

        yz_xu = np.einsum('nijkl, nkjl -> nikjl', y_xzu, z_xu)

        Iyz_xu_arr = conditional_mutual_information_vec(yz_xu.reshape(num_dist, y_states, z_states, x_states*u_states), xu.reshape(num_dist, x_states*u_states))

        xz = xzu.sum(axis=3)
        Ixz_arr = mutual_information_vec(xz)


        print("Experiment: ", folder_name)



        natural_bounds_arr = np.load('experiments/{}/natural_bounds.npy'.format(folder_name))
        tp_bounds_arr = np.load('experiments/{}/tp_bounds.npy'.format(folder_name))
        entr_bounds_arr = np.load('experiments/{}/entr_bounds.npy'.format(folder_name))
        entropy_arr = np.load('experiments/{}/entropy.npy'.format(folder_name))

        natural_bounds_all.append(natural_bounds_arr)
        tp_bounds_all.append(tp_bounds_arr)
        entr_bounds_all.append(entr_bounds_arr)
        entropy_all.append(entropy_arr)



    natural_bounds_all = np.vstack(natural_bounds_all)
    tp_bounds_all = np.vstack(tp_bounds_all)
    entr_bounds_all = np.vstack(entr_bounds_all)
    entropy_all = np.hstack(entropy_all)


    min_entr = entropy_all.min()
    max_entr = entropy_all.max()
    print("min_entr: ", min_entr)
    print("max_entr: ", max_entr)
    group_interval = (max_entr - min_entr)/10
    group_idx_list = []

    natural_gaps = natural_bounds_all[:, 1] - natural_bounds_all[:, 0]
    tp_gaps = tp_bounds_all[:, 1] - tp_bounds_all[:, 0]
    entr_gaps = entr_bounds_all[:, 1] - entr_bounds_all[:, 0]


    for i in range(10):
        group_idx = np.where((entropy_all >= (min_entr + i*group_interval)) & (entropy_all <= (min_entr + (i+1)*group_interval)))[0]
        group_idx_list.append(group_idx)


    ## add the rest to the last group
    group_idx_list[-1] = np.concatenate((group_idx_list[-1], np.where(entropy_all > (min_entr + 10*group_interval))[0]))

    if graph == 'iv':
        ## plot the average gap of each group
        natural_gap_list = []
        tp_gap_list = []
        entr_gap_list = []
        for group in group_idx_list:
            natural_gap_list.append(natural_gaps[group].mean())
            tp_gap_list.append(tp_gaps[group].mean())
            entr_gap_list.append(entr_gaps[group].mean())

        # plt.figure(figsize=(10, 10))
        ## make the axe font larger
        plt.rc('axes', labelsize=20)
        ## make the axes number thicker
        plt.rc('ytick', labelsize=20)
        plt.rc('xtick', labelsize=20)
        ## make the legend font larger
        plt.rc('legend', fontsize=20)

        plt.plot(np.arange(10), tp_gap_list, label='TP bounds |X|={},|Y|={},|Z|={}'.format(x_states, y_states, z_states), marker='o', markersize=5, color=colors[col])
        col+=1
        plt.plot(np.arange(10), natural_gap_list, label='Natural bounds |X|={},|Y|={},|Z|={}'.format(x_states, y_states, z_states), marker='x', markersize=5, color=colors[col])
        col+=1
        plt.plot(np.arange(10), entr_gap_list, label='Our bounds |X|={},|Y|={},|Z|={}'.format(x_states, y_states, z_states), marker='^', markersize=10, color=colors[col])
        col+=1
        print(col)
        plt.legend()
        plt.title('Average Gap', fontsize=20)
        plt.xlabel('Entropy', fontsize=20)
        plt.xticks(np.arange(10), ["[{:.2f}, {:.2f}]".format(min_entr + i*group_interval, min_entr + (i+1)*group_interval) for i in range(10)], fontsize=10)
        plt.ylabel('Average Gap', fontsize=20)
        # plt.show()

        ## Generate a bar plot for number tighter and totol number of samples
        width = 0.5

    else:
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Paired.colors)

        ## plot the average gap of each group
        tp_gap_list = []
        tp_gap_list = []
        entr_gap_list = []
        for group in group_idx_list:
            tp_gap_list.append(tp_gaps[group].mean())
            entr_gap_list.append(entr_gaps[group].mean())

        print(entr_gap_list)

        # plt.figure(figsize=(10,10))
        ## make the axe font larger
        plt.rc('axes', labelsize=20)
        ## make the axes number thicker
        plt.rc('ytick', labelsize=20)
        plt.rc('xtick', labelsize=20)
        ## make the legend font larger
        plt.rc('legend', fontsize=20)

        
        plt.plot(np.arange(10), tp_gap_list,  marker='o', markersize=15, label='TP bounds |X|={},|Y|={},|Z|={}'.format(x_states, y_states, z_states), linewidth=3)
        plt.plot(np.arange(10), entr_gap_list,  marker='^', markersize=15, label='Our bounds |X|={},|Y|={},|Z|={}'.format(x_states, y_states, z_states), linewidth=3)
        plt.legend()
        plt.title('Average Gap', fontsize=20)
        plt.xlabel('Entropy', fontsize=20)
        plt.ylabel('Average Gap', fontsize=20)

        # plt.show()

      
plt.show()