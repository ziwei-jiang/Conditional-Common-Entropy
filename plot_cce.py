import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm
from utils import generate_iv_samples, mutual_information, generate_conditional_dist, entropy, smoothing, conditional_latent_search, conditional_latent_search_iv, generate_invalid_iv_samples
from mpl_toolkits.mplot3d import Axes3D
import os
from utils import iv_inequality_constraint_check

np.random.seed(0)




## set the number of states
alpha_u = 0.1
x_states = 4
y_states = 4
z_states = 4
u_states = 4
num_dist = 100
# graph = 'invalid_iv'
# graph= 'iv'
graph = 'weak_zy'

## get a color map according to the value of H(W)
def get_color_map(Hw_iv_arr):
    cmap = plt.get_cmap('autumn')
    norm = plt.Normalize(Hw_iv_arr.min(), Hw_iv_arr.max())
    color = cmap(norm(Hw_iv_arr))
    return color


folder_name = '{}_x{}_y{}_z{}_u{}_a{}'.format(graph, x_states, y_states, z_states, u_states, str(alpha_u).replace('.', ''))
## load the data from npy file

## 4, 14, 23 looks good

# for i_dist in range(num_dist):
for i_dist in [4, 14, 23]:
    print(i_dist)
    Hw_iv_arr = np.load('experiments/{}/s{}_Hw_iv.npy'.format(folder_name, i_dist))
    Iyz_iv_arr = np.load('experiments/{}/s{}_Iyz_iv.npy'.format(folder_name, i_dist))
    Izw_iv_arr = np.load('experiments/{}/s{}_Izw_iv.npy'.format(folder_name, i_dist))

    ## draw a 3D plot with Hw_iv, Iyz_iv, Izw_iv
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(Izw_iv_arr, Iyz_iv_arr, Hw_iv_arr, c=get_color_map(Hw_iv_arr), marker='o')

    ax.set_xlabel('I(Z;W)')
    ax.set_ylabel('I(Y;Z|X,W)')
    ax.set_zlabel('H(W)')

    ## make the font and everything larger
    plt.rc('font', size=10)
    plt.rc('axes', titlesize=10)
    plt.rc('axes', labelsize=10)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.rc('legend', fontsize=10)


    plt.show()
    plt.close()
