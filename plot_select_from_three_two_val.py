import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import entropy
from tqdm import tqdm

np.random.seed(0)

eps = 3e-4


cce_v1_list = []
cce_v2_list = []
cce_z_list = []
Hu_list = []

correct_count = 0

# experiments_list = [[2,2,2,2,2,0.1], [2,2,2,2,2,0.5], [2,2,2,2,2,1]]
# experiments_list = [[3,3,3,3,3,0.1], [3,3,3,3,3,0.5], [3,3,3,3,3,1]]
experiments_list = [[2,2,2,2,2,2,1]]
# experiments_list = [[3,3,3,3,3,3,1]]

for experiment in experiments_list:

    x_states = experiment[0]
    y_states = experiment[1]
    z_states = experiment[2]
    u_states = experiment[3]
    v1_states = experiment[4]
    v2_states = experiment[5]
    alpha_u = experiment[6]
    num_dist = 100
    m_states = 2

    folder_name = 'x{}_y{}_z{}_1v{}_2v{}_u{}_a{}'.format(x_states, y_states, z_states, v1_states, v2_states, u_states, str(alpha_u).replace('.', ''))

    u = np.load('experiments/Three_cov_two_val/{}/u.npy'.format(folder_name))
    eps = 3e-4 

    for i_dist in tqdm(range(num_dist)):
        Hu = entropy(u[i_dist])
        Hu_list.append(Hu)
        Hw_z_arr = np.load('experiments/Three_cov_two_val/{}/s{}_Hw_z.npy'.format(folder_name, i_dist))
        Iyz_arr = np.load('experiments/Three_cov_two_val/{}/s{}_Iyz.npy'.format(folder_name, i_dist))
        Izw_arr = np.load('experiments/Three_cov_two_val/{}/s{}_Izw.npy'.format(folder_name, i_dist))
        
        Hw_v1_arr = np.load('experiments/Three_cov_two_val/{}/s{}_Hw_v1.npy'.format(folder_name, i_dist))
        Iyv1_arr = np.load('experiments/Three_cov_two_val/{}/s{}_Iyv1.npy'.format(folder_name, i_dist))
        Iv1w_arr = np.load('experiments/Three_cov_two_val/{}/s{}_Iv1w_.npy'.format(folder_name, i_dist))

        Hw_v2_arr = np.load('experiments/Three_cov_two_val/{}/s{}_Hw_v2.npy'.format(folder_name, i_dist))
        Iyv2_arr = np.load('experiments/Three_cov_two_val/{}/s{}_Iyv2.npy'.format(folder_name, i_dist))
        Iv2w_arr = np.load('experiments/Three_cov_two_val/{}/s{}_Iv2w.npy'.format(folder_name, i_dist))
        
        idx_z = np.where((Iyz_arr <=eps) & (Izw_arr <=eps))[0]
        cce_z = Hw_z_arr[idx_z].min()

        
        idx_v1 = np.where((Iyv1_arr <=eps) & (Iv1w_arr <=eps))[0]
        if len(idx_v1) == 0:
            cce_v1 = Hw_v1_arr.max()
        else:
            cce_v1 = Hw_v1_arr[idx_v1].min()
        
        idx_v2 = np.where((Iyv2_arr <=eps) & (Iv2w_arr <=eps))[0]
        if len(idx_v2) == 0:
            cce_v2 = Hw_v2_arr.max()
        else:
            cce_v2 = Hw_v2_arr[idx_v2].min()


        if (cce_v2 - cce_z >= eps) and (cce_v2 - cce_v1 >= eps):
            correct_count += 1

        cce_z_list.append(cce_z)
        cce_v1_list.append(cce_v1)
        cce_v2_list.append(cce_v2)


## plot the cce_z, cce_v, and Hu

cce_z_arr = np.array(cce_z_list)
cce_v1_arr = np.array(cce_v1_list)
cce_v2_arr = np.array(cce_v2_list)
Hu_arr = np.array(Hu_list)

## sort the cce_iv_arr and Hu_arr according to the cce_z_arr
idx_sort = np.argsort(cce_z_arr)
# idx_sort =  np.argsort(cce_v1_arr)
cce_z_arr = cce_z_arr[idx_sort]
cce_v1_arr = cce_v1_arr[idx_sort]
cce_v2_arr = cce_v2_arr[idx_sort]
Hu_arr = Hu_arr[idx_sort]
print('correct_count: {}/{}'.format(correct_count, len(cce_z_arr)))

plt.figure()

## make the axe font larger
plt.rc('axes', labelsize=15)


## make the axes number thicker
plt.rc('ytick', labelsize=15)
plt.rc('xtick', labelsize=15)
plt.rc('legend', fontsize=15)

plt.plot(cce_z_arr, 'o', label=r'CCE$_\mathcal{IV}(Z_1;Y|X,Z_2,V)$')
plt.plot(cce_v1_arr, 'o', label=r'CCE$_\mathcal{IV}(Z_2;Y|X,Z_1,V)$')
plt.plot(cce_v2_arr, 'o', label=r'CCE$_\mathcal{IV}(V;Y|X,Z_1,Z_2)$')

# plt.plot(Hu_arr, 'o', label='H(U)')

plt.xlabel('dist id')
plt.ylabel('bits')
plt.legend(loc='best')

plt.show()