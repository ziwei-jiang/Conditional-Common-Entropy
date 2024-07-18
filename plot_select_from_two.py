import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import entropy
from tqdm import tqdm




np.random.seed(0)

eps = 3e-4

cce_v_list = []
cce_z_list = []
Hu_list = []

correct_count = 0


experiments_list = [[2,2,2,2,2,0.1], [2,2,2,2,2,0.5], [2,2,2,2,2,1]]
experiments_list = [[3,3,3,3,3,0.1], [3,3,3,3,3,0.5], [3,3,3,3,3,1]]
experiments_list = [[4,4,4,4,4,0.1], [4,4,4,4,4,0.5], [4,4,4,4,4,1]]


for experiment in experiments_list:

    x_states = experiment[0]
    y_states = experiment[1]
    z_states = experiment[2]
    u_states = experiment[3]
    v_states = experiment[4]
    alpha_u = experiment[5]
    num_dist = 100
    m_states = 2

    folder_name = 'x{}_y{}_z{}_v{}_u{}_a{}'.format(x_states, y_states, z_states, v_states, u_states, str(alpha_u).replace('.', ''))

    u = np.load('experiments/Two_cov/{}/u.npy'.format(folder_name))

    eps = 3e-4

    for i_dist in tqdm(range(num_dist)):
        Hu = entropy(u[i_dist])
        Hu_list.append(Hu)
        Hw_z_arr = np.load('experiments/Two_cov/{}/s{}_Hw_z.npy'.format(folder_name, i_dist))
        Iyz_arr = np.load('experiments/Two_cov/{}/s{}_Iyz.npy'.format(folder_name, i_dist))
        Izw_arr = np.load('experiments/Two_cov/{}/s{}_Izw.npy'.format(folder_name, i_dist))
        Hw_v_arr = np.load('experiments/Two_cov/{}/s{}_Hw_v.npy'.format(folder_name, i_dist))
        Iyv_arr = np.load('experiments/Two_cov/{}/s{}_Iyv.npy'.format(folder_name, i_dist))
        Ivw_arr = np.load('experiments/Two_cov/{}/s{}_Ivw_.npy'.format(folder_name, i_dist))
        
        idx_z = np.where((Iyz_arr <=eps) & (Izw_arr <=eps))[0]
        cce_z = Hw_z_arr[idx_z].min()
        
        idx_v = np.where((Iyv_arr <=eps) & (Ivw_arr <=eps))[0]
        if len(idx_v) == 0:
            cce_v = Hw_v_arr.max()
        else:
            cce_v = Hw_v_arr[idx_v].min()
        
        if cce_v - cce_z >= eps:
            correct_count += 1

        cce_z_list.append(cce_z)
        cce_v_list.append(cce_v)


## plot the cce_z, cce_v, and Hu

cce_z_arr = np.array(cce_z_list)
cce_v_arr = np.array(cce_v_list)
Hu_arr = np.array(Hu_list)

## sort the cce_iv_arr and Hu_arr according to the cce_z_arr
idx_sort = np.argsort(cce_z_arr)
# idx_sort = np.argsort(Hu_arr)
cce_z_arr = cce_z_arr[idx_sort]
cce_v_arr = cce_v_arr[idx_sort]
Hu_arr = Hu_arr[idx_sort]
print('correct_count: {}/{}'.format(correct_count, len(cce_z_arr)))

plt.figure()

## make the axe font larger
plt.rc('axes', labelsize=15)


## make the axes number thicker
plt.rc('ytick', labelsize=15)
plt.rc('xtick', labelsize=15)
plt.rc('legend', fontsize=15)

plt.plot(cce_z_arr, 'o', label=r'CCE$_\mathcal{IV}(Z;Y|X,V)$')
plt.plot(cce_v_arr, 'o', label=r'CCE$_\mathcal{IV}(V;Y|X,Z)$')
# plt.plot(Hu_arr, 'o', label='H(U)')

plt.xlabel('dist id')
plt.ylabel('bits')
plt.legend(loc='best')

plt.show()