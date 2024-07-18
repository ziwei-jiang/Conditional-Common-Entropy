import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import entropy, iv_inequality_constraint_check, mutual_information_vec, conditional_mutual_information_vec
from tqdm import tqdm



np.random.seed(0)

eps = 3e-4

cce_iv_list = []
Hu_list = []
Iyz_xu_list = []
Iyz_x_list = []

CCE_count = 0
iv_ineq_count = 0
valid_count = 0
total = 0


graph= 'iv'
# graph = 'invalid_iv'

# experiments_list = [[2,2,2,2,0.2], [2,2,2,2,0.5]]
experiments_list = [[2,2,2,2,0.1], [2,2,2,2,0.2], [2,2,2,2,0.5]]
# experiments_list = [[2,4,4,4,0.1], [2,4,4,4,0.2], [2,4,4,4,0.5]]
# experiments_list = [[4,4,4,4,0.1], [4,4,4,4,0.2], [4,4,4,4,0.5]]


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


    xu = xzu.sum(axis=3)
    x = xu.sum(axis=2)
    zxu = xzu.transpose((0, 2, 1, 3))
    z_xu = np.einsum('nijk, njk -> nijk', zxu, np.divide(1, xu))

    yz_xu = np.einsum('nijkl, nkjl -> nikjl', y_xzu, z_xu)

    Iyz_xu_arr = conditional_mutual_information_vec(yz_xu.reshape(num_dist, y_states, z_states, x_states*u_states), xu.reshape(num_dist, x_states*u_states))

    xz = xzu.sum(axis=3)
    Ixz_arr = mutual_information_vec(xz)

    yxz = yxzu.sum(axis=4)
    yzx = yxz.transpose(0, 1, 3, 2)
    yz_x = np.einsum('nijk, nk -> nijk', yzx, np.divide(1, x, where=(x>0)))
    Iyz_x_arr = conditional_mutual_information_vec(yz_x, x)

    

    print("Experiment: ", folder_name)

    for i_dist in tqdm(range(num_dist)):
        valid = True

        total += 1
        Hu = entropy(u[i_dist])
        
        ## find the entries where Iyz_iv_arr and Izw_iv_arr both equal to zero
        Hw_iv_arr = np.load('experiments/{}/s{}_Hw_iv.npy'.format(folder_name, i_dist))
        Iyz_iv_arr = np.load('experiments/{}/s{}_Iyz_iv.npy'.format(folder_name, i_dist))
        Izw_iv_arr = np.load('experiments/{}/s{}_Izw_iv.npy'.format(folder_name, i_dist))


        valid_idx = np.where((Iyz_iv_arr <= eps) & (Izw_iv_arr <= eps))[0]

        if not iv_inequality_constraint_check(yx_z[i_dist]):
            iv_ineq_count += 1
            valid = False


        if len(valid_idx) == 0:
            CCE_count += 1
            valid = False
        else:
            cce_iv = Hw_iv_arr[valid_idx].min()
            if cce_iv - Hu >= eps:
                CCE_count += 1
                valid = False
            cce_iv_list.append(cce_iv)
            Hu_list.append(Hu)
            Iyz_xu_list.append(Iyz_xu_arr[i_dist])
            Iyz_x_list.append(Iyz_x_arr[i_dist])
        if valid:
            valid_count += 1   



print("Experiment: ", folder_name)
print("Number of samples: ", total)
print("Number of samples rejected by IV inequalities: ", iv_ineq_count)
print("Number of samples rejected by CCE: ", CCE_count)
print("Number of samples did not rejected by both: ", valid_count)
cce_iv_arr = np.array(cce_iv_list)

Hu_arr = np.array(Hu_list)
# print(invalid_count + (cce_iv_arr >= Hu_arr).sum())

Iyz_xu_arr = np.array(Iyz_xu_list)
Iyz_x_arr = np.array(Iyz_x_list)


## sort the cce_iv_arr and Hu_arr
# idx = np.argsort(cce_iv_arr)
idx = np.argsort(Hu_arr)
cce_iv_arr = cce_iv_arr[idx]
Hu_arr = Hu_arr[idx]
Iyz_xu_arr = Iyz_xu_arr[idx]
Iyz_x_arr = Iyz_x_arr[idx]


plt.figure()

## make the axe font larger
plt.rc('axes', labelsize=15)


## make the axes number thicker
plt.rc('ytick', labelsize=15)
plt.rc('xtick', labelsize=15)
plt.rc('legend', fontsize=15)


## plot the CCE_IV and H(U) with larger scatters 

plt.plot(cce_iv_arr, 'o', label=r'Approximation of CCE$_\mathcal{IV}$')
plt.plot(Hu_arr, 'o', label='H(U)', alpha=0.8, color='green')


plt.xlabel('dist id')
plt.ylabel('bits')

## make the legend font larger

plt.legend()

plt.show()