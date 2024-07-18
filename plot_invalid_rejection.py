import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import entropy, iv_inequality_constraint_check, mutual_information_vec, conditional_mutual_information_vec, entropy_vec
from tqdm import tqdm



np.random.seed(0)

eps = 3e-4

cce_iv_list = []
Hu_list = []

CCE_count = 0
iv_ineq_count = 0
valid_count = 0
total = 0




graph= 'invalid_iv'

# experiments_list = [[2,2,2,2,0.1], [2,2,2,2,0.2], [2,2,2,2,0.3], [2,2,2,2,0.5]]
# experiments_list = [[3,3,3,3,0.1], [3,3,3,3,0.2], [3,3,3,3,0.3], [3,3,3,3,0.5]]
experiments_list = [[4,4,4,4,0.1], [4,4,4,4,0.2], [4,4,4,4,0.3], [4,4,4,4,0.5]]




cce_reject_list = []
ineq_reject_list = []
valid_list = []
entr_list = [] ## this is for plotting the bar plot
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



    for i_dist in tqdm(range(num_dist)):
        valid = True
        
        total += 1
        Hu = entropy(u[i_dist])
        
        ## find the entries where Iyz_iv_arr and Izw_iv_arr both equal to zero

        Hw_iv_arr = np.load('experiments/{}/s{}_Hw_iv.npy'.format(folder_name, i_dist))
        Iyz_iv_arr = np.load('experiments/{}/s{}_Iyz_iv.npy'.format(folder_name, i_dist))
        Izw_iv_arr = np.load('experiments/{}/s{}_Izw_iv.npy'.format(folder_name, i_dist))

        entr_list.append(Hu)

        valid_idx = np.where((Iyz_iv_arr <= eps) & (Izw_iv_arr <= eps))[0]

        if not iv_inequality_constraint_check(yx_z[i_dist]):
            iv_ineq_count += 1
            valid = False
            ineq_reject_list.append(1)
        else:
            ineq_reject_list.append(0)

        if len(valid_idx) == 0:
            CCE_count += 1
            valid = False
            cce_reject_list.append(1)
        else:
            cce_iv = Hw_iv_arr[valid_idx].min()
            if cce_iv - Hu >= eps:
                CCE_count += 1
                valid = False
                cce_reject_list.append(1)
            else:
                cce_reject_list.append(0)
            cce_iv_list.append(cce_iv)
            Hu_list.append(Hu)
        if valid:
            valid_count += 1




print("Experiment: ", folder_name)
print("Number of samples: ", total)
print("Number of samples rejected by IV inequalities: ", iv_ineq_count)
print("Number of samples rejected by CCE: ", CCE_count)
print("Number of samples did not rejected by both: ", valid_count)
print("")


cce_iv_arr = np.array(cce_iv_list)
cce_reject_arr = np.array(cce_reject_list)
ineq_reject_arr = np.array(ineq_reject_list)
Hu_arr = np.array(Hu_list)

entr_arr = np.array(entr_list)

min_entr = entr_arr.min()
max_entr = entr_arr.max()
print("min_entr: ", min_entr)
print("max_entr: ", max_entr)
group_interval = (max_entr - min_entr)/10
group_idx_list = []




for i in range(10):
    group_idx = np.where((entr_arr >= (min_entr + i*group_interval)) & (entr_arr <= (min_entr + (i+1)*group_interval)))[0]
    group_idx_list.append(group_idx)



## add the rest to the last group
group_idx_list[-1] = np.concatenate((group_idx_list[-1], np.where(entr_arr > (min_entr + 10*group_interval))[0]))


check = 0
ineq_reject_count = []
cce_reject_count = []
total_reject_count = []
total_count = []





for group in group_idx_list:
    print("Number of samples in group: ", group.shape[0])
    total_count.append(group.shape[0])
    print("Number of samples rejected by IV inequalities: ", ineq_reject_arr[group].sum())
    ineq_reject_count.append(ineq_reject_arr[group].sum())
    print("Number of samples rejected by CCE: ", cce_reject_arr[group].sum())
    cce_reject_count.append(cce_reject_arr[group].sum())
    print("Number of samples rejected by both: ", (ineq_reject_arr[group] | cce_reject_arr[group]).sum())
    total_reject_count.append((ineq_reject_arr[group] | cce_reject_arr[group]).sum())
    print("")
    check += group.shape[0]
    ## get the count for where either ineq_reject_arr or cce_reject_arr is 1
    
print(check)
print(entr_arr.shape)


indices = np.arange(10)
width = 0.5


## make a bar plot 
plt.figure(figsize=(10, 10))
plt.bar(np.arange(10), ineq_reject_count, width=width, label='Number of samples rejected rejected by IV Inequality',align='center')
plt.bar(indices, cce_reject_count, width=width/3, label='Number of samples rejected by CCE', align='center')

# draw a dashed line over each bar to denote total number of samples in each group
for i in range(10):
    plt.plot([i-width/2, i+width/2], [total_count[i], total_count[i]], 'k-', linewidth=5)


x_label = ["[{:.2f}, {:.2f}]".format(min_entr + i*group_interval, min_entr + (i+1)*group_interval) for i in range(10)]


plt.xlabel('H(U)', fontsize=20)
plt.ylabel('Number of samples', fontsize=20)
plt.xticks(np.arange(10), x_label , fontsize=10)
plt.yticks(fontsize=20)

## add legend for dashed line
plt.plot([], 'k--', linewidth=2, label='Number of samples')
# plt.legend(fontsize=20)
## legend position
plt.legend(fontsize=15)

plt.show()




