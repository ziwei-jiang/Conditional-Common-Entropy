import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm
from utils import generate_iv_samples, mutual_information, mutual_information_vec, generate_conditional_dist, entropy, conditional_latent_search, conditional_latent_search_iv, generate_invalid_iv_samples, optimization_entr, get_natural_bounds, entropy_vec, mutual_information_vec, get_tp_bounds


np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


np.random.seed(0)

## set the number of states
alpha_u = 0.2
x_states = 3
y_states = 3
z_states = 3
u_states = 3
num_dist = 100
# graph = 'invalid_iv'
graph= 'iv'
# graph = 'weak_zy'





folder_name = '{}_x{}_y{}_z{}_u{}_a{}'.format(graph, x_states, y_states, z_states, u_states, str(alpha_u).replace('.', ''))
## load the data from npy file
y_xzu = np.load('experiments/{}/y_xzu.npy'.format(folder_name))
x_zu = np.load('experiments/{}/x_zu.npy'.format(folder_name))
z = np.load('experiments/{}/z.npy'.format(folder_name))
u = np.load('experiments/{}/u.npy'.format(folder_name))



if graph == 'weak_zy':
    m = np.load('experiments/{}/m.npy'.format(folder_name))
    Hm = entropy_vec(m)


try:
    
    print("Check if the bounds are already computed")
    ## load the results from npy file
    natural_bounds_arr = np.load('experiments/{}/natural_bounds.npy'.format(folder_name))
    tp_bounds_arr = np.load('experiments/{}/tp_bounds.npy'.format(folder_name))
    entr_bounds_arr = np.load('experiments/{}/entr_bounds.npy'.format(folder_name))
    ydox_arr = np.load('experiments/{}/ydox.npy'.format(folder_name))
    entropy_arr = np.load('experiments/{}/entropy.npy'.format(folder_name))

    print("Bounds are already computed")

except:

    print("Bounds has not been computed, computing the bounds")
    
    yx_zu  = np.einsum('nijkl, njkl -> nijkl', y_xzu, x_zu)
    yx_z = np.einsum('nijkl, nl -> nijk', yx_zu, u)
    x_z = np.einsum('njkl, nl -> njk', x_zu, u)

    zu = np.einsum('nk, nl -> nkl', z, u)
    xzu = np.einsum('njkl, nkl -> njkl', x_zu, zu)
    yxzu = np.einsum('nijkl, njkl -> nijkl', y_xzu, xzu)
    x = xzu.sum(axis=(2,3))
    xz = xzu.sum(axis=3)

    yxz = yxzu.sum(axis=4)
    yx = yxz.sum(axis=3)
    yzx = yxz.transpose(0, 1, 3, 2)
    yz_x = np.einsum('nijk, nk -> nijk', yzx, np.divide(1, x))
    y_x = yz_x.sum(axis=2)
    yz = yzx.sum(axis=3)
    Iyz_arr = mutual_information_vec(yz)
    Ixz_arr = mutual_information_vec(xz)
    Izu_arr = mutual_information_vec(zu)

    y_xu = y_xzu[:, :, :, 0, :]
    ydox = np.einsum('nijk, nk -> nij', y_xu, u)

    y_xz = np.einsum('nijk, nk -> nijk', yxz, np.divide(1, z))

    natural_bounds_list = []
    tp_bounds_list = []

    entr_bounds_list = []
    entropy_list = []
    ydox_list = []
    

    dist_range = tqdm(range(num_dist))
    print("Experiment: ", folder_name)
    for i_dist in dist_range:
    # for i_dist in range(9, num_dist):
        Iyz = Iyz_arr[i_dist]
        Ixz = Ixz_arr[i_dist]
        entr_u = entropy(u[i_dist])


        
        if graph == 'weak_zy':
            phi = Hm[i_dist]
        elif graph == 'iv':
            phi = 0

        
        # if Izu_arr[i_dist] >= 1e-5:
        #     print('Izu is zero')
        #     continue

        

        for y_idx in range(y_states):
            for x_idx in range(x_states):

                natural_bounds = get_natural_bounds(yx_z[i_dist], y_idx=y_idx, x_idx=x_idx)
                tp_bounds = get_tp_bounds(yx[i_dist], y_idx=y_idx, x_idx=x_idx)
                entr_bounds = optimization_entr(yx_z[i_dist], z[i_dist], y_idx=y_idx, x_idx=x_idx, entr=entr_u, izy=phi)
                

                ## check if the bound is not inf
                if np.isinf(entr_bounds[0]) or np.isinf(entr_bounds[1]):
                    print('distribution {} error'.format(i_dist))
                    print('entropy : {}'.format(entr_u))
                    print('entr_bounds {}'.format(entr_bounds))
                    print('natural_bounds {}'.format(natural_bounds))
                    print("phi: ", phi)
                    quit()


                natural_bounds_list.append(natural_bounds)
                tp_bounds_list.append(tp_bounds)
                entr_bounds_list.append(entr_bounds)
                
                ydox_list.append(ydox[i_dist, y_idx, x_idx])
                entropy_list.append(entr_u)

    natural_bounds_arr = np.array(natural_bounds_list)
    tp_bounds_arr = np.array(tp_bounds_list)
    entr_bounds_arr = np.array(entr_bounds_list)
    
    ydox_arr = np.array(ydox_list)
    
    entropy_arr = np.array(entropy_list)

    # save the results to npy file
    np.save('experiments/{}/natural_bounds.npy'.format(folder_name), natural_bounds_arr)
    np.save('experiments/{}/tp_bounds.npy'.format(folder_name), tp_bounds_arr)
    np.save('experiments/{}/entr_bounds.npy'.format(folder_name), entr_bounds_arr)
    np.save('experiments/{}/ydox.npy'.format(folder_name), ydox_arr)
    np.save('experiments/{}/entropy.npy'.format(folder_name), entropy_arr)
    



print("Experiment: ", folder_name)


## get the index of ydox
idx_ydox = np.argsort(ydox_arr)
## sort the bounds
natural_bounds_arr = natural_bounds_arr[idx_ydox]
entr_bounds_arr = entr_bounds_arr[idx_ydox]
ydox_arr = ydox_arr[idx_ydox]
entropy_arr = entropy_arr[idx_ydox]
tp_bounds_arr = tp_bounds_arr[idx_ydox]




if graph == 'weak_zy':

    ## count the number of entr_bounds that are tighter than natural_bounds
    entr_tighter = 0
    for i in range(entr_bounds_arr.shape[0]):
        if (entr_bounds_arr[i, 0] - tp_bounds_arr[i, 0]>1e-4) or (tp_bounds_arr[i, 1] - entr_bounds_arr[i, 1]  > 1e-4):
            entr_tighter += 1
            # print('query {} '.format(i))
            # print('distribution {}'.format(i//x_states*y_states))
            # print('entropy : {}'.format(entropy_arr[i]))
            # print('entr_bounds {}'.format(entr_bounds_arr[i]))
            # print('natural_bounds {}'.format(tp_bounds_arr[i]))
            # print("")

    print('entr_tighter: {}/{}'.format(entr_tighter, entr_bounds_arr.shape[0]))


else:
    ## count the number of entr_bounds that are tighter than natural_bounds
    entr_tighter = 0
    for i in range(entr_bounds_arr.shape[0]):
        if (entr_bounds_arr[i, 0] - natural_bounds_arr[i, 0]>1e-5) or (natural_bounds_arr[i, 1] - entr_bounds_arr[i, 1]  > 1e-5):
            entr_tighter += 1
            # print('query {} '.format(i))
            # print('distribution {}'.format(i//x_states*y_states))
            # print('entropy : {}'.format(entropy_arr[i]))
            # print('entr_bounds {}'.format(entr_bounds_arr[i]))
            # print('natural_bounds {}'.format(natural_bounds_arr[i]))
            # print("")

    print('entr_tighter: {}/{}'.format(entr_tighter, entr_bounds_arr.shape[0]))




## plot the bounds
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(natural_bounds_arr[:, 0], label='natural lower bound', color='red')
ax.plot(natural_bounds_arr[:, 1], label='natural upper bound', color='red')
ax.plot(entr_bounds_arr[:, 0], label='entropy lower bound', color='blue')
ax.plot(entr_bounds_arr[:, 1], label='entropy upper bound', color='blue')
ax.plot(ydox_arr, label='actual causal effect')


ax.set_xlabel('P(y|do(x))')
ax.set_ylabel('Causal effect')
ax.set_title('Causal effect vs Bounds')
ax.legend()
plt.show()


    
entr_u_arr = entropy_vec(u)

