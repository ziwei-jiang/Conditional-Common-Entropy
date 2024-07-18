import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm
from utils import generate_iv_samples, mutual_information, mutual_information_vec, conditional_mutual_information, conditional_mutual_information_vec, generate_conditional_dist, entropy, conditional_latent_search, conditional_latent_search_iv, generate_invalid_iv_samples, generate_weakly_invalid_iv_samples
from mpl_toolkits.mplot3d import Axes3D
import os




np.random.seed(0)

################################   generate random the data   ################################

## set the number of states
alpha_u = 0.2
x_states = 8
y_states = 8
z_states = 8
u_states = 4
num_dist = 100
# graph = 'invalid_iv'
# graph = 'weak_zy'
graph= 'iv'



if graph == 'iv':
    y_xzu, x_zu, z, u = generate_iv_samples(z_state_nums=z_states, u_state_nums=u_states, x_state_nums=x_states, y_state_nums=y_states, num_dist=num_dist, alpha_u=alpha_u, alpha_z=1, alpha_x=1, alpha_y=-1)

elif graph == 'weak_zy':
    y_xzu, x_zu, z, u, m = generate_weakly_invalid_iv_samples(z_state_nums=z_states, u_state_nums=u_states, m_state_nums=2, x_state_nums=x_states, y_state_nums=y_states, num_dist=num_dist, alpha_u=alpha_u, alpha_z=1, alpha_x=1, alpha_y=-1, alpha_m=alpha_u)

else:
    y_xzu, x_zu, z, u = generate_invalid_iv_samples(z_state_nums=z_states, u_state_nums=u_states, x_state_nums=x_states, y_state_nums=y_states, num_dist=num_dist, alpha_u=alpha_u, alpha_z=1, alpha_x=1, alpha_y=-1)



# qyxzu = np.einsum('nijkl, jkl, k, l-> ijkl', y_xzu, x_zu, z, u)
zu = np.einsum('nk, nl -> nkl', z, u)
xzu = np.einsum('njkl, nkl -> njkl', x_zu, zu)
yxzu = np.einsum('nijkl, njkl -> nijkl', y_xzu, xzu)

xz = xzu.sum(axis=3)
yz = yxzu.sum(axis=(2,4))
x = xz.sum(axis=2)
yxz = yxzu.sum(axis=4)
yzx = yxz.transpose(0,1,3,2)
yz_x = np.einsum('nijk, nk -> nijk', yzx, np.divide(1, x, where=(x>0)))
Iyz_x = conditional_mutual_information_vec(yz_x, x)

Iyz_arr = mutual_information_vec(yz)
Ixz_arr = mutual_information_vec(xz)

## Create a folder for this experiment trial inside the experiments folder
folder_name = '{}_x{}_y{}_z{}_u{}_a{}'.format(graph, x_states, y_states, z_states, u_states, str(alpha_u).replace('.', ''))

## make the folder
os.makedirs('experiments/{}'.format(folder_name), exist_ok=False)


## save the data to npy file
np.save('experiments/{}/y_xzu.npy'.format(folder_name), y_xzu)
np.save('experiments/{}/x_zu.npy'.format(folder_name), x_zu)
np.save('experiments/{}/z.npy'.format(folder_name), z)
np.save('experiments/{}/u.npy'.format(folder_name), u)

if graph == 'weak_zy':
    np.save('experiments/{}/m.npy'.format(folder_name), m)


