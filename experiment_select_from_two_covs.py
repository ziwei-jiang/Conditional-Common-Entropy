import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
from tqdm import tqdm
from utils import mutual_information, conditional_mutual_information, generate_conditional_dist, entropy, conditional_latent_search_iv, generate_two_cov_samples






np.random.seed(0)

################################   generate random the data   ################################

## set the number of states
alpha_u = 1
x_states = 3
y_states = 3
z_states = 3
v_states = 3
u_states = 3
num_dist = 100



y_xzvu, x_zvu, z, v, u = generate_two_cov_samples(z_state_nums=z_states, u_state_nums=u_states, v_state_nums=v_states, x_state_nums=x_states, y_state_nums=y_states, num_dist=num_dist, alpha_u=alpha_u, alpha_z=1, alpha_x=1, alpha_y=-1, alpha_v=1)



zvu = np.einsum('nk, nl, nm -> nklm', z, v, u)
xzvu = np.einsum('njklm, nklm -> njklm', x_zvu, zvu)
yxzvu = np.einsum('nijklm, njklm -> nijklm', y_xzvu, xzvu)



## combine the variable x and z into one variable 
yXvu = yxzvu.reshape(num_dist, y_states, x_states*z_states, v_states, u_states)

## combien the variable x and v into one variable 
yxvzu = yxzvu.transpose(0, 1, 2, 4, 3, 5)
yXzu = yxvzu.reshape(num_dist, y_states, x_states*v_states, z_states, u_states)


folder_name = 'x{}_y{}_z{}_v{}_u{}_a{}'.format(x_states, y_states, z_states, v_states, u_states, str(alpha_u).replace('.', ''))


## make the folder
os.makedirs('experiments/Two_cov/{}'.format(folder_name), exist_ok=False)


## save the data to npy file
np.save('experiments/Two_cov/{}/y_xzvu.npy'.format(folder_name), y_xzvu)
np.save('experiments/Two_cov/{}/x_zvu.npy'.format(folder_name), x_zvu)
np.save('experiments/Two_cov/{}/z.npy'.format(folder_name), z)
np.save('experiments/Two_cov/{}/v.npy'.format(folder_name), v)
np.save('experiments/Two_cov/{}/u.npy'.format(folder_name), u)



b = np.linspace(0.5, 0, 100)

print("Experiment: ", folder_name)


################################   Searching for Minimum Entropy Confounder for IV graph   ################################



for i_dist in range(num_dist):
    print("sample {}".format(i_dist))
    Hu = entropy(u[i_dist])

    pyxzu = yXzu[i_dist]
    pyxz = pyxzu.sum(axis=(3))
    pxyz = pyxz.transpose(1,0,2)
    
    pyxvu = yXvu[i_dist]
    pyxv = pyxvu.sum(axis=(3))
    pxyv = pyxv.transpose(1,0,2)


    Hw_z_list = []
    Iyz_list = []
    Izw_list = []
    b0s = []
    b1s = []

    Hw_v_list = []
    Iyv_list = []
    Ivw_list = []


    ## iterate over b
    for beta0 in tqdm(b):
        for beta1 in (b):
            ## generate the conditional distribution for yXzu
            qw_xyz = generate_conditional_dist(num_states=u_states, num_parents_states=v_states*x_states*y_states*z_states, num_dist=1, alpha=1)
            qw_xyz = qw_xyz.reshape(u_states, x_states*v_states, y_states, z_states)
            qw_xyz = conditional_latent_search_iv(pxyz, u_states, qw_xyz, iter_num=300, beta0=beta0+beta1, beta1=beta1) 
            ## get the entropy of W
            qw = np.einsum('lijk, ijk->l', qw_xyz, pxyz)
            ## get the conditional mutual information
            qxyzw = np.einsum('lijk, ijk->ijkl', qw_xyz, pxyz)
            qyzxw = qxyzw.transpose(1,2,0,3)
            qxw = qyzxw.sum(axis=(0,1))
            qyz_xw = np.einsum('ijkl, kl -> ijkl', qyzxw, np.divide(1, qxw, where=(qxw>0)))
            
            qxw = qxw.reshape(-1)
            qyz_xw = qyz_xw.reshape(y_states, z_states, -1)
            Iyz_xw = conditional_mutual_information(qyz_xw, qxw)

            qzw = qyzxw.sum(axis=(0,2))
            Izw = mutual_information(qzw)


            ## generate the conditional distribution for yXvu
            pw_xyv = generate_conditional_dist(num_states=u_states, num_parents_states=z_states*x_states*y_states*v_states, num_dist=1, alpha=1)
            pw_xyv = pw_xyv.reshape(u_states, x_states*z_states, y_states, v_states)
            pw_xyv = conditional_latent_search_iv(pxyv, u_states, pw_xyv, iter_num=300, beta0=beta0+beta1, beta1=beta1)
            ## get the entropy of W
            pw = np.einsum('lijk, ijk->l', pw_xyv, pxyv)
            ## get the conditional mutual information
            pxyvw = np.einsum('lijk, ijk->ijkl', pw_xyv, pxyv)
            pyvxw = pxyvw.transpose(1,2,0,3)
            pxw = pyvxw.sum(axis=(0,1))
            pyv_xw = np.einsum('ijkl, kl -> ijkl', pyvxw, np.divide(1, pxw, where=(pxw>0)))
            pyv_xw = pyv_xw.reshape(y_states, v_states, -1)
            pxw = pxw.reshape(-1)
            Iyv_xw = conditional_mutual_information(pyv_xw, pxw)

            pvw = pyvxw.sum(axis=(0,2))
            Ivw = mutual_information(pvw)
            
            
            b0s.append(beta0+beta1)
            b1s.append(beta1)
            
            Hw_z_list.append(entropy(qw))
            Hw_v_list.append(entropy(pw))

            Izw_list.append(Izw)
            Iyz_list.append(Iyz_xw)

            Ivw_list.append(Ivw)
            Iyv_list.append(Iyv_xw)


    ## save the entropy of W
    Hw_z_arr = np.array(Hw_z_list)
    Hw_v_arr = np.array(Hw_v_list)

    Iyz_arr = np.array(Iyz_list)
    Izw_arr = np.array(Izw_list)

    Iyv_arr = np.array(Iyv_list)
    Ivw_arr = np.array(Ivw_list)

    b0_arr = np.array(b0s)
    b1_arr = np.array(b1s)

    ## save z data to a signle csv file with only 3 digits after the decimal point
    df1 = pd.DataFrame({'b0': b0_arr, 'b1': b1_arr, 'Hw_z': Hw_z_arr, 'Iyz_xu': Iyz_arr, 'Izw': Izw_arr, 'Hu': Hu})
    df1.to_csv('experiments/Two_cov/{}/s{}_plot_z.csv'.format(folder_name, i_dist), index=False, float_format='%.3f')

    ## save v data to a signle csv file with only 3 digits after the decimal point
    df2 = pd.DataFrame({'b0': b0_arr, 'b1': b1_arr, 'Hw_v': Hw_v_arr, 'Iyv_xu': Iyv_arr, 'Ivw': Ivw_arr, 'Hu': Hu})
    df2.to_csv('experiments/Two_cov/{}/s{}_plot_v.csv'.format(folder_name, i_dist), index=False, float_format='%.3f')

    ## save H and Iyz_xu as npy file to the folder
    np.save('experiments/Two_cov/{}/s{}_Hw_z.npy'.format(folder_name, i_dist), Hw_z_arr)
    np.save('experiments/Two_cov/{}/s{}_Iyz.npy'.format(folder_name, i_dist), Iyz_arr)
    np.save('experiments/Two_cov/{}/s{}_Izw.npy'.format(folder_name, i_dist), Izw_arr)

    np.save('experiments/Two_cov/{}/s{}_Hw_v.npy'.format(folder_name, i_dist), Hw_v_arr)
    np.save('experiments/Two_cov/{}/s{}_Iyv.npy'.format(folder_name, i_dist), Iyv_arr)
    np.save('experiments/Two_cov/{}/s{}_Ivw_.npy'.format(folder_name, i_dist), Ivw_arr)

print("Experiment: ", folder_name)
