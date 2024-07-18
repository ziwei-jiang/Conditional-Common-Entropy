import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
from tqdm import tqdm
from utils import mutual_information, conditional_mutual_information, generate_conditional_dist, entropy, conditional_latent_search_iv, generate_three_cov_samples_one_val



np.random.seed(0)

################################   generate random the data   ################################

## set the number of states
alpha_u = 1
x_states = 4
y_states = 4
z_states = 4
v1_states = 4
v2_states = 4
u_states = 4
num_dist = 100


y_xzv1v2u, x_zv1v2u, z, v1, v2, u = generate_three_cov_samples_one_val(z_state_nums=z_states, u_state_nums=u_states, v1_state_nums=v1_states, v2_state_nums=v2_states, x_state_nums=x_states, y_state_nums=y_states, num_dist=num_dist, alpha_u=alpha_u, alpha_z=1, alpha_x=1, alpha_y=-1, alpha_v1=1, alpha_v2=1)


zv1v2u = np.einsum('nk, nl, nm, no -> nklmo', z, v1, v2, u)
xzv1v2u = np.einsum('njklmo, nklmo -> njklmo', x_zv1v2u, zv1v2u)
yxzv1v2u = np.einsum('nijklmo, njklmo -> nijklmo', y_xzv1v2u, xzv1v2u)


## combine the variable x and v1, v2 into one variable
yxv1v2zu = yxzv1v2u.transpose(0, 1, 2, 4, 5, 3, 6)
yXzu = yxv1v2zu.reshape(num_dist, y_states, x_states*v1_states*v2_states, z_states, u_states)

## combine the variable x and z, v2 into one variable
yxzv2v1u = yxzv1v2u.transpose(0, 1, 2, 3, 5, 4, 6)
yXv1u = yxzv1v2u.reshape(num_dist, y_states, x_states*z_states*v2_states, v1_states, u_states)

## combine the variable x and z, v1 into one variable
yXv2u = yxzv1v2u.reshape(num_dist, y_states, x_states*z_states*v1_states, v2_states, u_states)



folder_name = 'x{}_y{}_z{}_1v{}_2v{}_u{}_a{}'.format(x_states, y_states, z_states, v1_states, v2_states, u_states, str(alpha_u).replace('.', ''))


## make the folder
os.makedirs('experiments/Three_cov_one_val/{}'.format(folder_name), exist_ok=False)


## save the data to npy file
np.save('experiments/Three_cov_one_val/{}/y_xzv1v2u.npy'.format(folder_name), y_xzv1v2u)
np.save('experiments/Three_cov_one_val/{}/x_zv1v2u.npy'.format(folder_name), x_zv1v2u)
np.save('experiments/Three_cov_one_val/{}/z.npy'.format(folder_name), z)
np.save('experiments/Three_cov_one_val/{}/v1.npy'.format(folder_name), v1)
np.save('experiments/Three_cov_one_val/{}/v2.npy'.format(folder_name), v2)
np.save('experiments/Three_cov_one_val/{}/u.npy'.format(folder_name), u)



b = np.linspace(0.5, 0, 100)


print("Experiment: ", folder_name)


################################   Searching for Minimum Entropy Confounder for IV graph   ################################



for i_dist in range(num_dist):
    print("sample {}".format(i_dist))
    Hu = entropy(u[i_dist])

    pyxzu = yXzu[i_dist]
    pyxz = pyxzu.sum(axis=(3))
    pxyz = pyxz.transpose(1,0,2)
    
    pyxv1u = yXv1u[i_dist]
    pyxv1 = pyxv1u.sum(axis=(3))
    pxyv1 = pyxv1.transpose(1,0,2)

    pyxv2u = yXv2u[i_dist]
    pyxv2 = pyxv2u.sum(axis=(3))
    pxyv2 = pyxv2.transpose(1,0,2)


    Hw_z_list = []
    Iyz_list = []
    Izw_list = []
    b0s = []
    b1s = []

    Hw_v1_list = []
    Iyv1_list = []
    Iv1w_list = []

    Hw_v2_list = []
    Iyv2_list = []
    Iv2w_list = []
    


    ## iterate over b
    for beta0 in tqdm(b):
        for beta1 in (b):
            ## generate the conditional distribution for yXzu
            qw_xyz = generate_conditional_dist(num_states=u_states, num_parents_states=v1_states*v2_states*x_states*y_states*z_states, num_dist=1, alpha=1)
            qw_xyz = qw_xyz.reshape(u_states, x_states*v1_states*v2_states, y_states, z_states)
            qw_xyz = conditional_latent_search_iv(pxyz, u_states, qw_xyz, iter_num=200, beta0=beta0+beta1, beta1=beta1) 
            ## get the entropy of W
            qw__z = np.einsum('lijk, ijk->l', qw_xyz, pxyz)
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


            ## generate the conditional distribution for yXv1u
            pw_xyv1 = generate_conditional_dist(num_states=u_states, num_parents_states=z_states*x_states*y_states*v1_states*v2_states, num_dist=1, alpha=1)
            pw_xyv1 = pw_xyv1.reshape(u_states, x_states*z_states*v2_states, y_states, v1_states)
            pw_xyv1 = conditional_latent_search_iv(pxyv1, u_states, pw_xyv1, iter_num=200, beta0=beta0+beta1, beta1=beta1)
            ## get the entropy of W
            pw__v1 = np.einsum('lijk, ijk->l', pw_xyv1, pxyv1)
            ## get the conditional mutual information
            pxyv1w = np.einsum('lijk, ijk->ijkl', pw_xyv1, pxyv1)
            pyv1xw = pxyv1w.transpose(1,2,0,3)
            pxw = pyv1xw.sum(axis=(0,1))
            pyv1_xw = np.einsum('ijkl, kl -> ijkl', pyv1xw, np.divide(1, pxw, where=(pxw>0)))
            pyv1_xw = pyv1_xw.reshape(y_states, v1_states, -1)
            pxw = pxw.reshape(-1)
            Iyv1_xw = conditional_mutual_information(pyv1_xw, pxw)

            pv1w = pyv1xw.sum(axis=(0,2))
            Iv1w = mutual_information(pv1w)
            

            ## generate the conditional distribution for yXv2u
            pw_xyv2 = generate_conditional_dist(num_states=u_states, num_parents_states=z_states*x_states*y_states*v1_states*v2_states, num_dist=1, alpha=1)
            pw_xyv2 = pw_xyv2.reshape(u_states, x_states*z_states*v1_states, y_states, v2_states)
            pw_xyv2 = conditional_latent_search_iv(pxyv2, u_states, pw_xyv2, iter_num=200, beta0=beta0+beta1, beta1=beta1)
            ## get the entropy of W
            pw__v2 = np.einsum('lijk, ijk->l', pw_xyv2, pxyv2)
            ## get the conditional mutual information
            pxyv2w = np.einsum('lijk, ijk->ijkl', pw_xyv2, pxyv2)
            pyv2xw = pxyv2w.transpose(1,2,0,3)
            pxw = pyv2xw.sum(axis=(0,1))
            pyv2_xw = np.einsum('ijkl, kl -> ijkl', pyv2xw, np.divide(1, pxw, where=(pxw>0)))
            pyv2_xw = pyv2_xw.reshape(y_states, v2_states, -1)
            pxw = pxw.reshape(-1)
            Iyv2_xw = conditional_mutual_information(pyv2_xw, pxw)

            pv2w = pyv2xw.sum(axis=(0,2))
            Iv2w = mutual_information(pv2w)

            
            b0s.append(beta0+beta1)
            b1s.append(beta1)
            
            Hw_z_list.append(entropy(qw__z))
            Hw_v1_list.append(entropy(pw__v1))
            Hw_v2_list.append(entropy(pw__v2))

            Izw_list.append(Izw)
            Iyz_list.append(Iyz_xw)

            Iv1w_list.append(Iv1w)
            Iyv1_list.append(Iyv1_xw)

            Iv2w_list.append(Iv2w)
            Iyv2_list.append(Iyv2_xw)

    ## save the entropy of W
    Hw_z_arr = np.array(Hw_z_list)
    Hw_v1_arr = np.array(Hw_v1_list)
    Hw_v2_arr = np.array(Hw_v2_list)

    Iyz_arr = np.array(Iyz_list)
    Izw_arr = np.array(Izw_list)

    Iyv1_arr = np.array(Iyv1_list)
    Iv1w_arr = np.array(Iv1w_list)

    Iyv2_arr = np.array(Iyv2_list)
    Iv2w_arr = np.array(Iv2w_list)


    b0_arr = np.array(b0s)
    b1_arr = np.array(b1s)

    ## save z data to a signle csv file with only 3 digits after the decimal point
    df1 = pd.DataFrame({'b0': b0_arr, 'b1': b1_arr, 'Hw_z': Hw_z_arr, 'Iyz_xu': Iyz_arr, 'Izw': Izw_arr, 'Hu': Hu})
    df1.to_csv('experiments/Three_cov_one_val/{}/s{}_plot_z.csv'.format(folder_name, i_dist), index=False, float_format='%.3f')

    ## save v1 data to a signle csv file with only 3 digits after the decimal point
    df2 = pd.DataFrame({'b0': b0_arr, 'b1': b1_arr, 'Hw_v1': Hw_v1_arr, 'Iyv1_xu': Iyv1_arr, 'Iv1w': Iv1w_arr, 'Hu': Hu})
    df2.to_csv('experiments/Three_cov_one_val/{}/s{}_plot_v1.csv'.format(folder_name, i_dist), index=False, float_format='%.3f')

    ## save v2 data to a signle csv file with only 3 digits after the decimal point
    df3 = pd.DataFrame({'b0': b0_arr, 'b1': b1_arr, 'Hw_v2': Hw_v2_arr, 'Iyv2_xu': Iyv2_arr, 'Iv2w': Iv2w_arr, 'Hu': Hu})
    df3.to_csv('experiments/Three_cov_one_val/{}/s{}_plot_v2.csv'.format(folder_name, i_dist), index=False, float_format='%.3f')

    ## save H and Iyz_xu as npy file to the folder
    np.save('experiments/Three_cov_one_val/{}/s{}_Hw_z.npy'.format(folder_name, i_dist), Hw_z_arr)
    np.save('experiments/Three_cov_one_val/{}/s{}_Iyz.npy'.format(folder_name, i_dist), Iyz_arr)
    np.save('experiments/Three_cov_one_val/{}/s{}_Izw.npy'.format(folder_name, i_dist), Izw_arr)

    np.save('experiments/Three_cov_one_val/{}/s{}_Hw_v1.npy'.format(folder_name, i_dist), Hw_v1_arr)
    np.save('experiments/Three_cov_one_val/{}/s{}_Iyv1.npy'.format(folder_name, i_dist), Iyv1_arr)
    np.save('experiments/Three_cov_one_val/{}/s{}_Iv1w_.npy'.format(folder_name, i_dist), Iv1w_arr)

    np.save('experiments/Three_cov_one_val/{}/s{}_Hw_v2.npy'.format(folder_name, i_dist), Hw_v2_arr)
    np.save('experiments/Three_cov_one_val/{}/s{}_Iyv2.npy'.format(folder_name, i_dist), Iyv2_arr)
    np.save('experiments/Three_cov_one_val/{}/s{}_Iv2w.npy'.format(folder_name, i_dist), Iv2w_arr)
    

print("Experiment: ", folder_name)
