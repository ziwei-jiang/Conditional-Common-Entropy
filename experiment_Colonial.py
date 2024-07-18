import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm
from utils import generate_iv_samples, mutual_information, generate_conditional_dist, entropy, smoothing, conditional_latent_search, conditional_latent_search_iv, generate_invalid_iv_samples
from utils import mutual_information, conditional_mutual_information, generate_conditional_dist, generate_two_cov_samples

## set the seed 
np.random.seed(0)


# ################################   Load the data   ################################
## Read the csv file
df = pd.read_csv('ajrcomment.csv', sep=',', header=0)

## take the logmort0, loggdp, and risk columns 
df = df[['logmort0', 'loggdp', 'risk', 'latitude']]


## convert the data to integer
df['logmort0'] = np.digitize(df['logmort0'], bins=[4,6])
df['loggdp'] = np.digitize(df['loggdp'], bins=[7,8,9])
df['risk'] = np.digitize(df['risk'], bins=[5,6,7,8])
df['latitude'] = np.digitize(df['latitude'], bins=[0.1,0.2,0.3,0.4])



## counter the number of unique values in each column
logmort0 = df['logmort0'].unique()
loggdp = df['loggdp'].unique()
risk = df['risk'].unique()
latitude = df['latitude'].unique()


x_states = len(risk)
y_states = len(loggdp)
z_states = len(logmort0)
v_states = len(latitude)
u_states = 10


## form the joint distribution
pxyzv = np.zeros((x_states, y_states, z_states, v_states))
for i in range(len(risk)):
    for j in range(len(loggdp)):
        for k in range(len(logmort0)):
            for l in range(len(latitude)):
                pxyzv[i,j,k,l] = len(df[(df['risk']==i) & (df['loggdp']==j) & (df['logmort0']==k) & (df['latitude']==l)])

pxyzv = pxyzv/pxyzv.sum()

print(pxyzv.sum())
print(pxyzv.shape)
print(pxyzv.min(), pxyzv.max())

pyxzv = pxyzv.transpose(1,0,2,3)

pyXv = pyxzv.reshape(y_states, x_states*z_states, v_states)

pyxvz = pyxzv.transpose(0,1,3,2)
pyXz = pyxvz.reshape(y_states, x_states*v_states, z_states)


## iterate over b

Hw_z_list = []
Iyz_list = []
Izw_list = []
b0s = []
b1s = []

Hw_v_list = []
Iyv_list = []
Ivw_list = []

b = np.linspace(0.5, 0, 100)

pxyz = pyXz.transpose(1,0,2)
pxyv = pyXv.transpose(1,0,2)
for beta0 in tqdm(b):
    for beta1 in (b):
        ## generate the conditional distribution for yXzu
        qw_xyz = generate_conditional_dist(num_states=u_states, num_parents_states=v_states*x_states*y_states*z_states, num_dist=1, alpha=1)
        qw_xyz = qw_xyz.reshape(u_states, x_states*v_states, y_states, z_states)
        qw_xyz = conditional_latent_search_iv(pxyz, u_states, qw_xyz, iter_num=200, beta0=beta0+beta1, beta1=beta1) 
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
        pw_xyv = conditional_latent_search_iv(pxyv, u_states, pw_xyv, iter_num=200, beta0=beta0+beta1, beta1=beta1)
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
df1 = pd.DataFrame({'b0': b0_arr, 'b1': b1_arr, 'Hw_z': Hw_z_arr, 'Iyz_xu': Iyz_arr, 'Izw': Izw_arr})
df1.to_csv('experiments/colonial/plot_z.csv', index=False, float_format='%.3f')

## save v data to a signle csv file with only 3 digits after the decimal point
df2 = pd.DataFrame({'b0': b0_arr, 'b1': b1_arr, 'Hw_v': Hw_v_arr, 'Iyv_xu': Iyv_arr, 'Ivw': Ivw_arr})
df2.to_csv('experiments/colonial/plot_v.csv', index=False, float_format='%.3f')

## save H and Iyz_xu as npy file to the folder
np.save('experiments/colonial/Hw_z.npy', Hw_z_arr)
np.save('experiments/colonial/Iyz.npy', Iyz_arr)
np.save('experiments/colonial/Izw.npy', Izw_arr)

np.save('experiments/colonial/Hw_v.npy', Hw_v_arr)
np.save('experiments/colonial/Iyv.npy', Iyv_arr)
np.save('experiments/colonial/Ivw_.npy', Ivw_arr)

