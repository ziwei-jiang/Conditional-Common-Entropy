import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm
from utils import generate_iv_samples, mutual_information, generate_conditional_dist, entropy, smoothing, conditional_latent_search, conditional_latent_search_iv, generate_invalid_iv_samples
from utils import mutual_information, conditional_mutual_information, generate_conditional_dist, entropy, conditional_latent_search_iv, generate_two_cov_samples



## set the seed 
np.random.seed(0)


# # ################################   Load the data   ################################

# ## Read the csv file
df = pd.read_csv('diabetes.csv', sep=',', header=0)

'''
Variables in the data:

Glucose
BloodPressure

Insulin
BMI

Age

'''

# show a histogeams of for value of each variable


# preg = df['Pregnancies']
gluc = df['Glucose']
bp = df['BloodPressure']
skin = df['SkinThickness']
insulin = df['Insulin']
bmi = df['BMI']
age = df['Age']
# outcome = df['Outcome']


bin_nums = 2

### cluster the gluc variables into 3 groups, below 100, 100-125, and above 125
df['Glucose'] = pd.cut(df['Glucose'], bins=[0, 125, 200], labels=False, right=True)
### cluster other variables
df['BloodPressure'] = pd.cut(df['BloodPressure'] , bins=[0,80, 200], labels=False)
df['Insulin'] = pd.cut(df['Insulin'], bins=[0,85,400], labels=False)
df['BMI'] = pd.cut(df['BMI'], bins=[0,30,70], labels=False)


### Form a joint distribution of the data

x_states = len(df['Glucose'].unique())
y_states = len(df['BloodPressure'].unique())
z_states = len(df['Insulin'].unique())
v_states = len(df['BMI'].unique())
# u_states = len(df['Age'].unique())



pxyzv = np.zeros((x_states, y_states, z_states, v_states))

for i in range(x_states):
    for j in range(y_states):
        for k in range(z_states):
            for l in range(v_states):
                pxyzv[i,j,k,l] = len(df[(df['Glucose']==i) & (df['BloodPressure']==j) & (df['Insulin']==k) & (df['BMI']==l)])

pxyzv = pxyzv/pxyzv.sum()
pxyzv = smoothing(pxyzv, 1e-6)

pyxzv = pxyzv.transpose(1,0,2,3)

pyXv = pyxzv.reshape(y_states, x_states*z_states, v_states)
pyXz = pyxzv.reshape(y_states, x_states*v_states, z_states)

pxyz = pyXz.transpose(1,0,2)
pxyv = pyXv.transpose(1,0,2)

b = np.linspace(0.5, 0, 100)


Hw_z_list = []
Iyz_list = []
Izw_list = []
b0s = []
b1s = []

Hw_v_list = []
Iyv_list = []
Ivw_list = []

u_states = 4

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




## save the results
Hw_z_arr = np.array(Hw_z_list)
Hw_v_arr = np.array(Hw_v_list)

Iyz_arr = np.array(Iyz_list)
Izw_arr = np.array(Izw_list)

Iyv_arr = np.array(Iyv_list)
Ivw_arr = np.array(Ivw_list)

## save to npy files

np.save('Hw_z_arr.npy', Hw_z_arr)
np.save('Hw_v_arr.npy', Hw_v_arr)

np.save('Iyz_arr.npy', Iyz_arr)
np.save('Izw_arr.npy', Izw_arr)

np.save('Iyv_arr.npy', Iyv_arr)
np.save('Ivw_arr.npy', Ivw_arr)



Hw_z_arr = np.load('Hw_z_arr.npy')
Hw_v_arr = np.load('Hw_v_arr.npy')

Iyz_arr = np.load('Iyz_arr.npy')
Izw_arr = np.load('Izw_arr.npy')

Iyv_arr = np.load('Iyv_arr.npy')
Ivw_arr = np.load('Ivw_arr.npy')


eps = 3e-5
valid_idx_z = np.where((Iyz_arr <= eps) & (Izw_arr <= eps))[0]
valid_idx_v = np.where((Iyv_arr <= eps) & (Ivw_arr <= eps))[0]

cce_z = Hw_z_arr[valid_idx_z].min()
cce_v = Hw_v_arr[valid_idx_v].min()

print("CCE for z: ", cce_z)
print("CCE for v: ", cce_v)
    
    
    
    
    
    
    


