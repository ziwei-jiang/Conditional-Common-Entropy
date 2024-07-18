import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm
from utils import generate_iv_samples, mutual_information, generate_conditional_dist, entropy, smoothing, conditional_latent_search, conditional_latent_search_iv, generate_invalid_iv_samples
from utils import mutual_information, conditional_mutual_information, generate_conditional_dist, generate_two_cov_samples



## set the seed 
np.random.seed(0)


# # ################################   Load the data   ################################

# ## Read the csv file
df = pd.read_csv('ak91.csv', sep=',', header=0)


'''
Variables in the data:

log_wage
years_of_schooling
quarter_of_birth
state_of_birth

'''



# log_wage = df['log_wage'].values
# years_of_schooling = df['years_of_schooling'].values
# quarter_of_birth = df['quarter_of_birth'].values
# year_of_birth = df['year_of_birth'].values

# log_wage = np.ceil(log_wage)
# ## (0-12), (12-16), (16-20)
# years_of_schooling = np.digitize(years_of_schooling, bins=[12,16])

# wage_bins = np.unique(log_wage)
# schooling_bins = np.unique(years_of_schooling)
# quarter_bins = np.unique(quarter_of_birth)
# year_bins = np.unique(year_of_birth)


# pWageSchoolQuarterYear = np.zeros((len(wage_bins), len(schooling_bins), len(quarter_bins), len(year_bins)))

# print(pWageSchoolQuarterYear.shape)


# for data_idx in tqdm(range(len(log_wage))):
#     wage_idx = np.digitize(log_wage[data_idx], wage_bins)-1
#     schooling_idx = np.digitize(years_of_schooling[data_idx], schooling_bins)-1
#     quarter_idx = np.digitize(quarter_of_birth[data_idx], quarter_bins)-1
#     year_idx = np.digitize(year_of_birth[data_idx], year_bins)-1

#     pWageSchoolQuarterYear[wage_idx, schooling_idx, quarter_idx, year_idx] += 1

# pWageSchoolQuarterYear = pWageSchoolQuarterYear/pWageSchoolQuarterYear.sum()

# print(pWageSchoolQuarterYear.sum())
# print(pWageSchoolQuarterYear.min(), pWageSchoolQuarterYear.max())

# ## save the data
# np.save('ak91.npy', pWageSchoolQuarterYear)

pWageSchoolQuarterYear = np.load('ak91.npy')


pWageSchoolYearQuarter = pWageSchoolQuarterYear.transpose(0,1,3,2)

pWageSQYear = pWageSchoolQuarterYear.reshape(pWageSchoolQuarterYear.shape[0],-1, pWageSchoolQuarterYear.shape[-1])
pWageSYQuarter = pWageSchoolYearQuarter.reshape(pWageSchoolYearQuarter.shape[0],-1, pWageSchoolYearQuarter.shape[-1])

b = np.linspace(0.5, 0, 100)
pxyz = pWageSYQuarter.transpose(1,0,2)
pxyv = pWageSQYear.transpose(1,0,2)

Hw_z_list = []
Iyz_list = []
Izw_list = []
b0s = []
b1s = []

Hw_v_list = []
Iyv_list = []
Ivw_list = []


u_states = 20
x_states = pWageSchoolYearQuarter.shape[1]
y_states = pWageSchoolYearQuarter.shape[0]
z_states = pWageSchoolYearQuarter.shape[3]
v_states = pWageSchoolYearQuarter.shape[2]

## iterate over b
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
df1.to_csv('experiments/schooling/plot_z.csv', index=False, float_format='%.3f')

## save v data to a signle csv file with only 3 digits after the decimal point
df2 = pd.DataFrame({'b0': b0_arr, 'b1': b1_arr, 'Hw_v': Hw_v_arr, 'Iyv_xu': Iyv_arr, 'Ivw': Ivw_arr})
df2.to_csv('experiments/schooling/plot_v.csv', index=False, float_format='%.3f')

## save H and Iyz_xu as npy file to the folder
np.save('experiments/schooling/Hw_z.npy', Hw_z_arr)
np.save('experiments/schooling/Iyz.npy', Iyz_arr)
np.save('experiments/schooling/Izw.npy', Izw_arr)

np.save('experiments/schooling/Hw_v.npy', Hw_v_arr)
np.save('experiments/schooling/Iyv.npy', Iyv_arr)
np.save('experiments/schooling/Ivw_.npy', Ivw_arr)