import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm
from utils import generate_iv_samples, mutual_information, generate_conditional_dist, entropy, laplace_smoothing, conditional_latent_search, conditional_latent_search_iv, generate_invalid_iv_samples
from mpl_toolkits.mplot3d import Axes3D




## set the number of states
alpha_u = 0.5
x_states = 8
y_states = 8
z_states = 8
u_states = 8
num_dist = 100
graph = 'invalid_iv'
# graph= 'iv'

i_dist = 0


folder_name = '{}_x{}_y{}_z{}_u{}_a{}'.format(graph, x_states, y_states, z_states, u_states, str(alpha_u).replace('.', ''))


## load the data from npy file
y_xzu = np.load('experiments/{}/y_xzu.npy'.format(folder_name))
x_zu = np.load('experiments/{}/x_zu.npy'.format(folder_name))
z = np.load('experiments/{}/z.npy'.format(folder_name))
u = np.load('experiments/{}/u.npy'.format(folder_name))


## load the csv file

df = pd.read_csv('experiments/{}/s{}_plot_iv.csv'.format(folder_name, i_dist))

Hw_iv_out = df['Hw'].values
Iyz_iv_out = df['Iyz_xu'].values
Izw_iv_out = df['Izw'].values


Hw_out  = np.load('experiments/{}/s{}_Hw.npy'.format(folder_name, i_dist))
Iyz_out = np.load('experiments/{}/s{}_Iyz_xu.npy'.format(folder_name, i_dist))


b = np.linspace(0.5, 0, 100)

# ## scatter plot of CCE_IV

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Iyz_iv_out, Izw_iv_out, Hw_iv_out)
ax.set_xlabel('I(Y;Z|X,W)')
ax.set_ylabel('I(Z;U)')
ax.set_zlabel('H(U)')
ax.set_title('Entropy of U vs Conditional Mutual Information vs I(Z;U)')
plt.show()


# ## 2d scatter plot of CCE
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(Iyz_out, Hw_out)
ax.set_xlabel('I(Y;Z|X)')
ax.set_ylabel('H(U)')
ax.set_title('Entropy of U vs Conditional Mutual Information')
plt.show()


