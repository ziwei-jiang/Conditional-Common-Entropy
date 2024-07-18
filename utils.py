
import numpy as np
import cvxpy as cp
from scipy.special import xlogy
import scipy.stats as stats
import math





def common_entropy(pyx):
    px = np.sum(pyx, axis=0)
    py_x = np.einsum('ij, j -> ij', pyx, np.divide(1,px, where=(px>0)))
    a = py_x[0, 0]
    b = py_x[0, 1]
    trans = np.array([[0, 1], [1, 0]])
    if a <= b:
        py_w = np.array([[a, 1-a], [1, 0]]).T
        pw_x = np.array([[1, 0], [(1-b)/(1-a), 1- (1-b)/(1-a)]]).T

        qy_w = (np.array([[0, 1], [b, 1-b]]).T ) @trans
        qw_x = trans@(np.array([[1-a/b, a/b], [0, 1]]).T)

    else:
        py_w = (np.array([[a, 1-a], [0, 1]]).T)
        pw_x = (np.array([[1, 0], [b/a, 1-b/a]]).T)

        qy_w = (np.array([[1, 0], [b, 1-b]]).T)@trans
        qw_x = trans@(np.array([[1-(1-a)/(1-b), (1-a)/(1-b)], [0, 1]]).T)

    pw1 = np.einsum('ij, j -> i', pw_x, px)
    pw2 = np.einsum('ij, j -> i', qw_x, px)

    ce1 = entropy(pw1)
    ce2 = entropy(pw2)
    
    if ce1 < ce2:
        return ce1, py_w, pw_x, pw1
    else:
        return ce2, qy_w , qw_x, pw2


def conditional_common_entropy(pyz_x, px):
    num_x = px.shape[0]
    ## create an 2 x 1 array to accumulate the common entropy
    pw = np.zeros((2,))
    max_entr = 0
    for xi in range(num_x):
        ce, py_wxi, pw_zxi, pw_x =  common_entropy(pyz_x[:,:,xi])
        pw += pw_x * px[xi]
        
    return entropy(pw)



def iv_inequality_constraint_check(yx_z_dist):
    for j in range(yx_z_dist.shape[1]):
        max_z = np.max(yx_z_dist[:,j,:], axis=-1)
        if max_z.sum() > 1:
            return False
    return True



def common_entropy(pyx):
    px = np.sum(pyx, axis=0)
    py_x = np.einsum('ij, j -> ij', pyx, np.divide(1,px, where=(px>0)))
    a = py_x[0, 0]
    b = py_x[0, 1]
    trans = np.array([[0, 1], [1, 0]])
    if a <= b:
        py_w = np.array([[a, 1-a], [1, 0]]).T
        pw_x = np.array([[1, 0], [(1-b)/(1-a), 1- (1-b)/(1-a)]]).T

        qy_w = (np.array([[0, 1], [b, 1-b]]).T ) @trans
        qw_x = trans@(np.array([[1-a/b, a/b], [0, 1]]).T)

    else:
        py_w = (np.array([[a, 1-a], [0, 1]]).T)
        pw_x = (np.array([[1, 0], [b/a, 1-b/a]]).T)

        qy_w = (np.array([[1, 0], [b, 1-b]]).T)@trans
        qw_x = trans@(np.array([[1-(1-a)/(1-b), (1-a)/(1-b)], [0, 1]]).T)

    pw1 = np.einsum('ij, j -> i', pw_x, px)
    pw2 = np.einsum('ij, j -> i', qw_x, px)

    ce1 = entropy(pw1)
    ce2 = entropy(pw2)
    
    if ce1 < ce2:
        return ce1, py_w, pw_x, pw1
    else:
        return ce2, qy_w , qw_x, pw2


def conditional_common_entropy(pyz_x, px):
    num_x = px.shape[0]
    ## create an 2 x 1 array to accumulate the common entropy
    pw = np.zeros((2,))
    max_entr = 0
    for xi in range(num_x):
        ce, py_wxi, pw_zxi, pw_x =  common_entropy(pyz_x[:,:,xi])
        pw += pw_x * px[xi]
        
    return entropy(pw)


def entropy_vec(px):
    return -np.sum(px * np.log2(px), axis=(1), where=(px>0))

def entropy(px):
    return -np.sum(px * np.log2(px), axis=(0), where=(px>0)) 

def mutual_information(XY):
    X = XY.sum(axis=0)
    Y = XY.sum(axis=1)
    return entropy(X) + entropy(Y) - entropy(XY.reshape(-1))

def mutual_information_vec(XY):
    X = XY.sum(axis=1)
    Y = XY.sum(axis=2)
    return entropy_vec(X) + entropy_vec(Y) - entropy_vec(XY.reshape(XY.shape[0], -1))

def conditional_mutual_information(XY_Z, Z):
    num_z = XY_Z.shape[2]
    cmi = sum([mutual_information(XY_Z[:,:,i])*Z[i] for i in range(num_z)])
    return cmi

def conditional_mutual_information_vec(XY_Z, Z):
    num_z = XY_Z.shape[3]
    cmi = sum([mutual_information_vec(XY_Z[:,:,:,i])*Z[:,i] for i in range(num_z)])
    return cmi


## Generate a distribution of from dirichlet distribution
def generate_dirichlet(alpha, num_states, num_dist):
    return stats.dirichlet.rvs([alpha]*num_states, size=num_dist)


def generate_conditional_dist(num_states, num_parents_states, num_dist, alpha=-1):
    v = 1/np.arange(1,num_states+1)
    v = v/np.sum(v)
    conditional_dist = np.zeros((num_dist, num_states, num_parents_states))
    ## Create shifted vectors
    for i in range(num_parents_states):
        vi = np.roll(v, i)
        ## Ensure that parent-child relations are far from uniform
        if alpha == -1:
            dist = stats.dirichlet.rvs(vi, size=num_dist)
        else:
            dist = stats.dirichlet.rvs(np.ones_like(vi)*alpha, size=num_dist)
        conditional_dist[:,:,i] = dist
    return conditional_dist




# def generate_multi_cov_samples(z_num, u_num, z_state_nums, u_state_nums, x_state_nums, y_state_nums, num_dist, alpha_u=1, alpha_z=1, alpha_x=1, alpha_y=1):
#     z_list = []
#     u_list = []
#     for i in range(z_num):
#         z_list.append(generate_dirichlet(alpha=alpha_z, num_states=z_state_nums, num_dist=num_dist))
#     for i in range(u_num):
#         u_list.append(generate_dirichlet(alpha=alpha_u, num_states=u_state_nums, num_dist=num_dist))
 
#     x_zsus = np.zeros((num_dist, x_state_nums))
    
#     for i in range(z_num):
#         ## expand x_zsus in a new dimension z_num times
#         x_zsus = np.expand_dims(x_zsus, axis=-1)
#         x_zsus = np.repeat(x_zsus, z_state_nums, axis=-1)
#     for i in range(u_num):
#         ## expand x_zsus in a new dimension u_num times
#         x_zsus = np.expand_dims(x_zsus, axis=-1)
#         x_zsus = np.repeat(x_zsus, u_state_nums, axis=-1)
    
#     for i in range(z_num):
#         for j in range(u_num):
#             x_zsus[:,:,:,i,j] = generate_conditional_dist(num_states=x_state_nums, num_parents_states=z_state_nums, num_dist=num_dist, alpha=alpha_x)
    
    

def generate_two_cov_samples(z_state_nums, v_state_nums, u_state_nums, x_state_nums, y_state_nums, num_dist, alpha_u=1, alpha_v=1, alpha_z=1, alpha_x=1, alpha_y=1):

    z = generate_dirichlet(alpha=alpha_z, num_states=z_state_nums, num_dist=num_dist)
    u = generate_dirichlet(alpha=alpha_u, num_states=u_state_nums, num_dist=num_dist)
    v = generate_dirichlet(alpha=alpha_v, num_states=v_state_nums, num_dist=num_dist)
    

    x_zvu = np.zeros((num_dist, x_state_nums, z_state_nums, v_state_nums, u_state_nums))
    for i in range(z_state_nums):
        for j in range(v_state_nums):
            x_zvu[:,:,i,j,:] = generate_conditional_dist(num_states=x_state_nums, num_parents_states=u_state_nums, num_dist=num_dist, alpha=alpha_x)
    

    y_xvu = np.zeros((num_dist, y_state_nums, x_state_nums, v_state_nums,u_state_nums))
    
    for j in range(v_state_nums):
        for k in range(u_state_nums):
            y_xvu[:,:,:,j,k] = generate_conditional_dist(num_states=y_state_nums, num_parents_states=x_state_nums, num_dist=num_dist, alpha=1) 
    
    y_xzvu = np.expand_dims(y_xvu, axis=3)
    y_xzvu = np.repeat(y_xzvu, z_state_nums, axis=3)

    return y_xzvu, x_zvu, z, v, u 



def generate_three_cov_samples_one_val(z_state_nums, v1_state_nums, v2_state_nums, u_state_nums, x_state_nums, y_state_nums, num_dist, alpha_u=1, alpha_v1=1, alpha_v2=1, alpha_z=1, alpha_x=1, alpha_y=1):

    z = generate_dirichlet(alpha=alpha_z, num_states=z_state_nums, num_dist=num_dist)
    u = generate_dirichlet(alpha=alpha_u, num_states=u_state_nums, num_dist=num_dist)
    v1 = generate_dirichlet(alpha=alpha_v1, num_states=v1_state_nums, num_dist=num_dist)
    v2 = generate_dirichlet(alpha=alpha_v2, num_states=v2_state_nums, num_dist=num_dist)
    

    x_zv1v2u = np.zeros((num_dist, x_state_nums, z_state_nums, v1_state_nums, v2_state_nums ,u_state_nums))
    for i in range(z_state_nums):
        for j in range(v1_state_nums):
            for k in range(v2_state_nums):
                x_zv1v2u[:,:,i,j,k,:] = generate_conditional_dist(num_states=x_state_nums, num_parents_states=u_state_nums, num_dist=num_dist, alpha=alpha_x)
    

    y_xv1v2u = np.zeros((num_dist, y_state_nums, x_state_nums, v1_state_nums, v2_state_nums, u_state_nums))
    
    for j in range(v1_state_nums):
        for k in range(v2_state_nums):
            for l in range(u_state_nums):
                y_xv1v2u[:,:,:,j,k,l] = generate_conditional_dist(num_states=y_state_nums, num_parents_states=x_state_nums, num_dist=num_dist, alpha=alpha_y) 
    
    y_xzv1v2u = np.expand_dims(y_xv1v2u, axis=3)
    y_xzv1v2u = np.repeat(y_xzv1v2u, z_state_nums, axis=3)

    return y_xzv1v2u, x_zv1v2u, z, v1, v2, u 



def generate_three_cov_samples_two_val(z_state_nums, v1_state_nums, v2_state_nums, u_state_nums, x_state_nums, y_state_nums, num_dist, alpha_u=1, alpha_v1=1, alpha_v2=1, alpha_z=1, alpha_x=1, alpha_y=1):

    z = generate_dirichlet(alpha=alpha_z, num_states=z_state_nums, num_dist=num_dist)
    u = generate_dirichlet(alpha=alpha_u, num_states=u_state_nums, num_dist=num_dist)
    v1 = generate_dirichlet(alpha=alpha_v1, num_states=v1_state_nums, num_dist=num_dist)
    v2 = generate_dirichlet(alpha=alpha_v2, num_states=v2_state_nums, num_dist=num_dist)


    x_zv1v2u = np.zeros((num_dist, x_state_nums, z_state_nums, v1_state_nums, v2_state_nums ,u_state_nums))
    for i in range(z_state_nums):
        for j in range(v1_state_nums):
            for k in range(v2_state_nums):
                x_zv1v2u[:,:,i,j,k,:] = generate_conditional_dist(num_states=x_state_nums, num_parents_states=u_state_nums, num_dist=num_dist, alpha=alpha_x)
    

    y_xv2u = np.zeros((num_dist, y_state_nums, x_state_nums, v2_state_nums, u_state_nums))
    
    for k in range(v2_state_nums):
        for l in range(u_state_nums):
            y_xv2u[:,:,:,k,l] = generate_conditional_dist(num_states=y_state_nums, num_parents_states=x_state_nums, num_dist=num_dist, alpha=alpha_y) 
    
    y_xzv2u = np.expand_dims(y_xv2u, axis=3)
    y_xzv2u = np.repeat(y_xzv2u, z_state_nums, axis=3)

    y_xzv1v2u = np.expand_dims(y_xzv2u, axis=4)
    y_xzv1v2u = np.repeat(y_xzv1v2u, v1_state_nums, axis=4)

    return y_xzv1v2u, x_zv1v2u, z, v1, v2, u 




def generate_iv_samples(z_state_nums, u_state_nums, x_state_nums, y_state_nums, num_dist, alpha_u=1, alpha_z=1, alpha_x=1, alpha_y=1):
    z = generate_dirichlet(alpha=alpha_z, num_states=z_state_nums, num_dist=num_dist)
    u = generate_dirichlet(alpha=alpha_u, num_states=u_state_nums, num_dist=num_dist)
    y_xu = np.zeros((num_dist, y_state_nums, x_state_nums, u_state_nums))
    for i in range(u_state_nums):
        y_xu[:,:,:,i] = generate_conditional_dist(num_states=y_state_nums, num_parents_states=x_state_nums, num_dist=num_dist, alpha=alpha_y)

    x_zu = np.zeros((num_dist, x_state_nums, z_state_nums, u_state_nums))
    for i in range(u_state_nums):
        x_zu[:,:,:,i] = generate_conditional_dist(num_states=x_state_nums, num_parents_states=z_state_nums, num_dist=num_dist, alpha=alpha_x)
    
    
    y_xzu = np.expand_dims(y_xu, axis=3)
    y_xzu = np.repeat(y_xzu, z_state_nums, axis=3)

    return y_xzu, x_zu, z, u




def generate_invalid_iv_samples(z_state_nums, u_state_nums, x_state_nums, y_state_nums, num_dist, alpha_u=1, alpha_z=1, alpha_x=1, alpha_y=1):
    z = generate_dirichlet(alpha=alpha_z, num_states=z_state_nums, num_dist=num_dist)
    u = generate_dirichlet(alpha=alpha_u, num_states=u_state_nums, num_dist=num_dist)
    # zu = np.einsum('ni, nj -> nij', z, u)

    x_zu = np.zeros((num_dist, x_state_nums, z_state_nums, u_state_nums))
    for i in range(u_state_nums):
        x_zu[:,:,:,i] = generate_conditional_dist(num_states=x_state_nums, num_parents_states=z_state_nums, num_dist=num_dist, alpha=alpha_x)
    

    y_xzu = np.zeros((num_dist, y_state_nums, x_state_nums, z_state_nums, u_state_nums))
    for i in range(z_state_nums):
        for j in range(u_state_nums):
            y_xzu[:,:,:,i,j] = generate_conditional_dist(num_states=y_state_nums, num_parents_states=x_state_nums, num_dist=num_dist, alpha=alpha_y) 
    
    return y_xzu, x_zu, z, u 




### z -> m
# def generate_weakly_invalid_iv_samples(z_state_nums, u_state_nums, m_state_nums, x_state_nums, y_state_nums, num_dist, alpha_u=1, alpha_z=1, alpha_x=1, alpha_y=1, alpha_m=1):
#     z = generate_dirichlet(alpha=alpha_z, num_states=z_state_nums, num_dist=num_dist)
#     u = generate_dirichlet(alpha=alpha_u, num_states=u_state_nums, num_dist=num_dist)
#     m_z = generate_conditional_dist(num_states=m_state_nums, num_parents_states=z_state_nums, num_dist=num_dist, alpha=alpha_m)
#     zu = np.einsum('ni, nj -> nij', z, u)
#     # mz = np.einsum('nij, nj -> ni', m_z, z)
#     mzu = np.einsum('nij, nj, nk -> nijk', m_z, z, u)
#     m = mzu.sum(axis=(2,3))
#     zu = np.einsum('ni, nj -> nij', z, u)

#     x_zu = np.zeros((num_dist, x_state_nums, z_state_nums, u_state_nums))
#     for i in range(u_state_nums):
#         x_zu[:,:,:,i] = generate_conditional_dist(num_states=x_state_nums, num_parents_states=z_state_nums, num_dist=num_dist, alpha=1)


#     y_xmu = np.zeros((num_dist, y_state_nums, x_state_nums, m_state_nums, u_state_nums))
#     for i in range(m_state_nums):
#         for j in range(u_state_nums):
#             y_xmu[:,:,:,i,j] = generate_conditional_dist(num_states=y_state_nums, num_parents_states=x_state_nums, num_dist=num_dist, alpha=1) 

#     ## repeat x_zu in a new dimension m_state_nums times
#     x_mzu = np.expand_dims(x_zu, axis=2)
#     x_mzu = np.repeat(x_mzu, m_state_nums, axis=2)

#     xmzu = np.einsum('nijkl, njkl->nijkl', x_mzu, mzu)
#     xzu = xmzu.sum(axis=2)
#     xmu = xmzu.sum(axis=3)

#     y_xmzu = np.expand_dims(y_xmu, axis=4) 
#     y_xmzu = np.repeat(y_xmzu, z_state_nums, axis=4)
#     yxmzu = np.einsum('nijklm, njklm->nijklm', y_xmzu, xmzu)
#     yxzu = yxmzu.sum(axis=3)
#     y_xzu = np.einsum('nijkl, njkl->nijkl', yxzu, np.divide(1, xzu, where=(xzu>0)))
    
#     return y_xzu, x_zu, z, u, m


def generate_weakly_invalid_iv_samples(z_state_nums, u_state_nums, m_state_nums, x_state_nums, y_state_nums, num_dist, alpha_u=1, alpha_z=1, alpha_x=1, alpha_y=1, alpha_m=1):
    m = generate_dirichlet(alpha=alpha_m, num_states=m_state_nums, num_dist=num_dist)
    u = generate_dirichlet(alpha=alpha_u, num_states=u_state_nums, num_dist=num_dist)
    z_m = generate_conditional_dist(num_states=z_state_nums, num_parents_states=m_state_nums, num_dist=num_dist, alpha=alpha_z)
    

    mzu = np.einsum('ni, nji, nk-> nijk', m, z_m, u)
    m = mzu.sum(axis=(2,3))
    z = mzu.sum(axis=(1,3))

    x_zu = np.zeros((num_dist, x_state_nums, z_state_nums, u_state_nums))
    for i in range(u_state_nums):
        x_zu[:,:,:,i] = generate_conditional_dist(num_states=x_state_nums, num_parents_states=z_state_nums, num_dist=num_dist, alpha=1)


    y_xmu = np.zeros((num_dist, y_state_nums, x_state_nums, m_state_nums, u_state_nums))
    for i in range(m_state_nums):
        for j in range(u_state_nums):
            y_xmu[:,:,:,i,j] = generate_conditional_dist(num_states=y_state_nums, num_parents_states=x_state_nums, num_dist=num_dist, alpha=1) 

    ## repeat x_zu in a new dimension m_state_nums times
    x_mzu = np.expand_dims(x_zu, axis=2)
    x_mzu = np.repeat(x_mzu, m_state_nums, axis=2)

    xmzu = np.einsum('nijkl, njkl->nijkl', x_mzu, mzu)
    xzu = xmzu.sum(axis=2)
    xmu = xmzu.sum(axis=3)

    y_xmzu = np.expand_dims(y_xmu, axis=4) 
    y_xmzu = np.repeat(y_xmzu, z_state_nums, axis=4)
    yxmzu = np.einsum('nijklm, njklm->nijklm', y_xmzu, xmzu)
    yxzu = yxmzu.sum(axis=3)
    y_xzu = np.einsum('nijkl, njkl->nijkl', yxzu, np.divide(1, xzu, where=(xzu>0)))
    
    return y_xzu, x_zu, z, u, m



                
## define a function for laplace smoothing
def smoothing(pxyz, alpha=1):
    ## get the number of states
    x_states = pxyz.shape[0]
    y_states = pxyz.shape[1]
    z_states = pxyz.shape[2]

    ## get the number of observations
    N = pxyz.sum()

    ## get the new distribution
    pxyz = (pxyz + alpha)/(N + alpha*x_states*y_states*z_states)
    return pxyz



def conditional_latent_search(pxyz, latent_size, qu_xyz, iter_num=200, beta=1):
    qu = np.einsum('lijk, ijk->l', qu_xyz, pxyz)

    pxyz = smoothing(pxyz, alpha=+1e-18)
    pxz = pxyz.sum(axis=1)
    px = pxz.sum(axis=1)
    pxy = pxyz.sum(axis=2)
    for i in range(iter_num):
        ## form the joint
        pxyzu = np.einsum('ijk, lijk -> ijkl', pxyz, qu_xyz)
        # pxyzu = smoothing(pxyzu, alpha=+1e-18)
        puxz = np.einsum('ijkl->lik', pxyzu)
        pux = puxz.sum(axis=2)
        puxy = np.einsum('ijkl->lij', pxyzu)
        pu = pux.sum(axis=1)

        ## get the posterior
        pu_x = np.einsum('li, i -> li', pux, np.divide(1,px, where=(px>0)))
        pu_xz = np.einsum('lik, ik-> lik', puxz, np.divide(1,pxz, where=(pxz>0)))
        pu_xy = np.einsum('lij, ij-> lij', puxy, np.divide(1,pxy, where=(pxy>0)))
        
        ## update
        fxyz = np.einsum('lik, lij, l, li->ijk', pu_xz, pu_xy, pu**(beta), np.divide(1,pu_x, where=(pu_x>0)))
        fxyz = np.divide(1,fxyz, where=(fxyz>0))
        qu_xyz =  np.einsum('lik, lij, l, li, ijk->lijk', pu_xz, pu_xy, pu**(beta), np.divide(1,pu_x, where=(pu_x>0)), fxyz)
        qu = np.einsum('lijk, ijk->l', qu_xyz, pxyz)
        
    return qu_xyz



def conditional_latent_search_iv(pxyz, latent_size, qu_xyz, iter_num=200, beta0=1, beta1=1):
    qu = np.einsum('lijk, ijk->l', qu_xyz, pxyz)
    # beta_0 = beta0
    pxyz = smoothing(pxyz, alpha=+1e-18)
    pxz = pxyz.sum(axis=1)
    pz = pxz.sum(axis=0)
    px = pxz.sum(axis=1)
    pxy = pxyz.sum(axis=2)
    for i in range(iter_num):
        ## form the joint
        pxyzu = np.einsum('ijk, lijk -> ijkl', pxyz, qu_xyz)

        puxz = np.einsum('ijkl->lik', pxyzu)
        pux = puxz.sum(axis=2)
        puz = puxz.sum(axis=1)
        puxy = np.einsum('ijkl->lij', pxyzu)
        pu = pux.sum(axis=1)

        ## get the posterior
        pu_x = np.einsum('li, i -> li', pux, np.divide(1,px, where=(px>0)))
        pu_xz = np.einsum('lik, ik-> lik', puxz, np.divide(1,pxz, where=(pxz>0)))
        pu_xy = np.einsum('lij, ij-> lij', puxy, np.divide(1,pxy, where=(pxy>0)))
        pu_z = np.einsum('lk, k -> lk', puz, np.divide(1,pz, where=(pz>0)))
        
        ## update
        fxyz = np.einsum('lik, lij, l, li, lk->ijk', pu_xz, pu_xy, pu**(beta0), np.divide(1,pu_x, where=(pu_x>0)), np.divide(1, pu_z**(beta1), where=(pu_z**(beta1)>0)))
        fxyz = np.divide(1,fxyz, where=(fxyz>0))
        qu_xyz =  np.einsum('lik, lij, l, li, lk, ijk->lijk', pu_xz, pu_xy, pu**(beta0), np.divide(1,pu_x, where=(pu_x>0)), np.divide(1, pu_z**(beta1), where=((pu_z**(beta1))>0)), fxyz)
        qu = np.einsum('lijk, ijk->l', qu_xyz, pxyz)
        
    return qu_xyz




def get_natural_bounds(pyx_z, y_idx=0, x_idx=0):
    
    # z_state_nums = pyx_z.shape[2]
    px_z = pyx_z.sum(axis=0)
    lb = np.max(pyx_z[y_idx,x_idx,:])
    ub = np.min(1-px_z[x_idx,:]+pyx_z[y_idx,x_idx,:])
    return np.array([lb, ub])


def get_tp_bounds(pyx, y_idx=0, x_idx=0):
    px = pyx.sum(axis=0)
    lb = pyx[y_idx,x_idx]
    ub = 1-px[x_idx]+pyx[y_idx,x_idx]
    return np.array([lb, ub])



def optimization_entr(yx_z, z, y_idx, x_idx, entr=0, izy=0, iuz=0):
    
    z_state_nums = yx_z.shape[2]
    x_state_nums = yx_z.shape[1]
    y_state_nums = yx_z.shape[0]

    x = np.einsum('ijk, k -> j', yx_z, z)
    # if np.any(x==0):
    #     x = (x + 1e-18)/(1 + 1e-18*x_state_nums)
    x_z = yx_z.sum(axis=0)
    YX_Z = cp.Variable((y_state_nums*x_state_nums, z_state_nums))

    pydox = cp.sum(YX_Z[y_idx*x_state_nums:(y_idx+1)*x_state_nums,:], axis=0) @ z

    ## multiply with vector y
    z_vec = np.vstack([z]* y_state_nums) # y_state_nums by z_state_nums
    
    ## compute P(Yx|Z) by summing over every (x_state_nums) rows

    Y_Z = sum([YX_Z[i::x_state_nums] for i in range(x_state_nums)])
    ## [P(Yx), ...] repeat Z times
    Y_vec = Y_Z @ (np.vstack([z]* z_state_nums).T)




    # ### =============== Computing I(Yx; X) ===============
    # x_vec = np.vstack([x]* y_state_nums) # y_state_nums by x_state_nums
    # ## P(Yx, X)
    # YX = cp.transpose(cp.reshape((YX_Z @ z), (y_state_nums, x_state_nums)))
    # ## q(Y, X) = P(Y)P(X)
    # qYX = cp.multiply(Y_vec, x_vec)
    # ## I(Y, X)
    # IYX = cp.sum(cp.kl_div(YX, qYX)/math.log(2))


    # ### =============== Computing I(Yx; Z|X) =============== 
    # ### need to be fix
    # ## P(Yx, X, Z) 
    # YXZ = cp.multiply(YX_Z, np.vstack([z_vec]* x_state_nums))
    # ## P(Yx, Z|X)
    # x_vec2 = np.vstack([x_vec.T]* z_state_nums)
    
    # YZ_X = cp.multiply(YXZ, np.divide(1, x_vec2, where=(x_vec2>0)))
    # ## reshape YZ_X to (y_state_nums*z_state_nums) by x_state_nums
    # YZ_X = cp.vstack([cp.vec(YZ_X[i::x_state_nums]) for i in range(y_state_nums)]).T
    # ## P(Yx|X)
    # Y_X = sum([YZ_X[i::z_state_nums] for i in range(y_state_nums)])
    # ## repeat Y_X rows n times
    # Y_X_vec = cp.vstack([row for row in Y_X for i in range(z_state_nums)])
    # ## P(Z|X)
    # xz = np.einsum('ij, j -> ij', x_z, z)
    # z_x = np.divide(xz, x, where=(x>0))
    # z_x_vec = cp.vstack([z_x]* y_state_nums)
    # ## q(Yx, Z|X) = P(Yx|X)P(Z|X)
    # qYZ_X = cp.multiply(Y_X_vec, z_x_vec)
    # ## I(Yx; Z|X)
    # IYZ_x = cp.sum(cp.kl_div(YZ_X, qYZ_X)/math.log(2), axis=0)
    # IYZ_X = IYZ_x @ xc
    
    ### =============== Computing I(Yx; X|Z) ===============
    x_z_vec = np.vstack([x_z]* y_state_nums)
    ## repeat Y_Z rows n times
    Y_Z_vec = cp.vstack([row for row in Y_Z for i in range(x_state_nums)])
    ## q(Y, X| Z) = P(Y|Z)P(X|Z)
    qYX_Z = cp.multiply(Y_Z_vec, x_z_vec)
    ## I(Yx; X|Z)
    IYX_z = cp.sum(cp.kl_div(YX_Z, qYX_Z)/math.log(2), axis=0)
    
    
    # ### =============== Computing I(Yx; Z) ===============
    # ## P(Yx, Z)
    # YZ = cp.multiply(Y_Z, z_vec)
    # ## q(Yx, Z) = P(Yx)P(Z)
    # qYZ = cp.multiply(Y_vec, z_vec)
    # ## I(Yx, Z)
    # IYZ = cp.sum(cp.kl_div(YZ, qYZ)/math.log(2))



        


    if (iuz == 0) and (izy == 0):
        ## valid IV
        entr_constraint = [cp.max(IYX_z)<= entr]
        entr_constraint += [cp.sum(YX_Z[y_idx*x_state_nums:(y_idx+1)*x_state_nums,i-1]) == cp.sum(YX_Z[y_idx*x_state_nums:(y_idx+1)*x_state_nums,i]) for i in range(1, z_state_nums)]
    else:
        ### =============== Computing I(Yx; Z) ===============
        ## P(Yx, Z)
        YZ = cp.multiply(Y_Z, z_vec)
        ## q(Yx, Z) = P(Yx)P(Z)
        qYZ = cp.multiply(Y_vec, z_vec)
        ## I(Yx, Z)
        IYZ = cp.sum(cp.kl_div(YZ, qYZ)/math.log(2))

        if (iuz == 0):
            ## unconfounded
            entr_constraint = [cp.max(IYX_z)<= entr, IYZ<= izy]
        else:
            IYX_Z = IYX_z @ z
            entr_constraint = [IYX_Z + IYZ<= entr + izy]


    constraints  = [YX_Z >= 0, YX_Z <= 1]
    constraints += entr_constraint
    constraints += [cp.sum(YX_Z[:,i]) == 1 for i in range(z_state_nums)]
    constraints += [YX_Z[i*x_state_nums+x_idx,:] == yx_z[i,x_idx,:] for i in range(y_state_nums)]
    # constraints += [cp.sum(YX_Z[y_idx*x_state_nums:(y_idx+1)*x_state_nums,i-1]) == cp.sum(YX_Z[y_idx*x_state_nums:(y_idx+1)*x_state_nums,i]) for i in range(1, z_state_nums)]
    
    max_pydox = cp.Maximize((pydox))
    min_pydox = cp.Minimize((pydox))
    max_prob = cp.Problem(max_pydox, constraints)
    max_prob.solve(solver=cp.SCS)

    min_prob = cp.Problem(min_pydox, constraints)
    min_prob.solve(solver=cp.SCS)

    return np.array([min_prob.value, max_prob.value])

    



def optimization_entropy(yx_z_dist, z, Izy=-1, entr=-1):

    x = np.einsum('ijk, k -> j', yx_z_dist, z)


    YxX_Z_dox0 = cp.Variable((4, 2))
    YxX_Z_dox1 = cp.Variable((4, 2))

    py1dox0 = cp.sum(YxX_Z_dox0[2:,:], axis=0) @ z
    py1dox1 = cp.sum(YxX_Z_dox1[2:,:], axis=0) @ z

    z_vec = np.vstack([z, z])
    x_vec = np.vstack([x, x])
    ## P(Yx|Z)
    Yx_Z_dox0 = cp.vstack([YxX_Z_dox0[0,:]+ YxX_Z_dox0[1,:], YxX_Z_dox0[2,:]+ YxX_Z_dox0[3,:]])
    Yx_Z_dox1 = cp.vstack([YxX_Z_dox1[0,:]+ YxX_Z_dox1[1,:], YxX_Z_dox1[2,:]+ YxX_Z_dox1[3,:]])

    ## P(Yx, dup)
    Yx_dox0_vec = Yx_Z_dox0 @ z_vec.T
    Yx_dox1_vec = Yx_Z_dox1 @ z_vec.T

    ## P(Yx, Z)
    YxZ_dox0 = cp.multiply(Yx_Z_dox0, z_vec)
    YxZ_dox1 = cp.multiply(Yx_Z_dox1, z_vec)


    ## q(Yx, Z) = P(Yx)P(Z)
    qYxZ_dox0 = cp.multiply(Yx_dox0_vec, z_vec)
    qYxZ_dox1 = cp.multiply(Yx_dox1_vec, z_vec)
    
    ## I(Yx, Z)
    IYxZ_dox0 = cp.sum(cp.kl_div(YxZ_dox0, qYxZ_dox0)/math.log(2))
    IYxZ_dox1 = cp.sum(cp.kl_div(YxZ_dox1, qYxZ_dox1)/math.log(2))

    ## P(Yx, X)
    YxX_dox0 = cp.transpose(cp.reshape((YxX_Z_dox0 @ z), (2,2)))
    YxX_dox1 = cp.transpose(cp.reshape((YxX_Z_dox1 @ z), (2,2)))

    ## q(Yx, X) = P(Yx)P(X)
    qYxX_dox0 = cp.multiply(Yx_dox0_vec, x_vec)
    qYxX_dox1 = cp.multiply(Yx_dox1_vec, x_vec)

    ## I(Yx, X)
    IYxX_dox0 = cp.sum(cp.kl_div(YxX_dox0, qYxX_dox0)/math.log(2))
    IYxX_dox1 = cp.sum(cp.kl_div(YxX_dox1, qYxX_dox1)/math.log(2))

    if (Izy == -1):
        Izy_constraint1 = True 
        Izy_constraint2 = True
    else:
        Izy_constraint1 = (IYxZ_dox0 <= Izy)
        Izy_constraint2 = (IYxZ_dox1 <= Izy)
    if (entr == -1):
        entr_constraint1 = True 
        entr_constraint2 = True
    else:
        entr_constraint1 = (IYxX_dox0 <= entr) 
        entr_constraint2 = (IYxX_dox1 <= entr)


    constraints  = [YxX_Z_dox0[0,0] == yx_z_dist[0,0,0],
                    YxX_Z_dox0[0,1] == yx_z_dist[0,0,1],
                    YxX_Z_dox0[2,0] == yx_z_dist[1,0,0],
                    YxX_Z_dox0[2,1] == yx_z_dist[1,0,1],
                    YxX_Z_dox0 >= 0,   
                    YxX_Z_dox0 <= 1,                   
                    cp.sum(YxX_Z_dox0[:,0]) == 1,
                    cp.sum(YxX_Z_dox0[:,1]) == 1,             ## constraints from the first table
                    ##==================================
                    YxX_Z_dox1[1,0] == yx_z_dist[0,1,0],
                    YxX_Z_dox1[1,1] == yx_z_dist[0,1,1],
                    YxX_Z_dox1[3,0] == yx_z_dist[1,1,0],
                    YxX_Z_dox1[3,1] == yx_z_dist[1,1,1],
                    YxX_Z_dox1 >= 0,  
                    YxX_Z_dox1 <= 1,  
                    cp.sum(YxX_Z_dox1[:,0]) == 1,
                    cp.sum(YxX_Z_dox1[:,1]) == 1,              ## constraints from the second table
                    ##==================================
                    Izy_constraint1,
                    Izy_constraint2,
                    entr_constraint1,
                    entr_constraint2
                    ]            
    
    max_10 = cp.Maximize((py1dox0))
    min_10 = cp.Minimize((py1dox0))
    max10_prob = cp.Problem(max_10, constraints)
    max10_prob.solve(solver=cp.SCS)

    min10_prob = cp.Problem(min_10, constraints)
    min10_prob.solve(solver=cp.SCS)

    max_11 = cp.Maximize((py1dox1))
    min_11 = cp.Minimize((py1dox1))

    max11_prob = cp.Problem(max_11, constraints)
    max11_prob.solve(solver=cp.SCS)

    min11_prob = cp.Problem(min_11, constraints)
    min11_prob.solve(solver=cp.SCS)

    return [min10_prob.value, max10_prob.value], [min11_prob.value, max11_prob.value]