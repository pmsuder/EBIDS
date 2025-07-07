import numpy as np
import random

def mat_norm(A: np.ndarray, x: np.ndarray): # return \|x\|_A^2
    return np.dot(x, np.dot(A, x))

def rd_argmax(vector):
    m = np.amax(vector)
    return random.choice(np.nonzero(vector == m)[0])

def info_ratio(q,a,b,c,d):
    return ((q*a + (1-q)*b)**2) / (q*c + (1-q)*d)


def optimize_info_ratio(a, b, c, d):
    if abs(a-b) < 1e-10:
        if c >= d:
            return 0
        else:
            return 1
    
    if abs(c-d) < 1e-10:
        return 1
    
    sol = (b*c + b*d - 2*a*d) / ((a-b)*(c-d))
    if info_ratio(0,a,b,c,d) < info_ratio(sol, a,b,c,d) or info_ratio(1,a,b,c,d) < info_ratio(sol, a,b,c,d) or sol < 0 or sol > 1:
        if info_ratio(1,a,b,c,d) < info_ratio(0,a,b,c,d):
            return 1
        else:
            return 0
    else:
        return sol


def IDS_action(regret_vec, info_vec):
    num_actions = len(regret_vec)
    scores = np.ones((num_actions, num_actions,2))*np.inf
    for i in range(num_actions):
        for j in range(i+1,num_actions):
            a = regret_vec[i]
            b = regret_vec[j]
            c = info_vec[i]
            d = info_vec[j]
            
            minimizer = optimize_info_ratio(a,b,c,d)
            scores[i,j,0] = minimizer
            scores[i,j,1] = info_ratio(minimizer,a,b,c,d)
    
    m = np.amin(scores[:,:,1])
    indexes_i = np.nonzero(scores[:,:,1] == m)[0]
    indexes_j = np.nonzero(scores[:,:,1] == m)[1]
    chosen_ind = np.random.randint(low = 0, high = len(indexes_i)) # RANDOMLY BREAKING TIES
    
    i_opt = indexes_i[chosen_ind]
    j_opt = indexes_j[chosen_ind]
    q_opt = scores[i_opt, j_opt,0]
    
    u = np.random.uniform(0, 1)
    if u < q_opt:
        return i_opt
    else:
        return j_opt
    
    
def BH_algo(X):
    """
    BH algorithm (Betke and Henk, 1993; Kumar and Yildirim, 2005) as given in (Gales et al., 2022)
    """
    K = X.shape[0]
    d = X.shape[1]
    if K <= 2*d:
        return [i for i in range(K)], X
    X_0 = np.empty((0,d))
    X_0_indexes = []
    full_rank = False
    v_mat = np.zeros((1,d))
    v_perp_mat = np.zeros((1,d))
    i = 0
    while not full_rank:
        i += 1
        if i == 1:
            b = np.zeros(d)
            b[0] = 1
        else:
            if i == 2:
                new_v_perp = v_mat[i-1,:]
            else:
                new_v_perp = v_mat[i-1,:]
                for j in range(1,i-1):
                    new_v_perp -= (np.dot(v_perp_mat[j,:], v_mat[i-1,:]) / np.sum(v_perp_mat[j,:]**2))*v_perp_mat[j,:]
            v_perp_mat = np.vstack([v_perp_mat, new_v_perp])
            e_i = np.zeros(d)
            e_i[i-1]=1
            b = e_i
            for j in range(1,i):
                b -= (np.dot(v_perp_mat[j,:], e_i) / np.sum(v_perp_mat[j,:]**2))*v_perp_mat[j,:]
        dot_prod_vec = np.dot(X, b)
        
        p_index = np.argmax(dot_prod_vec)
        q_index = np.argmin(dot_prod_vec)
        p_vec = X[p_index,:]
        q_vec = X[q_index,:]
        X_0 = np.vstack([X_0, p_vec])
        X_0 = np.vstack([X_0, q_vec])
        X_0_indexes.append(p_index)
        X_0_indexes.append(q_index)
        
        v_mat = np.vstack([v_mat, p_vec - q_vec])
        if np.linalg.matrix_rank(v_mat) == d:
            full_rank = True
    return X_0_indexes, X_0
        
        
        