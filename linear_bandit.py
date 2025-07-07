import numpy as np
import random
from joblib import Parallel, delayed
from copy import deepcopy

from bandit_utils import rd_argmax, IDS_action, mat_norm, BH_algo

class LinearBandit:
    """
    theta - true value of the linear coefficient theta of the model
    eta - vector of variances specific to each arm
    features - list of vectors of features spefic to each arm
    """ 
    def __init__(self, real_theta, eta, features = None, changing_features = False, features_tensor = None):
        self.real_theta = real_theta
        self.dim_theta = len(self.real_theta)
        self.eta = eta
        self.features = features
        self.n_actions = features.shape[0]
        self.n_features = features.shape[1]
        
        self.changing_features = changing_features
        self.features_tensor = deepcopy(features_tensor)
        self.features_time_counter = 0
        
        
    def reward(self, arm, theta = None):
        if theta is None:
            theta = self.real_theta
        theta = theta.flatten()
        return np.dot(self.features[arm], theta) + np.random.normal(0, self.eta[arm], 1)
    
    def reset_features_time_counter(self):
        self.features_time_counter = 0
    
    def update_features(self):
        if self.changing_features:
            self.features = self.features_tensor[self.features_time_counter,:,:]
            self.features_time_counter += 1
    

    # @property
    # def n_features(self):
    #     return self.features.shape[1]

    # @property
    # def n_actions(self):
    #     return self.features.shape[0]

    def regret(self, reward, T):
        if self.changing_features:
            best_arm_vals = np.empty(T)
            for t in range(T):
                best_arm_vals[t] = np.max(np.dot(self.features_tensor[t,:,:], self.real_theta))
            return np.cumsum(best_arm_vals) - np.cumsum(reward)
        else:
            best_arm_reward = np.max(np.dot(self.features, self.real_theta))
            return best_arm_reward * np.arange(1, T + 1) - np.cumsum(reward)
    
        
    ##############################################################################
    ########## ALGORITHMS ########################################################
    ##############################################################################     

    def eps_greedy(self, T, gamma, eps_init = 1.0, decay_rate = 0.5, explore_steps = 1):
        self.reset_features_time_counter()
        rewards, arm_sequence, feature_sequence = np.zeros(T), np.zeros(T), np.zeros((T, self.dim_theta))

        indexes = np.arange(0, self.n_actions)

        W_t = gamma * np.eye(self.dim_theta)
        V_t = gamma * np.eye(self.dim_theta)
        b_t = np.zeros(self.dim_theta)
        
        pred_rewards = np.zeros(self.n_actions)
        
        for t in range(T):
            self.update_features()
            eps = eps_init * (1/(t+1))**(decay_rate)
            u = np.random.uniform(0,1)
            if t < explore_steps or u < eps:
                arm = np.random.choice(indexes)
            else:
                W_t_inv = np.linalg.inv(W_t)
                theta_hat = np.dot(W_t_inv, b_t)

                for a in range(self.n_actions):
                    pred_rewards[a] = np.dot(self.features[a], theta_hat)
                    
                a_t_greedy = rd_argmax(pred_rewards)
                arm = a_t_greedy
                
            reward = self.reward(arm)
            rewards[t] = reward
            arm_sequence[t] = arm
            
            
            W_t += (1/(self.eta[arm]**2))*np.outer(self.features[arm], self.features[arm])
            V_t += np.outer(self.features[arm], self.features[arm])
            b_t += (1/(self.eta[arm]**2))*reward*self.features[arm]

        return rewards, arm_sequence    

    def IDS_UCB(self, T, gamma, B, deterministic = False, explore_steps = 1):
        """
        IDS-UCB algorithm by Kirschner and Krause (2018)
        """
        
        self.reset_features_time_counter()
        rewards, arm_sequence, feature_sequence = np.zeros(T), np.zeros(T), np.zeros((T, self.dim_theta))

        indexes = np.arange(0, self.n_actions)
        
        sampled_arms = None
        if explore_steps > 0:
            sampled_arms = np.random.choice(indexes, explore_steps, replace=False)

        W_t = gamma * np.eye(self.dim_theta)
        b_t = np.zeros(self.dim_theta)
        det_W_1 = gamma**self.dim_theta

        Delta_hat = np.zeros(self.n_actions)
        I_t_UCB = np.zeros(self.n_actions)
        Upper_bounds = np.zeros(self.n_actions)
        Lower_bounds = np.zeros(self.n_actions)

        for t in range(T):
            self.update_features()
            if t < explore_steps:
                arm = sampled_arms[t]
            else:
                det_W_t = np.linalg.det(W_t)
                W_t_inv = np.linalg.inv(W_t)
                theta_hat = np.dot(W_t_inv, b_t)

                beta_t_sqrt = np.sqrt(2*np.log((t+1)**2) + np.log(det_W_t) - np.log(det_W_1)) + np.sqrt(gamma)*B
                
                for a in range(self.n_actions):
                    Upper_bounds[a] = np.dot(self.features[a], theta_hat) +  beta_t_sqrt*mat_norm(W_t_inv, self.features[a])
                    Lower_bounds[a] = np.dot(self.features[a], theta_hat) -  beta_t_sqrt*mat_norm(W_t_inv, self.features[a])

                a_t_UCB = rd_argmax(Upper_bounds)
                Delta_hat = max(Upper_bounds) - Lower_bounds

                for a in range(self.n_actions):
                    # This is W_t_inv instead of V_t_inv - Thesis has a mistake, check original paper
                    W_t_plus = W_t + (1/(self.eta[a]**2))*np.outer(self.features[a], self.features[a])
                    W_t_plus_inv = np.linalg.inv(W_t_plus)

                    first_norm = mat_norm(W_t_inv, self.features[a_t_UCB])
                    second_norm = mat_norm(W_t_plus_inv, self.features[a_t_UCB])

                    I_t_UCB[a] = (1/2)*(np.log(first_norm) - np.log(second_norm)) + 1e-8 # small constant added for numerical stability

                if deterministic:
                    arm = rd_argmax(I_t_UCB / Delta_hat**2)
                else:
                    arm = IDS_action(Delta_hat, I_t_UCB)
                
            reward = self.reward(arm)
            rewards[t] = reward
            arm_sequence[t] = arm


            W_t += (1/(self.eta[arm]**2))*np.outer(self.features[arm], self.features[arm])
            b_t += (1/(self.eta[arm]**2))*reward*self.features[arm]
            if t == 0:
                det_W_1 = np.linalg.det(W_t) 

        return rewards, arm_sequence
    
    
    def EB_OFUL(self, T, gamma, B, estim_B_steps=10, B_bound_err = 1e-3):
        """
        Empirical bound version of the OFUL UCB algorithm
        """
        
        self.reset_features_time_counter()
        rewards, arm_sequence, feature_sequence = np.zeros(T), np.zeros(T), np.zeros((T, self.dim_theta))

        W_t = gamma * np.eye(self.dim_theta)
        V_t = gamma * np.eye(self.dim_theta)
        b_t = np.zeros(self.dim_theta)
        det_W_1 = gamma**self.dim_theta

        Upper_bounds = np.zeros(self.n_actions)
        
        B_estimates_vec = np.empty(0)
        

        for t in range(T):
            self.update_features()
            det_W_t = np.linalg.det(W_t)
            W_t_inv = np.linalg.inv(W_t)
            theta_hat = np.dot(W_t_inv, b_t)
            theta_hat_norm = np.sqrt(np.sum(theta_hat**2))

            eigenvalues, eigenvectors = np.linalg.eig(W_t)

            # Find the index of the smallest eigenvalue
            min_eig_index = np.argmin(eigenvalues)
            lambda_min = eigenvalues[min_eig_index]

            # Extract the corresponding eigenvector
            v_min = eigenvectors[:, min_eig_index]

            zeta = min(B_bound_err, 1/(t+1)**2)
            beta_sqrt = np.sqrt(2*np.log(1/zeta) + np.log(det_W_t) - np.log(det_W_1)) + np.sqrt(gamma)*B
            B_hat = min(B, theta_hat_norm + (beta_sqrt / np.sqrt(lambda_min)))

            beta_hat_sqrt = np.sqrt(2*np.log(1/zeta) + np.log(det_W_t) - np.log(det_W_1)) + np.sqrt(gamma)*B_hat

            if t < estim_B_steps:
                for a in range(self.n_actions):
                    Upper_bounds[a] = np.dot(self.features[a], theta_hat) +  beta_hat_sqrt*mat_norm(W_t_inv, self.features[a])

                a_t_EB_UCB = rd_argmax(Upper_bounds)
    
            else:
                B_tilde = min(B, np.min(B_estimates_vec))
                beta_tilde_sqrt = np.sqrt(2*np.log(1/zeta) + np.log(det_W_t) - np.log(det_W_1)) + np.sqrt(gamma)*B_tilde
                for a in range(self.n_actions):
                    Upper_bounds[a] = np.dot(self.features[a], theta_hat) +  beta_tilde_sqrt*mat_norm(W_t_inv, self.features[a])

                a_t_EB_UCB = rd_argmax(Upper_bounds)

            
            arm = a_t_EB_UCB

            reward = self.reward(arm)
            rewards[t] = reward
            arm_sequence[t] = arm
            
            W_t += (1/(self.eta[arm]**2))*np.outer(self.features[arm], self.features[arm])
            V_t += np.outer(self.features[arm], self.features[arm])
            b_t += (1/(self.eta[arm]**2))*reward*self.features[arm]

            B_estimates_vec = np.append(B_estimates_vec, theta_hat_norm + (beta_hat_sqrt / np.sqrt(lambda_min)))

        return rewards, arm_sequence

    def OFUL(self, T, gamma, B, explore_steps = 1):
        """
        OFUL UCB algorithm for linear bandit by Abbasi-Yadkori et al. (2011)
        """
        self.reset_features_time_counter()
        rewards, arm_sequence, feature_sequence = np.zeros(T), np.zeros(T), np.zeros((T, self.dim_theta))

        indexes = np.arange(0, self.n_actions)
        sampled_arms = np.random.choice(indexes, explore_steps, replace=False)
        
        W_t = gamma * np.eye(self.dim_theta)
        V_t = gamma * np.eye(self.dim_theta)
        b_t = np.zeros(self.dim_theta)
        det_W_1 = gamma**self.dim_theta
        
        Upper_bounds = np.zeros(self.n_actions)
        
        for t in range(T):
            self.update_features()
            if t < explore_steps:
                arm = sampled_arms[t]
            else:
                det_W_t = np.linalg.det(W_t)
                W_t_inv = np.linalg.inv(W_t)
                theta_hat = np.dot(W_t_inv, b_t)

                beta_t_sqrt = np.sqrt(4*np.log(t+1) + np.log(det_W_t) - np.log(det_W_1)) + np.sqrt(gamma)*B

                for a in range(self.n_actions):
                    Upper_bounds[a] = np.dot(self.features[a], theta_hat) +  beta_t_sqrt*mat_norm(W_t_inv, self.features[a])
                    
                a_t_UCB = rd_argmax(Upper_bounds)
                arm = a_t_UCB
                
            reward = self.reward(arm)
            rewards[t] = reward
            arm_sequence[t] = arm
            
            
            W_t += (1/(self.eta[arm]**2))*np.outer(self.features[arm], self.features[arm])
            V_t += np.outer(self.features[arm], self.features[arm])
            b_t += (1/(self.eta[arm]**2))*reward*self.features[arm]
            if t == 0:
                det_W_1 = np.linalg.det(W_t) 
                
        return rewards, arm_sequence
    
    def NAOFUL(self, T, alpha=2):
        """
        NAOFUL algorithm by Gales et al. (2022)
        """
        self.reset_features_time_counter()
        R = np.max(self.eta)
        rewards, arm_sequence, feature_sequence = np.zeros(T), np.zeros(T), np.zeros((T,self.dim_theta))
        V_bar_t = np.zeros((self.dim_theta, self.dim_theta))
        b_t = np.zeros(self.dim_theta)
        Upper_bounds = np.zeros(self.n_actions)
        
        for t in range(T):
            self.update_features()
            lambda_t = (R**2)*self.dim_theta / ((t+1)**alpha)
            delta_t = 6/(((t+1)*np.pi)**2)
            V_t = V_bar_t + lambda_t*np.eye(self.dim_theta)
            
            V_t_inv = np.linalg.inv(V_t)
            theta_hat = np.dot(V_t_inv, b_t)
            
            beta_t_sqrt = R*(np.sqrt(np.log(np.linalg.det(V_t)) - self.dim_theta*np.log(lambda_t) - 2*np.log(delta_t)) + np.sqrt(self.dim_theta))
            for a in range(self.n_actions):
                Upper_bounds[a] = np.dot(self.features[a], theta_hat) +  beta_t_sqrt*mat_norm(V_t_inv, self.features[a])
                
            arm = rd_argmax(Upper_bounds)
            
            reward = self.reward(arm)
            rewards[t] = reward
            arm_sequence[t] = arm
            
            V_bar_t += np.outer(self.features[arm], self.features[arm])
            b_t += reward*self.features[arm]
            
        return rewards, arm_sequence
            
    def OLSOFUL(self, T, delta = 1e-3):
        """
        OLSOFUL algorithm by Gales et al. (2022)
        """
        self.reset_features_time_counter()
        R = np.max(self.eta)
        rewards, arm_sequence, feature_sequence = np.zeros(T), np.zeros(T), np.zeros((T,self.dim_theta))
        V_bar_t = np.zeros((self.dim_theta, self.dim_theta))
        b_t = np.zeros(self.dim_theta)
        Upper_bounds = np.zeros(self.n_actions)
        arm_scores = np.zeros(self.n_actions)
        criterion_met = False
        self.update_features()
        init_indexes, _ = BH_algo(self.features)
        for t in range(T):
            self.update_features()
            if t < len(init_indexes):
                arm = init_indexes[t]
            else:
                V_bar_t_inv = np.linalg.inv(V_bar_t)
                if not criterion_met:
                    for a in range(self.n_actions):
                        arm_scores[a] = mat_norm(V_bar_t_inv, self.features[a])
                    if np.max(arm_scores) <= 1:
                        criterion_met = True
                        V_bar_tau = np.copy(V_bar_t)
                        det_V_bar_tau = np.linalg.det(V_bar_tau)
                if not criterion_met:
                        arm = rd_argmax(arm_scores)
                else:
                    theta_hat = theta_hat = np.dot(V_bar_t_inv, b_t)
                    beta_t_sqrt = R*(np.sqrt(np.log(np.linalg.det(V_bar_t) - det_V_bar_tau - 2*np.log(delta))) + np.sqrt(np.log(4)*self.dim_theta - 4*np.log(delta)))
                    for a in range(self.n_actions):
                        Upper_bounds[a] = np.dot(self.features[a], theta_hat) +  beta_t_sqrt*mat_norm(V_bar_t_inv, self.features[a])

                    arm = rd_argmax(Upper_bounds)
                    
            reward = self.reward(arm)
            rewards[t] = reward
            arm_sequence[t] = arm

            V_bar_t += np.outer(self.features[arm], self.features[arm])
            b_t += reward*self.features[arm]
            
        return rewards, arm_sequence
    
    def EB_IDS(self, T, gamma, B, alpha, estim_B_steps = 10, B_bound_err = 1e-3):
        """
        Emprical bound information-directed sampling (EBIDS) algorithm we propose in our paper
        """
        self.reset_features_time_counter()
        rewards, arm_sequence, feature_sequence = np.zeros(T), np.zeros(T), np.zeros((T, self.dim_theta))

        W_t = gamma * np.eye(self.dim_theta)
        V_t = gamma * np.eye(self.dim_theta)
        b_t = np.zeros(self.dim_theta)
        det_W_1 = gamma**self.dim_theta

        Delta_hat = np.zeros(self.n_actions)
        I_t_EB_UCB = np.zeros(self.n_actions)
        I_t_B = np.zeros(self.n_actions)
        I_t_BAM = np.zeros(self.n_actions)
        
        Upper_bounds = np.zeros(self.n_actions)
        Lower_bounds = np.zeros(self.n_actions)
        
        B_estimates_vec = np.empty(0)
        
        for t in range(T):
            self.update_features()
            det_W_t = np.linalg.det(W_t)
            W_t_inv = np.linalg.inv(W_t)
            theta_hat = np.dot(W_t_inv, b_t)
            theta_hat_norm = np.sqrt(np.sum(theta_hat**2))

            eigenvalues, eigenvectors = np.linalg.eig(W_t)

            # Find the index of the smallest eigenvalue
            min_eig_index = np.argmin(eigenvalues)
            lambda_min = eigenvalues[min_eig_index]

            # Extract the corresponding eigenvector
            v_min = eigenvectors[:, min_eig_index]

            zeta = min(B_bound_err, 1/(t+1)**2)
            beta_sqrt = np.sqrt(2*np.log(1/zeta) + np.log(det_W_t) - np.log(det_W_1)) + np.sqrt(gamma)*B
            B_hat = min(B, theta_hat_norm + (beta_sqrt / np.sqrt(lambda_min)))

            beta_hat_sqrt = np.sqrt(2*np.log(1/zeta) + np.log(det_W_t) - np.log(det_W_1)) + np.sqrt(gamma)*B_hat

            if t < estim_B_steps:
                for a in range(self.n_actions):
                    Upper_bounds[a] = np.dot(self.features[a], theta_hat) +  beta_hat_sqrt*mat_norm(W_t_inv, self.features[a])
                    Lower_bounds[a] = np.dot(self.features[a], theta_hat) -  beta_hat_sqrt*mat_norm(W_t_inv, self.features[a])

                a_t_EB_UCB = rd_argmax(Upper_bounds)
                for a in range(self.n_actions):
                    W_t_plus = W_t + (1/(self.eta[a]**2))*np.outer(self.features[a], self.features[a])
                    W_t_plus_inv = np.linalg.inv(W_t_plus)

                    first_norm = mat_norm(W_t_inv, self.features[a_t_EB_UCB])
                    second_norm = mat_norm(W_t_plus_inv, self.features[a_t_EB_UCB])


                    I_t_EB_UCB[a] = (1/2)*(np.log(first_norm) - np.log(second_norm)) + 1e-8 # small constant added for numerical stability
                    I_t_B[a] =  (1/2)*(np.log(mat_norm(W_t_plus, v_min))) - (1/2)*np.log(lambda_min)

                I_t_BAM = alpha*I_t_B + (1-alpha)*I_t_EB_UCB
                Delta_hat = max(Upper_bounds) - Lower_bounds
                arm = rd_argmax(I_t_BAM / Delta_hat**2)
            else:
                B_tilde = min(B, np.min(B_estimates_vec))
                beta_tilde_sqrt = np.sqrt(2*np.log(1/zeta) + np.log(det_W_t) - np.log(det_W_1)) + np.sqrt(gamma)*B_tilde
                for a in range(self.n_actions):
                    Upper_bounds[a] = np.dot(self.features[a], theta_hat) +  beta_tilde_sqrt*mat_norm(W_t_inv, self.features[a])
                    Lower_bounds[a] = np.dot(self.features[a], theta_hat) -  beta_tilde_sqrt*mat_norm(W_t_inv, self.features[a])

                a_t_EB_UCB = rd_argmax(Upper_bounds)
                for a in range(self.n_actions):
                    W_t_plus = W_t + (1/(self.eta[a]**2))*np.outer(self.features[a], self.features[a])
                    W_t_plus_inv = np.linalg.inv(W_t_plus)

                    first_norm = mat_norm(W_t_inv, self.features[a_t_EB_UCB])
                    second_norm = mat_norm(W_t_plus_inv, self.features[a_t_EB_UCB])

                    I_t_EB_UCB[a] = (1/2)*(np.log(first_norm) - np.log(second_norm)) + 1e-8 # small constant added for numerical stability

                Delta_hat = max(Upper_bounds) - Lower_bounds
                arm = rd_argmax(I_t_EB_UCB / Delta_hat**2)

            reward = self.reward(arm)
            rewards[t] = reward
            arm_sequence[t] = arm
            
            W_t += (1/(self.eta[arm]**2))*np.outer(self.features[arm], self.features[arm])
            V_t += np.outer(self.features[arm], self.features[arm])
            b_t += (1/(self.eta[arm]**2))*reward*self.features[arm]

            B_estimates_vec = np.append(B_estimates_vec, theta_hat_norm + (beta_hat_sqrt / np.sqrt(lambda_min)))

        return rewards, arm_sequence
