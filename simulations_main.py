import numpy as np
import random
import pickle
import argparse
import os

from sim_utils import run_simulations

ALL_SETTINGS = ['conservative_B', 'anti_conservative_B', '10_arms_fixed_variances', '10_arms_random_variances', '20_arms_fixed_variances', '20_arms_random_variances', 'splines']

def simulation_main(setting):
    
    #TASK_ID = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    TASK_ID = 0
    
    # Shared parameters for different settings ------------------------------------------------------------
    T = 500
    reps = 1
    
    theta_truth = np.array([-5, 1, 1, 1.5, 2])
    B_exact = np.sqrt(np.sum(theta_truth**2))
    features_range = 1 / np.sqrt(5)
    B_assumed = 100
    
    # These are applicable for all settings except for splines where we set them accordingly --------------
    splines = False
    discretization_steps = None
    alpha_0 = None
    alpha_1 = None
    # ------------------------------------------------------------------------------------------------------
        
    
    # parameters for algorithms comparison (excluding conservative-B and anti-conservative-B scenarios) --------------------------------------------
    params_dict = {}
    params_dict['EBIDS'] = {'algo': 'EB-IDS', 'gamma' : 0.2, 'B': B_assumed, 'alpha' : 0.5, 'estim_B_steps' : 50, 'B_bound_err' : 1e-3}
    params_dict['IDS-UCB'] = {'algo': 'IDS-UCB', 'gamma' : 0.2, 'B': B_assumed, 'deterministic' : False, 'explore_steps': 0}
    params_dict['UCB'] = {'algo': 'UCB', 'gamma' : 0.2, 'B': B_assumed, 'explore_steps': 0}
    params_dict['EB-UCB'] = {'algo': 'EB-UCB', 'gamma' : 0.2, 'B': B_assumed, 'estim_B_steps' : 50, 'B_bound_err' : 1e-3}
    params_dict['NAOFUL'] = {'algo': 'NAOFUL', 'alpha' : 2}
    params_dict['OLSOFUL'] =  {'algo': 'OLSOFUL', 'delta' : 1e-3}

    # Oracles
    params_dict['EBIDS (oracle)'] = {'algo': 'EB-IDS', 'gamma' : 0.2, 'B': B_exact, 'alpha' : 0.5, 'estim_B_steps' : 50, 'B_bound_err' : 1e-3}
    params_dict['IDS-UCB (oracle)'] = {'algo': 'IDS-UCB', 'gamma' : 0.2, 'B': B_exact, 'deterministic' : False, 'explore_steps': 0}
    params_dict['UCB (oracle)'] = {'algo': 'UCB', 'gamma' : 0.2, 'B': B_exact, 'explore_steps': 0}

    
    # parameters for snesitivity ablation study -------------------------------------------------------------------------------------------------------
    params_dict_sensitivity = {}
    params_dict_sensitivity['$\\alpha = 0.7, T_B = 50$'] = {'algo' : 'EB-IDS', 'gamma' : 0.2, 'B': B_assumed, 'alpha' : 0.7, 'estim_B_steps' : 50, 'B_bound_err' : 1e-3, 'explore_steps': 0}
    params_dict_sensitivity['$\\alpha = 0.5, T_B = 50$'] = {'algo' : 'EB-IDS', 'gamma' : 0.2, 'B': B_assumed, 'alpha' : 0.5, 'estim_B_steps' : 50, 'B_bound_err' : 1e-3, 'explore_steps': 0}
    params_dict_sensitivity['$\\alpha = 0.3, T_B = 50$'] = {'algo' : 'EB-IDS', 'gamma' : 0.2, 'B': B_assumed, 'alpha' : 0.3, 'estim_B_steps' : 50, 'B_bound_err' : 1e-3, 'explore_steps': 0}
    params_dict_sensitivity['$\\alpha = 0.1, T_B = 50$'] = {'algo' : 'EB-IDS', 'gamma' : 0.2, 'B': B_assumed, 'alpha' : 0.1, 'estim_B_steps' : 50, 'B_bound_err' : 1e-3, 'explore_steps': 0}

    params_dict_sensitivity['$\\alpha = 0.7, T_B = 100$'] = {'algo' : 'EB-IDS', 'gamma' : 0.2, 'B': B_assumed, 'alpha' : 0.7, 'estim_B_steps' : 100, 'B_bound_err' : 1e-3, 'explore_steps': 0}
    params_dict_sensitivity['$\\alpha = 0.5, T_B = 100$'] = {'algo' : 'EB-IDS', 'gamma' : 0.2, 'B': B_assumed, 'alpha' : 0.5, 'estim_B_steps' : 100, 'B_bound_err' : 1e-3, 'explore_steps': 0}
    params_dict_sensitivity['$\\alpha = 0.3, T_B = 100$'] = {'algo' : 'EB-IDS', 'gamma' : 0.2, 'B': B_assumed, 'alpha' : 0.3, 'estim_B_steps' : 100, 'B_bound_err' : 1e-3, 'explore_steps': 0}
    params_dict_sensitivity['$\\alpha = 0.1, T_B = 100$'] = {'algo' : 'EB-IDS', 'gamma' : 0.2, 'B': B_assumed, 'alpha' : 0.1, 'estim_B_steps' : 100, 'B_bound_err' : 1e-3, 'explore_steps': 0}

    
    if setting == 'conservative_B':
        n_actions = 10
        eta_vec = np.array([1 for i in range(5)] + [0.2 for i in range(5)])
        rand_eta_vec = False
        eta_vec_range = None
        
        features_range = 1 / np.sqrt(5)
        B_assumed = 100
        
        # conservative-B and anti-conservative-B are the only cases where we use a different set of algorithms with different parameters than for other comparisons
        params_dict = {}
        params_dict['IDS-UCB ($B=100$)'] = {'algo': 'IDS-UCB', 'gamma' : 0.2, 'B': B_assumed, 'deterministic' : False, 'explore_steps': 0}
        params_dict['UCB ($B=100$)'] = {'algo': 'UCB', 'gamma' : 0.2, 'B': B_assumed, 'explore_steps': 0}

        # Oracles
        params_dict['IDS-UCB (oracle)'] = {'algo': 'IDS-UCB', 'gamma' : 0.2, 'B': B_exact, 'deterministic' : False, 'explore_steps': 0}
        params_dict['UCB (oracle)'] = {'algo': 'UCB', 'gamma' : 0.2, 'B': B_exact, 'explore_steps': 0}

    elif setting == 'anti_conservative_B':
        n_actions = 10
        eta_vec = np.array([1 for i in range(5)] + [0.2 for i in range(5)])
        rand_eta_vec = False
        eta_vec_range = None
        
        features_range = 1 / np.sqrt(5)
        B_assumed = 1.0

        # conservative-B and anti-conservative-B are the only cases where we use a different set of algorithms with different parameters than for other comparisons
        params_dict = {}
        params_dict['IDS-UCB'] = {'algo': 'IDS-UCB', 'gamma' : 2.0, 'B': B_assumed, 'deterministic' : False, 'explore_steps': 0}
        params_dict['UCB'] = {'algo': 'UCB', 'gamma' : 2.0, 'B': B_assumed, 'explore_steps': 0}

        # Oracles
        params_dict['IDS-UCB (oracle)'] = {'algo': 'IDS-UCB', 'gamma' : 0.2, 'B': B_exact, 'deterministic' : False, 'explore_steps': 0}
        params_dict['UCB (oracle)'] = {'algo': 'UCB', 'gamma' : 0.2, 'B': B_exact, 'explore_steps': 0}

    elif setting == '10_arms_fixed_variances':
        n_actions = 10
        eta_vec = np.array([1 for i in range(5)] + [0.2 for i in range(5)])
        rand_eta_vec = False
        eta_vec_range = None
        
        features_range = 1 / np.sqrt(5)
        
    elif setting == '10_arms_random_variances':
        n_actions = 10
        eta_vec = None
        rand_eta_vec = True
        eta_vec_range = (0.1,1)
        
        features_range = 1 / np.sqrt(5)
    
    elif setting == '20_arms_fixed_variances':
        n_actions=20
        eta_vec = np.array([1 for i in range(10)] + [0.2 for i in range(10)])
        rand_eta_vec = False
        eta_vec_range = None

    elif setting == '20_arms_random_variances':
        n_actions = 20
        eta_vec = None
        rand_eta_vec = True
        eta_vec_range = (0.1,1)
        
        features_range = 1 / np.sqrt(5)
        
    elif setting == 'splines':
        n_actions = None # The number of arms will be determined by the discretization_steps parameter
        eta_vec = None # In case of splines the variance function of arms is determined by parameters alpha_0 and alpha_1
        rand_eta_vec = False
        eta_vec_range = (0.2,1)
        
        features_range = None
        
        # Setting spline-specific parameters for this setting
        splines = True
        discretization_steps = 1000
        alpha_0 = 0.5
        alpha_1 = -3
            
    else:
        raise Exception("Invalid \"setting\" option chosen. Available options: \"conservative_B\", \"anti_conservative_B\", \"10_arms_fixed_variances\", \"10_arms_random_variances\", \"20_arms_fixed_variances\", \"20_arms_random_variances\", \"splines\".")
        
    # ALGORITHMS COMPARISON --------------------------------------------------------------------------------------------------------
    np.random.seed(TASK_ID)
    random.seed(TASK_ID)

    mean_results, all_results = run_simulations(T=T, features_range=features_range, theta_truth=theta_truth, 
                            eta_vec=eta_vec, n_actions=n_actions, params_dict=params_dict, reps=reps, rand_eta_vec = rand_eta_vec, 
                            eta_vec_range = eta_vec_range, splines = splines, discretization_steps = discretization_steps, alpha_0 = alpha_0, alpha_1 = alpha_1)

    all_results_lists = {key: value.tolist() for key, value in all_results.items()}

    with open("results_slurm/" + setting + "/all_results_" + str(TASK_ID) + ".pkl", "wb") as file:
        pickle.dump(all_results_lists, file)
        file.close()

    # ABLATION STUDY ON SENSITIVITY OF EBIDS - run it for all settings except for the initial simulation illustration --------------
    if setting not in ['conservative_B', 'anti_conservative_B']:
        np.random.seed(TASK_ID)
        random.seed(TASK_ID)
        mean_results_sensitivity, all_results_sensitivity = run_simulations(T=T, features_range=features_range, theta_truth=theta_truth, 
                            eta_vec=eta_vec, n_actions = n_actions, params_dict = params_dict_sensitivity, reps=reps, rand_eta_vec = rand_eta_vec, 
                            eta_vec_range = eta_vec_range, splines = splines, discretization_steps = discretization_steps, alpha_0 = alpha_0, alpha_1 = alpha_1)

        all_results_sensitivity_lists = {key: value.tolist() for key, value in all_results_sensitivity.items()}

        with open("results_slurm/" + setting + "_sensitivity/all_results_sensitivity_" + str(TASK_ID) + ".pkl", "wb") as file:
            pickle.dump(all_results_sensitivity_lists, file)
            file.close()

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for obtaining regret measurements for all algorithms.")
    parser.add_argument('--setting', type=str, required=True, help='Available setting options: \"conservative_B\", \"anti_conservative_B\", \"10_arms_fixed_variances\", \"10_arms_random_variances\", \"20_arms_fixed_variances\", \"20_arms_random_variances\", \"splines\".')
    args = parser.parse_args()

    if args.setting == 'all':
        for setting in ALL_SETTINGS:
            simulation_main(setting)
    else:
        simulation_main(args.setting)