import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.interpolate import BSpline
from joblib import Parallel, delayed

from linear_bandit import LinearBandit

def get_normal_confidence_bounds(all_results, quantiles = (0.025, 0.975)):
    lower_bounds = {}
    upper_bounds = {}
    for key in all_results.keys():
        results_matrix = all_results[key]
        num_trials = results_matrix.shape[0]
        sim_len = results_matrix.shape[1]
        lower_vec = np.empty(sim_len)
        upper_vec = np.empty(sim_len)
        for t in range(sim_len):
            results = results_matrix[:,t]
            mean = np.mean(results)
            sigma_2 = np.var(results, ddof=1) / num_trials
            sigma = np.sqrt(sigma_2)
            lower, upper = stats.norm.ppf(quantiles, loc=mean, scale=sigma)
            lower_vec[t] = lower
            upper_vec[t] = upper
        lower_bounds[key] = lower_vec
        upper_bounds[key] = upper_vec
    return lower_bounds, upper_bounds


    
def get_bootstrap_confidence_bounds(all_results, num_bootstrap = 1000, quantiles = (0.025, 0.975)):
    lower_bounds = {}
    upper_bounds = {}
    for key in all_results.keys():
        results_matrix = all_results[key]
        num_trials = results_matrix.shape[0]
        sim_len = results_matrix.shape[1]
        lower_vec = np.empty(sim_len)
        upper_vec = np.empty(sim_len)
        for t in range(sim_len):
            mean_results = np.empty(num_bootstrap)
            for b in range(num_bootstrap):
                boot_indexes = [random.randint(0, num_trials-1) for _ in range(num_trials)]
                results = np.array([results_matrix[boot_indexes[i], t] for i in range(num_trials)])
                mean_results[b] = np.mean(results)
            [lower, upper] = np.quantile(mean_results, quantiles)
            lower_vec[t] = lower
            upper_vec[t] = upper
        lower_bounds[key] = lower_vec
        upper_bounds[key] = upper_vec
    return lower_bounds, upper_bounds

    
def generate_latex_table(filename, table):
    """
    Generate a LaTeX table with full row and column lines,
    and write it to the specified filename.

    Parameters:
    - filename: str, path to the output .txt file
    - table: dict of dicts, where table[row][column] = (x, y)
    """

    # Extract row and column labels from the table
    row_labels = list(table.keys())
    column_labels = list(next(iter(table.values())).keys())

    # Define column format with vertical bars
    col_format = "|c|" + "|".join(["c"] * len(column_labels)) + "|"

    # Start building LaTeX code
    latex = f"\\begin{{tabular}}{{{col_format}}}\n"
    latex += "\\hline\n"
    latex += " & " + " & ".join(str(column_labels)) + " \\\\\n"
    latex += "\\hline\n"

    for row in row_labels:
        row_entries = [f"{table[row][col][0]} \\pm {table[row][col][1]}" for col in column_labels]
        latex += f"{row} & " + " & ".join(row_entries) + " \\\\\n"
        latex += "\\hline\n"

    latex += "\\end{tabular}"

    # Write to file
    with open(filename, "w") as f:
        f.write(latex)

    print(f"LaTeX table written to '{filename}'")

    
def plot_regrets(outfile, mean_results, lower_conf_results = None, upper_conf_results = None, figsize = (14, 6), num_fontsize = 16, legend_fontsize = 14, axis_label_fontsize = 18, add_legend = True, add_y_label = True):
    """
    results: dictionary with keys being the names that should be displayed on the plot and values being the numpy vectors with regret at each iteration
    """
    plt.figure(figsize=figsize)

    # Iterate over the dictionary to plot each vector with a label
    for algo, info in mean_results.items():
        mean_regret = info['mean-regret']
        color = info['color']
        linestyle = info['linestyle']
        plt.plot(mean_regret, label=algo, color=color, linestyle=linestyle)

        if lower_conf_results and upper_conf_results:
            lower_bound = lower_conf_results[algo]
            upper_bound = upper_conf_results[algo]
            plt.fill_between(range(len(mean_regret)), lower_bound, upper_bound, color=color, alpha=0.2)
        

    # Set axis labels with larger font size
    plt.xlabel('Time', fontsize=axis_label_fontsize)
    
    if add_y_label:
        plt.ylabel('Regret', fontsize=axis_label_fontsize)

    # Increase font size for tick labels on axes
    plt.tick_params(axis='both', labelsize=num_fontsize)

    # Make the legend smaller and position it outside the plot
    if add_legend:
        plt.legend(fontsize=legend_fontsize, bbox_to_anchor=(1.00, 0.5), loc='center left')
        plt.subplots_adjust(right=0.6)  # Shrinks the plot area to make room for the legend


    # Save or display the plot
    plt.savefig(outfile, format='pdf', bbox_inches='tight')
    plt.show()
            
    
def get_algorithm(params, linear_bandit, T):
        if params['algo'] == 'IDS-UCB':
            return lambda: linear_bandit.IDS_UCB(T, params['gamma'], params['B'], params.get('deterministic', False), params['explore_steps'])
        elif params['algo'] == 'UCB': 
            return lambda: linear_bandit.OFUL(T, params['gamma'], params['B'], params['explore_steps'])
        elif params['algo'] == 'EB-IDS': 
            return lambda: linear_bandit.EB_IDS(T, params['gamma'], params['B'], params['alpha'], params['estim_B_steps'], params['B_bound_err'])
        elif params['algo'] == 'EB-UCB':
            return lambda: linear_bandit.EB_OFUL(T, params['gamma'], params['B'], params['estim_B_steps'], params['B_bound_err'])
        elif params['algo'] == 'eps-greedy':
            return lambda: linear_bandit.eps_greedy(T, params['gamma'], params['eps_init'], params['decay_rate'], params['explore_steps'])
        elif params['algo'] == 'NAOFUL': 
            return lambda: linear_bandit.NAOFUL(T, params['alpha'])
        elif params['algo'] == 'OLSOFUL': 
            return lambda: linear_bandit.OLSOFUL(T, params['delta'])
        else:
            raise Exception('Unrecognized algorithm selected!')
            
            
def run_simulations(T, features_range, theta_truth, eta_vec, n_actions, params_dict, reps, rand_eta_vec = False, eta_vec_range = (0.2,1), splines = False, discretization_steps = 100, alpha_0 = -1, alpha_1 = 1, changing_features = False, features_tensor = False, n_jobs=-1):
    def simulate(rand_seed):
        np.random.seed(rand_seed)
        random.seed(rand_seed)

        if splines:
            spline_function = sample_splines_function(degree=3,num_knots=10, dim=n_features)
            features = np.empty((discretization_steps, n_features))
            eta_vec_instance = np.empty(discretization_steps)
            for i in range(discretization_steps):
                a = i / discretization_steps
                features_vec = spline_function(a)
                features[i,:] = features_vec
                eta_vec_instance[i] = np.exp(alpha_0 + alpha_1*a)
        else:
            features = np.random.uniform(-features_range, features_range, (n_actions, n_features))
            if rand_eta_vec:
                eta_vec_instance = np.random.uniform(eta_vec_range[0], eta_vec_range[1], n_actions)
            else:
                eta_vec_instance = eta_vec
                
        linear_bandit = LinearBandit(real_theta = theta_truth, eta = eta_vec_instance, features = features, changing_features = changing_features, features_tensor = features_tensor)

        results_dict = {}
        for algo_name, params in params_dict.items():
            algorithm = get_algorithm(params, linear_bandit, T)
            rewards, _ = algorithm()
            regret = linear_bandit.regret(rewards, T)
            results_dict[algo_name] = regret
        return results_dict
    
    n_features = len(theta_truth)
    
    algo_names_list = params_dict.keys()
    seed_multiplier = random.randint(1, 5000)

    # Parallel execution
    if reps > 1:
        regrets_list = Parallel(n_jobs=n_jobs)(delayed(simulate)(k * seed_multiplier) for k in range(reps))
    else:
        regrets_results = simulate(seed_multiplier)
        regrets_list = [regrets_results]
    all_results_dict = {algo: np.empty((reps, T)) for algo in algo_names_list}

    for k, results_dict in enumerate(regrets_list):
        for algo_name in results_dict.keys():
            all_results_dict[algo_name][k, :] = np.copy(results_dict[algo_name])

    # Compute average regret
    mean_results_dict = {}
    for algo_name in all_results_dict.keys():
        mean_results_dict[algo_name] = np.mean(all_results_dict[algo_name], axis=0)

    return mean_results_dict, all_results_dict

                
def sample_splines_function(degree, num_knots, dim):
    # Generate 10 equally spaced knots in [0,1]
    knots = np.linspace(0, 1, num_knots)

    # Extend knots to satisfy B-spline requirements (adding k+1 repetitions at start & end)
    t = np.concatenate(([0] * (degree+1), knots, [1] * (degree+1)))
    
    spline_vec = [None for i in range(dim)]
    for i in range(dim):

        # Define coefficients (random or fixed)
        c = np.random.rand(len(t) - degree - 1)  # Coefficients for the spline basis functions

        # Create the cubic B-spline
        spline = BSpline(t, c, degree)
        spline_vec[i] = spline

    # Function that evaluates the spline for each component of x
    def spline_function(x):
        return np.array([spline_vec[i](x) for i in range(dim)])
    
    return spline_function


