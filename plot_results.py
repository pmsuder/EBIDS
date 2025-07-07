import numpy as np
import pickle
import os

from sim_utils import plot_regrets, get_normal_confidence_bounds, generate_latex_table

ALL_SETTINGS = ['conservative_B', 'anti_conservative_B',  '10_arms_fixed_variances', '10_arms_random_variances', '20_arms_fixed_variances', '20_arms_random_variances', 'splines']

def make_plots(setting):

    print('Plottig results for setting: ' + setting)
    
    # Define color-blicd friendly palettes
    CB_colors = ['#377eb8', '#ff7f00', '#4daf4a',
                 '#e41a1c', '#a65628', '#984ea3',
                 '#f781bf', '#999999', '#dede00']

    CB_colors_ordered = ['#377eb8', '#984ea3', '#e41a1c',  '#ff7f00', '#dede00']

    palette = CB_colors
    palette_sensitivity = CB_colors_ordered

    num_results = 200
    sim_len = 500
    table_skip = 20
    
    table_timepts = [i for i in range(table_skip-1, sim_len, table_skip)]
    
    solid_lines_keys = ['EBIDS', 'IDS-UCB', 'UCB', 'EB-UCB', 'NAOFUL', 'OLSOFUL']
    dashed_lines_keys = ['EBIDS (oracle)', 'IDS-UCB (oracle)', 'UCB (oracle)']

    TB_50_names = ['$\\alpha = 0.1, T_B = 50$', '$\\alpha = 0.3, T_B = 50$', '$\\alpha = 0.5, T_B = 50$', '$\\alpha = 0.7, T_B = 50$']
    TB_100_names = ['$\\alpha = 0.1, T_B = 100$', '$\\alpha = 0.3, T_B = 100$', '$\\alpha = 0.5, T_B = 100$', '$\\alpha = 0.7, T_B = 100$']

    if setting == 'conservative_B':
        figsize = (8.5, 6)
        fontsize = 22
        add_legend = False
        add_y_label = True

        solid_lines_keys = ['IDS-UCB ($B=100$)', 'UCB ($B=100$)']
        dashed_lines_keys = ['IDS-UCB (oracle)', 'UCB (oracle)']

        # modify the color palette
        palette = [CB_colors[i] for i in range(1,3)]

    elif setting == 'anti_conservative_B':
        figsize = (14, 6)
        fontsize = 22
        add_legend = True
        add_y_label = False

        solid_lines_keys = ['IDS-UCB', 'UCB']
        dashed_lines_keys = ['IDS-UCB (oracle)', 'UCB (oracle)']

        # modify the color palette
        palette = [CB_colors[i] for i in range(1,3)]

    elif setting == '10_arms_fixed_variances':
        figsize = (14, 6)
        fontsize = 18
        add_legend = True
        add_y_label = True

    elif setting == '10_arms_random_variances':
        figsize = (8.5, 6)
        fontsize = 22
        add_legend = False
        add_y_label = True

    elif setting == '20_arms_fixed_variances':
        figsize = (8.5, 6)
        fontsize = 22
        add_legend = False
        add_y_label = True

    elif setting == '20_arms_random_variances':
        figsize = (14, 6)
        fontsize = 22
        add_legend = True
        add_y_label = False

    elif setting == 'splines':
        figsize = (14, 6)
        fontsize = 22
        add_legend = True
        add_y_label = False

    else:
        raise Exception("Invalid \"setting\" option chosen. Available options: \"conservative_B\", \"anti_conservative_B\", \"10_arms_fixed_variances\", \"10_arms_random_variances\", \"20_arms_fixed_variances\", \"20_arms_random_variances\", \"splines\".")


    all_results = {}
    results_dir = os.path.join('results_slurm', setting)
    for i in range(num_results):
        with open(os.path.join(results_dir, 'all_results_' + str(i+1)) + '.pkl', 'rb') as file:
            all_results_i = pickle.load(file)

        for key in all_results_i.keys():
            if i == 0:
                all_results[key] = np.empty((num_results, sim_len))
            all_results[key][i,:] = np.array(all_results_i[key])


    mean_results = {}
    for key in all_results.keys():
        mean_results[key] = np.mean(all_results[key], axis = 0)

    results_plot = {}

    for index in range(len(solid_lines_keys)):
        key = solid_lines_keys[index]
        results_plot[key] = {'color' : palette[index],  'linestyle' : '-', 'mean-regret' : mean_results[key], 'regret' : all_results[key]}

    for index in range(len(dashed_lines_keys)):
        key = dashed_lines_keys[index]
        results_plot[key] = {'color' : palette[index],  'linestyle' : '--', 'mean-regret' : mean_results[key], 'regret' : all_results[key]}

    lower_conf_results, upper_conf_results = get_normal_confidence_bounds(all_results, quantiles = (0.025, 0.975))

    plot_regrets(outfile = 'plots/' + setting + '.pdf', mean_results = results_plot, lower_conf_results=lower_conf_results, upper_conf_results=upper_conf_results, figsize = figsize, num_fontsize = fontsize, legend_fontsize = fontsize, axis_label_fontsize = fontsize, add_legend = add_legend , add_y_label = add_y_label)


    # ABLATION STUDY ON SENSITIVITY OF EBIDS - run it for all settings except for the initial simulation illustration --------------
    if setting not in ['conservative_B', 'anti_conservative_B']:
        all_results = {}
        results_dir = os.path.join('results_slurm', setting + '_sensitivity')
        for i in range(num_results):
            with open(os.path.join(results_dir, 'all_results_sensitivity_' + str(i+1)) + '.pkl', 'rb') as file:
                all_results_i = pickle.load(file)
            for key in all_results_i.keys():
                if i == 0:
                    all_results[key] = np.empty((num_results, sim_len))
                all_results[key][i,:] = np.array(all_results_i[key])


        mean_results = {}
        for key in all_results.keys():
            mean_results[key] = np.mean(all_results[key], axis = 0)

        lower_conf_results, upper_conf_results = get_normal_confidence_bounds(all_results, quantiles = (0.025, 0.975))

        results_plot = {}

        for index in range(len(TB_50_names)):
            key = TB_50_names[index]
            results_plot[key] = {'color' : palette_sensitivity[index], 'linestyle' : '-', 'mean-regret' : mean_results[key], 'regret' : all_results[key]}

        for index in range(len(TB_100_names)):
            key = TB_100_names[index]
            results_plot[key] = {'color' : palette_sensitivity[index], 'linestyle' : '--', 'mean-regret' : mean_results[key], 'regret' : all_results[key]}


        plot_regrets(outfile = 'plots/' + setting + '_EBIDS_tuning.pdf', mean_results = results_plot, lower_conf_results = None, upper_conf_results = None, figsize = figsize, num_fontsize = fontsize, legend_fontsize = fontsize, axis_label_fontsize = fontsize, add_legend = add_legend, add_y_label = add_y_label)
        
        # PREPARE TABLE FOR LATEX ------------------------------------------------------------------------
        table_latex = {}
        for key in all_results.keys():
            table_latex[key] = {time: (mean_results[key][time], mean_results[key][time] - lower_conf_results[key][time]) for time in table_timepts}
        
        generate_latex_table(filename = 'tables/' + setting + '.txt', table = table_latex)
    
if __name__ == '__main__':
    for setting in ALL_SETTINGS:
        make_plots(setting)