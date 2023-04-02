import numpy as np 
import sys
from datetime import datetime
import os
from parameter import Parameter
from simulation_functions import SimulationSession


def main():

    output_dir = os.path.join('data', 'firing_rates')
    settings_file = 'parameter.yml'
    filename_connectivity = 'MODEL_Cmatrix_grouped_cre-False_hemi-3_grouping-median.csv'
    drn_connect_file = 'drn_connectivity_cre-True_hemi-3_grouping-median.csv'
    nrAreas = 14

    if not os.path.exists('./data/firing_rates/'):
        os.mkdir('./data/firing_rates/')

    # create an array with many G parameters within 1 array 
    G_parameters = np.arange(1.1, 3, 0.1)
    plot_results = False

    for G in G_parameters:

        G_param = np.round(G, 1)
        print(f'\nSimulation with G = {G_param}')

        # create the simulation session 
        sim_session = SimulationSession(output_dir, nrAreas, filename_connectivity, settings_file, drn_connect_file, G_param)
        sim_session.start_sim()
        if plot_results:
            sim_session.plot_results()


if __name__ == '__main__':
   main()