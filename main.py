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
    nr_sessions = 10 # to get some error bars we run several sessions

    if not os.path.exists('./data/firing_rates/'):
        os.mkdir('./data/firing_rates/')

    # serotonin stimulation 
    S_parameters = np.arange(0,100,10)
    # create an array with many G parameters within 1 array 
    G_parameters = np.arange(0, 5, 0.5)

    plot_results = False

    for G in G_parameters:

        G_param = np.round(G, 1)
        print(f'\nSimulation with G = {G_param}')

        for S in S_parameters:

            for session in range(nr_sessions):
                print(f"Run: S={S}, G={G}, session={session}")
                # create the simulation session 
                S_param =  np.round(S, 1)
                sim_session = SimulationSession(output_dir, nrAreas, filename_connectivity, settings_file, drn_connect_file, G_param, S_param, session)
                sim_session.start_sim()
                if plot_results:
                    sim_session.plot_results()


if __name__ == '__main__':
   main()