import numpy as np 
import os
from parameter import Parameter
from simulation_functions import SimulationSession


def main():
    # read in arguments
    total_sim = 5 # in sec
    sigmaN = 3.5
    betaE = 6 # 0.7
    t_start=0
    t_end=total_sim*1000 # in milliseconds
    dt=0.2
    x0=0
    thetaE = -1
    tauN = 1

    # for the midbrain and hippocampus we need a separate thetaE & betaE parameter combination
    # they usually only show up-states
    thetaE_UP = -5
    betaE_UP = 1

    # get the parameter
    par = Parameter()
    par.thetaE = par.Edesp - thetaE 
    par.tauN = tauN    
    par.sigmaN = sigmaN
    par.betaE = betaE
    par.thetaE_UP = par.Edesp - thetaE_UP
    par.betaE_UP = betaE_UP
    sim_params = [t_start, t_end, dt]
    output_dir = 'data/'
    filename_connectivity = 'MODEL_Cmatrix_grouped_cre-False_hemi-3_grouping-max.csv'
    nrAreas = 14


    # create the simulation session 
    sim_session = SimulationSession(par, sim_params, output_dir, nrAreas, filename_connectivity)
    sim_session.start_sim()
    sim_session.plot_results()


if __name__ == '__main__':
   main()