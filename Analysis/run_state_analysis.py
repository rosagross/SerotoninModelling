# %%
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from analysis_functions import *

#import analysis_functions as af

'''
Analyse states: downstate probability, probability of state change
Required: files that ran the HMM analysis for every session to analyse
'''

# directory where state- and firing data is stored
output_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), 'data/firing_rates'))

# directory for saving analysed data
analysed_data_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), 'Analysis/analysed_data'))

# %%

G_params = [2]
S = 0
beta = 6
thetaE = -1
extra = 'RateAdj1_K'
all_sessions = np.arange(0,10,1)
window_duration = 4000
pre_stim = 1000
serotonin_analysis = pd.DataFrame()

# load data for all sessions 
for G in G_params:
    print('G:', G)
    for session in all_sessions:

        # get session folder name
        session_name = f'14areas_G{float(G)}_S{float(S)}_thetaE{thetaE}_beta{beta}{extra}_sessions'
        file_dir = os.path.join(output_dir, session_name)
        file_name = f'14areas_G{float(G)}_S{float(S)}_thetaE{thetaE}_beta{beta}{extra}_sessions_{session}'

        # compute down state probability 
        data_df = compute_transitions([G], [S], extra=extra, window_length=window_duration, pre_stim=pre_stim)
        serotonin_analysis = pd.concat((serotonin_analysis, data_df))

serotonin_analysis.to_csv(os.path.join(analysed_data_dir, f'serotonin_analysis_all_regions_S{S}_G{G_params}_{extra}.csv'), index=False)

# take the mean for the combined regions 
regions = np.unique(serotonin_analysis['region_name'])
state_transitions = serotonin_analysis.groupby(['region_name', 'S', 'session', 'G', 'time'], as_index=False).mean()

# compute bins
# for this find how many sessions there are in this set and how long one session is 
bin_size = 200
nr_sessions = len(np.unique(state_transitions['session']))
max_time = len(np.unique(state_transitions['time']))
nr_bins = max_time/bin_size
bins = np.linspace(-pre_stim, max_time-pre_stim, int(nr_bins), endpoint=False)

# compute the time points of the bins
digitized = np.digitize(np.unique(state_transitions['time']), bins, right=True)
state_transitions['bin'] = np.tile(digitized, nr_sessions*len(regions)*len([S])*len(G_params))

# average over bins
data_grouped_bins = state_transitions.groupby(['region_name', 'session', 'bin', 'G', 'S'], as_index=False).mean()
data_grouped_bins['time'] = data_grouped_bins['time']/1000

# save averaged time bin data 
data_grouped_bins.to_csv(os.path.join(analysed_data_dir, f'state_analysis_bins_S{S}_G{G_params}_{extra}.csv'), index=False)
 # %%

# run state analysis for all different option of serotonin targets (targeting either E or I population 
# with an inhibitory or excitatory current)

S = 0
G = [2]

serotonin_analysis = pd.read_csv(os.path.join(analysed_data_dir, f'serotonin_analysis_all_regions_S{S}_G{G}_RateAdj1.csv'))
serotonin_analysis_I = pd.read_csv(os.path.join(analysed_data_dir, f'serotonin_analysis_all_regions_S{S}_G{G}_RateAdj1_I.csv'))
serotonin_analysis_J = pd.read_csv(os.path.join(analysed_data_dir, f'serotonin_analysis_all_regions_S{S}_G{G}_RateAdj1_J.csv'))
serotonin_analysis_K = pd.read_csv(os.path.join(analysed_data_dir, f'serotonin_analysis_all_regions_S{S}_G{G}_RateAdj1_K.csv'))

targets = {'E-' : serotonin_analysis, 'I+' : serotonin_analysis_I, 'I-' : serotonin_analysis_J, 'E+' : serotonin_analysis_K}

# group data into time bins 
bin_size = 200
all_targets_df = pd.DataFrame()

for target_name, target_df in targets.items():
    # 
    # take the mean for the combined regions 
    state_transitions = target_df.groupby(['region_name', 'S', 'session', 'G', 'time'], as_index=False).mean()

    # compute bins
    # for this find how many sessions there are in this set and how long one session is 
    nr_sessions = len(np.unique(state_transitions['session']))
    max_time = len(np.unique(state_transitions['time']))
    nr_bins = max_time/bin_size
    bins = np.linspace(-pre_stim, max_time-pre_stim, int(nr_bins), endpoint=False)

    # compute the time points of the bins and add corresponding target name (e.g., 'E+')
    digitized = np.digitize(np.unique(state_transitions['time']), bins, right=True)
    state_transitions['bin'] = np.tile(digitized, nr_sessions*len(regions)*len([S])*len(G))
    data_grouped_bins = state_transitions.groupby(['region_name', 'session', 'bin', 'window', 'G'], as_index=False).mean()
    data_grouped_bins['time'] = data_grouped_bins['time']/1000
    data_grouped_bins['target_name'] = target_name
    
    all_targets_df = pd.concat((all_targets_df, data_grouped_bins))

# save data
all_targets_df.to_csv(os.path.join(analysed_data_dir, f'serotonin_targets_state_analysis_S{S}_G{G}.csv'))

# %% Run synchrony analysis



def compute_brunel(output_dir, file_name):
    """
    Quantify synchrony of the network by computing the parameter X as defined in Brunel & Hansel (2006).
    """
    
    # get the session number (if file is part of several session)
    session = extract_session_nr(file_name)
    
    # load the activity for the inhibitory poputations
    _, ratesI, _ = load_firing_rates(output_dir, file_name)
    
    # get infos about session (for saving it later)
    G, S = extract_file_info(output_dir, file_name)
    
    # compute the variance of the population averaged activity
    network_rate_mean_pertime = np.mean(ratesI, axis=1)
    network_rate_mean_perregion = np.mean(ratesI, axis=0)

    # this is the same as np.var(network_rate_mean_pertime)
    network_variance = np.mean((network_rate_mean_pertime)**2) - np.mean(network_rate_mean_pertime)**2
    #print('network variance:', network_variance)

    # calculate the variance of each area individually 
    # the individual variance is completely independent of the other region's variance 
    individual_variance = np.mean((ratesI)**2) - np.mean(ratesI)**2

    # X parameter
    # if the average individual variance is similar to the overall regions variance 
    mean_individual_var = np.mean(individual_variance)
    #print('average of individual variance:', mean_individual_var)

    brunel_x = np.sqrt(network_variance/mean_individual_var)
    #print('synchrony:', brunel_x)
    
    return pd.DataFrame(data={'brunel_X' : brunel_x, 'G' : G, 'S' : S, 'session' : session}, index=[0])
# %%     

# calculate the states and synchrony quantification (brunel X parameter) for many different G values 
G_parameters = [0,1,2,3]
S = 0
brunel_X_df = pd.DataFrame()
extra = 'RateAdj1'
thetaE = -1
beta = 6
session = 0

for G in G_parameters:
    G_param = np.round(G, 1)
    session_name = f'14areas_G{float(G)}_S{float(S)}_thetaE{thetaE}_beta{beta}{extra}_sessions'
    file_dir = os.path.join(output_dir, session_name)
    file_name = f'14areas_G{float(G_param)}_S{float(S)}_thetaE{thetaE}_beta{beta}{extra}_sessions_{session}'
    brunel_X_df = pd.concat((brunel_X_df, compute_brunel(file_dir, file_name)))

brunel_X_df.to_csv(os.path.join(analysed_data_dir, f'brunelX_S{S}_G{G_parameters}{extra}.csv'))

# %% run state analysis (on whole trajectory)

def compute_state_analysis(output_dir, file_name):
    """
    File should be the From the data frame with state info calculate the most important parameter for every 
    region time series.
    G value, thetaE, beta, thetaE stimulation, serotonin stimulation times, and the most important parameters:
    Description of the parameters:
    f_s - frequency of states
    d_down - average duration of down-states
    d_up - average duration of up-states
    p_down - proportion time in state 0
    p_up - proportion time in state 1 
    brunel_X - quantifies synchrony: 0 low, 1 high
    """
    
    # check if it is a session
    session = extract_session_nr(file_name)
            
    # check if any files in the directory end with 'states.csv'
    if any(f.endswith(f'{session}_states.csv') for f in os.listdir(output_dir)):
        pass
    else:
        print('The folder does not contain a file with the ending "states.csv"')
        _ = get_all_states(output_dir, file_name)
        
        
    region_states = pd.read_csv(os.path.join(output_dir, f'{file_name}_states.csv'))
    state_statistics = pd.DataFrame()
    
    # get infos about session (for saving it later)
    G, S = extract_file_info(output_dir, file_name)

    # load atlas with releveant regions for plotting
    atlas = pd.read_csv(os.path.join(atlas_dir, 'relevant_areas.csv'))
    atlas.drop(['Unnamed: 0'], inplace=True, axis=1)
    atlas = np.array(atlas)
    
    for i, region in enumerate(np.unique(region_states['region'])):
    
        # get the time series for this region
        time_series = list(region_states[region_states['region']==region]['state'])
        trial_duration = len(time_series)

        # count the number of occurrences of 0 and 1
        num_zeros = time_series.count(0)
        num_ones = time_series.count(1)

        # calculate the duration of each state (in number of time steps)
        zero_durations = [sum(1 for _ in group) for key, group in itertools.groupby(time_series) if key == 0]
        one_durations = [sum(1 for _ in group) for key, group in itertools.groupby(time_series) if key == 1]
                
        # calculate the frequency of the states (I use the up-state here) 
        f_s = (len(one_durations)/trial_duration)*1000
        
        # calculate the average state durations 
        d_down = np.mean(zero_durations)
        d_up = np.mean(one_durations)

        # calculate the total duration of each state (in number of time steps)
        total_zero_duration = sum(zero_durations)
        total_one_duration = sum(one_durations)

        # calculate the proportion of time spent in each state
        prop_zero = total_zero_duration / len(time_series)
        prop_one = total_one_duration / len(time_series)

        # collect the values in the dataframe 
        state_statistics = pd.concat((state_statistics, pd.DataFrame(data={'state_frequency': f_s, 'd_down': d_down,
                                                                           'd_up' : d_up, 'p_down' : prop_zero,
                                                                           'p_up' : prop_one,
                                                                           'region' : atlas[i][0], 'G' : G, 'S' : S,
                                                                           'session': session}, index=[0])))
        
    return state_statistics

sessions = np.arange(0, 10, 1)
S = 0
G = 2
session_dir = f"14areas_G{float(G)}_S{float(S)}_thetaE-1_beta6RateAdj1_sessions"
states_df = pd.DataFrame()

for ses in sessions:

    file_name = f"14areas_G2.0_S0.0_thetaE-1_beta6RateAdj1_sessions_{ses}"
    file_dir = os.path.join(output_dir, session_dir)
    state_stats = compute_state_analysis(file_dir, file_name)
    states_df = pd.concat((states_df, state_stats))

states_df.to_csv(os.path.join(analysed_data_dir, f'totalduration_state_analysis_S{S}_G{G}RateAdj1.csv'))