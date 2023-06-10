# %%
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from analysis_functions import compute_transitions


'''
Analyse states: downstate probability, probability of state change
Required: files that ran the HMM analysis for every session to analyse
'''

# directory where state- and firing data is stored
output_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), 'data/firing_rates'))

# directory for saving analysed data
analysed_data_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), 'Analysis/analysed_data'))

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

# %%
