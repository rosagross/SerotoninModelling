import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import ssm 
from stim_functions import figure_style
import yaml
import re

# set directories
output_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), 'data', 'firing_rates'))
atlas_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), 'Analysis', 'atlas_data'))
model_dir = os.path.abspath(os.path.dirname(os.getcwd()))


def load_firing_rates(output_dir, file_name):
    """
    Load data of firing rates
    output_dir: folder where the firing rates are saved
    Returns: Data Frame with state for all regions 
    """
    
    # load data 
    frate_E_sync = pd.read_csv(os.path.join(output_dir, f'frateE_{file_name}.csv'))
    frate_I_sync = pd.read_csv(os.path.join(output_dir, f'frateI_{file_name}.csv'))
    frate_A_sync = pd.read_csv(os.path.join(output_dir, f'frateA_{file_name}.csv')) 

    ratesG = np.array((frate_E_sync, frate_I_sync, frate_A_sync))
    return ratesG

def extract_file_info(output_dir, file_name):
    
    # load settingsfile 
    settings_file = os.path.join(output_dir, file_name+'_expsettings.yml')
    
    with open(settings_file, 'r', encoding='utf8') as f_in:
        settings = yaml.safe_load(f_in)
        
    G = settings['Parameter']['G']
    S = settings['Parameter']['S']
    
    return [G, S]


def extract_session_nr(filename):
    '''
    Extract the session number from the file name.
    '''
        
    pattern = r'sessions_(\d+)'

    # use re.search to find the first occurrence of the pattern in the filename
    match = re.search(pattern, filename)

    if match:
        session = match.group(1)
    else:
        session = None
        
    return session

def calc_HMM_states(firing_rate_I):
    '''
    Calculate the Up and down states of a time series
    '''
    
    # parameters for the HMM for finding Up and Down states 
    K = 2    # number of discrete states
    D = 1    # dimension of the observations

    # make an hmm and sample from it
    simple_hmm = ssm.HMM(K, D, observations='gaussian')

    # Fit HMM on all data
    trial_data = np.expand_dims(np.array(firing_rate_I), axis=1)
    lls = simple_hmm.fit(trial_data, method='em', transitions='sticky')
    posterior = simple_hmm.filter(trial_data)
    zhat = simple_hmm.most_likely_states(trial_data)
    # check if 1 and 0 state has to be turned around
    # so that 1 is up and 0 is down 
    if np.mean(trial_data[zhat==0]) > np.mean(trial_data[zhat==1]):
        zhat = np.where((zhat==0)|(zhat==1), zhat^1, zhat)
        p_down = posterior[:, 1]
    else:
        p_down = posterior[:, 0]
    
    return zhat, p_down

def get_all_states(output_dir, file_name, save_states=True):
    """
    Compute the up and down states for all of the regions in this session.
    output_dir: folder where the firing rates are saved
    file_name: name of the csv file, without the rate specification
    Returns: Data Frame with state for all regions 
    """
    
    # load data 
    regions_states = pd.DataFrame()
    frate_E_sync = pd.read_csv(os.path.join(output_dir, f'frateE_{file_name}.csv'))
    frate_I_sync = pd.read_csv(os.path.join(output_dir, f'frateI_{file_name}.csv'))
    frate_A_sync = pd.read_csv(os.path.join(output_dir, f'frateA_{file_name}.csv')) 

    ratesG = np.array((frate_E_sync, frate_I_sync, frate_A_sync))

    for i in range(ratesG.shape[2]):

        zhat, p_down = calc_HMM_states(ratesG[1, :, i])

        # safe the states in this dataframe
        regions_states = pd.concat((regions_states, pd.DataFrame(data={
        'state': zhat, 'p_down': p_down, 'region': i})))
    
    if save_states:
        regions_states.to_csv(os.path.join(output_dir, file_name+'_states.csv'))
    
    return regions_states



def compute_transitions(G_params, S_params, regions='all', window_length=4000, pre_stim=1000, sessions=range(10), thetaE=-1, beta=6, extra=''):
    ''' 
    For each window of serotonin stimulation, calculate the downstate probability and state change probablility.
    G_params: list of G parameter 
    S_params: list of S parameter 
    region: brain region
    window_length: length of window around stimulation onset
    pre_stim: time to plot before stimulation onset (subtracted from entire window in the end)
    '''
    
    times = np.arange(-pre_stim, window_length-pre_stim, 1)
    sessions = sessions 
    state_all_windows = pd.DataFrame()

    # regions acronyms
    acronyms = ['Amyg', 'Hipp', 'mPFC', 'MRN', 'OLF', 'OFC', 'PAG', 'Pir', 'RSP', 'M2', 'SC', 'Str', 'Thal', 'VIS']
    
    if regions=='all':
        regions = range(14)


    for S in S_params:
        for G in G_params:
            p_state_change_all = []
           
            session_name = f'14areas_G{float(G)}_S{float(S)}_thetaE{thetaE}_beta{beta}{extra}_sessions' 
            for session in sessions:
                
                for region in regions:
                
                    file_dir = os.path.join(output_dir, session_name)
                    file_name = f'14areas_G{float(G)}_S{float(S)}_thetaE{thetaE}_beta{beta}{extra}_sessions_{session}'
                    states = pd.read_csv(os.path.join(file_dir, file_name+'_states.csv'), index_col=0)
                    states_region = states[states['region']==region]
                    # get the time window around the stimulation 
                    times_array = pd.read_csv(os.path.join(file_dir, file_name+'_stimulation_times.csv')).to_numpy()

                    # first check if all time windows are big enough 
                    # (e.g., sometimes the last simulation is too close to the end of the trial)
                    windows = []
                    for win in times_array:
                        start = win[0] - pre_stim
                        state_window = states_region[start:start+window_length]
                        if len(state_window) == window_length:
                            windows.append(start)

                    for i, start in enumerate(windows):
                        state_window = states_region[start:start+window_length]

                        # the state changes
                        zhat = state_window['state']
                        state_change_up = np.concatenate((np.diff(zhat) > 0, [False])).astype(int)
                        state_change_down = np.concatenate((np.diff(zhat) < 0, [False])).astype(int)

                        state_all_windows = pd.concat((state_all_windows, pd.DataFrame(data={'session': session, 'window': i, 'p_down':state_window['p_down'],
                                                                                 'p_state_change_up': state_change_up, 'p_state_change_down': state_change_down,
                                                                                 'time':times, 'state':zhat,
                                                                                 'region':region, 'abr_region': acronyms[region],
                                                                                 'G' : G, 'S' : S})))
                    
            
    # add the names of the regions to the dataset
    # for this load the atlas with releveant regions for plotting
    atlas = pd.read_csv(os.path.join(atlas_dir, 'relevant_areas.csv'))
    atlas.drop(['Unnamed: 0'], inplace=True, axis=1)
    atlas = np.array(atlas)
    state_all_windows['region_name'] = atlas[np.array(state_all_windows['region'].values, dtype=int)]
    
    # add info about drn connectivity
    drn_connect = pd.read_csv(os.path.join(model_dir, 'drn_connectivity_cre-True_hemi-3_grouping-median_thresh-0.005.csv'))
    drn_connect = np.array(drn_connect)
    state_all_windows['drn_connect'] = drn_connect[np.array(state_all_windows['region'].values, dtype=int)]
    
    return state_all_windows 
