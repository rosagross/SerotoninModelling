# %%

import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import ssm 
from ssm.plots import gradient_cmap
from stim_functions import figure_style

"""
Replication of UP and DOWN state dynamic 
# three plots for one joint figure:
1) example UP/DOWN state of one area in G = 2
2) different UP/ DOWN state frequencies
3) synchrony 
"""

# set directories
frate_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, os.pardir, 'data', 'firing_rates'))
figure_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, os.pardir, os.pardir, os.pardir, 'Writing', 'Figures', 'Results'))
atlas_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, 'atlas_data'))
analysed_data_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, 'analysed_data'))

color_names = [
    "windows blue",
    "faded green"
    ]
colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)

# load data 
file_dir = "14areas_G2.0_S0.0_thetaE-1_beta6RateAdj1_sessions"
file_name = "14areas_G2.0_S0.0_thetaE-1_beta6RateAdj1_sessions_0"
start = 0
stop = 10*1000
frate_E_sync = pd.read_csv(os.path.join(frate_dir, file_dir, f'frateE_{file_name}.csv'))
frate_I_sync = pd.read_csv(os.path.join(frate_dir, file_dir, f'frateI_{file_name}.csv'))
frate_A_sync = pd.read_csv(os.path.join(frate_dir, file_dir, f'frateA_{file_name}.csv')) 
example_rates = np.array((frate_E_sync, frate_I_sync, frate_A_sync))
example_rates = example_rates[:,start:stop,:]

# load state data
states_data = pd.read_csv(os.path.join(frate_dir, file_dir, file_name+'_states.csv'), index_col=0)

# choose area to plot 
region = 2
region_UP = 3

# load atlas with releveant regions for plotting
atlas = pd.read_csv(os.path.join(atlas_dir, 'relevant_areas.csv'))
atlas.drop(['Unnamed: 0'], inplace=True, axis=1)
atlas = np.array(atlas)

# %% plot one area

colors, dpi = figure_style()
f, axs = plt.subplots(1, 2, figsize=(3, 1.75), dpi=dpi, sharey=True)

# plot rates
axs[0].plot(example_rates[1,:,region], label='I', color='#ff7f0e')
axs[0].plot(example_rates[0,:,region], label='E', color='#1f77b4')

# plot states
data = states_data[states_data['region']==region].reset_index()
data = data.iloc[start:stop]
data['time'] = data['index']/1000
lim = 1.05 * abs(example_rates[1,:,region]).max()

axs[0].imshow(np.expand_dims(data['state'], axis=0),
           aspect="auto",
           cmap=cmap,
           alpha=0.2,
           # vmin=0,
           # vmax=len(colors)-1,
           extent=(0, len(example_rates[1,:,region]), -2, 60))

axs[0].set_xticks(ticks=[0, 2000, 4000, 6000, 8000, 10000])
axs[0].set_xticklabels(labels=[0, 2, 4, 6, 8, 10])
axs[0].set_yticks(ticks=[0, 20, 40, 60])
axs[0].set_xlabel('time (s)')
axs[0].set_ylabel('firing rate (Hz)')
sns.despine(trim=True)
axs[0].set_title(atlas[region][0])


# plot rates UP state region
axs[1].plot(example_rates[1,:,region_UP], label='I', color='#ff7f0e')
axs[1].plot(example_rates[0,:,region_UP], label='E', color='#1f77b4')

data_UP = states_data[states_data['region']==region_UP].reset_index()
data_UP = data_UP.iloc[start:stop]
data_UP['time'] = data_UP['index']/1000

axs[1].imshow(np.expand_dims(data_UP['state'], axis=0),
           aspect="auto",
           cmap=cmap,
           alpha=0.2,
           # vmin=0,
           # vmax=len(colors)-1,
           extent=(0, len(example_rates[1,:,region]), -2, 60))

axs[1].set_xticks(ticks=[0, 2000, 4000, 6000, 8000, 10000])
axs[1].set_xticklabels(labels=[0, 2, 4, 6, 8, 10])
axs[1].set_yticks(ticks=[0, 20, 40, 60])
axs[1].set_xlabel('time (s)')
sns.despine(trim=True)
axs[1].set_title(atlas[region_UP][0])

plt.savefig(os.path.join(figure_dir, f'UPDOWN_{atlas[region][0]}-{atlas[region_UP][0]}S0_G2.pdf'), dpi=600, bbox_inches="tight")


# %% Load heterogeneity data
updown_regions = ['Amyg', 'mPFC', 'OLF', 'OFC', 'Pir', 'RSC', 'M2', 'Str', 'Thal', 'Vis']
up_regions = ['Hipp', 'MRN', 'PAG', 'SC']
all_regions = ['Amyg', 'Hipp', 'mPFC', 'MRN', 'OLF', 'OFC', 'PAG', 'Pir', 'RSC', 'M2', 'SC', 'Str', 'Thal', 'Vis']
state_data = pd.read_csv(os.path.join(analysed_data_dir, 'totalduration_state_analysis_S0_G2RateAdj1.csv'))
up_areas = ['Midbrain reticular nucleus', 'Hippocampus', 'Superior colliculus', 'Periaqueductal gray']
updown_df = state_data[~state_data['region'].isin(up_areas)]

colors, dpi = figure_style()
f, axs = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)
plot_order = state_data.sort_values(by='state_frequency', ascending=False).region.values
sns.pointplot(state_data, x='region', y='state_frequency', ax=axs, order=plot_order)
axs.set_xticklabels(all_regions, rotation=40)
sns.despine(trim=True)

# %% Load synchrony data

brunelX = pd.read_csv(os.path.join(analysed_data_dir, 'brunelX_S0_G[0, 1, 2, 3]RateAdj1.csv'))
colors, dpi = figure_style()
f, axs = plt.subplots(1, 1, figsize=(2, 1.75), dpi=dpi)

sns.lineplot(brunelX, x='G', y='brunel_X')
axs.set_xticks([0,1,2,3])
axs.set_xticklabels(labels=[0,1,2,3])
axs.set_xlabel('Coupling strength (G)')
axs.set_ylabel('Synchrony (X)')
sns.despine(trim=True)
plt.savefig(os.path.join(figure_dir, f'network_synchrony_brunelX.pdf'), dpi=600)
# %% Plot 


