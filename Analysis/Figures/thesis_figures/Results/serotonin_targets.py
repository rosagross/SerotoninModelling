# %%
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from stim_functions import figure_style

'''
Plot:
1) I with + current, show traces of three grouped regions
2) plot with all options averaged over all regions 
3) paradoxical effect: example region targeted I with - current (inhibiting I population leads to UP state)
'''

# set directories
analysed_data_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, 'analysed_data'))
figure_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, os.pardir, os.pardir, os.pardir, 'Writing', 'Figures', 'Results', 'Figure4'))
frate_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, os.pardir, os.pardir, 'ModelData', 'data', 'firing_rates'))
atlas_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, 'atlas_data'))

# load firing rate 
file_dir = "14areas_G2.0_S40.0_thetaE-1_beta6RateAdj1_J_sessions"
file_name = "14areas_G2.0_S40.0_thetaE-1_beta6RateAdj1_J_sessions_0"
start = 2*1000
stop = 8*1000
frate_E_sync = pd.read_csv(os.path.join(frate_dir, file_dir, f'frateE_{file_name}.csv'))
frate_I_sync = pd.read_csv(os.path.join(frate_dir, file_dir, f'frateI_{file_name}.csv'))
frate_A_sync = pd.read_csv(os.path.join(frate_dir, file_dir, f'frateA_{file_name}.csv')) 
example_rates = np.array((frate_E_sync, frate_I_sync, frate_A_sync))
example_rates = example_rates[:,start:stop,:]
stim_times = pd.read_csv(os.path.join(frate_dir, file_dir, file_name + '_stimulation_times.csv')).to_numpy() - start


# load state analysis data 
S = 40
S_baseline = 0
G = [2]
targets_df = pd.read_csv(os.path.join(analysed_data_dir, f'serotonin_targets_state_analysis_S{S}_G{G}.csv'))
targets_baseline_df = pd.read_csv(os.path.join(analysed_data_dir, f'serotonin_targets_state_analysis_S{S_baseline}_G{G}.csv'))

# load atlas with releveant regions for plotting
atlas = pd.read_csv(os.path.join(atlas_dir, 'relevant_areas.csv'))
atlas.drop(['Unnamed: 0'], inplace=True, axis=1)
atlas = np.array(atlas)

# compute the delta downstate 
targets_baseline_df['p_down_mean'] = targets_baseline_df.groupby(['region_name', 'G', 'target_name'])['p_down'].transform('mean')
targets_baseline_df['p_down_delta'] = targets_baseline_df['p_down'] - targets_baseline_df['p_down_mean']
targets_df['p_down_delta'] = (targets_df['p_down'].values - targets_baseline_df['p_down_mean'].values - targets_baseline_df['p_down_delta'].values)

# summarize regions according to higher level regions
frontal = ['Medial prefrontal cortex', 'Orbitofrontal cortex', 'Secondary motor cortex'] # = 'Frontal'
sensory = ['Piriform', 'Visual cortex'] # = 'Sensory'
midbrain = ['Periaqueductal gray', 'Midbrain reticular nucleus', 'Superior colliculus'] # = 'Midbrain'
relevant_area_IDs = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13]
targets_df['combined_region'] = targets_df['region_name']
targets_df['combined_region'].iloc[np.where(targets_df['region_name'].isin(frontal))] = 'Frontal cortex' 
targets_df['combined_region'].iloc[np.where(targets_df['region_name'].isin(sensory))] = 'Sensory cortex' 
targets_df['combined_region'].iloc[np.where(targets_df['region_name'].isin(midbrain))] = 'Midbrain' 
targets_df_grouped = targets_df.groupby(['combined_region', 'S', 'session', 'G', 'time', 'target_name'], as_index=False).mean()
targets_baseline_df['combined_region'] = targets_baseline_df['region_name']
targets_baseline_df['combined_region'].iloc[np.where(targets_baseline_df['region_name'].isin(frontal))] = 'Frontal cortex' 
targets_baseline_df['combined_region'].iloc[np.where(targets_baseline_df['region_name'].isin(sensory))] = 'Sensory cortex' 
targets_baseline_df['combined_region'].iloc[np.where(targets_baseline_df['region_name'].isin(midbrain))] = 'Midbrain' 
targets_baseline_df_grouped = targets_baseline_df.groupby(['combined_region', 'S', 'session', 'G', 'time', 'target_name'], as_index=False).mean()

# %% Plot 1)

colors, dpi = figure_style()
f, axs = plt.subplots(1, 7, figsize=(7.5, 1.75), sharey=True)

regions_plot = ['Frontal cortex', 'Amygdala', 'Tail of the striatum', 'Sensory cortex', 'Hippocampus', 'Thalamus', 'Midbrain']
target = 'I+'
serotonin_window = [0.5,1]

for i, region in enumerate(regions_plot):
    # first axis: trajectory of one area
    stimulation = targets_df_grouped[targets_df_grouped['combined_region'] == region]
    stimulation = stimulation[stimulation['target_name']==target]
    no_stimulation = targets_baseline_df_grouped[targets_baseline_df_grouped['combined_region'] == region]

    axs[i].axvspan(0, 1, alpha=0.25, color='royalblue', lw=0)
    axs[i].plot([-1, 3], [0, 0], ls='--', color='grey')

    sns.lineplot(data=stimulation, x='time', y='p_down_delta',
                    color=colors['stim'], errorbar='se', err_kws={'lw': 0}, ax=axs[i], label='Stimulation')
    sns.lineplot(data=no_stimulation, x='time', y='p_down_delta', err_kws={'lw': 0},
                    color=colors['no-stim'], errorbar='se', ax=axs[i], label='No stimulation')
    axs[i].set(xlabel='Time (s)', title=region, ylim=[-0.265, 0.3],
                yticks=[-0.25, 0, 0.3], yticklabels=[-25, 0, 30])
    axs[i].set_title(region)
    axs[i].set_ylabel(u'Δ down state probability (%)', labelpad=0)
    axs[i].get_xaxis().set_visible(False)
    sns.despine(trim=True, bottom=True, ax=axs[i])
    axs[i].legend('')

axs[0].plot([0, 2], [-0.27, -0.27], color='k', lw=0.5, clip_on=False)
axs[0].text(1, -0.29, '2s', ha='center', va='top')
plt.subplots_adjust(wspace=0.1)
axs[6].legend(title=None, loc='upper right', bbox_to_anchor=(1.3, 1), frameon=True, framealpha=0.6)    

plt.savefig(os.path.join(figure_dir, f'I+_trajectories_G{G}_S{S}.pdf'), bbox_inches="tight")


# %% Plot 2)

# compute delta downstate probability per region
targets_baseline_df['p_down_mean'] = targets_baseline_df.groupby(['region_name', 'G', 'session', 'target_name'])['p_down'].transform('mean')
targets_baseline_df['p_down_delta'] = targets_baseline_df['p_down'] - targets_baseline_df['p_down_mean']
targets_df['p_down_delta'] = (targets_df['p_down'].values - targets_baseline_df['p_down_mean'].values - targets_baseline_df['p_down_delta'].values)

# plot 
colors, dpi = figure_style()
fig, ax = plt.subplots(1, 1, figsize=(1.7, 1.75), dpi=dpi)

plt.axvspan(0, 1, alpha=0.25, ymin=-0.25, color='royalblue', lw=0)
ax.plot([-1, 3], [0, 0], ls='--', color='grey')
ax.plot([0, 2], [-0.25, -0.25], color='k', lw=0.5, clip_on=False)
ax.text(1, -0.27, '2s', ha='center', va='top')
sns.lineplot(targets_df, x='time', y='p_down_delta', hue='target_name', 
             err_kws={'lw': 0}, palette=sns.color_palette('Dark2'), ax=ax)

ax.get_xaxis().set_visible(False)
ax.set_ylim([-0.25, 0.50])
ax.set_yticks([-0.25, 0, 0.25, 0.5])
ax.set_yticklabels([-25, 0, 25, 50])
ax.set_ylabel(u'Δ down state probability (%)')
ax.legend(title=None, loc='upper right', bbox_to_anchor=(0.95, 1.05))
sns.despine(trim=True, bottom=True)
plt.savefig(os.path.join(figure_dir, f'trajectory_serotonin_targets_G{G}_S{S}.pdf'), bbox_inches="tight")
plt.show()

# %% Plot 3)

colors, dpi = figure_style()
f, axs = plt.subplots(1, 1, figsize=(2, 1.75), dpi=dpi, sharey=True)

region = 2

# plot rates
axs.axvspan(stim_times[0][0], stim_times[0][0]+1000, alpha=0.25, ymin=-0.25, color='royalblue', lw=0)
axs.plot(example_rates[1,:,region], label='I', color='#ff7f0e')
axs.plot(example_rates[0,:,region], label='E', color='#1f77b4')

# plot states
axs.set_xticks(ticks=[0, 2000, 4000, 6000])
axs.set_xticklabels(labels=np.array([0, 2, 4, 6]))
axs.set_yticks(ticks=[0, 25, 50])
axs.set_xlabel('time (s)')
axs.set_ylabel('firing rate (Hz)')
sns.despine(trim=True)
axs.set_title(atlas[region][0])

# save
plt.savefig(os.path.join(figure_dir, f'example_I-target_G{G}_S{S}.pdf'), bbox_inches="tight")
plt.show()

