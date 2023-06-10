# %%
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from stim_functions import figure_style

# set directories
analysed_data_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, 'analysed_data'))
figure_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), 'paper_panel'))

# load data
S = 40 
G = [2]
data_df = pd.read_csv(os.path.join(analysed_data_dir, f'state_analysis_bins_S{S}_G[0, 1, 2, 3]_RateAdj1.csv'))
baseline_df = pd.read_csv(os.path.join(analysed_data_dir, f'state_analysis_bins_S0_G{G}_RateAdj1.csv'))

# summarize regions according to higher level regions
frontal = ['Medial prefrontal cortex', 'Orbitofrontal cortex', 'Secondary motor cortex'] # = 'Frontal'
sensory = ['Piriform', 'Visual cortex'] # = 'Sensory'
midbrain = ['Periaqueductal gray', 'Midbrain reticular nucleus', 'Superior colliculus'] # = 'Midbrain'
relevant_area_IDs = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13]
data_df['combined_region'] = data_df['region_name']
data_df['combined_region'].iloc[np.where(data_df['region_name'].isin(frontal))] = 'Frontal cortex' 
data_df['combined_region'].iloc[np.where(data_df['region_name'].isin(sensory))] = 'Sensory cortex' 
data_df['combined_region'].iloc[np.where(data_df['region_name'].isin(midbrain))] = 'Midbrain' 
data_df = data_df.groupby(['combined_region', 'S', 'session', 'G', 'time'], as_index=False).mean()
baseline_df['combined_region'] = baseline_df['region_name']
baseline_df['combined_region'].iloc[np.where(baseline_df['region_name'].isin(frontal))] = 'Frontal cortex' 
baseline_df['combined_region'].iloc[np.where(baseline_df['region_name'].isin(sensory))] = 'Sensory cortex' 
baseline_df['combined_region'].iloc[np.where(baseline_df['region_name'].isin(midbrain))] = 'Midbrain' 
baseline_df = baseline_df.groupby(['combined_region', 'S', 'session', 'G', 'time'], as_index=False).mean()
serotonin_df = data_df[data_df['G'] == 2]

# compute the delta downstate for all regions 
baseline_df['p_down_mean'] = baseline_df.groupby(['combined_region', 'G'])['p_down'].transform('mean')
baseline_df['p_down_delta'] = baseline_df['p_down'] - baseline_df['p_down_mean']
serotonin_df['p_down_delta'] = (serotonin_df['p_down'].values - baseline_df['p_down_mean'].values - baseline_df['p_down_delta'].values)

# plot in this order
regions = ['Frontal cortex', 'Amygdala',  'Tail of the striatum', 'Sensory cortex', 
           'Hippocampus', 'Thalamus','Midbrain']
# %%

# Plot: Downstate probability
colors, dpi = figure_style()
f, axs = plt.subplots(1, 7, figsize=(7, 1.75), dpi=dpi, sharey=True)

for i, region in enumerate(regions):
    
    axs[i].axvspan(0, 1, alpha=0.25, color='royalblue', lw=0)
    
    axs[i].plot([-1, 3], [0.5, 0.5], ls='--', color='grey')
    sns.lineplot(data=serotonin_df[serotonin_df['combined_region'] == region], x='time', y='p_down',
                 color=colors['suppressed'], errorbar='se', err_kws={'lw': 0}, ax=axs[i])
    sns.lineplot(data=baseline_df[baseline_df['combined_region'] == region], x='time', y='p_down',
                 color=colors['grey'], errorbar='se', ax=axs[i], label='baseline')
    axs[i].set(xlabel='Time (s)', title=region, ylim=[-0.02, 1],
               yticks=[0, 0.5, 1], yticklabels=[0, 50, 100])
    axs[i].set_title(region, fontsize=10)
    
    if i == 0:
        axs[i].set_ylabel('Down state probability (%)')
        axs[i].get_xaxis().set_visible(False)
        sns.despine(trim=True, bottom=True, ax=axs[i])
        axs[i].plot([0, 2], [-0.01, -0.01], color='k', lw=0.5)
        axs[i].text(1, -0.03, '2s', ha='center', va='top')
    else:
        axs[i].get_yaxis().set_visible(False)
        axs[i].axis('off')
    
    if i < 6:
        axs[i].get_legend().remove()
        
plt.subplots_adjust(left=0.08, bottom=0.15, right=1, top=0.85, wspace=0, hspace=0.4)
plt.tight_layout(h_pad=-10, w_pad=1.08)
plt.savefig(os.path.join(figure_dir, f'p_down_anesthesia_S{S}_G{G}.pdf'), dpi=600)


# %% Plot: delta downstate probability 

colors, dpi = figure_style()
f, axs = plt.subplots(1, 7, figsize=(7, 1.75), sharey=True)

for i, region in enumerate(regions):
    
    axs[i].axvspan(0, 1, alpha=0.25, color='royalblue', lw=0)
    axs[i].plot([-1, 3], [0, 0], ls='--', color='grey')
    
    stimulation = serotonin_df[serotonin_df['combined_region'] == region]
    no_stimulation = baseline_df[baseline_df['combined_region'] == region]
    
    sns.lineplot(data=stimulation, x='time', y='p_down_delta',
                 color=colors['stim'], errorbar='se', err_kws={'lw': 0}, ax=axs[i], label='Stimulation')
    sns.lineplot(data=no_stimulation, x='time', y='p_down_delta',
                 color=colors['no-stim'], errorbar='se', ax=axs[i], label='No stimulation')
    axs[i].set(xlabel='Time (s)', title=region, ylim=[-0.265, 0.5],
               yticks=[-0.25, 0, 0.5, 1], yticklabels=[-25, 0, 50, 100])
    axs[i].set_title(region, fontsize=10)
    
    if i == 0:
        axs[i].set_ylabel(u'Δ down state probability (%)', labelpad=0)
        axs[i].get_xaxis().set_visible(False)
        sns.despine(trim=True, bottom=True, ax=axs[i])
        axs[i].plot([0, 2], [-0.25, -0.25], color='k', lw=0.5)
        axs[i].text(1, -0.27, '2s', ha='center', va='top')
    else:
        axs[i].get_yaxis().set_visible(False)
        axs[i].axis('off')
    
    if i < 6:
        axs[i].get_legend().remove()

plt.subplots_adjust(left=0.08, bottom=0.15, right=1, top=0.85, wspace=0, hspace=0.4)
plt.tight_layout(h_pad=-10, w_pad=1.08)
plt.savefig(os.path.join(figure_dir, f'p_delta_downstate_S{S}_G{G}.jpg'), dpi=600)

