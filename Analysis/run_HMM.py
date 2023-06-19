# %%
import pandas as pd
import numpy as np
import os 
from analysis_functions import get_all_states
import ssm 

output_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()),  os.pardir, 'ModelData', 'data/firing_rates'))

G_params = [2]
S_params = [0, 40]
beta = 6
thetaE = -1
session_params = pd.DataFrame()
brunel_X_df = pd.DataFrame()
extra = "RateAdj1_C_sessions"
filename_out = ''
sessions = np.arange(0,10,1)
all_analyses = pd.DataFrame()

# %%
for G in G_params:
    for S in S_params:           
            
        for session in sessions:
            
            print(f"Run: S={S}, G={G}, session={session}")
            # load the file name
            session_name = f'14areas_G{float(G)}_S{float(S)}_thetaE{thetaE}_beta{beta}{extra}'
            file_dir = os.path.join(output_dir, session_name)
            file_name = f'14areas_G{float(G)}_S{float(S)}_thetaE{thetaE}_beta{beta}{extra}_{session}'    
    
            # check if any files in the directory end with 'states.csv'
            if any(f.endswith(f'{session}_states.csv') for f in os.listdir(file_dir)):
                pass
            else:
                print('The folder does not contain a file with the ending "states.csv"')
                _ = get_all_states(file_dir, file_name)
# %%
