import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from matplotlib.backends.backend_pdf import PdfPages
import xarray as xr
from scipy.stats import mannwhitneyu
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from scipy.stats import bootstrap
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

# sys.path.append(r'H:/anthony/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/Pulin_et_al')
# from nwb_wrappers import nwb_reader_functions as nwb_read
import src.utils.utils_imaging as imaging_utils
import src.utils.utils_io as io
from src.utils.utils_plot import *
from src.utils.utils_behavior import *

sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1,
            rc={'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'})


# #############################################################################
# 1. PSTH's.
# #############################################################################

# Parameters.
# -----------

sampling_rate = 30
win_sec = (-0.5, 4)  
baseline_win = (0, 1)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days = [10]
days_str = ['+10']


_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',)
# mice = [m for m in mice if m not in ['AR163']]
mice_count = db[['mouse_id', 'reward_group']].drop_duplicates()
# print(mice_count)
# print(mice_count.groupby('reward_group').count().reset_index())


# Load the data.
# --------------

psth = []

for mouse_id in mice:
    # Disregard these mice as the number of trials is too low.
    # if mouse_id in ['GF307', 'GF310', 'GF333', 'AR144', 'AR135']:
    #     continue
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)

    file_name = 'tensor_xarray_learning_data.nc'
    folder = os.path.join(io.processed_dir, 'mice')
    xarr = imaging_utils.load_mouse_xarray(mouse_id, folder, file_name, substracted=False)
    xarr = imaging_utils.substract_baseline(xarr, 2, baseline_win)
    
    # # Keep days of interest.
    # xarr = xarr.sel(trial=xrr['day'].isin(days))
    # Select trials with whisker_stim != 1.
    xarr = xarr.sel(trial=xarr['whisker_stim'] == 1)

    # Select PSTH trace length.
    xarr = xarr.sel(time=slice(win_sec[0], win_sec[1]))
    # Average trials per days.
    xarr = xarr.groupby('day').mean(dim='trial')
    
    xarr.name = 'psth'
    xarr = xarr.to_dataframe().reset_index()
    xarr['mouse_id'] = mouse_id
    xarr['reward_group'] = reward_group
    psth.append(xarr)
psth = pd.concat(psth)




# Grand average psth's for all cells and projection neurons.
# ##########################################################

# GF305 has baseline artefact on day -1 at auditory trials.
# data = data.loc[~data.mouse_id.isin(['GF305'])]

# mice_AR = [m for m in mice if m.startswith('AR')]
# mice_GF = [m for m in mice if m.startswith('GF') or m.startswith('MI')]
# data = data.loc[data.mouse_id.isin(mice_AR)]
# len(mice_GF)

sns.lineplot(data=psth, x='time', y='psth', errorbar='ci', estimator='mean')
plt.axvline(0, color="#030303", linestyle='--')

variance = 'cells'  # 'mice' or 'cells'

if variance == "mice":
    min_cells = 3
    data = filter_data_by_cell_count(psth, min_cells)
    data = data.groupby(['mouse_id', 'day', 'time', 'cell_type',])['psth'].agg('mean').reset_index()
else:
    data = psth.groupby(['mouse_id', 'day', 'time', 'cell_type', 'roi'])['psth'].agg('mean').reset_index()

# Convert data to percent dF/F0.
data['psth'] = data['psth'] * 100

# Plot for all cells.
fig, axes = plt.subplots(1, len(days), figsize=(18, 5), sharey=True)

for j, day in enumerate(days):
    d = data.loc[data['day'] == day]
    sns.lineplot(data=d, x='time', y='psth', errorbar='ci', palette=reward_palette, estimator='mean')
    axes[j].axvline(0, color="#030303", linestyle='--')
    axes[j].set_title(f'Day {day} - All Cells')
    axes[j].set_ylabel('DF/F0 (%)')
plt.ylim(-1, 12)
# Adjust spacing between subplots to prevent title overlap
plt.tight_layout()
sns.despine()
