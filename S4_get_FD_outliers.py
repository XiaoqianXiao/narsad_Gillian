import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from nilearn import image as nimg
from nilearn import plotting as nplot
import bids
from nltools.file_reader import onsets_to_dm
from nltools.stats import regress, zscore
from nltools.data import Brain_Data, Design_Matrix
from nltools.stats import find_spikes 
from nilearn.plotting import view_img, glass_brain, plot_stat_map
from bids import BIDSLayout, BIDSValidator
import os
from pathlib import Path
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from nilearn import plotting
from nilearn.plotting import plot_contrast_matrix
from nilearn.glm.first_level import FirstLevelModel
from nilearn import plotting
# creating mean img for plotting purposes 
from nilearn.image import mean_img
from nilearn.image import load_img
from nibabel import load
from nibabel.gifti import GiftiDataArray, GiftiImage
from nilearn.glm.first_level import run_glm as run_glm
from nilearn.glm import compute_contrast
import nilearn
fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage5')

# Initialize BIDS layout
layout = bids.BIDSLayout('/gscratch/fang/NARSAD/MRI/derivatives/fmriprep', validate=False,
                         config=['bids', 'derivatives'])

# Output file
output_csv = "/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/ROI/FD.csv"
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# Prepare DataFrame to store results
fd_summary = []

# List of tasks (phases)
tasks = ['phase2', 'phase3']
fd_threshold = 0.5

for task in tasks:
    subjects = layout.get_subjects()
    for sub in subjects:
        # Get confounds file
        confound_files = layout.get(
            subject=sub,
            datatype='func',
            task=task,
            desc='confounds',
            extension='tsv',
            return_type='file'
        )
        if not confound_files:
            continue  # Skip if no confound file found
        
        # Read confound file
        confound_file = confound_files[0]
        df_conf = pd.read_csv(confound_file, sep='\t')
        if 'framewise_displacement' not in df_conf.columns:
            print(f"No FD column found for sub-{sub}, task-{task}")
            continue

        # Compute FD > 0.5 stats
        n_scans = len(df_conf)
        n_high_fd = np.sum(df_conf['framewise_displacement'] > fd_threshold)
        perc_high_fd = (n_high_fd / n_scans) * 100 if n_scans > 0 else np.nan
        mean_fd = df_conf['framewise_displacement'].mean()
        max_fd = df_conf['framewise_displacement'].max()

        # Append to results
        fd_summary.append({
            "subject": sub,
            "task": task,
            "n_scans": n_scans,
            "FD>0.5_count": n_high_fd,
            "FD>0.5_%": round(perc_high_fd, 2),
            "mean_FD": round(mean_fd, 4),
            "max_FD": round(max_fd, 4)
        })

# Convert to DataFrame
fd_df = pd.DataFrame(fd_summary)

# Save to CSV
fd_df.to_csv(output_csv, index=False)

print(f"FD summary saved to {output_csv}")
print(fd_df.head())
