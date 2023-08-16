import os
import mne
from util import get_markers, subset_intervals, print_dict
import sys
import shutil

aim_data_directory = r'C:/Users/kanwali/Box/MICU_AIM_data/Copy_ICM_AIM/icm_clean_vmrk_vhdr'

for fn in os.listdir(aim_data_directory):
    f_suffix = fn[-5:]
    f_id = fn.replace(f_suffix, '')

    if f_suffix == '.vhdr':
        print(fn)
        vhdr_file = os.path.abspath(aim_data_directory + '/' + fn)
        raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)
        raw_df = raw.to_data_frame()

        # if f_id.startswith("NHC"):
        #     raw_df.to_csv('C:/Users/kanwali/Box/MICU_AIM_data/nhc_aim_csvs/all_sample_csvs/%s_all_samples.csv' % (f_id))

        if f_id.startswith("ICM"):
            raw_df.to_csv('C:/Users/kanwali/Box/MICU_AIM_data/icm_aim_csvs/all_sample_csvs/%s_all_samples.csv' % (f_id))

