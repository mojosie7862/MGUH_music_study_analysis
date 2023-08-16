import os
import mne
from util import get_markers, subset_intervals, print_dict
import sys
import shutil

# This script processes raw brainvision format files into .csvs
# .vmrk and .vhdr files must be cleaned first (organize_aim_data.py)
# Takes 1 additional argument, differentiating processing for patient vs. healthy participants

# for patient (icm) files - python process_aim_tocsv.py icm
# for healthy (nhc) files - python process_aim_tocsv.py nhc

marker_dict = {}
group_v = sys.argv[1]
if group_v == 'nhc':
    aim_data_directory = r'C:/Users/kanwali/Box/MICU_AIM_data/Copy_NHC_AIM/'

if group_v == 'icm':
    aim_data_directory = r'C:/Users/kanwali/Box/MICU_AIM_data/Copy_ICM_AIM'

#extract marker data
for fn in os.listdir(aim_data_directory):
    if fn.startswith('NHC') or fn.startswith('ICM'):
        filename = aim_data_directory+'/'+fn
        f_suffix = fn[-5:]
        if fn.startswith("NHC"):
            f_id = fn.replace(f_suffix, '')[-8:]
        if fn.startswith("ICM"):
            f_id = fn.replace(f_suffix, '')[-13:]
        print('--------', fn, '--------')
        print('ID:', f_id)

        if f_suffix == '.vmrk':
            b1_start, music_start, music_stop, b2_start = get_markers(filename)
            marker_dict[f_id] = [b1_start, music_start, music_stop, b2_start]

#create 3 subsets (before, during, and after music) of trial data from nme objects and brain vision files
for fn in os.listdir(aim_data_directory):
    f_suffix = fn[-5:]
    f_id = fn.replace(f_suffix, '')

    if f_suffix == '.vhdr':
        print(fn)
        vhdr_file = os.path.abspath(aim_data_directory + '/' + fn)
        raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)
        raw_df = raw.to_data_frame()
        b1_raw, music_raw, b2_raw = subset_intervals(raw_df, marker_dict[f_id][0], marker_dict[f_id][1],
                                                     marker_dict[f_id][2], marker_dict[f_id][3])

        print("b1 # samples", len(b1_raw))
        print("music # samples", len(music_raw))
        print("b2 # samples", len(b2_raw))

        if f_id.startswith("NHC"):
            b1_raw.to_csv('C:/Users/kanwali/Box/MICU_AIM_data/nhc_aim_csvs/interval_csvs/%s_b1_AIM.csv' % (f_id))
            music_raw.to_csv('C:/Users/kanwali/Box/MICU_AIM_data/nhc_aim_csvs/interval_csvs/%s_mu_AIM.csv' % (f_id))
            b2_raw.to_csv('C:/Users/kanwali/Box/MICU_AIM_data/nhc_aim_csvs/interval_csvs/%s_b2_AIM.csv' % (f_id))
            clean_directory = r'C:/Users/kanwali/Box/MICU_AIM_data/Copy_NHC_AIM/nhc_clean_vmrk_vhdr'
            
        if f_id.startswith("ICM"):
            b1_raw.to_csv('C:/Users/kanwali/Box/MICU_AIM_data/icm_aim_csvs/interval_csvs/%s_b1_AIM.csv' % (f_id))
            music_raw.to_csv('C:/Users/kanwali/Box/MICU_AIM_data/icm_aim_csvs/interval_csvs/%s_mu_AIM.csv' % (f_id))
            b2_raw.to_csv('C:/Users/kanwali/Box/MICU_AIM_data/icm_aim_csvs/interval_csvs/%s_b2_AIM.csv' % (f_id))
            clean_directory = r'C:/Users/kanwali/Box/MICU_AIM_data/Copy_ICM_AIM/icm_clean_vmrk_vhdr'

        # move respective ICM and NHC brainvision files to finished processing folder
        for file in os.listdir(aim_data_directory):
            if file.startswith(f_id):
                original_path = aim_data_directory + '/' + file
                target_path = clean_directory + '/' + file
                shutil.move(original_path, target_path)

print('Done.')




