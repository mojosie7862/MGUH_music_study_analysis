import os
import mne
import util
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import pandas as pd
from util import get_markers, subset_intervals, print_dict, add_int_prefix
from heartpy.preprocessing import flip_signal
from scipy.stats.stats import pearsonr
from statsmodels.sandbox.stats.multicomp import multipletests

icm_aim_data_directory = 'C:/Users/kanwali/Box/MICU_AIM_data/aim_csvs'


# ICM-5KJ3_exp2_b1_AIM.csv
#organize files in AIM data folder (id, # of experiments, complete 6 trials?)
#adding experiment and interval columns to combine all .csv files

pt_exp_dict = {}
pt_exp_int_dfs = []
hrv_dfs = []
inspected = []
inverted_ids = [('ICM-9XYD', '3'), ('ICM-9XYD', '6'), ('ICM-HU9L', '3'), ('ICM-HU9L', '5'), ('ICM-IPL6', '5'), ('ICM-IPL6', '6'),
                ('ICM-K0XH', '1'), ('ICM-LGDL', '2'), ('ICM-LGDL', '3'), ('ICM-SL7D', '1'), ('ICM-V4VM', '1'), ('ICM-V4VM', '4'),
                ('ICM-V4VM', '6'), ('ICM-WZ9G', '1'), ('ICM-WZ9G', '4'), ('ICM-WZ9G', '5'), ('ICM-WZ9G', '6')]
exclude = [('ICM-9XYD', '1'), ('ICM-HU9L', '1'), ('ICM-LGDL', '4'), ('ICM-LGDL', '5'), ('ICM-V4VM', '3'),
           ('ICM-WZ9G', '2'), ('ICM-WZ9G', '3'), ('ICM-WZ9G', '4')]
for filename in os.listdir(icm_aim_data_directory):
    pt_id = filename[:8]
    exp = filename[12]
    interval = filename[14:16]
    pt_exp_int_df = pd.read_csv(icm_aim_data_directory+'/'+filename, usecols=range(1,10), index_col='index')
    pt_exp_int_df['participant_id'] = pt_id
    pt_exp_int_df['experiment'] = int(exp)
    pt_exp_int_df['interval'] = interval
    cols = ['participant_id', 'experiment', 'interval', 'time', 'ECG',
            'Respiration', 'PPG', 'SpO2', 'Heart rate', 'GSR', 'Temperature']
    pt_exp_int_df = pt_exp_int_df[cols]
    pt_exp_int_dfs.append(pt_exp_int_df)


    exp_int_id = (pt_id, exp, interval)
    if exp_int_id[0] not in inspected:
        if exp_int_id[0:2] not in exclude:
            if exp_int_id[0:2] in inverted_ids:
                pt_exp_int_df['ECG'] = flip_signal(pt_exp_int_df['ECG'])
            print(exp_int_id)
            # clean waveforms and evaluate quality
            ecg_cleaned = nk.ecg_clean(pt_exp_int_df['ECG'], sampling_rate=500, method="neurokit")
            ecg_quality = nk.ecg_quality(ecg_cleaned, sampling_rate=500, method="zhao2018", approach="fuzzy")
            ecg_signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=500)
            peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=500)

            # ecg_plot = pd.DataFrame({'ECG_clean': ecg_signals['ECG_Clean']})
            # {'ecg': pt_exp_int_df['ECG']}
            # ecg_plot.plot()
            # for i, event in enumerate(ecg_signals['ECG_R_Peaks']):
            #     if event == 1:
            #         plt.axvline(i, c='r', linestyle='--')
            # plt.show()

            # transform NNs to hrv indices
            hrv_indices = nk.hrv(ecg_signals, sampling_rate=500)
            hrv_indices.insert(loc=0, column='patient_id', value=pt_id)
            hrv_indices.insert(loc=1, column='trial_id', value=exp)
            hrv_indices.insert(loc=2, column='interval', value=interval)
            hrv_indices.insert(loc=3, column='ecg_quality', value=ecg_quality)
            hrv_dfs.append(hrv_indices)
icm_hrv_indicies = pd.concat(hrv_dfs)
# icm_hrv_indicies.to_csv('icm_cohort1_hrv_indicies.csv')



#
#     b1_df_list.append(b1_hrv_indices)
#     music_df_list.append(music_hrv_indices)
#     b2_df_list.append(b2_hrv_indices)
#
# b1_hrv_data = pd.concat(b1_df_list)
# music_hrv_data = pd.concat(music_df_list)
# b2_hrv_data = pd.concat(b2_df_list)
#
# diff_dict = {'participant_id':[]}
# for b1_iter, mu_iter, b2_iter in zip(b1_hrv_data.iterrows(), music_hrv_data.iterrows(), b2_hrv_data.iterrows()):
#     for i, (b1_col_v, mu_col_v, b2_col_v) in enumerate(zip(b1_iter[1], mu_iter[1], b2_iter[1])):
#         if i == 0:
#             diff_dict['participant_id'].append(b1_col_v)
#             continue
#         b1_mu_diff = mu_col_v - b1_col_v
#         b1_b2_diff = b2_col_v - b1_col_v
#         mu_b2_diff = b2_col_v - mu_col_v
#         b1_mu_label = 'b1_mu_diff_' + diff_cols[i-1]
#         b1_b2_label = 'b1_b2_diff_' + diff_cols[i-1]
#         mu_b2_label = 'mu_b2_diff_' + diff_cols[i-1]
#         if b1_mu_label not in diff_dict.keys():
#             diff_dict[b1_mu_label] = [b1_mu_diff]
#         else:
#             diff_dict[b1_mu_label].append(b1_mu_diff)
#         if b1_b2_label not in diff_dict.keys():
#             diff_dict[b1_b2_label] = [b1_b2_diff]
#         else:
#             diff_dict[b1_b2_label].append(b1_b2_diff)
#         if mu_b2_label not in diff_dict.keys():
#             diff_dict[mu_b2_label] = [mu_b2_diff]
#         else:
#             diff_dict[mu_b2_label].append(mu_b2_diff)
#
# diff_df = pd.DataFrame(diff_dict)
# hrv_df = b1_hrv_data.merge(music_hrv_data, on='participant_id', how='left')
# hrv_df = hrv_df.merge(b2_hrv_data, on='participant_id', how='left')
# hrv_diff_df = pd.concat([hrv_df, diff_df], axis=1)






