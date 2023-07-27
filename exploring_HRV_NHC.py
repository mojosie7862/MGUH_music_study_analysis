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


marker_dict = {}
aim_data_directory = 'C:/Users/kanwali/Box/MICU_AIM_data/aim_csvs'

#extract marker data
for filename in os.listdir(aim_data_directory):
    f_suffix = filename[-5:]
    f_id = filename.replace(f_suffix, '')
    patient_id = filename[:8]
    if f_suffix == '.vmrk':
        vmrk_file = aim_data_directory+'/'+filename
        b1_start, music_start, music_stop, b2_start = get_markers(vmrk_file)
        marker_dict[f_id] = [b1_start, music_start, music_stop, b2_start]

#create 3 subsets (before, during, and after music) of trial data from nme objects and brain vision files
b1_df_list = []
music_df_list = []
b2_df_list = []

for filename in os.listdir(aim_data_directory):
    f_suffix = filename[-5:]
    f_id = filename.replace(f_suffix, '')
    abnormal_QRS = ['NHC-AECK', 'NHC-X0AH']
    exp_confounded = ['NHC-Q2TW']
    excluded = abnormal_QRS+exp_confounded
    if f_id in excluded:
        continue

    if f_suffix == '.vhdr':
        print(filename)
        vhdr_file = os.path.abspath(aim_data_directory + '/' + filename)
        raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)
        raw_df = raw.to_data_frame()
    else:
        continue
    # split intervals

    b1_raw, music_raw, b2_raw = subset_intervals(raw_df, marker_dict[f_id][0], marker_dict[f_id][1],
                                                 marker_dict[f_id][2], marker_dict[f_id][3])

    print('b1', len(b1_raw))
    print('music', len(music_raw))
    print('b2', len(b2_raw))

    # flip inverted signals
    b1_ecg_signal = b1_raw["ECG"]
    music_ecg_signal = music_raw["ECG"]
    b2_ecg_signal = b2_raw["ECG"]
    inverted_ids = ['NHC-325V', 'NHC-HAW2', 'NHC-X0AH']
    if f_id in inverted_ids:
        b1_ecg_signal = flip_signal(b1_ecg_signal)
        music_ecg_signal = flip_signal(music_ecg_signal)
        b2_ecg_signal = flip_signal(b2_ecg_signal)

    #unsplit data
    # ecg_cleaned = nk.ecg_clean(raw_df['ECG'], sampling_rate=500, method="neurokit")
    # ecg_signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=500)
    # ecg_rate = pd.DataFrame({'ECG_rate': ecg_signals['ECG_Rate']})
    #
    # ecg_rate.plot()
    # plt.show()

    # clean waveforms and evaluate quality
    b1_ecg_cleaned = nk.ecg_clean(b1_ecg_signal, sampling_rate=500, method="neurokit")
    b1_ecg_quality = nk.ecg_quality(b1_ecg_cleaned, sampling_rate=500, method="zhao2018", approach="fuzzy")

    music_ecg_cleaned = nk.ecg_clean(music_ecg_signal, sampling_rate=500, method="neurokit")
    music_ecg_quality = nk.ecg_quality(music_ecg_cleaned, sampling_rate=500, method="zhao2018", approach="fuzzy")

    b2_ecg_cleaned = nk.ecg_clean(b2_ecg_signal, sampling_rate=500, method="neurokit")
    b2_ecg_quality = nk.ecg_quality(b2_ecg_cleaned, sampling_rate=500, method="zhao2018", approach="fuzzy")

    # get R peaks
    b1_ecg_peaks, b1_info = nk.ecg_peaks(b1_ecg_cleaned, sampling_rate=500)
    music_ecg_peaks, music_info = nk.ecg_peaks(music_ecg_cleaned, sampling_rate=500)
    b2_ecg_peaks, b2_info = nk.ecg_peaks(b2_ecg_cleaned, sampling_rate=500)
    #
    # hrv = nk.hrv_nonlinear(b1_ecg_peaks, sampling_rate=500, show=True)
    # hrv_welch_b1 = nk.hrv_frequency(b1_ecg_peaks, sampling_rate=500, show=True, psd_method="welch")
    # plt.show()
    #
    # hrv_welch_b2 = nk.hrv_frequency(b2_ecg_peaks, sampling_rate=500, show=True, psd_method="welch")
    # plt.show()


    # transform NNs to hrv indices
    b1_hrv_indices = nk.hrv(b1_ecg_peaks, sampling_rate=500)
    music_hrv_indices = nk.hrv(music_ecg_peaks, sampling_rate=500)
    b2_hrv_indices = nk.hrv(b2_ecg_peaks, sampling_rate=500)

    # add prefixes to column names, identifiers, and interval of waveform
    diff_cols = list(b1_hrv_indices.columns)

    b1_cols = []
    for col in b1_hrv_indices.columns:
        new_col = add_int_prefix(col, 'b1')
        b1_cols.append(new_col)
    b1_hrv_indices.set_axis(b1_cols, axis=1, inplace=True)
    # b1_hrv_indices.insert(loc=0, column='patient_id', value=patient_id)
    # b1_hrv_indices.insert(loc=1, column='trial_id', value=f_id)
    b1_hrv_indices.insert(loc=0, column='participant_id', value=f_id)
    # b1_hrv_indices.insert(loc=1, column='interval', value='b1')
    # b1_hrv_indices.insert(loc=2, column='ecg_quality', value=b1_ecg_quality)

    music_cols = []
    for col in music_hrv_indices.columns:
        new_col = add_int_prefix(col, 'mu')
        music_cols.append(new_col)
    music_hrv_indices.set_axis(music_cols, axis=1, inplace=True)
    # music_hrv_indices.insert(loc=0, column='patient_id', value=patient_id)
    # music_hrv_indices.insert(loc=1, column='trial_id', value=f_id)
    music_hrv_indices.insert(loc=0, column='participant_id', value=f_id)
    # music_hrv_indices.insert(loc=1, column='interval', value='music')
    # music_hrv_indices.insert(loc=2, column='ecg_quality', value=music_ecg_quality)

    b2_cols = []
    for col in b2_hrv_indices.columns:
        new_col = add_int_prefix(col, 'b2')
        b2_cols.append(new_col)
    b2_hrv_indices.set_axis(b2_cols, axis=1, inplace=True)
    # b2_hrv_indices.insert(loc=0, column='patient_id', value=patient_id)
    # b2_hrv_indices.insert(loc=1, column='trial_id', value=f_id)
    b2_hrv_indices.insert(loc=0, column='participant_id', value=f_id)
    # b2_hrv_indices.insert(loc=1, column='interval', value='b2')
    # b2_hrv_indices.insert(loc=2, column='ecg_quality', value=b2_ecg_quality)

    b1_df_list.append(b1_hrv_indices)
    music_df_list.append(music_hrv_indices)
    b2_df_list.append(b2_hrv_indices)

b1_hrv_data = pd.concat(b1_df_list)
music_hrv_data = pd.concat(music_df_list)
b2_hrv_data = pd.concat(b2_df_list)

diff_dict = {'participant_id':[]}
for b1_iter, mu_iter, b2_iter in zip(b1_hrv_data.iterrows(), music_hrv_data.iterrows(), b2_hrv_data.iterrows()):
    for i, (b1_col_v, mu_col_v, b2_col_v) in enumerate(zip(b1_iter[1], mu_iter[1], b2_iter[1])):
        if i == 0:
            diff_dict['participant_id'].append(b1_col_v)
            continue
        b1_mu_diff = mu_col_v - b1_col_v
        b1_b2_diff = b2_col_v - b1_col_v
        mu_b2_diff = b2_col_v - mu_col_v
        b1_mu_label = 'b1_mu_diff_' + diff_cols[i-1]
        b1_b2_label = 'b1_b2_diff_' + diff_cols[i-1]
        mu_b2_label = 'mu_b2_diff_' + diff_cols[i-1]
        if b1_mu_label not in diff_dict.keys():
            diff_dict[b1_mu_label] = [b1_mu_diff]
        else:
            diff_dict[b1_mu_label].append(b1_mu_diff)
        if b1_b2_label not in diff_dict.keys():
            diff_dict[b1_b2_label] = [b1_b2_diff]
        else:
            diff_dict[b1_b2_label].append(b1_b2_diff)
        if mu_b2_label not in diff_dict.keys():
            diff_dict[mu_b2_label] = [mu_b2_diff]
        else:
            diff_dict[mu_b2_label].append(mu_b2_diff)
diff_df = pd.DataFrame(diff_dict)
hrv_df = b1_hrv_data.merge(music_hrv_data, on='participant_id', how='left')
hrv_df = hrv_df.merge(b2_hrv_data, on='participant_id', how='left')
hrv_diff_df = pd.concat([hrv_df, diff_df], axis=1)

qs_data = pd.read_csv("NHC_q_data_cohort1.csv")

qs_data['delt_cortisol'] = qs_data['cortisol_2_mean'] - qs_data['cortisol_1_mean']
qs_data['delt_STAI_state'] = qs_data['STAI_2 State'] - qs_data['STAI_1 State']
qs_data['delt_STAI_trait'] = qs_data['STAI_2 Trait'] - qs_data['STAI_1 State']

cols = ['participant_id', 'STAI_1 State', 'STAI_1 Trait', 'delt_STAI_state', 'delt_STAI_trait', 'Emotional sensitivity to music', 'Personal commitment to music',
 'Music memory and imagery', 'Listening sophistication', 'Indifference to music', 'Musical transcendance', 'Emotion regulation',
 'Social', 'Music identity and expression', 'Cognitive regulation', 'head_circ', 'head_ni', 'delt_cortisol', 'cortisol_1_mean', 'cortisol_2_mean']
qs_data = qs_data[cols]

# diff_q_df = diff_df.merge(qs_data, on='participant_id', how='left')
diff_q_df = pd.concat([qs_data, diff_df], axis=1)
diff_q_df = diff_q_df.dropna(axis=1)
diff_q_df = diff_q_df.drop(['participant_id', 'participant_id'], axis=1)
# diff_q_df.to_csv('NHC_AIM_Data5-17.csv')
#
# diff_q_df = pd.read_csv('NHC_AIM_Data5-17.csv')
# multi_test = []
# track_corrs = []
# for x in diff_q_df.columns:
#     for y in cols[1:]:
#         if y != x:
#             col1 = list(diff_q_df[x])
#             col2 = list(diff_q_df[y])
#             corr_res = pearsonr(col1, col2)
#             multi_test.append(corr_res[1])
#             corr = [x,y,corr_res[0], corr_res[1]]
#             track_corrs.append(corr)

b1_b2_diff_HRV_LF = diff_q_df['b1_b2_diff_HRV_LF']
b1_b2_diff_HRV_LFHF = diff_q_df['b1_b2_diff_HRV_LFHF']
per_comm_mus = diff_q_df['Personal commitment to music']
listen_soph = diff_q_df['Listening sophistication']
mu_b2_diff_HRV_CVI = diff_q_df['mu_b2_diff_HRV_CVI']
music_tran = diff_q_df['Musical transcendance']
emotion_reg = diff_q_df['Emotion regulation']
cog_reg = diff_q_df['Cognitive regulation']
delt_cort = diff_q_df['delt_cortisol']
b1_b2_diff_HRV_HF = diff_q_df['b1_b2_diff_HRV_HF']
indiff_music = diff_q_df['Indifference to music']
mu_b2_diff_HRV_TINN = diff_q_df['mu_b2_diff_HRV_TINN']
#
plt.scatter(delt_cort, mu_b2_diff_HRV_TINN)
plt.xlabel('Δ Salivary Cortisol (ug/dL)')
plt.ylabel('Δ TINN Index Between Music and B2')
r, p = pearsonr(delt_cort, mu_b2_diff_HRV_TINN)
plt.annotate('r = {:.2f}'.format(r), xy=(0.7, 0.9), xycoords='axes fraction')
z = np.polyfit(delt_cort, mu_b2_diff_HRV_TINN, 1)
p = np.poly1d(z)
plt.plot(delt_cort, p(delt_cort), linestyle='dashed', color='lightsteelblue')
plt.show()
#
# plt.scatter(b1_b2_diff_HRV_LF, listen_soph, color='blueviolet')
# plt.xlabel('Δ ECG LF (0.04-0.15 Hz) Power Between B1 and B2')
# plt.ylabel('Listening Sophistication (MUSEBAQ)')
# r, p = pearsonr(b1_b2_diff_HRV_LF, listen_soph)
# plt.annotate('r = {:.2f}'.format(r), xy=(0.7, 0.9), xycoords='axes fraction')
# z = np.polyfit(b1_b2_diff_HRV_LF, listen_soph, 1)
# p = np.poly1d(z)
# plt.plot(b1_b2_diff_HRV_LF, p(b1_b2_diff_HRV_LF), linestyle='dashed', color='purple')
# plt.show()
#
# plt.scatter(delt_cort, cog_reg, color='blueviolet')
# plt.xlabel('Δ Salivary Cortisol (ug/dL)')
# plt.ylabel('Cognitive Regulation (MUSEBAQ)')
# r, p = pearsonr(delt_cort, cog_reg)
# plt.annotate('r = {:.2f}'.format(r), xy=(0.7, 0.9), xycoords='axes fraction')
# z = np.polyfit(delt_cort, cog_reg, 1)
# p = np.poly1d(z)
# plt.plot(delt_cort, p(delt_cort), linestyle='dashed', color='purple')
# plt.show()

# plt.scatter(mu_b2_diff_HRV_CVI, emotion_reg, color='black')
# plt.xlabel('Δ Cardiac Vagal Index (CVI) Between Music and B2')
# plt.ylabel('Emotion Regulation (MUSEBAQ)')
# r, p = pearsonr(mu_b2_diff_HRV_CVI, emotion_reg)
# plt.annotate('r = {:.2f}'.format(r), xy=(0.7, 0.9), xycoords='axes fraction')
# z = np.polyfit(mu_b2_diff_HRV_CVI, emotion_reg, 1)
# p = np.poly1d(z)
# plt.plot(mu_b2_diff_HRV_CVI, p(mu_b2_diff_HRV_CVI), linestyle='dashed', color='gray')
# plt.show()

# print(len(multi_test))
# bonferroni_ps = multipletests(multi_test)
# print(bonferroni_ps[1])
#
# for i, p in enumerate(bonferroni_ps[1]):
#     if p < 0.8:
#         print(p)
#         print(track_corrs[i])





