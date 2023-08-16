import os
import mne
import util
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import pandas as pd
from util import get_markers, subset_intervals, print_dict

marker_dict = {}
aim_data_directory = 'MICU_AIM_data'

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
df_list = []
# for filename in os.listdir(aim_data_directory):
filename = 'NHC-5LK3.vhdr'
f_suffix = filename[-5:]
f_id = filename.replace(f_suffix, '')
abnormal_QRS = ['NHC-AECK', 'NHC-X0AH']

    # if f_id in abnormal_QRS:
    #     continue

if f_suffix == '.vhdr':
    print(filename)
    vhdr_file = os.path.abspath(aim_data_directory + '/' + filename)
    raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)
    raw_df = raw.to_data_frame()

    # else:
    #     continue
    # split intervals
b1_raw, music_raw, b2_raw = subset_intervals(raw_df, marker_dict[f_id][0], marker_dict[f_id][1],
                                             marker_dict[f_id][2], marker_dict[f_id][3])

print('b1', len(b1_raw))
print('music', len(music_raw))
print('b2', len(b2_raw))

b1_rsp_signal = b1_raw["Respiration"][0:5000]
music_rsp_signal = music_raw["Respiration"][0:5000]
b2_rsp_signal = b2_raw["Respiration"][0:5000]

# b1_rsp_cleaned = nk.rsp_clean(b1_rsp_signal)
signals = pd.DataFrame({
    "RSP_Raw": b1_rsp_signal,
    "RSP_Khodadad2018": nk.rsp_clean(b1_rsp_signal, sampling_rate=50, method="khodadad2018"),
    "RSP_BioSPPy": nk.rsp_clean(b1_rsp_signal, sampling_rate=50, method="biosppy"),
    "RSP_Hampel": nk.rsp_clean(b1_rsp_signal, sampling_rate=50, method="hampel", threshold=3)
})
signals.plot()
plt.show()


'''
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

# transform NNs to hrv indices
b1_hrv_indices = nk.hrv(b1_ecg_peaks, sampling_rate=500)
music_hrv_indices = nk.hrv(music_ecg_peaks, sampling_rate=500)
b2_hrv_indices = nk.hrv(b2_ecg_peaks, sampling_rate=500)

# add identifiers and interval of waveform
# b1_hrv_indices.insert(loc=0, column='patient_id', value=patient_id)
# b1_hrv_indices.insert(loc=1, column='trial_id', value=f_id)
b1_hrv_indices.insert(loc=0, column='participant_id', value=f_id)
b1_hrv_indices.insert(loc=1, column='interval', value='b1')
b1_hrv_indices.insert(loc=2, column='ecg_quality', value=b1_ecg_quality)

# music_hrv_indices.insert(loc=0, column='patient_id', value=patient_id)
# music_hrv_indices.insert(loc=1, column='trial_id', value=f_id)
music_hrv_indices.insert(loc=0, column='participant_id', value=f_id)
music_hrv_indices.insert(loc=1, column='interval', value='music')
music_hrv_indices.insert(loc=2, column='ecg_quality', value=music_ecg_quality)

# b2_hrv_indices.insert(loc=0, column='patient_id', value=patient_id)
# b2_hrv_indices.insert(loc=1, column='trial_id', value=f_id)
b2_hrv_indices.insert(loc=0, column='participant_id', value=f_id)
b2_hrv_indices.insert(loc=1, column='interval', value='b2')
b2_hrv_indices.insert(loc=2, column='ecg_quality', value=b2_ecg_quality)

df_list.append(b1_hrv_indices)
df_list.append(music_hrv_indices)
df_list.append(b2_hrv_indices)



dfs = pd.concat(df_list)
dfs.to_csv('NHC_AIM_Analysis.csv')'''
