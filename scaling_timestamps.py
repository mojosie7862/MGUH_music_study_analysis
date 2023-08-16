import numpy as np
import pandas as pd
import mne
ch_names = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']
sfreq = 256
eeg_data = pd.read_csv('recording_2023-03-09-14.55.41.csv', dtype={'timestamps':float}).to_numpy()
timestamps = eeg_data[:, 0]
t1=0
for i, t in enumerate(timestamps):
    print(t, t-t1)
    t1 = t


'''eeg_data = eeg_data.transpose()[1:9]
eeg_data = eeg_data / 1000000  # BrainFlow returns uV, convert to V for MNE

info = mne.create_info(ch_names, 256)
raw = mne.io.RawArray(eeg_data, info)
raw.plot()'''
