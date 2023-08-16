import mne
import numpy as np

# need to run python -i script.py for MNE matplotlib interactive plots in nme

# sample_data_folder = mne.datasets.sample.data_path()
# sample_data_raw_file = (sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif")
sample_data_raw_file = 'NHC-16R0_EPOCX_183950_2023.07.13T09.55.38.04.00.edf'
raw = mne.io.read_raw_edf(sample_data_raw_file)
print('# of timepoints', raw.n_times)
epoc_ch_names = [ 'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
bands = {'Delta (0-4 Hz)': (0, 4), 'Theta (4-8 Hz)': (4, 8),
         'Alpha (8-12 Hz)': (8, 12), 'Beta (12-30 Hz)': (12, 30),
         'Gamma (30-45 Hz)': (30, 45)}

t_int = 300

#drop non-channel streams from channels
drop_ch = []
for ch in raw.info.ch_names:
    if ch not in epoc_ch_names:
        drop_ch.append(ch)
raw.drop_channels(drop_ch)


raw.compute_psd(fmax=50, tmax=1639-t_int, tmin=t_int).plot()
# raw.plot(duration=5, n_channels=14)

# events = mne.find_events(raw, stim_channel="STI 014")
# print(events[:5])  # show the first 5