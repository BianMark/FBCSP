import mne
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np


def get_psds(data, fs=250, f_range=[4, 40]):
    '''
    Calculate signal power using Welch method.

    Input: data- mxn matrix (m: number of channels, n: samples of signals)
           fs: Sampling frequency
           f_range: Frequency range
    Output: Power values and PSD values
    '''
    powers = []
    psds = list()
    for sig in data:
        freq, psd = signal.welch(sig, fs)
        idx = np.logical_and(freq >= f_range[0], freq <= f_range[1])
        powers = np.append(powers, sum(psd[idx]))
        psds.append(psd[idx])

    return powers, psds


# Import BCIC4-2a Data
filePath = 'E:\\EEG\\DataSets\\BCI Competition IV\\BCICIV_2a_gdf'
Subject = 1
fileName = 'A0' + str(Subject) + 'T.gdf'
eeg_data = mne.io.read_raw_gdf(filePath + '/' + fileName)  # load the raw dataset
fs = eeg_data.info.get('sfreq')
ch_names = eeg_data.ch_names
new_ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
ch_positions = np.array([[0, 85], [-30, 55], [-30, 70], [0, 70], [30, 70], [30, 55], [-45, 45], [-30, 35], [-30, 20], [0, 20], [30, 20], [30, 35], [45, 45],
                         [-30, 0], [-30, -15], [0, -15], [30, -15], [30, 0], [-45, -45], [0, -50], [45, -45], [0, -85]])
mapping = dict(zip(ch_names, new_ch_names))
eeg_data.rename_channels(mapping)  # re-name the channels
eeg_data.pick_channels(ch_names=new_ch_names)  # exclude 3 EOG channels

# Extract epochs and create montage
montage = mne.channels.make_standard_montage('standard_1005')
eeg_data.set_montage(montage)
eeg_data.set_eeg_reference('average', projection=True)
events, events_id = mne.events_from_annotations(eeg_data, regexp="769|770|771|772")
epochs = mne.Epochs(eeg_data, events, events_id, tmin=0.5, tmax=2.5, baseline=None, preload=True)
all_epochs_data = epochs.get_data()*1e6
events_label = events[..., 2]

# Plot the topograph
cmap = 'RdBu_r'
num_label = len(events_id)
num_cols = 2  # 2 sub-plot in each row
num_rows = (num_label + 1) // num_cols  # the number of rows needed for all the sub-plot
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, num_rows * 4))  # create the big fig to include all the sub-plot
axes = axes.flatten()  # convert to 1-D

for idx, label in enumerate(events_id):
    target_event_id = label
    epochs_target = epochs[target_event_id].copy().get_data() * 1e6
    channel_data = np.reshape(epochs_target, (22, -1))
    pwrs, _ = get_psds(channel_data, fs)

    ax = axes[idx]
    im, _ = mne.viz.plot_topomap(pwrs, ch_positions, axes=ax, show=False, cmap=cmap, names=new_ch_names, vlim=(np.min(pwrs), np.max(pwrs)))
    ax.set_title("Event: " + label)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Power')

for idx in range(num_label, num_rows * num_cols):  # hide extra sub-plot
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.show()
# fig.savefig("topograph.png", bbox_inches='tight')
