import os
import mne


def get_mne_test_data():
    # Prepare some mne data following the example at https://mne.tools/stable/auto_tutorials/evoked/30_eeg_erp.html?highlight=epoch         # noqa
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                        'sample_audvis_filt-0-40_raw.fif')
    raw = mne.io.read_raw_fif(sample_data_raw_file, preload=False)

    sample_data_events_file = os.path.join(
        sample_data_folder, 'MEG', 'sample',
        'sample_audvis_filt-0-40_raw-eve.fif')
    events = mne.read_events(sample_data_events_file)

    raw.crop(tmax=90)  # in seconds; happens in-place
    # discard events >90 seconds (not strictly necessary: avoids some warnings)
    events = events[events[:, 0] <= raw.last_samp]
    raw.pick(['eeg', 'eog']).load_data()
    event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
                  'visual/right': 4, 'face': 5, 'buttonpress': 32}
    epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.3, tmax=0.7,
                        preload=True)

    return {'raw': raw, 'epos': epochs}


