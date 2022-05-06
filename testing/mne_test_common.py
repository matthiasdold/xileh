import mne
import numpy as np


def get_mne_test_data():

    # prepare some random data as mne entities, as downloading the samples folder
    # (about 1.65Gb) is just to big / potentially unnecessary for these tests
    chs = ['Cz', 'Fz', 'Pz']
    info = mne.create_info(chs, 100, ch_types='eeg')
    raw = mne.io.RawArray(np.random.randn(3, 10000), info, verbose=False)
    tev = np.arange(40, 9300, 300)
    events = np.asarray([tev, np.zeros(tev.shape), np.ones(tev.shape)],
                        dtype='int').T
    events[::2, 2] = 2
    ev_ids = {'left': 2, 'right': 1}
    epochs = mne.Epochs(raw, events, event_id=ev_ids, tmin=-0.3, tmax=0.7,
                        preload=True, verbose=False)

    return {'raw': raw, 'epos': epochs}


