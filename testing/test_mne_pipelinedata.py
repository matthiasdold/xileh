#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# author: Matthias Dold
# date: 20220506
#
# Test only for mne data entities

import os
import mne
import pytest
import tempfile
import numpy as np

from pathlib import Path

from xileh.core.pipelinedata import xPData, from_container
from xileh.core.pipelinedata import CheckedList
from xileh.core.pipelinedata import from_dict

from testing.compare_utils import _compare_container



# ------------------------------ data -----------------------------------------


# Prepare some mne data following the example at https://mne.tools/stable/auto_tutorials/evoked/30_eeg_erp.html?highlight=epoch         # noqa
sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_filt-0-40_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=False)

sample_data_events_file = os.path.join(sample_data_folder, 'MEG', 'sample',
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


@pytest.fixture
def get_test_data():
    tdata = xPData(
        data=np.eye(5),
        header={'name': 'test_data',
                'description': 'Some data description'},
        meta={'mean': 5}
    )
    return tdata


@pytest.fixture
def get_nested_test_data():
    tdata = xPData(
        data=[
            xPData(
                data=[
                    xPData(data=np.eye(3), header={'name': 'test'}),
                    xPData(data=[1, 23, 4],
                           header={'name': 'somename'}),
                    xPData(epochs, name='epos')
                ],
                header={'name': '1st_level_child'},
            ),
            xPData(
                data=[
                    xPData(data=np.eye(3), header={'name': 'test2'}),
                    xPData(data=[1, 23, 4],
                           header={'name': 'somename2'})
                ],
                header={'name': 'string_nest_c'},
            ),
            xPData(
                data=np.ones(3),
                header={'name': 'search_target',
                        'discription': 'We will search for this'}
            ),
            xPData(raw, name='raw'),
            xPData(data=np.zeros(5), header={'name': 'not the target'})
        ],
        header={'name': 'outer_container',
                'description': 'A parent container without name'},
        meta={'some_meta': [1, 2, 3]}
    )
    return tdata




