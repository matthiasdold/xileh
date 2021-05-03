#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# projection methods

import mne
import numpy as np

from pathlib import Path
from xileh.core.pipelinedata import xPData as PData


def basic_projection(pdata, proj_matrix):
    """ The basic projection procedure core to any
    projection

    Parameters
    ----------
    pdata : pipelinedata.PData
        The PData object used as starting point
    proj_matrix : numpy array

    Returns
    -------
    pdata : pipelinedata.PData
        The PData object with its .data property modified
        by projection with proj_matrix

    """

    pdata.data = np.dot(proj_matrix.T, pdata.data, proj_matrix)

    return pdata


def mne_ica_filter(pdata, ica_model, filter_manual_selection=False):
    """ Use a mne ica model to filter dropping channels flagged
    in the model accordingly

    Note: This will transform back to the initial space after
    zeroing certain components

    Parameters
    ----------
    pdata : pipelinedata.PData
        The PData object used as starting point
    ica_model : mne.preprocessing.ICA
        The ICA model defining the target space
    filter_manual_selection : bool, False
        To also filter channels potentially removed manually by
        comparison of ch_names

    Returns
    -------
    pdata : pipelinedata.PData
        The PData object with its .data property modified
        by projection to ica space

    """

    try:
        assert isinstance(pdata.data, mne.io.BaseRaw)
    except Exception as e:
        raise TypeError(f"{e} \n\n For mne_ica_filter the input pdata needs"
                        " to be inherited from mne.io.BaseRaw object to filter"
                        f" using ch_names. pdata.data = {type(pdata.data)}")

    if filter_manual_selection:
        pdata.data.pick_channels(ica_model.ch_names)

    ica_model.apply(pdata.data)

    return pdata


if __name__ == "__main__":
    x = np.random.randn(70, 50)
    data_dir = Path('./testing')
    ica_f = [f for f in data_dir.rglob('*model-ica.fif')][0]
    ica_model = mne.preprocessing.read_ica(ica_f)

    info = mne.create_info(
        ica_model.ch_names
        + [f'c_{i}' for i in range(x.shape[0] - len(ica_model.ch_names))],
        sfreq=100)
    r = mne.io.RawArray(x, info=info)
    r.load_data()

    pda = PData(data=r, header={'desc': 'some test data'}, meta={})

    pdp = mne_ica_filter(pda, ica_model)
