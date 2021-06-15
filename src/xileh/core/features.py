#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Feature extraction methods

import os
import catch22
import mat73        # necessary for reading the hdf5 of the hctsa

import pandas as pd
import numpy as np

# from progress.bar import IncrementalBar
from scipy.io import savemat
from pathlib import Path
from tsfresh import extract_features

from xileh.core.pipelinedata import xPData as PData
from xileh.utils.logger import xileh_log_this


@xileh_log_this()
def create_features(pdata, algo='c22', src_container=None, logger=None, **kwargs):
    """ Extract features on the provided pdatas data object
    and stores as meta data inplace of the provied pdata object

    Parameters
    ----------
    pdata : PData
        A pipeline data object containing the relevant time series
    algo : str, default c22
        Algorithm to be used for the feature extraction, viable options are
            c22 => catch22
            hctsa => hctsa using the matlab implementation
            tsfresh => tsfresh features
    src_container : str
        Container name to work on, if None use the outer most
    """

    extr_map = {
        'c22': catch22_extraction,
        'hctsa': hctsa_extraction,
        'tsfresh': tsfresh_extraction
    }

    if src_container is None:
        tsd = pdata.data
    else:
        tsd = pdata.get_by_name(src_container).data

    # check if ts_dim in header (description of which index) to be
    # used to identify single time series - else assume first
    idx = pdata.header['ts_var_dim'] if 'ts_var_dim' in pdata.header else 0

    # make sure data is of shape n_variables x n_times, put
    # the idx to first
    if idx != 0:
        tsd = tsd.transpose(tuple([idx] + [i for i in range(len(tsd.shape))
                                           if i != idx]))

    feat = extr_map[algo](tsd, **kwargs)
    pdata.update_meta(feat)

    return pdata


def hctsa_extraction(ts,
                     hctsa_path='/home/doda/workspace/matlab/hctsa',
                     labels=[],
                     keywords=[],
                     **kwargs):
    """ Use htcs algorithm to extract features

    Parameters
    ----------
    ts : numpy array
        array of time series (n_variables x n_times)

    Returns
    -------
        ften : numpy array
            feature tensor (n_variables x features)
    """

    hctsa_path = Path(hctsa_path)
    labels = (np.array([np.array([f'v_{i}'], dtype=object)
                        for i in range(ts.shape[0])])
              if labels == [] else
              np.array([np.array([la], dtype=object)
                        for la in labels])).T

    keywords = (np.array([np.array([''], dtype=object)
                          for i in range(ts.shape[0])])
                if keywords == [] else
                np.array([np.array([kw], dtype=object)
                          for kw in keywords])).T

    # check dimensions, i.e. one label for each variable
    assert ts.shape[0] == labels.shape[1] == keywords.shape[1]
    # write temp.mat for hctsa to be used -> use the linked TimeSeries dir
    # from within the hctsa dir, set by startup()
    # format is aligned to 'INP_test_ts.mat' in the TimeSeries dir

    # trick numpy to create an array of arrays
    tt = np.empty(labels.shape, object)
    for i, tsi in enumerate(ts):
        tt[0, i] = tsi

    savemat(hctsa_path.joinpath('TimeSeries', 'tmp.mat'),
            {'timeSeriesData': tt,
             'labels': labels,
             'keywords': keywords})

    eng = connect_matlab()
    eng.cd(str(hctsa_path))
    eng.startup(nargout=0)          # note that nargout=0 is needed to it work

    # remove older feature .mat files to avoid dialog
    os.remove(hctsa_path.joinpath('HCTSA.mat'))
    eng.TS_Init_no_dialog('tmp.mat', nargout=0)

    do_parallel = 'true'
    eng.TS_Compute_no_dialog(do_parallel, nargout=0)

    # ---------- Note ---------------
    # In matlabs SQL_Add.m the time series is currently limited to a length
    # of 50k = n_times
    # This could be an issue -> potentially adjust matlab implementation

    # Load mat and transform to dictionary structure
    d = mat73.loadmat(hctsa_path.joinpath('HCTSA.mat'))

    # conserve the dimensions adding a 1 for signifying that the time
    # dimension was averaged over
    data = d['TS_DataMat'].T.reshape(d['TS_DataMat'].T.shape + (1,))
    n = [n for nl in d['OperationsName'] for n in nl]

    return dict(zip(n, data))


def catch22_extraction(ts, **kwargs):
    """ Use the catch22 algorithm to extract features

    Parameters
    ----------
    ts : numpy array
        array of time series (n_variables x n_times)

    Returns
    -------
        ften : numpy array
            feature tensor (n_variables x features)
    """

    extractor = catch22.catch22_all

    ftens = []
    n = []
    dstore = np.empty((1, 1))       # will be recreated correctly later

    for i, var in enumerate(ts):
        c22 = extractor(var)
        c22['names'].append('var')
        c22['values'].append(i)

        if i == 0:
            # initialize the data array
            n = c22['names']
            d = np.array(c22['values'])
            dstore = np.empty((d.shape[0], ts.shape[0]))

        # make sure it is always the same features
        assert n == c22['names']

        dstore[:, i] = c22['values']

    # combine the data and cast to df
    # assert all names are aligned properly
    assert all([ftens[i]['names'] == ftens[i + 1]['names']
                for i in range(len(ftens) - 1)])

    # data in the form n_feat x n_variables
    # add an empty dim for n_times
    dstore = dstore.reshape(dstore.shape + (1,))
    n22 = ['catch22__' + s for s in n]

    return dict(zip(n22, dstore))


def tsfresh_extraction(ts, id_col_grps={}, **kwargs):
    """ Use the tsfresh algorithm to extract features

    Parameters
    ----------
    ts : numpy array
        array of time series (n_variables x n_times)
    id_col_grps : dict
        dict of groups labels and variable names
        for the variables to consider different entities,
        e.g. epochs in a EEG experiment, or different sensors

        e.g. if position (s) and speed (v) in two coordinates (x, y)
        for 2 robots (r1, r2), ts should be sorted in the first dim as
        r1_s_x, r1_s_y, r1_v_x, r1_v_y, r2_s_x, ...

        the id_col_grps would the encode the robots and look like
        {'grps': [1, 1, 1, 1, 2, 2, 2, 2], 'vars': ['s_x, s_y, v_x, v_y']}

        Note that we would require the variables within each group
        to be aligned, e.f. s_x, s_y, v_x, v_y in exactly this order

        If id_col_grps is left empty, assume each col is a separate
        entity


    Returns
    -------
        ften : numpy array
            feature tensor (n_variables x features)
    """

    # tsfresh works on pandas with time and as row index
    df = pd.DataFrame(
        ts.T,
        columns=[f"var_{i}" for i in range(ts.shape[0])]
    )

    drop_prefix = False
    if id_col_grps == {}:
        dfu = (df.unstack().reset_index()
               .rename(columns={'level_0': 'id', 'level_1': 'time'}))

        # if we do not provide grps, i.e. each row of ts corresponds
        # to an independent variable (e.g. channel), dfu would be a
        # series with one column named '0' in this case drop the pref
        # in the generated feature names
        drop_prefix = True

    else:

        # validate that multiplicity for each of the groups is the same
        grp_c = np.unique(id_col_grps['grps'], return_counts=True)
        assert all(grp_c[1] == len(id_col_grps['vars']))

        cidx = pd.MultiIndex.from_tuples(
            zip(id_col_grps['grps'], id_col_grps['vars'] * len(grp_c[0])),
            names=['grps', 'vars'])

        df.columns = cidx
        dfu = (df.T.unstack('grps').T
               .reset_index(level=1)
               .reset_index()
               .rename(columns={'index': 'time', 'grps': 'id'})
               )

        dfu.columns.name = ''

    dff = extract_features(dfu, column_id='id', column_sort='time')

    if drop_prefix:
        dff = dff.rename(columns=dict(zip(dff.columns,
                                          [c[3:] for c in dff.columns])))

    n = ['tsfresh__' + s for s in dff.columns.to_list()]
    d = dff.to_numpy().T
    d = d.reshape(d.shape + (1,))

    return dict(zip(n, d))


def connect_matlab():
    """ connect to matlab engine

    Returns
    -------
    eng : matlab.engine
        the matlab processing engine
    """
    import matlab.engine

    print(" Connecting to matlab engine")
    eng = matlab.engine.start_matlab()

    return eng


if __name__ == "__main__":
    x = np.random.randn(3, 50)
    pda = PData(data=x, header={'desc': 'some test data'}, meta={})
    create_features(pda, algo='c22')
    create_features(pda, algo='tsfresh')
