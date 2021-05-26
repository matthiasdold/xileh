#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# author: Matthias Dold
# date: 20210503
#
# Wrapper module for scipy.signals filter including

import numpy as np
from scipy import signal

from xileh.core.pipelinedata import xPData
from xileh.core.pipeline import xPipeline


def bandfilter(pdata, src_container='', mode='append',
               suffix='bandfiltered', lfilter_cfg={}):
    """ The bandfiltered info

    Parameters
    ----------
    pdata : xPData
        the data container to hand over
    src_container : str
        name of container to work with
    mode : str, (append, inplace)
        whether to modify the container or append a suffixed version
    suffix : str
        suffix to use if mode=append
    lfilter_cfg : dict, optional
        parameters for scipy.signal.lfilter, needs to contain 'a', 'b'
        and 'zi' with appropriate arrays. The zi will be updated if zi
        is not none

    Returns
    -------
    pdata : xPData
        the modified or appended container
    """

    xc = pdata.get_by_name(src_container)

    # if append, create new trg
    if mode == 'append':
        new_name = xc.header['name'] + '_' + suffix

        # Note: this will raise an error if pdata.data is not a list
        # this is correct as we would only to append to a container that
        # is capable of append -> saves an extra "if" check
        pdata.data.append(
            xPData(
                np.empty(xc.data.shape),
                header={'name': new_name},
                meta={'lfilter_cfg': lfilter_cfg}
            )
        )

        trg = pdata.data[-1]        # shortcut since we just appended
    else:
        trg = xc

    a = lfilter_cfg['a']
    b = lfilter_cfg['b']
    zi = lfilter_cfg['zi']

    if zi is None:
        # The matlab version would have been initialized with ones
        zi = np.ones((a.shape[0]-1,))
        trg.data, _ = vectorized_filter(xc.data, b, a, zi)
    else:
        trg.data, lfilter_cfg = vectorized_filter(xc.data, b, a, zi)

    return pdata


def downsample(pdata, src_container='', mode='append',
               suffix='downsampled', downsampling_cfg={}):
    """ The bandfiltered info

    Parameters
    ----------
    pdata : xPData
        the data container to hand over
    src_container : str
        name of container to work with
    mode : str, (append, inplace)
        whether to modify the container or append a suffixed version
    suffix : str
        suffix to use if mode=append
    downsampling_cfg : dict, optional
        parameters for scipy.signal.resample_poly, needs to contain
        'from_f' and 'to_f'

    Returns
    -------
    pdata : xPData
        the modified or appended container
    """

    xc = pdata.get_by_name(src_container)

    # if append, create new trg
    if mode == 'append':
        new_name = xc.header['name'] + '_' + suffix

        # Note: this will raise an error if pdata.data is not a list
        # this is correct as we would only to append to a container that
        # is capable of append -> saves an extra "if" check
        pdata.data.append(
            xPData(
                np.empty(xc.data.shape),
                header={'name': new_name},
                meta={'downsampling_cfg': downsampling_cfg}
            )
        )

        trg = pdata.data[-1]        # shortcut since we just appended
    else:
        trg = xc

    trg.data = vectorized_downsampling(xc.data, from_f=downsampling_cfg['from_f'],
                                       to_f=downsampling_cfg['to_f'])

    return pdata


def vectorized_filter(x, b, a, zi):
    """ The scipy filter is only able of dealing with 1d vectors
        use this wrapper to track n dimensional states via partial()
        Not really vectorized of course

    Parameters
    ----------
    x : np.array, (ntimes, nchan)
        data matrix
    b : np.array
        filter parameters b
    a : np.array
        filter parameters a
    zi : np.array, (max(len(a), len(b))-1, nchan)
        filter states separate per channels

    Returns:
    --------
        fdata : np.array, (ntimes, nchan)
            the filtered data per channel
        zi : np.array, (max(len(a), len(b))-1, nchan)
            the updated filter state
    """
    if len(x.shape) < 2:
        x = x.reshape((-1, 1))

    fdata = np.empty(x.shape)

    # TODO: This is the point where parallization could come in
    # check latest rumba etc for doing this in parallel
    for i, rec in enumerate(x.T):
        # get filtered data and persist the filter state for later use
        fdata[:, i], zi[:, i] = signal.lfilter(b, a, rec, zi=zi[:, i])

    return fdata, zi


def vectorized_downsampling(x, from_f, to_f):
    """ Perform a 0 phase downsampling, to match the matlab bbci proc_resampling
    This is using a polyphase filtering with a kaiser window

    Parameters
    ----------
    x : np.array, (ntimes, nchans)
        data matrix
    from_f : int
        sampling frequency of x
    to_f : int
        target frequency to downsample to

    Returns:
    --------
        fdata : np.array, (ntimes, nchans)
            the resampled data
    """
    # use the scipy downsampling with 0 phase to match bbci proc_resampling
    fdata = signal.resample_poly(x, to_f, from_f, axis=0)

    return fdata


if __name__ == '__main__':

    # The underlying functions
    x = np.ones((50, 5))
    fband = [10, 14]
    order = 5
    sfreq = 200
    b, a = signal.butter(order, np.array(fband)/(sfreq/2), btype='bandpass')
    zi = np.ones((max(len(a), len(b))-1, x.shape[1]))

    fdata, zi = vectorized_filter(x, b, a, zi)

    to_f = 50
    from_f = 200
    dsdata = vectorized_downsampling(x, from_f, to_f)

    # Example within a pipeline
    pline = xPipeline('filter_and_downsample')
    pline.add_step(('bandfiltering_with_bw', bandfilter,
                    {'src_container': 'raw_data',
                     'suffix': 'bf',
                     'lfilter_cfg': {'a': a, 'b': b, 'zi': zi}}))

    pline.add_step(('downsample_filtered', downsample,
                    {'src_container': 'raw_data_bf',
                     'downsampling_cfg': {'to_f': 20, 'from_f': 200}}))

    # pack the data

    pdata = xPData(
        [
            xPData(x, header={'name': 'raw_data'})
        ],
        header={'name': 'outer_container'}
    )

    pline.eval(pdata)
