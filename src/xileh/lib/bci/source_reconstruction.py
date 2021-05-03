#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# author: Matthias Dold
# date: 2021-03-25
#
# This module contains source reconstruction processing steps
# for the xileh pipeline, operating on the
#

import mne

from logging import Logger
from xileh.utils.logger import xileh_log_this

from xileh.core.pipelinedata import xPData as PData
from xileh.core.pipeline import xPipeline


@xileh_log_this()                             # provides a logger to func call
def surface_source(pdata, logger=Logger('_')):
    """ Compute the source space confined to a 2d surface

    Parameters
    ----------
    pdata : pipelinedata object
        pipeline data containing eeg / emg recordings
    logger : logging.Logger, optional
        a logger, usually set by the xileh_log_this wrapper


    Returns
    -------
        pipeline data with added meta properties for:
            meta['fwm'] : mne.... TODO
                The forward model

    """

    # check if surface topology is provided

    pass


@xileh_log_this()
def calc_foward_model(pdata, logger=Logger('FW_model')):
    """ Calculate the forward model

    Parameters
    ----------
    pdata : pipelinedata object
        pipeline data containing eeg / emg recordings
    logger : logging.Logger, optional
        a logger, usually set by the xileh_log_this wrapper

    Returns
    -------
        pdata : pipeline data with added data for the forward model
            with header['description'] == 'mne forward model'

    """

    # setup default values extract if in meta
    cfg = pdata.check_meta('fw_cfg',
                           missing={
                               'meg': True,
                               'eeg': False,
                               'mindist': 5.0,
                               'n_jobs': 1,
                               'verbose': False
                           })

    raw_frame = pdata.get_by_name('raw_path')
    trans = pdata.get_by_name('transformation')
    src_2d = pdata.get_by_name('src')
    bem = pdata.get_by_name('bem_solution')

    fwd = mne.make_forward_solution(raw_fname, trans=trans,
                                    src=src_2d, bem=bem,
                                    **cfg
                                    )

    return pdata.data.append(
        PData(
            fwd,
            header={'name': 'forward_model',
                    'description': 'forward model computed by'
                                   ' mne.make_forward_solution'},
            meta={'cfg': cfg}
        )
    )


# TODO Implement the 3d surface


# TODO The forward calculation
# --> compute the forward model and add to the data

if __name__ == "__main__":
    # for source recon with mne, we need:
    # 1) raw data
    # 2) transformation data -> optained from coregistration or
    # 3) source_space -> a source space
    # 4) BEM surface -> a tessalated boundry surface

    # a mne example
    # load mne data samples - note this is ~1.54Gb
    sample_path = mne.datasets.sample.data_path()
    subjects_dir = sample_path + '/subjects'
    epo = 'test'
    subject = 'sample'

    # 1) raw data
    raw_fname = sample_path + '/MEG/sample/sample_audvis_raw.fif'

    # 2) the transformation
    trans = sample_path + '/MEG/sample/sample_audvis_raw-trans.fif'

    # 3) the source space
    src_2d = mne.setup_source_space(subject,
                                    spacing='oct4',
                                    add_dist='patch',
                                    subjects_dir=subjects_dir)

    # 4) the bem model
    conductivity = (0.3,)  # for single layer -> e.g for MEG
    # conductivity = (0.3, 0.006, 0.3)  # for three layers -> e.g for EEG
    model = mne.make_bem_model(subject='sample', ico=4,
                               conductivity=conductivity,
                               subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)

    # package the data
    tdata = PData(
        [
            PData(raw_fname, header={'name': 'raw_path',
                                     'decription': 'For raw.info'}),
            PData(trans, header={'name': 'transformation'}),
            PData(src_2d,
                  header={'name': 'src',
                          'description': '2d surface'},
                  meta={'spacing': 'oct4', 'add_dist': 'patch'}
                  ),
            PData(bem,
                  header={'name': 'bem_solution'},
                  meta={'ico': 4,
                        'conductivity': conductivity}),
        ],
        header={'name': 'forward_data',
                'description': 'Data necessary for computing the fwd solution',
                'processing_function': 'mne.make_forward_solution'
                },
        meta={'fw_cfg': {'meg': True,
                         'eeg': False,
                         'mindist': 5.0,
                         'n_jobs': 1,
                         'verbose': True
                         }
              }
    )

    # setup the pipeline
    xpl = xPipeline('Source_localization_pipelien')
    xpl.add_step(('compute_fwd', calc_foward_model,
                  {'logger': Logger('xpl_fwd_calc')}))

    # evaluate given data
    xpl.eval(tdata)

    if False:
        # quick check with mne only
        fwd = mne.make_forward_solution(raw_fname, trans=trans,
                                        src=src_2d, bem=bem,
                                        meg=True, eeg=False,
                                        mindist=5.0, n_jobs=1, verbose=True)
